"""flowertune-llm: A Flower / FlowerTune app."""

import os
import warnings
from typing import Dict, Tuple, Optional
from datetime import datetime, date
import json
from pathlib import Path
import logging
from logging import INFO, DEBUG, ERROR
import pdb

import wandb
import evaluate
import torch
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from flwr.common.logger import log
from omegaconf import DictConfig

from transformers import (
    TrainingArguments, 
    EvalPrediction
)

from trl import SFTTrainer, SFTConfig

from flowertune_llm.dataset import (
    load_partition_data,
    replace_keys,
    get_seq2seq_data_collator,
    get_causallm_data_collator,
    get_tokenizer,
    formatting_prompts_func
)
from flowertune_llm.models import (
    cosine_annealing,
    get_model,
    get_peft_config,
    set_parameters,
    get_parameters,
    compute_metrics,
    SEQUENCE_MODELS
)

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("myFlowertuneLLM")
logger.propagate = False

tokenizer = None
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

CAUSALLM = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
]

SEQ2SEQ = [
    "google-t5/t5-small",
    "google-t5/t5-base"
]


class FlowerClient(NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model_name: str,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        evalset,
        tokenizer,
        num_rounds,
        formatting_prompts_func=None,
        data_collator=None,
        client_id=None,
        strategy_name=None
    ):  # pylint: disable=too-many-arguments
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_arguments = SFTConfig(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.num_rounds = num_rounds
        self.trainset = trainset
        self.evalset = evalset
        self.client_id = client_id
        self.strategy_name = strategy_name
        self.model_name = model_name

        current_date = date.today()
        current_time = datetime.now()
        log_folder_name = current_date.strftime("%Y-%m-%d")
        log_save_path = os.path.join(os.getcwd(), f"logs/{log_folder_name}")
        os.makedirs(log_save_path, exist_ok=True)
        log_file_name = f"{log_save_path}/{current_time.time().strftime('%H-%M')}_client_app.log"
        fl.common.logger.configure(identifier="myFlowertuneLLM", filename=log_file_name)

        # instantiate model
        self.model = get_model(model_cfg)
        self.peft_config = get_peft_config(model_cfg)

        if data_collator:
            self.data_collator = data_collator
        elif self.model_name in SEQ2SEQ:
            self.data_collator = get_seq2seq_data_collator(tokenizer, self.model)
        elif self.model_name in CAUSALLM:
            self.data_collator = get_causallm_data_collator(tokenizer)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""

        current_round = int(config["current_round"])

        wandb.init(
            project="ntu-msds-sd6106",
            # project="test",
            group="federated_learning_experiment",
            name=f"TinyLlama-1.1B-Chat_{self.strategy_name}_client_{self.client_id}_round_{current_round}",
            config={
                "client_id": self.client_id,
                "round": current_round,
                "num_rounds": self.num_rounds,
                # Add any other config parameters you want to track
            },
            reinit=True  # Allow multiple initializations in the same process
        )

        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_arguments.learning_rate = new_lr
        self.training_arguments.output_dir = config["save_path"]
        self.training_arguments.label_names = ["labels"]

        # Construct trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            # max_seq_length=self.train_cfg.seq_length,
            train_dataset=self.trainset,
            eval_dataset=self.evalset,
            formatting_func=formatting_prompts_func,
            data_collator=self.data_collator,
            peft_config=self.peft_config,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        # Do local training
        results = trainer.train()

        eval_results = trainer.evaluate()

        # Combine training and evaluation results
        combined_results = {
            "train_loss": results.training_loss,
            "eval_loss": eval_results["eval_loss"],
            "R1": eval_results["eval_R1"],
            "R2": eval_results["eval_R2"],
            "RL": eval_results["eval_RL"],
            "RLsum": eval_results["eval_RLsum"]
        }

        log(INFO, f"combined_results: {combined_results}")

        return (
            get_parameters(self.model),
            len(self.trainset),
            # {"train_loss": results.training_loss},
            combined_results
        )
    
def compute_metrics(eval_pred: EvalPrediction):
    """
    Function to compute loss metrics using CrossEntropyLoss
    """

    rouge = evaluate.load('rouge')
    
    pred_str = tokenizer.batch_decode(eval_pred.predictions, skip_special_tokens=True)
    labels_ids = eval_pred.label_ids.copy()
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)


    rouge_output = rouge.compute(
        predictions=pred_str,
        references=label_str,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
    )


    return {
        "R1": round(rouge_output["rouge1"], 4),
        "R2": round(rouge_output["rouge2"], 4),
        "RL": round(rouge_output["rougeL"], 4),
        "RLsum": round(rouge_output["rougeLsum"], 4),
    }

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.

    Function obtained from https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
    """

    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def client_fn(context: Context) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    client_id = context.node_config["partition-id"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Let's get the client partition
    client_trainset, client_validationset, client_testset = load_partition_data(partition_id, num_partitions, cfg.dataset.name, cfg.model.name)

    log(INFO, f"Model Name: {cfg.model.name}")

    log(INFO, f"output_dir: {cfg.train.training_arguments}")

    strategy_name = cfg.strategy.name

    global tokenizer
    # if cfg.model.name in SEQUENCE_MODELS:
    tokenizer = get_tokenizer(model_name=cfg.model.name)
        
    return FlowerClient(
        model_cfg=cfg.model,
        train_cfg=cfg.train,
        trainset=client_trainset,
        evalset=client_validationset,
        tokenizer=tokenizer,
        num_rounds=num_rounds,
        client_id=client_id,
        strategy_name=strategy_name,
        model_name=cfg.model.name
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
