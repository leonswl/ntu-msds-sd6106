from rich.console import Console
from rich.theme import Theme
from typing import Tuple
import pdb
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    HfArgumentParser
)
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig,
    TaskType
)

from trl import SFTTrainer, SFTConfig

from src.train.callbacks import BatchSizeCallback, MetricsLoggingCallback
from src.train.finetune_helpers import ModelArguments, ScriptArguments

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red"
})

tokenizer = None

def data_collator(tokenizer, peft_model):

    # ignore tokenizer pad token in the loss
    label_pad_token_id=-100

    # padding the sentence of the entire datasets
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=peft_model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    return data_collator


def parse_arguments() -> Tuple[ModelArguments, ScriptArguments]:
    """
    Parse command-line arguments and return them as dataclass instances.

    This function uses the HuggingFace ArgumentParser to convert
    command-line arguments into dataclass instances for easier handling
    of configuration parameters.

    Returns:
        Tuple[ModelArguments, ScriptArguments]: A tuple containing instances
        of ModelArguments and ScriptArguments dataclasses.
    """
    # Create an instance of HfArgumentParser with the argument dataclasses
    parser = HfArgumentParser((ModelArguments, ScriptArguments))

    # Parse command-line arguments and convert them to dataclass instances
    # This method automatically handles argument parsing and type conversion
    return parser.parse_args_into_dataclasses()


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


class FinetuneSeq2SeqLLMs:

    def __init__ (
            self,
            model_args: ModelArguments,
            script_args: ScriptArguments,
            ):
        
        self.model_args= model_args
        self.script_args = script_args

        self.console = Console(theme=custom_theme)

        self.train_set = None
        self.eval_set = None
        self.test_set = None

    def get_peft_config(self):

        # LoRA configuration
        peft_config = LoraConfig(
                r=self.model_args.lora_r,
                lora_alpha=self.model_args.lora_alpha,
                target_modules=["q", "k", "v", "o"],  # Adjust target modules for T5
                lora_dropout=self.model_args.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
        
        return peft_config
    
    def get_tokenizer(self):
        """
        Get tokenizer
        """

        global tokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name,
            load_in_8bit=True,
            device_map="auto", 
            use_fast=True,
            padding_side="right"
            )

        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def get_model(self, peft_config):

        compute_dtype = getattr(torch, self.model_args.bnb_4bit_compute_dtype)
        
        # alert for bfloat16 acceleration support
        if compute_dtype == torch.float16 and self.model_args.use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                self.console.log("=" * 80)
                self.console.log("Current GPU supports bfloat16, training can be accelerated using bfloat16.")
                self.console.log("=" * 80)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.model_args.use_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.model_args.bnb_4bit_quant_dtype,
            bnb_4bit_use_double_quant=self.model_args.use_nested_quant
            )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_args.model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )

        # Freeze the original parameters
        model=prepare_model_for_kbit_training(model)

        peft_model = get_peft_model(model, peft_config)

        self.console.log(peft_model.print_trainable_parameters())

        return peft_model

    
    # Tokenization
    @staticmethod
    def preprocess_func(examples):

        input_texts = [instruction + " " + input for instruction, input in zip(examples["instruction"], examples["input"])]
        target_texts = examples["output"]
        
        # Tokenize input and target texts
        model_inputs = tokenizer(
            input_texts,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
            )
        
        labels = tokenizer(
            target_texts,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
            )
        
        labels["input_ids"] = torch.where(
            labels["input_ids"] == tokenizer.pad_token_id,
            torch.tensor(-100, dtype=labels["input_ids"].dtype),
            labels["input_ids"]
        )
        
        # Convert everything back to Python lists for compatibility with Hugging Face datasets
        model_inputs = {key: val.tolist() for key, val in model_inputs.items()}
        # Add labels to the model inputs
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def tokenize_split_dataset(self, sample: bool=False):
        """
        Apply tokenization and split dataset into train, eval and valid/test sets.

        Args:
            dataset_name (str): name or path of dataset

        Returns:
            train_set (Dataset): training set
            eval_set (Dataset): evaluation set use to evaluate model performance during training
            test_set (Dataset)
        """

        # load dataset
        dataset = load_dataset(self.script_args.dataset_name)

        if sample:
            # Shuffle the dataset and select the first 1/3 samples
            sampled_dataset = dataset['train'].shuffle(seed=42).select(range(17979))
            
        else:
            sampled_dataset = dataset['train'].shuffle(seed=42).select(range(17979)) 
        
        # apply preprocessing on loaded dataset
        # tokenized_dataset = sampled_dataset.map(FinetuneSeq2SeqLLMs.preprocess_func, batched=True, num_proc=round((os.cpu_count() * 0.8)))
        tokenized_dataset = sampled_dataset.map(FinetuneSeq2SeqLLMs.preprocess_func, batched=True)

         # Perform a 80-10-10 split
        train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)  # 60% train, 40% (to split further)

        test_valid_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)  # 20% test, 20% validation

        self.train_set = train_test_split["train"]
        self.eval_set = test_valid_split["train"]
        self.test_set = test_valid_split["test"]

        return self.train_set, self.eval_set, self.test_set
    
    def train(self, peft_model, peft_config, data_collator):
        """
        Build training arguments and trainer to perform training
        """

        # Training arguments
        training_args = SFTConfig(
            output_dir=f"./results/{self.model_args.model_name}",
            eval_strategy=self.script_args.eval_strategy,
            eval_steps=self.script_args.eval_steps,
            save_steps=self.script_args.save_steps,
            logging_steps=self.script_args.logging_steps,
            save_total_limit=2,
            per_device_train_batch_size=self.script_args.per_device_train_batch_size,
            per_device_eval_batch_size=self.script_args.per_device_eval_batch_size,
            auto_find_batch_size=self.script_args.auto_find_batch_size,
            gradient_accumulation_steps=self.script_args.gradient_accumulation_steps,
            eval_accumulation_steps=self.script_args.eval_accumulation_steps,
            gradient_checkpointing=self.script_args.gradient_checkpointing,
            max_steps=self.script_args.max_steps,
            num_train_epochs=self.script_args.num_train_epochs,
            learning_rate=self.script_args.learning_rate,
            warmup_steps=self.script_args.warmup_steps,
            weight_decay=self.script_args.weight_decay,
            fp16=self.script_args.fp16,
            bf16=self.script_args.bf16,
            packing=self.script_args.packing,
            dataloader_num_workers=self.script_args.dataloader_num_workers,
            report_to=self.script_args.report_to,
            load_best_model_at_end=self.script_args.load_best_model_at_end,
            metric_for_best_model="eval_loss",
            run_name=self.script_args.run_name,
            save_safetensors=self.script_args.save_safetensors,
            max_seq_length = self.script_args.max_seq_length,
            lr_scheduler_type=self.script_args.lr_scheduler_type,
            optim=self.script_args.optim,
            label_names=self.script_args.label_names
            )

        # Trainer
        trainer = SFTTrainer(
            model=peft_model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=self.train_set,
            eval_dataset=self.eval_set,
            data_collator=data_collator,
            peft_config=peft_config,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                BatchSizeCallback,
                MetricsLoggingCallback
                ],
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        peft_model.config.use_cache=False

        trainer.train()

if __name__ == "__main__":

    # model_args, script_args = parse_arguments()

    model_args = ModelArguments(
        model_name="google-t5/t5-small",
        use_4bit=True,
        use_nested_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_dtype="nf4",
        lora_alpha=128,
        lora_dropout=0.1,
        lora_r=8
    )

    script_args = ScriptArguments(
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        max_seq_length=512,
        dataset_name="4DR1455/finance_questions",
        num_train_epochs=3,
        fp16=True,
        optim='adafactor',
        lr_scheduler_type='cosine',
        max_steps=10,
        save_steps=10,
        logging_steps=10,
        eval_steps=10,
        warmup_steps=500,
        eval_strategy='steps',
        run_name="finetuning-llm",
        report_to=None,
        label_names=['labels'],
        load_best_model_at_end=True,
        dataloader_num_workers=8,
    )

    # dataset = load_dataset(dataset_name)

    seq2seq_llm = FinetuneSeq2SeqLLMs(model_args=model_args, script_args=script_args)

    tokenizer = seq2seq_llm.get_tokenizer()
    peft_config = seq2seq_llm.get_peft_config()
    peft_model = seq2seq_llm.get_model(peft_config)

    data_collator = data_collator(tokenizer=tokenizer, peft_model=peft_model)

    seq2seq_llm.tokenize_split_dataset()

    seq2seq_llm.train()
