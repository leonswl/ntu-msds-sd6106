# LLM evaluation

import os
import logging
import torch

import pandas as pd
from datasets import load_from_disk
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

from src.utility.dataset import train_valid_test_split, CustomFederatedDataset
from src.utility.logging import setup_logger
from src.evaluation.evaluate_metrics import (
    compute_bleu,
    compute_rouge
)
from src.settings import HF_SEQUENCE_MODELS


os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "10" 

# hf_model_name = "layonsan/flowertune-llm-google-t5-small"
# hf_model_name = "layonsan/flowertune-llm-google-t5-base"
# hf_model_name = "layonsan/flowertune-llm-tinyllama"

# set variables
dataset_name = "4DR1455/finance_questions"
# num_partitions = 3
num_partitions = 1
# seed = 28
seed = 42

# Create a logger specific to this module
logger = logging.getLogger(__name__)

def calculate_metric(metric_fn, dataset):

    result_dict = {}
    for partition_id in range(num_partitions):
        print(f"Partition: {partition_id}")
        train_set, validation_set, test_set = train_valid_test_split(dataset=dataset[partition_id], seed=seed)

        # Map over the dataset to compute perplexity for each batch
        results = test_set.map(metric_fn, batched=True,batch_size=8)

        result_dict[partition_id] = results

    return result_dict

def main(dataset_dir: str):

    # Check if MPS is available, otherwise use CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    setup_logger(
        log_folder_name="evaluate",
        file_name="evaluate"
    )

    # # Load the model and tokenizer with authentication
    # if hf_model_name in HF_SEQUENCE_MODELS:
    #     model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(hf_model_name)

    # logger.info(f"Model Name: {hf_model_name}")

    # tokenizer = AutoTokenizer.from_pretrained(
    #     hf_model_name, 
    #     use_fast=True,
    #     adding_side="right"
    # )

    # Load dataset
    dataset = load_from_disk(dataset_dir)

    dataset_metrics = dataset.select(range(len(dataset)))

    logger.info("Evaluating BLEU")
    dataset_metrics = dataset_metrics.map(compute_bleu, batched=True, batch_size=64, num_proc=8)
    logger.info("Evaluating ROUGE")
    dataset_metrics = dataset_metrics.map(compute_rouge, batched=True, batch_size=64, num_proc=8)

    # logger.info("Successfully completed evaluation. Commencing cleanup of metrics")
    metrics_df_values = [
        torch.tensor(dataset_metrics['bleu']).mean().item(),
        torch.tensor(dataset_metrics['rouge1']).mean().item(),
        torch.tensor(dataset_metrics['rouge2']).mean().item(),
        torch.tensor(dataset_metrics['rougeL']).mean().item(),
        torch.tensor(dataset_metrics['rougeLsum']).mean().item()
    ]
    metrics_df_columns = ['avg_bleu', 'avg_rouge1', 'avg_rouge2', 'avg_rougeL', 'avg_rougeLsum']

    metrics_df = pd.DataFrame(
        data=[metrics_df_values],
        columns=metrics_df_columns
    )


    # # create partitioner
    # partitioner = IidPartitioner(num_partitions=num_partitions)

    # # create federated dataset using partitioner
    # FDS = FederatedDataset(
    #     dataset=dataset_name,
    #     partitioners={"train": partitioner},
    #     seed=seed,
    #     shuffle=False
    # )

    # # split dataset into partitions
    # partition_trainset_dict = {}
    # for partition_id in range(num_partitions):
    #     partition_trainset = FDS.load_partition(partition_id, "train")
    #     partition_trainset_dict[partition_id] = partition_trainset

    
    # # evaluate bleu
    # bleu_result_dict = calculate_metric(metric_fn=compute_bleu, dataset=dataset)

    # avg_bleu_list = []
    # for partition_id, results in bleu_result_dict.items():
    #     # Calculate the average of BLEU
    #     avg_bleu_list.append(torch.tensor(results['bleu']).mean().item())

    # logger.info("Evaluating ROUGE")
    # # evaluate rouge
    # rouge_result_dict = calculate_metric(metric_fn=compute_rouge, dataset=dataset)

    # logger.info("Successfully completed evaluation. Commencing cleanup of metrics")

    # avg_rouge1_list = []
    # avg_rouge2_list = []
    # avg_rougeL_list = []
    # avg_rougeLsum_list = []

    # for partition_id, results in rouge_result_dict.items():
    #     # Calculate the average of ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum
    #     avg_rouge1_list.append(torch.tensor(results['rouge1']).mean().item())
    #     avg_rouge2_list.append(torch.tensor(results['rouge2']).mean().item())
    #     avg_rougeL_list.append(torch.tensor(results['rougeL']).mean().item())
    #     avg_rougeLsum_list.append(torch.tensor(results['rougeLsum']).mean().item())
    
    # metrics_df = pd.DataFrame({
    #     'avg_bleu': avg_bleu_list,
    #     'avg_rouge1': avg_rouge1_list,
    #     'avg_rouge2': avg_rouge2_list,
    #     'avg_rougeL': avg_rougeL_list,
    #     'avg_rougeLsum': avg_rougeLsum_list
    # }).reset_index(names='partition')

    logger.info(f"Summary of metrics:\n{metrics_df}")

if __name__ == "__main__":
    main(dataset_dir="layonsan/flwr-llm-google-t5-small-fedavgm/data")