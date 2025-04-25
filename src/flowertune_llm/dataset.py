from typing import Tuple

import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset, load_dataset
from trl import DataCollatorForCompletionOnlyLM

from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset

FDS = None  # Cache FederatedDataset

def formatting_prompts_func(example):
    output_texts = []
    # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for i in range(len(example["instruction"])):
        text = f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n### Output: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

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

def get_seq2seq_data_collator(tokenizer, peft_model):
    """
    Seq2Seq data collator
    """

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

def get_causallm_data_collator(tokenizer):

    # ignore tokenizer pad token in the loss
    # label_pad_token_id=-100

    response_template = "Output:"

    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False) 

    # padding the sentence of the entire datasets
    data_collator=DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, 
        response_template=response_template_ids,
        pad_to_multiple_of=8,
    )
    
    return data_collator



def get_tokenizer(model_name):
        """
        Get tokenizer
        """

        global tokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto", 
            use_fast=True,
            padding_side="right"
            )

        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

def get_tokenizer_and_data_collator_and_propt_formatting(model_name: str):
    # From: https://huggingface.co/docs/trl/en/sft_trainer

    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    response_template_with_context = "\n### Response:"  # alpaca response tag
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    return tokenizer, data_collator, formatting_prompts_func

def train_valid_test_split(dataset:Dataset, seed:int=None):
    """
    Split dataset into train (80%), valid (10%) and test set (10%)
    
    Args:
        dataset (Dataset): loaded huggingface dataset
        seed (int): seed value for train_test_split()

    Returns
        train_set (Dataset)
        validation_set (Dataset)
        test_set (Dataset)
    """
    # split dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.2, seed=seed)
    # split test set into valid and test
    validation_test_split = train_test_split['test'].train_test_split(test_size=0.5, seed=seed)

    # New splits
    train_set = train_test_split['train']
    validation_set = validation_test_split['train']
    test_set = validation_test_split['test']

    return train_set, validation_set, test_set

def load_partition_data(partition_id: int, num_partitions: int, dataset_name: str, model_name):
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    client_trainset = FDS.load_partition(partition_id, "train")
    client_trainset = client_trainset.rename_column("output", "response")

    global tokenizer
    tokenizer = get_tokenizer(model_name=model_name)

    tokenized_dataset = client_trainset.map(preprocess_func, batched=True)

    train_set, validation_set, test_set = train_valid_test_split(dataset=tokenized_dataset, seed=42)

    return train_set, validation_set, test_set

def load_data(dataset_name: str) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load data from HuggingFace Hub
    
    Args:
        dataset_name (str): name of dataset in huggingface

    Returns:
        train_set (Dataset)
        validation_set (Dataset)
        test_set (Dataset)
    """
    trainset = load_dataset(dataset_name)
    trainset = trainset.rename_column("output", "response")
    
    train_set, validation_set, test_set = train_valid_test_split(dataset=trainset['train'], seed=42)

    return train_set, validation_set, test_set

def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict



def preprocess_func(examples):

    input_texts = [instruction + " " + input for instruction, input in zip(examples["instruction"], examples["input"])]
    target_texts = examples["response"]
    
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