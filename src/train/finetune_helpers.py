
from rich.console import Console
from dataclasses import dataclass, field
from typing import Optional, List
from datasets import DatasetDict, Dataset
import torch
from transformers import EvalPrediction

console = Console()

@dataclass
class ModelArguments:
    """
    Arguments for creating and preparing the model
    """

    model_name: str = field(
        default="google-t5/t5-small",
        metadata={"help": "The model name or path from the HuggingFace Hub"},
    )

    use_4bit: bool = field(
        default=False,
        metadata={"help": "Activate 4bit precision base model loading"},
    )

    use_nested_quant: bool = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )

    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    
    bnb_4bit_quant_dtype: str = field(
        default="nf4",
        metadata={"help": "Quantization type: fp4 or nf4"},
    )

    lora_alpha: int = field(
        default=16
    )
    
    lora_dropout: float = field(
        default=0.1
    )

    lora_r: int = field(
        default=0.1
    )

@dataclass
class ScriptArguments:
    """
    Arguments for model training and data handling
    """

    per_device_train_batch_size: int = field(
        default=4
    )

    per_device_eval_batch_size: int = field(
        default=4
    )

    auto_find_batch_size: bool = field(
        default=False,
        metadata={"help": "operates with exponential decay, decreasing the batch size in half after each failed run"}
    )

    gradient_accumulation_steps: Optional[int] = field(
        default=4
    )

    eval_accumulation_steps: Optional[int] = field(
        default=4
    )

    learning_rate: Optional[int] = field(
        default=2e-4
    )

    max_grad_norm: Optional[float] = field(
        default=1
    )

    weight_decay: Optional[float] = field(
        default=0.001
    )

    max_seq_length: Optional[float] = field(
        default=512
    )

    dataset_name: Optional[str] = field(
        default="4DR1455/finance_questions",
        metadata={"help": "Finance related dataset"}
    )

    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs to fine-tune the model"}
    )

    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training"}
    )

    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training"}
    )

    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing to create ConstantLengthDataset"}
    )

    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing"}
    )

    optim: Optional[str] = field(
        default="adafactor",
        metadata={"help": "Choice of optimizer"}
    )

    lr_scheduler_type: str = field(
        default="default",
        metadata={"help": "Learning rate schedule. For example, constant/cosine"}
    )

    max_steps: Optional[int] = field(
        default=-1,
        metadata={"help": "Number of optimizer update steps to take"}
    )

    save_steps: int = field(
        default=10,
        metadata={"help": "Save checkpoint every X updates steps."}
    )

    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X update steps"}
    )

    eval_steps: int = field(
        default=10,
        metadata={"help": "Perform evaluation eery X update steps"}
    )

    eval_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": ""}
    )

    warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of steps to perform warmup"}
    )

    group_by_length: bool = field(
        default=True,
        metadata={"help": "Group sequences into batches of the same length to save memory and accelerate training"}
    )

    run_name: Optional[str] = field(
        default="default",
        metadata={"help": "Name for run time"}
    )

    label_names: Optional[List[str]] = field(
        default_factory=lambda: ["labels"],
        metadata={"help": ""}
    )

    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": ""}
    )

    dataloader_num_workers: int = field(
        default=1,
        metadata={"help": ""}
    )

    save_total_limit: Optional[int] = field(
        default=None,
        metadata={"help":""}
    )

    report_to: Optional[str] = field(
        default=None,
        metadata={"help":""}
    )

    save_safetensors: Optional[bool] = field(
        default=False
    )



def seq_length_stats(dataset:DatasetDict):
    """
    Calculate dataset's sequence length statistics

    Args:
        dataset (DatasetDict): dataset dictionary containing 'train' set
    """
    lengths = [len(example['input_ids']) for example in dataset['train']]
    average_length = sum(lengths) / len(lengths)
    median_length = sorted(lengths)[len(lengths) // 2]
    percentile_95 = sorted(lengths)[int(len(lengths) * 0.95)]

    console.log(f"Average length: {average_length}")
    console.log(f"Median length: {median_length}")
    console.log(f"95th percentile length: {percentile_95}")


def check_dataset_examples(tokenized_dataset: Dataset):
    """
    Check if None values are present in tokenized datasets' tensors
    """
    # Validation steps
    console.log("Dataset size:", len(tokenized_dataset))
    console.log("Column names:", tokenized_dataset.column_names)

    # Check for None values and tensor shapes
    for key in ["input_ids", "attention_mask", "labels"]:
        if key not in tokenized_dataset.column_names:
            console.log(f"Warning: {key} not found in dataset")
        else:
            none_count = sum(1 for item in tokenized_dataset[key] if item is None)
            console.log(f"{key} None count: {none_count}")
            if none_count == 0:
                console.log(f"{key} Length:", len(tokenized_dataset[key][0]))

    # console.log a few examples
    console.log("\nFirst 3 examples:")
    for i in range(3):
        console.log(f"\nExample {i+1}:")
        for key in tokenized_dataset.column_names:
            console.log(f"{key}:", tokenized_dataset[i][key])


def compute_metrics(eval_pred: EvalPrediction):
    """
    Function to compute loss metrics for MSE
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logits = torch.tensor(eval_pred.predictions).to(device).float()
    labels = torch.tensor(eval_pred.label_ids).to(device).float()
    
    loss = torch.mean((logits - labels) ** 2).item()
    
    return {"loss": loss}