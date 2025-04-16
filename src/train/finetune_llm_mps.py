import os
from rich.console import Console
from rich.theme import Theme

import torch
from datasets import load_dataset, DatasetDict
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    EvalPrediction
)
from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
    PeftConfig,
    get_peft_model,
    LoraConfig,
    TaskType
)

from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

# from src.settings import SEQUENCE_MODELS
from src.train.callbacks import BatchSizeCallback, MetricsLoggingCallback
from src.train.finetune_helpers import ModelArguments, ScriptArguments

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red"
})

console = Console(theme=custom_theme)



class FinetuneSeq2SeqLLMs:

    def __init__ (self, model_arguments: ModelArguments, script_arguments: ScriptArguments):
        self.model_arguments = model_arguments
        self.script_arguments = script_arguments

    # Tokenization
    def __preprocess_function_t5(self, examples):
        input_texts = [instruction + " " + inp for instruction, inp in zip(examples["instruction"], examples["input"])]
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
        
        # Add labels to the model inputs
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def __preprocess_function_clm(self, examples):
        prompts = [instruction + " " + inp for instruction, inp in zip(examples["instruction"], examples["input"])]
        targets = examples["output"]

        # Tokenize prompts and targets
        inputs = tokenizer(prompts, text_target=targets, max_length=512, truncation=True, padding="max_length")

        return inputs
    
    def get_peft_config(self, args: ModelArguments):
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit,
            bnb_4bit_compute_dtype=compute_dtype
            bnb_4bit_quant_type=args.bnb_4bit_quant_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant
            )
        
        # alert for bfloat16 acceleration support
        if compute_dtype == torch.float16 and args.use_4bit:
            major, _ = torch.cuda_get_device_capability()
            if major >= 8:
                console.log("=" * 80)
                console.log("Current GPU supports bfloat16, training can be accelerated using bfloat16.")
                console.log("=" * 80)

        # LoRA configuration
        if model_name in SEQUENCE_MODELS:
            peft_config = LoraConfig(
                    r=8,
                    lora_alpha=128,
                    target_modules=["q", "k", "v", "o"],  # Adjust target modules for T5
                    lora_dropout=0.1,
                    bias="none"
                )
        else:
            peft_config = LoraConfig(
                r=16,  # Rank of the low-rank adapters
                lora_alpha=32,  # Scaling factor for the LoRA layers
                target_modules=["q_proj", "v_proj"],  # Specific modules to apply LoRA
                lora_dropout=0.075,  # Dropout rate for the LoRA layers,
                task_type="CAUSAL_LM",
                bias="none"  # Whether to include biases in LoRA layers ("none", "all", "lora_only")
            )

        return peft_config

    def get_model(self):
        # Freeze the original parameters
        model=prepare_model_for_kbit_training(model)
    
    def train(self):

        pass




def data_collator(tokenizer, peft_model):

    # ignore tokenizer pad token in the loss
    label_pad_token_id=-100

    # padding the sentence of the entire datasets
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=peft_model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    return data_collator


# Fine-tuning function
def finetune_model(model_name, dataset, use_lora=False):



    global tokenizer  # Required for the preprocess function
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto", 
        use_fast=True,
        padding_side="right"
        )

    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        #load_in_4bit=True,
        load_in_4bit=False,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )

    if model_name in SEQUENCE_MODELS:
        console.log("Using AutoModelForSeq2SeqLM")
        preprocess_function = preprocess_function_t5

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            quantization_config=bnb_config
        )
    else:
        console.log("Using AutoModelForCausalLM")
        preprocess_function = preprocess_function_clm

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float32,
            quantization_config={"bits": 4} if use_lora else None,
        ).to(device)

    console.log("Instantiated model")
    tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=8)

    # Remove unnecessary columns for training
    columns_to_keep = ["input_ids", "attention_mask", "labels"]

    # Remove additional columns
    # tokenized_dataset = tokenized_dataset.remove_columns(
    #     [col for col in tokenized_dataset.column_names if col not in columns_to_keep]
    # )

    # Perform a 60-20-20 split
    train_test_split = tokenized_dataset.train_test_split(test_size=0.4, seed=42)  # 60% train, 40% (to split further)
    test_valid_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)  # 20% test, 20% validation

    # Combine into a single DatasetDict
    split_dataset = DatasetDict({
        "train": train_test_split["train"],
        "validation": test_valid_split["train"],
        "test": test_valid_split["test"]
    })

    # LoRA configuration
    if use_lora:
        if model_name in SEQUENCE_MODELS:
            peft_config = LoraConfig(
                    r=8,
                    lora_alpha=128,
                    target_modules=["q", "k", "v", "o"],  # Adjust target modules for T5
                    lora_dropout=0.1,
                    bias="none"
                )
        else:
            peft_config = LoraConfig(
                r=16,  # Rank of the low-rank adapters
                lora_alpha=32,  # Scaling factor for the LoRA layers
                target_modules=["q_proj", "v_proj"],  # Specific modules to apply LoRA
                lora_dropout=0.075,  # Dropout rate for the LoRA layers,
                task_type="CAUSAL_LM",
                bias="none"  # Whether to include biases in LoRA layers ("none", "all", "lora_only")
            )
        model = get_peft_model(model, peft_config)

    

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name}",
        evaluation_strategy="steps",
        eval_steps=2,
        save_steps=2,
        logging_steps=5,
        save_total_limit=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=4,
        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        max_steps=2,
        # num_train_epochs=3,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=False,
        bf16=False,
        dataloader_num_workers=2,
        report_to=None,  # Disable logging services like wandb
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )


    def compute_metrics(eval_pred: EvalPrediction):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        logits = torch.tensor(eval_pred.predictions).to(device)
        labels = torch.tensor(eval_pred.label_ids).to(device)
        
        loss = torch.mean((logits - labels) ** 2).item()
        
        return {"eval_loss": loss}

    console.log("Instantiating trainer...")
    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["validation"],
        # data_collator=data_collator,
        formatting_func=formatting_prompts_func,
        peft_config=peft_config,
        packing=False, # create a ConstantLengthDataset
        max_seq_length=256,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            BatchSizeCallback,
            MetricsLoggingCallback
            ],
        compute_metrics=compute_metrics
    )

    # train
    console.log("Starting training")
    trainer.train()

    # save
    model.save_pretrained(f"./results/{model_name}")
    tokenizer.save_pretrained(f"./results/{model_name}")

    console.log("Completed training")

if __name__ == '__main__':
    model_name = "google-t5/t5-small"
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Load dataset
    dataset_name = "4DR1455/finance_questions"
    dataset = load_dataset(dataset_name)

    # T5 fine-tuning
    console.log(f"Fine-tuning {model_name} using {dataset_name} dataset")

    # Select device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.log(f"Using device: {device}")

    finetune_model(
        model_name=model_name,
        dataset=dataset['train'].select(range(int(len(dataset['train']) * 0.1))),
        use_lora=True
        )
