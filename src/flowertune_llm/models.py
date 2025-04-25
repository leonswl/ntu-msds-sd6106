import math
from logging import INFO, DEBUG, ERROR
import logging
import platform

import evaluate
import numpy as np
import torch
from omegaconf import DictConfig
from collections import OrderedDict
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, EvalPrediction, AutoModelForSeq2SeqLM
# import evaluate
from flwr.common.typing import NDArrays
from flwr.common.logger import log

SEQUENCE_MODELS = [
    "google-t5/t5-base",
    "google-t5/t5-small"
    ]

# Check if the operating system is macOS
platform_system = platform.system()
if  platform_system == "Darwin":
    print("Running on macOS. Skipping BitsAndBytesConfig import.")
else:
    # Safe to import BitsAndBytesConfig or perform other operations
    try:
        from transformers import BitsAndBytesConfig
        print("BitsAndBytesConfig imported successfully.")
    except ImportError:
        print("Failed to import BitsAndBytesConfig.")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """

    # avoid quan
    # if  platform_system not in ["Darwin", "Linux"]:
    if platform_system not in ["Darwin"]:
        if model_cfg.quantization == 4:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif model_cfg.quantization == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(
                f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
            )
    
    logger.info(f"\nModel Name: {model_cfg.name}")
    
    if model_cfg.name in SEQUENCE_MODELS:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_cfg.name,
            # quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )

        # LoRA configuration
        peft_config = LoraConfig(
            r=model_cfg.lora.peft_lora_r,
            lora_alpha=model_cfg.lora.peft_lora_alpha,
            target_modules=["q", "k", "v", "o"],  # Adjust target modules for T5
            lora_dropout=0.075,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )

        peft_config = LoraConfig(
            r=model_cfg.lora.peft_lora_r,
            lora_alpha=model_cfg.lora.peft_lora_alpha,
            lora_dropout=0.075,
            task_type="CAUSAL_LM",
        )
        
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    return get_peft_model(model, peft_config)

def get_peft_config(model_cfg: DictConfig):
    if model_cfg.name in SEQUENCE_MODELS:

        # LoRA configuration
        peft_config = LoraConfig(
                r=model_cfg.lora.peft_lora_r,
                lora_alpha=model_cfg.lora.peft_lora_alpha,
                target_modules=["q", "k", "v", "o"],  # Adjust target modules for T5
                lora_dropout=0.075,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
    else:
        peft_config = LoraConfig(
                r=model_cfg.lora.peft_lora_r,
                lora_alpha=model_cfg.lora.peft_lora_alpha,
                lora_dropout=0.075,
                task_type="CAUSAL_LM",
            )

    return peft_config

def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def get_parameters(model) -> NDArrays:
    """Return the parameters of the current net."""
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]


# # Load metrics using the new library
# bleu = evaluate.load("bleu")
# rouge = evaluate.load("rouge")

# def compute_metrics(eval_pred: EvalPrediction, tokenizer):
#     predictions, labels = eval_pred.predictions, eval_pred.label_ids

#     # Convert logits to predicted token ids (choose the token with the highest probability)
#     # predicted_token_ids = np.argmax(predictions, axis=-1)

#     # Decode the token ids into text
#     # decoded_preds = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
#     # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)

#     # Calculate BLEU and ROUGE
#     # bleu_result = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
#     # rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

#     # Perplexity Calculation
#     # If predictions is a tensor, convert to numpy and compute softmax in a memory-efficient way
#     if isinstance(predictions, torch.Tensor):
#         predictions = predictions.cpu().detach().numpy()

#     # Compute the softmax to get probabilities directly using torch for memory efficiency
#     probs = torch.softmax(torch.tensor(predictions), dim=-1).numpy()  # Convert softmax tensor to numpy array

#     # Gather the probabilities corresponding to the true labels using numpy
#     label_probs = np.take_along_axis(probs, labels[..., None], axis=-1).squeeze()

#     # Mask for padding tokens
#     mask = labels != tokenizer.pad_token_id
#     if mask.any():
#         # Use masked indices for calculating perplexity
#         perplexity = np.exp(-np.mean(np.log(label_probs[mask] + 1e-10)))  # Add a small constant to avoid log(0)
#     else:
#         perplexity = float('inf')  # If no valid labels, set perplexity to infinity

#     return {
#         # "bleu": bleu_result["bleu"],
#         # "rouge1": rouge_result["rouge1"].mid.fmeasure,
#         # "rougeL": rouge_result["rougeL"].mid.fmeasure,
#         "perplexity": perplexity
#     }


# def compute_metrics(eval_pred: EvalPrediction, tokenizer):
#     predictions, labels = eval_pred.predictions, eval_pred.label_ids

#     # Convert predictions to torch tensor if needed
#     if not isinstance(predictions, torch.Tensor):
#         predictions = torch.tensor(predictions)
    
#     # Move to CPU and ensure no gradients are calculated
#     predictions = predictions.cpu().detach()

#     # Mask for padding tokens to avoid unnecessary computation
#     mask = labels != tokenizer.pad_token_id
#     if not mask.any():
#         return {"perplexity": float('inf')}  # Handle case where no valid tokens are present

#     # Compute log probabilities for only the correct labels
#     with torch.no_grad():
#         log_probs = torch.nn.functional.log_softmax(predictions, dim=-1)

#         # Convert labels to tensor for indexing and compute log probabilities for the true labels
#         labels_tensor = torch.tensor(labels, device=log_probs.device)
#         label_log_probs = torch.gather(log_probs, dim=-1, index=labels_tensor.unsqueeze(-1)).squeeze(-1)

#         # Apply mask to ignore padding tokens
#         label_log_probs_masked = label_log_probs[mask]

#         # Calculate perplexity: exp(-mean(log_probs of correct labels))
#         perplexity = torch.exp(-label_log_probs_masked.mean()).item()

#     return {
#         "perplexity": perplexity
#     }

def compute_server_metrics(model, tokenizer, eval_dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Server-side centralised evaluation, metrics include loss and accuracy.
    """
    log(INFO, f"Device: {device}")
    model.to(device)
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_predictions = 0
    total_predictions = 0

    # Iterate through the entire eval_dataloader
    for batch in eval_dataloader:
        # Extract fields from the batch
        instructions = batch['instruction']
        inputs = batch['input']
        responses = batch['response']

        # Concatenate 'instruction' and 'input' with separators
        input_texts = [
            f"{instruction.strip()} {input_text.strip()}"
            for instruction, input_text in zip(instructions, inputs)
        ]
        target_texts = responses

        # Tokenize inputs and targets
        inputs_tokenized = tokenizer(
            input_texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
        )
        targets_tokenized = tokenizer(
            target_texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
        )

        del input_texts, target_texts

        # Move tokenized inputs and targets to the specified device
        inputs_tokenized = {k: v.to(device) for k, v in inputs_tokenized.items()}
        targets_tokenized = {k: v.to(device) for k, v in targets_tokenized.items()}

        # Compute the loss
        with torch.no_grad():
            outputs = model(
                input_ids=inputs_tokenized['input_ids'],
                attention_mask=inputs_tokenized['attention_mask'],
                labels=targets_tokenized['input_ids']
            )
            loss = outputs.loss
            predictions = outputs.logits.argmax(dim=-1)  # Get predicted classes

        # Accumulate the loss and the number of tokens
        total_loss += loss.item() * targets_tokenized['input_ids'].numel()
        total_tokens += targets_tokenized['input_ids'].numel()

        # Calculate correct predictions (ignoring padding tokens)
        mask = targets_tokenized['input_ids'] != tokenizer.pad_token_id  # Create a mask for non-padding tokens
        correct_predictions += (predictions[mask] == targets_tokenized['input_ids'][mask]).sum().item()
        total_predictions += mask.sum().item()

        # Free up memory by deleting variables
        del inputs_tokenized, targets_tokenized, outputs, loss
        torch.cuda.empty_cache()  # Clear unused memory

    avg_loss = total_loss / total_tokens  # Normalize by total tokens
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0  # Calculate accuracy

    return avg_loss, accuracy



def compute_metrics(eval_pred: EvalPrediction, tokenizer):
    """
    Function to compute loss metrics for client using eval/validation dataset
    
    """

    try:
        predictions, labels = eval_pred.predictions, eval_pred.labels
        
        # Convert predictions to logits if they aren't already
        if len(predictions.shape) == 3:
            logits = predictions
        else:
            logits = predictions[0] if isinstance(predictions, tuple) else predictions
        
        # Create a mask for non-padded tokens
        mask = labels != tokenizer.pad_token_id
        
        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Apply mask and calculate mean loss
        masked_loss = loss[mask.view(-1)]
        mean_loss = masked_loss.mean().item()

        log(INFO, f"Completed loss metrics computation: {mean_loss}")

        return {"loss": mean_loss}
    
    except Exception as e:
        log(ERROR, f"Failed to compute loss metrics. Error: {e}")
