import gc
import logging

import torch
from evaluate import load

# Create a logger specific to this module
logger = logging.getLogger(__name__)

# Function to calculate perplexity using concatenation of fields
def compute_perplexity(batch):
    # Concatenate the fields 'instruction', 'input', and 'response' for each example in the batch
    text = [instr + " " + inp + " " + resp for instr, inp, resp in zip(batch['instruction'], batch['input'], batch['output'])]
    
    # Tokenize the concatenated text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    # Free memory used by 'text'
    del text
    gc.collect()
    
    # Move model to the MPS device if not already
    model.to(device)
    
    # Forward pass through the model without updating weights
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    # Free memory
    del inputs
    
    # Return a list of losses (one per batch element)
    return {"loss": [outputs.loss.item()] * len(batch['instruction'])}



# Load the ROUGE metric from the evaluate library
rouge_metric = load("rouge")

def compute_rouge(batch):
    # Concatenate the fields 'instruction', 'input', and 'response' for each example in the batch
    # references = [inp + " " + instr for instr, inp in zip(batch['instruction'], batch['input'])]
    references = batch['output']
    predictions = batch['predictions']

    # prediction_outputs = [prediction['output'] for prediction in predictions]

    # Compute ROUGE score using the evaluate library
    results = rouge_metric.compute(predictions=predictions, references=references)

    # Free memory
    del predictions, references
    gc.collect()
    
    return {
        "rouge1": [results["rouge1"]] * len(batch['instruction']),
        "rouge2": [results["rouge2"]] * len(batch['instruction']),
        "rougeL": [results["rougeL"]] * len(batch['instruction']),
        "rougeLsum": [results["rougeLsum"]] * len(batch['instruction']),
    }

# Load the BLEU metric from the evaluate library
bleu_metric = load("bleu")

def compute_bleu(batch):
    # Concatenate the fields 'instruction', 'input', and 'response' for each example in the batch
    # references = [inp + " " + instr for instr, inp in zip(batch['instruction'], batch['input'])]
    references = batch['output']
    predictions = batch['predictions']

    # prediction_outputs = [prediction['output'] for prediction in predictions]
    
    # Compute BLEU score using the evaluate library
    results = bleu_metric.compute(predictions=predictions, references=references)

    # Free memory
    # Free memory used by 'references' if necessary
    del references, predictions
    gc.collect()
    
    return {"bleu": [results["bleu"]] * len(batch['instruction'])}

