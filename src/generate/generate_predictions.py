
import logging
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from tqdm import tqdm

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration, AutoConfig
import torch

from src.utility.dataset import train_valid_test_split
from src.utility.logging import setup_logger
from src.settings import HF_SEQUENCE_MODELS

# Create a logger specific to this module
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name:str=None):
    """Function to load model and tokeniser from HF"""
    
    if model_name is None:
        raise("Model Name is empty, please insert a valid model name")

    # Load the model and tokenizer with authentication
    logger.info(f"Loading Model Name from HuggingFace: {model_name}")

    if model_name in HF_SEQUENCE_MODELS:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True,
        adding_side="right"
    )

    config = AutoConfig.from_pretrained(model_name)

    return model, tokenizer, config

class FinetunedModel:
    """
    FinetunedModel

    Attributes
    ------------
        load_split_dataset():  load model and tokeniser from huggingface and perform dataset splitting to obtain test set 
        __generate_prediction(): generate predictions using loaded model against test set of loaded dataset
        update_prediction(): update dataset with generated predictions
        save_dataset_locally(): persist dataset with the generated predictions
    
    """
    def __init__(self, model_name):


        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = None
        self.model_name = model_name

        # set device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Check if MPS is available, otherwise use CPU

        logger.info(f"Device: {self.device}")

        # set model, tokenizer and config
        self.model, self.tokenizer, self.config = load_model_and_tokenizer(model_name=self.model_name)
        self.model = self.model.half() # use lower-precision floating-point formats

        logger.info(f"Successfully instantiated Finetuned Model using {self.model_name}")

    def load_split_dataset(self,dataset_name, seed):
        """Method to load dataset from Huggingface and split"""
        self.dataset_name = dataset_name
        self.seed = seed
        
        # load huggingface dataset
        self.dataset = load_dataset(self.dataset_name)['train']

        # train valid test split
        _, _, self.test_set = train_valid_test_split(dataset=self.dataset, seed=self.seed)

        logger.info("Succesfully split dataset into train, valid and test sets.")

        return self.test_set

    # Define a function to generate predictions for each row
    def __generate_prediction(self, examples):
        """Method to apply prediction generation to each row in the dataset"""

        if self.model_name in HF_SEQUENCE_MODELS:
            max_context_length = getattr(self.tokenizer, "model_max_length", 512)  # Fallback to 512
            logger.info(f"Using tokenizer model_max_length: {max_context_length}")
        else:
            max_context_length = getattr(self.config, "max_position_embeddings", 512)  # Fallback to 512
            logger.info(f"Using config max_position_embeddings: {max_context_length}")
        
        # Ensure max_context_length is reasonable
        if max_context_length > 2048:  # Assume 2048 is a reasonable limit for most models
            logger.warning(f"max_context_length too large ({max_context_length}), falling back to 2048.")
            max_context_length = 2048

        max_prompt_length = int(max_context_length * 0.75)
        max_response_length = max_context_length - max_prompt_length

        # Combine 'instruction' and 'input' as the prompt
        prompts = [
            example["instruction"] + " " + example["input"] + "\nAnswer:"
            for example in examples
            ]
        
        # Tokenize input and generate prediction
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
            padding=True # ensures inputs are batched
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_response_length,  # Ensure total stays within 2048
            repetition_penalty=1.2,  # Penalizes repeated phrases
            # temperature=0.7,  # Controls randomness; lower for more focused answers)  
            no_repeat_ngram_size=3
        )
        # Decode predictions and align with examples
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Trim the prompt from predictions if included
        trimmed_predictions = []
        for prompt, prediction in zip(prompts, predictions):
            # Check if the prediction starts with the prompt
            if prediction.startswith(prompt):
                prediction = prediction[len(prompt):].strip()
            trimmed_predictions.append(prediction)

        examples = examples.map(
            lambda example, idx: {**example, "prediction": trimmed_predictions[idx]}, 
            with_indices=True
        )

        return examples
    
    def update_prediction(self, dataset):
        """Method to update dataset with generated predictions using batch processing and parallelism"""
        logger.info(f"Identified {len(dataset)} prompts from the dataset. Generating predictions...")

        batch_size = 32 # number of rows per batch

        num_batches = (len(dataset) + batch_size -1) // batch_size # total number of batches

        # function to process a single batch
        def process_batch(start_idx):
            end_idx = min(start_idx + batch_size, len(dataset))
            batch = dataset.select(range(start_idx, end_idx))
            
            # Generate predictions for the batch
            updated_batch = self.__generate_prediction(batch)

            # Return the updated batch and the indices it corresponds to
            return updated_batch, range(start_idx, end_idx)

        
        predictions = [None] * len(dataset)  # Preallocate a list for storing predictions in order
        
        # Use ThreadPoolExecutor to process batches in parallel
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(process_batch, i * batch_size)
                for i in range(num_batches)
            ]

            # Use tqdm to show progress as futures complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Batches"):
                updated_batch, indices = future.result()
                for idx, updated_row in zip(indices, updated_batch):
                    predictions[idx] = updated_row["prediction"]

        # Add predictions to the original dataset
        dataset = dataset.add_column("predictions", predictions)

        logger.info("Successfully updated dataset with newly generated predictions")

        return dataset
    
    def save_dataset_locally(self, dataset, model_name):
        """"""

        # Save the updated dataset locally in Arrow format
        current_date = date.today()
        current_time = datetime.now().strftime("%H-%M-%S")
        # save_path = f"data/{current_date}/{current_time}"
        save_path = f"{model_name}/data"

        dataset.save_to_disk(save_path)
        logger.info(f"Successfully persisted updated dataset with predictions in {save_path}")

def main(model_name: str):

    seed = 42

    # setup logger
    setup_logger(
        log_folder_name="generate_predictions",
        file_name="generate_predictions"
    )

    # generate model instance
    finetuned_model = FinetunedModel(model_name=model_name)

    # load and split for test set
    test_set = finetuned_model.load_split_dataset(dataset_name="4DR1455/finance_questions", seed=seed)

    # test_set = test_set.select(range(3))

    # updated test set with generated predictions
    test_set_with_predictions = finetuned_model.update_prediction(dataset=test_set)

    finetuned_model.save_dataset_locally(dataset=test_set_with_predictions, model_name=model_name)

if __name__ == "__main__":
    # model_name = "layonsan/flowertune-llm-google-t5-base"
    # model_name = "layonsan/flowertune-llm-tinyllama"
    # model_name = "layonsan/flowertune-llm-google-t5-small"
    # model_name = "layonsan/flowertune-llm-google-t5-small-fedavgm"
    # model_name = "layonsan/google-t5-small"
    model_name = "layonsan/flwr-llm-google-t5-small-fedavgm"

    main(model_name=model_name)