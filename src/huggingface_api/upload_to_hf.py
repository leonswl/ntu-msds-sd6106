# Upload fine-tuned model to HF
from typing import Optional
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import HfApi
import torch # need to import torch to reload model

from src.settings import SEQUENCE_MODELS

# Configure the logging system
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def upload_gguf (
            api: HfApi,
            output_dir: str,
            file_name: str,
            repo_id: str
    ):
    """
    Upload GGUF model artifact to HuggingFace repository
    """
        
    file_name = "t5-small-F16.gguf"  
    hf_file_path = f"{output_dir}/{file_name}"

    # Upload file
    api.upload_file(
        path_or_fileobj=hf_file_path,
        path_in_repo=file_name,
        repo_id=repo_id,
        repo_type="model"  # Specify repo type (model, dataset, etc.)
    )


def run(model, output_dir, strategy: Optional[str]=None):
    """
    Args:
        model (str): HF model name
        output_dir (str): output directory containing artifacts. E.g. models/google-t5/t5-small/2025-01-03/02-16-53
        strategy (str): name of federated learning strategy. E.g. fedavg, fedprox, fedavgm, fedadam, fedadagrad
    """

    if strategy:
        model_hf_repo_mapping = {
            "google-t5/t5-small": "layonsan/flwr-llm-google-t5-small",
            "google-t5/t5-base": "layonsan/flowertune-llm-google-t5-base",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "layonsan/flowertune-llm-tinyllama"
        }

        repo_id = f"{model_hf_repo_mapping[model]}-{strategy}"
    
    else:
        model_hf_repo_mapping = {
            "google-t5/t5-small": "layonsan/google-t5-small",
        }

        repo_id = model_hf_repo_mapping[model]

    logger.info(f"HuggingFace repository id: {repo_id}")

    # Reload model from the saved directory
    output_dir = f"{output_dir}"

    # Load the model and tokenizer
    if model in SEQUENCE_MODELS:
        reloaded_model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
    else:
        reloaded_model = AutoModelForCausalLM.from_pretrained(output_dir)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # Verify the model works by running a test input (optional)
    test_input = reloaded_tokenizer("Hello, how are you?", return_tensors="pt")
    outputs = reloaded_model.generate(**test_input)
    logger.info(reloaded_tokenizer.decode(outputs[0], skip_special_tokens=True))

    # Set up your Hugging Face API and repository details
    api = HfApi()

    # Create a new repository on Hugging Face Hub (if it doesn't exist)
    api.create_repo(repo_id=repo_id, exist_ok=True)

    # upload gguf model to Hugging Face repo
    upload_gguf(
        api=api,
        output_dir=output_dir,
        file_name="t5-small-F16.gguf",
        repo_id=repo_id
    )

    # Push the reloaded model and tokenizer to the repository
    reloaded_model.push_to_hub(repo_id)
    reloaded_tokenizer.push_to_hub(repo_id)

if __name__ == "__main__":

    # specify model
    model = "google-t5/t5-small"
    # model = "google-t5/t5-base"
    # model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # specify strategy
    # strategy = 'fedavg'
    strategy = 'fedavgm'
    
    # output_dir="models/google-t5/t5-small/2025-01-03/08-59-57",
    output_dir = "models/google-t5/t5-small/FedAvgM"

    run(
        model=model,
        output_dir=output_dir,
        strategy=strategy
    )