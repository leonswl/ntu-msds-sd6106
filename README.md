# ntu-msds-ai6106
Repository for SD6106 capstone project


### [Convert to GGUF](convert_hf_to_gguf.sh)
Convert finetuned model to gguf format
```
sh convert_hf_to_gguf.sh  
```

### [Upload GGUF to HuggingFace](src/huggingface_api/upload_to_hf.py)
Upload gguf model to HuggingFace
```
python -m src.huggingface_api.upload_to_hf
```


### [Generate Predictions](src/generate/generate_predictions.py)

Load HuggingFace model to generate predictions on dataset test set before persisting updated dataset locally.
```
python -m src.generate.generate_predictions

```

### [Evaluation](src/evaluation/evaluation.py)
Perform evaluation on test dataset. 
```
python -m src.evaluation.evaluation  
```