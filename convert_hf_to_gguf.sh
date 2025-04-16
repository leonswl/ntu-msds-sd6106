#!/bin/bash

# Define paths and variables
model_path="models/google-t5/t5-small/FedAvgM"
outfile="models/google-t5/t5-small/FedAvgM"
outtype="f16"

# Run the conversion script
python3 llama.cpp/convert_hf_to_gguf.py  "$model_path" --outfile "$outfile" --outtype "$outtype"

# Check if the conversion was successful
if [ $? -eq 0 ]; then
    echo "Conversion completed successfully."
else
    echo "Conversion failed."
    exit 1
fi
