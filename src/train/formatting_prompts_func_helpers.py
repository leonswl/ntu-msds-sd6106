
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        if example['input'][i]:
            text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            {example['instruction'][i]}

            ### Input:
            {example['input'][i]}

            ### Response:
            {example['output'][i]}"""
            
        else:
            text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

            ### Instruction:
            {example['instruction'][i]}

            ### Response:
            {example['output'][i]}"""

        output_texts.append(text)
    return output_texts

def formatting_prompts_func(example):
    output_texts = []
    # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for i in range(len(example["instruction"])):
        text = f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n### Output: {example['output'][i]}"
        output_texts.append(text)
    return output_texts