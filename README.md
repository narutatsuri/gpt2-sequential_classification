# Binary Sequence Classification with GPT-2

Code inspired from ```gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/```.

## Requirements
Use requirements.txt to install all requirements.
```
pandas==1.4.2
scikit_learn==1.1.2
torch==1.12.1
tqdm==4.64.0
transformers==4.21.1
```

## Setup
Add dataset ```dataset.csv``` to ```data/```. 
```dataset.csv``` must contain two columns, 
the first column corresponding to one label and the 
second corresponding to another label. 

## Running the Code
```
python main.py
```
