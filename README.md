# CCA-RR
Code Context-Aware Reviewer Recommendation

# Dependency
* Python==2.7  
* PyTorch==1.6.0  
* numpy==1.16.3  
* tqdm ==4.48.2

# Code Structure
* attention: Self-attention network and submodule networks.
* method: Representation for file path, source files, and textual information.
* configs.py: Basic configuration for the attention and method folder. Each function defines the hyper-parameters for the corresponding model.
* datasets.py: Dataset loader.
* train.py: Train and validate representation models.
* utils.py: Utilities for models and training.

# Usage
## Data
You can download our shared dataset from Google Drive, the link is: https://drive.google.com/drive/folders/1ZNLJYYBXLuRsm38LYRftyZO7zWrJa3Yx?usp=sharing

## Configuration
Edit hyper-parameters and settings in '''config.py

## Train
>python train --mode train

## Eval
>python train --mode eval
