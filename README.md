# CCA-RR
# CCA-RR
Code Context-Aware Reviewer Recommendation

# Dependency
（看看是不是Latex的语法 如果是的话 直接复制下面的）
\begin{itemize}
    \item Python==2.7
    \item PyTorch==1.6.0
    \item numpy==1.16.3
\end{itemize}
（如果不是的话 你看看怎么弄有那个小点点 然后内容是下面的 反正呈现效果按照我给你的示例网址来）
Python==2.7
PyTorch==1.6.0
numpy==1.16.3

# Code Structure
\begin{itemize}
    \item attention: Self-attention network and submodule networks.
    \item method: Representation for file path, source files, and textual information.
    \item configs.py: Basic configuration for the attention and method folder. Each function defines the hyper-parameters for the corresponding model.
    \item datasets.py: Dataset loader.
    \item train.py: Train and validate representation models.
    \item utils.py: Utilities for models and training.
\end{itemize}

# Usage
## Data
You can download our shared dataset from Google Drive, the link is: https://drive.google.com/drive/folders/1ZNLJYYBXLuRsm38LYRftyZO7zWrJa3Yx?usp=sharing

## Configuration
Edit hyper-parameters and settings in config.py

## Train
python train --mode train

## Eval
python train --mode eval
