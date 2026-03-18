# Guidance for Running HiDe-LLaVA Experiments

This document provides a comprehensive step-by-step guide to running the HiDe-LLaVA experiments from scratch, covering environment setup, data preparation, configuring the scripts, training, and evaluation. It supplements the official README.md with practical fixes for script paths.

## 1. Environment Setup

Begin by creating and activating the Conda environment required for training and evaluation.

```bash
conda create -n hide python=3.10 -y
conda activate hide
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

For evaluation tasks (e.g., measuring caption metrics), install the following packages:
```bash
pip install nltk==3.9.1
pip install pycocotools==2.0.8
pip install pycocoevalcap==1.2
```

*Note:* As recommended in the README, replace the `eval.py` file under `/envs/hide/lib/python3.10/site-packages/pycocoevalcap/` with the `eval.py` provided in the repository to avoid errors.

## 2. Dataset and Pre-trained Weights Preparation

1. **Dataset:** Download the images and instructions for the **UCIT Benchmark** as per the README and organize them into `datasets/` and `instructions/` directories respectively.
2. **Pre-trained Weights:** Download [LLaVA-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) and [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336). 
   - **Important:** Replace the original `config.json` in your downloaded LLaVA directory with the `config.json` provided in this repository.

## 3. Configuring the Scripts (Important Path Updates)

The provided shell scripts contain placeholder absolute paths (`/your_path/`, `/mnt/haiyangguo/...`). You must update these to match your local system.

We provide an automated script to do this for you. Run the `update_paths.py` script from the root of this repository:

```bash
python update_paths.py
```

The script will interactively ask you for the absolute paths to:
1. `llava-v1.5-7b`
2. `clip-vit-large-patch14-336`
3. Your `datasets` folder
4. Your `instructions` folder
5. Where you want to save the `HiDe/` LoRA weights

*Tip: If you have all of these organized inside a **single base folder**, you can skip the interactive prompts by running:*
```bash
python update_paths.py --base_path /your/base/folder
```

This script will automatically update `train_all.sh`, all six training scripts (`Task1.sh` through `Task6.sh`), and the evaluation script (`Eval_all.sh`) with the correct, matching paths.

## 4. Run Training

Once all paths have been correctly configured, you can launch the sequential training pipeline:
```bash
sh ./scripts/HiDe/Train_UCIT/train_all.sh
```
*(Alternatively, you can run each task script individually to monitor logs.)*

## 5. Run Evaluation

After the sequential training completes on all 6 tasks, evaluate the performance across the benchmark:
```bash
sh ./scripts/HiDe/Eval_UCIT/Eval_all.sh
```
