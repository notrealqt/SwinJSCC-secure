# Secure Semantic Communications with Adversarial Training and Active Eavesdropper

## Overview

This repository contains the implementation of the paper "Secure Semantic Communications with Adversarial Training and Active Eavesdropper".

## System Model

![System Model](images/system_model.png)

## Getting Started

### Requirements

Install requirements with:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data

Download and preprocess the dataset used in the paper. Example:

```bash
python scripts/download_data.py --dataset <dataset_name> --output-dir data/
python scripts/preprocess_data.py --input-dir data/raw --output-dir data/processed
```

### 2. Train the Secure Semantic System

```bash
python train.py --config configs/secure_jscc.yaml
```

This should train:

- the semantic encoder/decoder pair
- the active eavesdropper network
- the adversarial training loop for confidentiality

### 3. Evaluate Performance

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --config configs/eval.yaml
```

Expected outputs include:

- semantic reconstruction metrics
- bit-level or perceptual quality metrics
- eavesdropper reconstruction accuracy
- secrecy performance curves

### 4. Run Active Eavesdropper Attack

```bash
python attack.py --checkpoint checkpoints/best_model.pth --attack-mode active
```

## Results

Include evaluation results, plots, and tables such as:

- semantic accuracy vs. SNR
- legitimate receiver quality vs. eavesdropper error
- robustness under active attack

## Citation

If you use this codebase, please cite these papers:

```bibtex
@ARTICLE{10589474,
  author={Yang, Ke and Wang, Sixian and Dai, Jincheng and Qin, Xiaoqi and Niu, Kai and Zhang, Ping},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Transformers;Adaptation models;Signal to noise ratio;Convolutional neural networks;Wireless communication;Vectors;Image coding;Joint source-channel coding;Swin Transformer;attention mechanism;image communications},
  doi={10.1109/TCCN.2024.3424842}
}
```

## Acknowledgement

This implementation is built upon and inspired by the [SwinJSCC](https://github.com/semcomm/SwinJSCC) framework.  
We sincerely thank the authors of SwinJSCC for making their code publicly available and supporting open research in semantic communications.
