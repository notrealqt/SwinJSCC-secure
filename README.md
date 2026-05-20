# Secure Semantic Communications with Adversarial Training and Active Eavesdropper

## Overview

This repository contains the implementation of the paper "Secure Semantic Communications with Adversarial Training and Active Eavesdropper".

## Getting Started

### Requirements

- Python 3.8 or later
- PyTorch
- NumPy
- tqdm
- matplotlib
- scikit-learn
- scikit-image

Install requirements with:

```bash
pip install -r requirements.txt
```

### Recommended Setup

```bash
python -m venv venv
venv\Scripts\activate
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

If you use this codebase, please cite the paper:

```bibtex
@INPROCEEDINGS{yourpaper2026,
  author={Your Name and Coauthor Name},
  booktitle={Conference Name 2026},
  title={Secure Semantic Communications with Adversarial Training and Active Eavesdropper},
  year={2026},
  pages={xxx--xxx},
  doi={10.XXXX/XXXXX}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
