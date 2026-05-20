import numpy as np
import math
import torch
import random
import os
import logging
import time
import json
import csv
from collections import defaultdict


def calculate_euclidean_distance(img1, img2):
    """
    Calculate RMS (Root Mean Square) error between two images.

    Formula: sqrt(mean((I1 - I2)^2))

    Args:
        img1 (torch.Tensor): First image (in [0, 1] range)
        img2 (torch.Tensor): Second image (in [0, 1] range)

    Returns:
        float: RMS error (Euclidean distance normalized by pixel count)
    """
    # Scale images to [0, 255] range to match MSE calculation
    img1_scaled = img1 * 255.
    img2_scaled = img2.clamp(0., 1.) * 255.

    # Calculate RMS error: sqrt(mean((diff)^2))
    diff = img1_scaled - img2_scaled
    rms_error = torch.sqrt(torch.mean(diff ** 2))

    return rms_error.item()


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def logger_configuration(config, save_log=False, test_mode=False):
    # 配置 logger
    logger = logging.getLogger("Deep joint source channel coder")
    if test_mode:
        config.workdir += '_test'
    if save_log:
        makedirs(config.workdir)
        makedirs(config.samples)
        makedirs(config.models)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(config.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    config.logger = logger
    return config.logger

def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model(model, save_path, extra_state=None):
    """Save model checkpoint.

    By default writes a dict with:
      - state_dict: model weights
      - metadata: debugging metadata

    If extra_state is provided, its key/values are merged into the saved dict.
    This is used to store training state for resume (optimizer/global_step/epoch).
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state_dict = model.state_dict()

    metadata = {
        'total_parameters': len(state_dict),
        'encoder_parameters': [k for k in state_dict.keys() if k.startswith('encoder')],
        'decoder_parameters': [k for k in state_dict.keys() if k.startswith('decoder')],
        'channel_parameters': [k for k in state_dict.keys() if 'channel' in k.lower()],
        'distortion_parameters': [k for k in state_dict.keys() if 'distortion' in k.lower()],
    }

    save_dict = {
        'state_dict': state_dict,
        'metadata': metadata,
    }

    if extra_state is not None:
        if not isinstance(extra_state, dict):
            raise TypeError("extra_state must be a dict or None")
        if 'state_dict' in extra_state:
            raise ValueError("extra_state must not contain 'state_dict'")
        save_dict.update(extra_state)

    torch.save(save_dict, save_path)
    print(f"Model saved to {save_path} with {len(state_dict)} parameters")
    print(f"  - Encoder parameters: {len(metadata['encoder_parameters'])}")
    print(f"  - Decoder parameters: {len(metadata['decoder_parameters'])}")
    print(f"  - Channel parameters: {len(metadata['channel_parameters'])}")
    print(f"  - Distortion parameters: {len(metadata['distortion_parameters'])}")


def optimizer_to_device(optimizer, device: torch.device):
    """Move optimizer state tensors to the given device.

    Useful after `optimizer.load_state_dict()` when checkpoint was loaded on CPU.
    """
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
