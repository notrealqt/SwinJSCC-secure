import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch.optim as optim
from net.network import SwinJSCC, SwinJSCC_adv
from data.datasets import get_loader
from utils import *
from collections import OrderedDict
torch.backends.cudnn.benchmark = True
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import MS_SSIM, SSIM
import time
import torchvision
import numpy as np

from metrics import (
    CSVLogger,
    MetricsAggregator,
    OptionalMetricBackends,
    clip_score_from_images,
    ed_from_images,
    lpips_from_images,
    psnr_from_images,
    ssim_from_images,
    get_model_efficiency,
)
from common_utils.efficiency_logger import log_model_efficiency

parser = argparse.ArgumentParser(description='SwinJSCC')
parser.add_argument('--training', action='store_true',
                    help='training or testing')
parser.add_argument('--trainset', type=str, default='DIV2K',
                    choices=['CIFAR10', 'DIV2K', 'ImageNet'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='kodak',
                    choices=['kodak', 'CLIC21', 'ffhq', 'ImageNet'],
                    help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--model', type=str, default='SwinJSCC_w_SA',
                    choices=['SwinJSCC_w_o_SAandRA', 'SwinJSCC_w_SA', 'SwinJSCC_w_RA', 'SwinJSCC_w_SAandRA'],
                    help='SwinJSCC model type')
parser.add_argument('--channel-type', type=str, default='rayleigh',
                    choices=['awgn', 'rayleigh'],
                    help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=str, default='32', 
                    help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13,16,19',
                    help='random or fixed snr')
parser.add_argument('--model_size', type=str, default='base',
                    choices=['small', 'base', 'large'], help='SwinJSCC model size')
# Metrics logging
parser.add_argument('--metrics-dir', type=str, default='./metrics', help='directory to write metrics CSV files')
parser.add_argument('--metrics-disable-lpips', action='store_true', help='disable LPIPS computation')
parser.add_argument('--metrics-disable-clip', action='store_true', help='disable CLIP Score computation')
# Adversarial training arguments
parser.add_argument('--enable-adversarial', action='store_true',
                    help='Enable adversarial training')
parser.add_argument('--fgsm-epsilon', type=float, default=0.1,
                    help='Adversarial epsilon (fraction of signal RMS in normalized space)')
parser.add_argument('--adversarial-alpha', type=float, default=0.5,
                    help='Alpha weight for clean vs adversarial loss (higher = more clean focus, e.g., 0.8 = 80%% clean, 20%% adversarial)')
# Adversarial attack method (training + testing)
parser.add_argument('--adv-attack', type=str, default='pgd', choices=['fgsm', 'pgd'],
                    help='Adversarial attack method to use (training default and testing fallback)')
parser.add_argument('--pgd-steps', type=int, default=10,
                    help='Number of PGD steps (used when adv-attack=pgd)')
parser.add_argument('--pgd-step-size', type=float, default=None,
                    help='PGD step size (if None, uses epsilon/pgd_steps)')
parser.add_argument('--pgd-random-start', action='store_true',
                    help='Enable PGD random start (uniform in [-epsilon, +epsilon])')

# Adversarial-debug (inner maximization sanity checks)
parser.add_argument('--adv-debug-inner', action='store_true',
                    help='Print inner-maximization attack-loss progression (debug)')
parser.add_argument('--adv-debug-inner-every', type=int, default=0,
                    help='Log inner-max debug every N attack generations (0 = only the first)')
parser.add_argument('--adv-debug-inner-max-steps', type=int, default=0,
                    help='Limit the number of PGD steps printed in debug (0 = all)')

parser.add_argument('--model-path', type=str, default=None,
                    help='Path to specific model file for testing')
parser.add_argument('--resume', type=str, default=None,
                    help='Resume training from a checkpoint (loads model + optimizer + epoch when available)')
parser.add_argument('--pretrained-path', type=str, 
                    default='/home/vtlphuong/Tinh/checkpoints/swin_base_patch4_window7_224.pth',
                    help='Path to pretrained model for fine-tuning')
parser.add_argument('--save-recon', action='store_true',
                    help='Save reconstructed images during testing')

# Adversarial testing argument
parser.add_argument('--adv-eval', action='store_true',
                    help='Enable adversarial evaluation during testing (FGSM/PGD depending on --test-attack/--adv-attack)')
parser.add_argument('--test-epsilon', type=float, default=0.1,
                    help='Attack epsilon for testing (if not specified, uses --fgsm-epsilon)')
parser.add_argument('--test-attack', type=str, default='pgd', choices=['fgsm', 'pgd'],
                    help='Attack method for adversarial testing (default: use --adv-attack)')
parser.add_argument('--attack-mode', type=str, default='before_channel',
                    choices=['before_channel', 'after_channel', 'jamming', 'snr_degradation'],
                    help='Attack mode: before_channel (standard), after_channel (realistic), jamming (military), snr_degradation (stealthy)')
parser.add_argument('--jamming-power', type=float, default=0.1,
                    help='Jamming power multiplier for jamming attack mode (default: 0.1)')
parser.add_argument('--snr-drop', type=float, default=5.0,
                    help='SNR degradation in dB for snr_degradation attack mode (default: 5.0)')

# Artificial Noise (AN) / Bob-Eve evaluation
parser.add_argument('--enable-an', action='store_true',
                    help='Enable artificial noise (Bob can cancel; Eve cannot)')
parser.add_argument('--sigma-an', type=float, default=10.0,
                    help='Std dev of complex artificial noise')
parser.add_argument('--shared-seed', type=int, default=42,
                    help='Shared seed (key) used to generate AN')
parser.add_argument('--train-party', type=str, default='bob', choices=['bob', 'eve'],
                    help='Training role when --enable-an is set: bob cancels AN, eve does not')
parser.add_argument('--test-bob', action='store_true',
                    help='During testing, evaluate Bob role (AN canceled)')
parser.add_argument('--test-eve', action='store_true',
                    help='During testing, evaluate Eve role (AN not canceled)')
parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run')
args = parser.parse_args()

class config():
    seed = 42
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    # logger
    print_step = 1
    plot_step = 10000
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Replace slashes in model name to avoid unwanted subdirectories
    safe_model_name = str(args.model).replace("/", "_")
    run_name = f"{safe_model_name}_{args.trainset}_SNR{args.multiple_snr}_C{args.C}_{timestamp}"
    workdir = f'./history/{run_name}'
    log = f'{workdir}/train.log'
    samples = f'{workdir}/samples'
    models = f'{workdir}/models'
    logger = None
    filename = run_name

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = args.epochs

    if args.trainset == 'CIFAR10':
        save_model_freq = 5
        image_dims = (3, 32, 32)
        train_data_dir = ""
        test_data_dir = ""
        batch_size = 2048
        downsample = 2
        channel_number = int(args.C)
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=channel_number,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=channel_number,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.trainset == 'ImageNet':
        save_model_freq = 2
        image_dims = (3, 224, 224)
        base_path = ""
        train_data_dir = base_path + ""
        test_data_dir = base_path + ""  
        batch_size = 64  
        downsample = 4
        if args.model == 'SwinJSCC_w_o_SAandRA' or args.model == 'SwinJSCC_w_SA':
            channel_number = int(args.C)
        else:
            channel_number = None
            encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=4, in_chans=3,
            embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], C=channel_number,
            window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[768, 384, 192, 96], depths=[2, 6, 2, 2], num_heads=[24, 12, 6, 3], C=channel_number,
            window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 10
        image_dims = (3, 256, 256)
        base_path = ""
        if args.testset == 'kodak':
            test_data_dir = [base_path + "/kodak/"]
        elif args.testset == 'CLIC21':
            test_data_dir = [base_path + "/HR_Image_dataset/clic2021/test/"]
        elif args.testset == 'ffhq':
            test_data_dir = [base_path + "/ffhq/"]

        train_data_dir = [base_path + '/DIV2K/train']
        batch_size = 8
        downsample = 4
        if args.model == 'SwinJSCC_w_o_SAandRA' or args.model == 'SwinJSCC_w_SA':
            channel_number = int(args.C)
        else:
            channel_number = None

        if args.model_size == 'small':
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 2, 2], num_heads=[4, 6, 8, 10], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 2, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
        elif args.model_size == 'base':
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
        elif args.model_size =='large':
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 18, 2], num_heads=[4, 6, 8, 10], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 18, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )

if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).to(config.device)
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).to(config.device)

CalcuSSIM_1 = SSIM(window_size=11, window_sigma=1.5, data_range=1.0, channel=3).to(config.device)

def load_weights(model_path):
    if os.path.exists(model_path):
        # Load on CPU to avoid accidental GPU memory spikes; model is moved later.
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Handle both old format (direct state_dict) and new format (with metadata)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            pretrained = checkpoint['state_dict']
            metadata = checkpoint.get('metadata', {})
            print(f"Loaded checkpoint with metadata from {model_path}")
            if metadata:
                print(f"  - Total parameters: {metadata.get('total_parameters', 'N/A')}")
                print(f"  - Encoder parameters: {len(metadata.get('encoder_parameters', []))}")
                print(f"  - Decoder parameters: {len(metadata.get('decoder_parameters', []))}")
                print(f"  - Channel parameters: {len(metadata.get('channel_parameters', []))}")
                print(f"  - Distortion parameters: {len(metadata.get('distortion_parameters', []))}")
        else:
            # Old format - direct state_dict
            pretrained = checkpoint
            print(f"Loaded old format checkpoint from {model_path}")
        
        # Filter out profiling keys that might be in the checkpoint
        filtered_pretrained = {k: v for k, v in pretrained.items() if not ('total_ops' in k or 'total_params' in k)}
        
        net.load_state_dict(filtered_pretrained, strict=True)
        del pretrained
        print(f"Successfully loaded pretrained weights from {model_path} (with strict=True for full loading)")
    else:
        print(f"Checkpoint not found at {model_path}, continuing without loading pretrained weights")


def _parse_epoch_from_filename(path: str):
    base = os.path.basename(path)
    import re
    m = re.search(r"(?:model|checkpoint)_epoch_(\d+)\.pth$", base)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def infer_workdir_from_checkpoint(ckpt_path: str) -> str:
    """Infer run workdir from a checkpoint path.

    Expected layout: <workdir>/models/<checkpoint>.pth
    Returns <workdir> when it matches, otherwise returns empty string.
    """
    if not ckpt_path:
        return ""
    ckpt_path = os.path.abspath(ckpt_path)
    models_dir = os.path.dirname(ckpt_path)
    if os.path.basename(models_dir) != "models":
        return ""
    return os.path.dirname(models_dir)


def resume_from_checkpoint(*, ckpt_path: str, net, optimizer, device, logger):
    """Resume training state from checkpoint.

    Returns: (start_epoch, global_step)
    """
    if ckpt_path is None:
        return 0, 0
    if not os.path.exists(ckpt_path):
        logger.warning(f"Resume checkpoint not found: {ckpt_path}. Starting fresh.")
        return 0, 0

    logger.info(f"Resuming from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Determine model weights payload
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        # Legacy: raw state_dict
        state_dict = checkpoint

    # Filter out profiling keys that might be in the checkpoint
    state_dict = {k: v for k, v in state_dict.items() if not ('total_ops' in k or 'total_params' in k)}

    try:
        net.load_state_dict(state_dict, strict=True)
        logger.info("Loaded model weights (strict=True)")
    except Exception as e:
        logger.warning(f"Strict load failed ({e}); retrying with strict=False")
        net.load_state_dict(state_dict, strict=False)

    start_epoch = 0
    global_step = 0

    if isinstance(checkpoint, dict):
        # Our training checkpoints save the next epoch to run in `epoch`.
        if 'epoch' in checkpoint:
            try:
                start_epoch = int(checkpoint['epoch'])
            except Exception:
                start_epoch = 0
        if 'global_step' in checkpoint:
            try:
                global_step = int(checkpoint['global_step'])
            except Exception:
                global_step = 0

        # Backward-compat: older checkpoints saved via `save_model()` only contain
        # weights + metadata, but the filename encodes the epoch.
        if 'epoch' not in checkpoint or start_epoch == 0:
            epoch_guess = _parse_epoch_from_filename(ckpt_path)
            if epoch_guess is not None:
                start_epoch = int(epoch_guess)
                logger.info(
                    f"Checkpoint has no saved training epoch; inferred start_epoch={start_epoch} from filename"
                )

        if optimizer is not None and 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                optimizer_to_device(optimizer, device)
                logger.info("Loaded optimizer state")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state ({e}); continuing with fresh optimizer")
    else:
        # Try to infer epoch from filename if it's a model_epoch_X checkpoint.
        epoch_guess = _parse_epoch_from_filename(ckpt_path)
        if epoch_guess is not None:
            start_epoch = int(epoch_guess)
            logger.info(f"Inferred start_epoch={start_epoch} from filename")

    return start_epoch, global_step

def train_one_epoch(args, epoch, optimizer):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    # Add separate meters for adversarial training loss components
    clean_losses, adv_losses, total_losses = [AverageMeter() for _ in range(3)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs, clean_losses, adv_losses, total_losses]
    train_agg = MetricsAggregator()
    backends = OptionalMetricBackends(device=config.device)
    global global_step
    train_is_bob = (not args.enable_an) or (args.train_party == 'bob')
    train_is_eve = bool(args.enable_an) and (args.train_party == 'eve')
    if args.trainset == 'CIFAR10' or args.trainset == 'ImageNet':
        for batch_idx, (input, label) in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.to(config.device)
            # Ensure a clean gradient state before any inner-maximization logic
            # inside the model forward (adversarial example generation).
            optimizer.zero_grad(set_to_none=True)
            # Use adversarial training if enabled
            if args.enable_adversarial:
                recon_image, CBR, SNR, mse, loss_G = net(
                    input,
                    adversarial_training=True,
                    is_bob=train_is_bob,
                    is_eve=train_is_eve,
                )

                # Get separate loss components for adversarial training
                clean_loss, adv_loss, total_loss = net.get_loss_components()
                if batch_idx == 0:  # Log once per epoch
                    logger.info(
                        f"Using adversarial training with attack={args.adv_attack}, epsilon={args.fgsm_epsilon}, alpha={args.adversarial_alpha}"
                    )
            else:
                recon_image, CBR, SNR, mse, loss_G = net(input, is_bob=train_is_bob, is_eve=train_is_eve)
                clean_loss, adv_loss, total_loss = None, None, None
            loss = loss_G
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            # Reset loss components after backprop
            if args.enable_adversarial:
                net.reset_loss_components()

            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)

            # Update separate loss meters if adversarial training
            if clean_loss is not None:
                clean_losses.update(clean_loss.item())
                adv_losses.update(adv_loss.item())
                total_losses.update(total_loss.item())

            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)

            # Metrics for CSV (batch-weighted)
            with torch.no_grad():
                recon = recon_image.clamp(0.0, 1.0)
                ssim_v = ssim_from_images(input, recon, CalcuSSIM_1)
                # Convert MS-SSIM to dB for training log too
                msssim_val = 1 - CalcuSSIM(input, recon).mean().item()
                msssim_db = -10 * np.log10(max(1e-10, 1 - msssim_val))
                
                batch_metrics = {
                    "loss": loss.detach(),
                    "psnr": psnr_from_images(input, recon),
                    "ssim": ssim_v,
                    "msssim_db": msssim_db,
                    "ed": ed_from_images(input, recon),
                }
                if clean_loss is not None:
                    batch_metrics["clean_loss"] = clean_loss.detach()
                if adv_loss is not None:
                    batch_metrics["adv_loss"] = adv_loss.detach()
                if total_loss is not None:
                    batch_metrics["total_loss"] = total_loss.detach()
                if not args.metrics_disable_lpips:
                    batch_metrics["lpips"] = lpips_from_images(input, recon, backends)
                if not args.metrics_disable_clip:
                    batch_metrics["clip_score"] = clip_score_from_images(input, recon, backends)
                train_agg.update(batch_metrics, n=input.shape[0])

            # Free memory by deleting large tensors
            del recon_image

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                
                # Build log message with adversarial loss info if available
                if args.enable_adversarial and clean_losses.count > 0:
                    log = (' | '.join([
                        f'Epoch {epoch}',
                        f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                        f'Time {elapsed.val:.3f}',
                        f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                        f'Clean {clean_losses.val:.3f} ({clean_losses.avg:.3f})',
                        f'Adv {adv_losses.val:.3f} ({adv_losses.avg:.3f})',
                        f'Total {total_losses.val:.3f} ({total_losses.avg:.3f})',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'ED {train_agg.as_dict().get("ed", float("nan")):.3f}',
                        f'Lr {cur_lr}',
                    ]))
                else:
                    log = (' | '.join([
                        f'Epoch {epoch}',
                        f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                        f'Time {elapsed.val:.3f}',
                        f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'ED {train_agg.as_dict().get("ed", float("nan")):.3f}',
                        f'Lr {cur_lr}',
                    ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    else:
        for batch_idx, input in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.to(config.device)
            # Ensure a clean gradient state before any inner-maximization logic
            # inside the model forward (adversarial example generation).
            optimizer.zero_grad(set_to_none=True)
            # Use adversarial training if enabled
            if args.enable_adversarial:
                recon_image, CBR, SNR, mse, loss_G = net(
                    input,
                    adversarial_training=True,
                    is_bob=train_is_bob,
                    is_eve=train_is_eve,
                )
                
                # Get separate loss components for adversarial training
                clean_loss, adv_loss, total_loss = net.get_loss_components()
                
                if batch_idx == 0:  # Log once per epoch
                    logger.info(
                        f"Using adversarial training with attack={args.adv_attack}, epsilon={args.fgsm_epsilon}, alpha={args.adversarial_alpha}"
                    )
            else:
                recon_image, CBR, SNR, mse, loss_G = net(input, is_bob=train_is_bob, is_eve=train_is_eve)
                clean_loss, adv_loss, total_loss = None, None, None
                
            loss = loss_G
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Reset loss components after backprop
            if args.enable_adversarial:
                net.reset_loss_components()
            
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            
            # Update separate loss meters if adversarial training
            if clean_loss is not None:
                clean_losses.update(clean_loss.item())
                adv_losses.update(adv_loss.item())
                total_losses.update(total_loss.item())
            
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)

            # Metrics for CSV (batch-weighted)
            with torch.no_grad():
                recon = recon_image.clamp(0.0, 1.0)
                ssim_v = ssim_from_images(input, recon, CalcuSSIM_1)
                # Convert MS-SSIM to dB for training log too
                msssim_val = 1 - CalcuSSIM(input, recon).mean().item()
                msssim_db = -10 * np.log10(max(1e-10, 1 - msssim_val))

                batch_metrics = {
                    "loss": loss.detach(),
                    "psnr": psnr_from_images(input, recon),
                    "ssim": ssim_v,
                    "msssim_db": msssim_db,
                    "ed": ed_from_images(input, recon),
                }
                if clean_loss is not None:
                    batch_metrics["clean_loss"] = clean_loss.detach()
                if adv_loss is not None:
                    batch_metrics["adv_loss"] = adv_loss.detach()
                if total_loss is not None:
                    batch_metrics["total_loss"] = total_loss.detach()
                if not args.metrics_disable_lpips:
                    batch_metrics["lpips"] = lpips_from_images(input, recon, backends)
                if not args.metrics_disable_clip:
                    batch_metrics["clip_score"] = clip_score_from_images(input, recon, backends)
                train_agg.update(batch_metrics, n=input.shape[0])

            # Free memory by deleting large tensors
            del recon_image

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                
                # Build log message with adversarial loss info if available
                if args.enable_adversarial and clean_losses.count > 0:
                    log = (' | '.join([
                        f'Epoch {epoch}',
                        f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                        f'Time {elapsed.val:.3f}',
                        f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                        f'Clean {clean_losses.val:.3f} ({clean_losses.avg:.3f})',
                        f'Adv {adv_losses.val:.3f} ({adv_losses.avg:.3f})',
                        f'Total {total_losses.val:.3f} ({total_losses.avg:.3f})',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'ED {train_agg.as_dict().get("ed", float("nan")):.3f}',
                        f'Lr {cur_lr}',
                    ]))
                else:
                    log = (' | '.join([
                        f'Epoch {epoch}',
                        f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                        f'Time {elapsed.val:.3f}',
                        f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'ED {train_agg.as_dict().get("ed", float("nan")):.3f}',
                        f'Lr {cur_lr}',
                    ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    for i in metrics:
        i.clear()

    # Write per-epoch averaged metrics
    train_csv = os.path.join(args.metrics_dir, "train_metrics.csv")
    train_logger = CSVLogger(train_csv)
    vals = train_agg.as_dict()
    train_logger.write_row({
        "split": "train",
        "epoch": epoch,
        "trainset": args.trainset,
        "testset": args.testset,
        "model": args.model,
        "channel_type": args.channel_type,
        "C": args.C,
        "multiple_snr": args.multiple_snr,
        "party": args.train_party if args.enable_an else "none",
        "enable_an": bool(args.enable_an),
        "sigma_an": float(args.sigma_an) if args.enable_an else 0.0,
        "shared_seed": int(args.shared_seed) if args.enable_an else None,
        "training_mode": "adversarial" if args.enable_adversarial else "clean",
        "attack_method": args.adv_attack if args.enable_adversarial else "none",
        "epsilon": args.fgsm_epsilon if args.enable_adversarial else None,
        "pgd_steps": args.pgd_steps if (args.enable_adversarial and args.adv_attack == 'pgd') else None,
        "pgd_step_size": args.pgd_step_size if (args.enable_adversarial and args.adv_attack == 'pgd') else None,
        "pgd_random_start": args.pgd_random_start if (args.enable_adversarial and args.adv_attack == 'pgd') else None,
        "loss": vals.get("loss"),
        "psnr": vals.get("psnr"),
        "ssim": vals.get("ssim"),
        "msssim_db": vals.get("msssim_db"),
        "ed": vals.get("ed"),
        "lpips": vals.get("lpips"),
        "clip_score": vals.get("clip_score"),
        "clean_loss": vals.get("clean_loss"),
        "adv_loss": vals.get("adv_loss"),
        "total_loss": vals.get("total_loss"),
    })


def test(save_recon=False, model_path=None, custom_snrs=None, logger=None, device=None):
    config.isTrain = False

    # Use provided logger and device, or fall back to global ones
    if logger is None:
        logger = globals().get('logger')
    if device is None:
        device = config.device

    backends = OptionalMetricBackends(device=device)

    def _extract_state_dict(ckpt):
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            return ckpt['state_dict'], ckpt.get('metadata', {})
        if isinstance(ckpt, dict) and 'model' in ckpt:
            return ckpt['model'], ckpt.get('metadata', {})
        return ckpt, {}

    # Handle custom model loading if specified
    if model_path is not None:
        # Initialize network based on model type
        # Use SwinJSCC_adv if adversarial training OR adversarial evaluation is enabled
        if args.enable_adversarial or (hasattr(args, 'adv_eval') and args.adv_eval):
            test_net = SwinJSCC_adv(args, config)
            logger.info("Initialized SwinJSCC with adversarial capabilities for testing")
        else:
            test_net = SwinJSCC(args, config)
            logger.info("Initialized standard SwinJSCC network for testing")

        # Load the model
        if os.path.exists(model_path):
            logger.info(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict, metadata = _extract_state_dict(checkpoint)
            if metadata:
                logger.info(f"Model metadata: {metadata.get('total_parameters', 'N/A')} total parameters")

            # Filter out profiling keys that might be in the checkpoint
            state_dict = {k: v for k, v in state_dict.items() if not ('total_ops' in k or 'total_params' in k)}

            # Load weights (strict first, fall back to non-strict with warning)
            try:
                missing, unexpected = test_net.load_state_dict(state_dict, strict=True)
                if missing or unexpected:
                    logger.warning(f"Strict load reported missing={len(missing)} unexpected={len(unexpected)}")
                logger.info("Loaded model weights (strict=True)")
            except Exception as e:
                logger.warning(f"Strict load failed ({e}); retrying with strict=False")
                missing, unexpected = test_net.load_state_dict(state_dict, strict=False)
                logger.warning(f"Non-strict load: missing={len(missing)} unexpected={len(unexpected)}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        test_net = test_net.to(device)
        test_net.eval()
    else:
        # Use the global net for testing
        test_net = net
        test_net.eval()

    # Use custom SNRs if provided, otherwise use args.multiple_snr
    if custom_snrs is not None:
        multiple_snr = custom_snrs
    else:
        multiple_snr = args.multiple_snr.split(",")
        for i in range(len(multiple_snr)):
            multiple_snr[i] = int(multiple_snr[i])

    channel_number = args.C.split(",")
    for i in range(len(channel_number)):
        channel_number[i] = int(channel_number[i])

    # Determine testing mode based on arguments
    adversarial_testing = args.adv_eval if hasattr(args, 'adv_eval') else False
    testing_mode = 'adversarial' if adversarial_testing else 'clean'
    
    # Determine epsilon for testing
    test_epsilon = args.test_epsilon if (hasattr(args, 'test_epsilon') and args.test_epsilon is not None) else args.fgsm_epsilon
    test_attack = args.test_attack if (hasattr(args, 'test_attack') and args.test_attack is not None) else args.adv_attack

    logger.info(f"Testing with SNR values: {multiple_snr}")
    logger.info(f"Testing with channel rates: {channel_number}")
    if adversarial_testing:
        logger.info(f"⚠️  ADVERSARIAL TESTING ENABLED - attack={test_attack}, epsilon={test_epsilon}")
    else:
        logger.info("Testing mode: CLEAN (no adversarial attacks)")

    test_csv = os.path.join(args.metrics_dir, "test_metrics.csv")
    test_logger = CSVLogger(test_csv)

    if args.enable_an:
        if args.test_bob or args.test_eve:
            parties = []
            if args.test_bob:
                parties.append(("bob", True, False))
            if args.test_eve:
                parties.append(("eve", False, True))
        else:
            # Default matches AN_c: training/testing uses Bob unless explicitly asked.
            parties = [("bob", True, False)]
    else:
        parties = [("none", True, False)]

    for party, is_bob, is_eve in parties:
        logger.info(f"Testing party={party} (enable_an={bool(args.enable_an)})")

        # Use appropriate context manager based on testing mode
        grad_context = torch.enable_grad() if adversarial_testing else torch.no_grad()

        for i, SNR in enumerate(multiple_snr):
            snr_agg = MetricsAggregator()

            for j, rate in enumerate(channel_number):
                logger.info(f"Testing SNR={SNR}dB, Channel Rate={rate}")

                elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
                metrics = [elapsed, psnrs, msssims, snrs, cbrs]

                group_agg = MetricsAggregator()

                with grad_context:
                    if args.trainset == 'CIFAR10' or args.trainset == 'ImageNet':
                        for batch_idx, (input, label) in enumerate(test_loader):
                            start_time = time.time()
                            input = input.to(device)

                            if adversarial_testing:
                                recon_image, CBR, SNR_actual, mse, loss_G = test_net(
                                    input, SNR, rate,
                                    adversarial_testing=True,
                                    fgsm_epsilon=test_epsilon,
                                    attack_method=test_attack,
                                    attack_mode=args.attack_mode,
                                    jamming_power=args.jamming_power,
                                    snr_drop=args.snr_drop,
                                    is_bob=is_bob,
                                    is_eve=is_eve,
                                )
                            else:
                                recon_image, CBR, SNR_actual, mse, loss_G = test_net(
                                    input,
                                    SNR,
                                    rate,
                                    is_bob=is_bob,
                                    is_eve=is_eve,
                                )

                            if save_recon:
                                mode_str = "adversarial" if adversarial_testing else "clean"
                                recon_dir = f"./data/recon/{mode_str}/snr_{SNR}dB_rate_{rate}/"
                                os.makedirs(recon_dir, exist_ok=True)
                                recon_path = os.path.join(recon_dir, f"batch_{batch_idx}.png")
                                torchvision.utils.save_image(recon_image, recon_path)
                                input_dir = f"./data/recon/input/"
                                os.makedirs(input_dir, exist_ok=True)
                                input_path = os.path.join(input_dir, f"batch_{batch_idx}.png")
                                if not os.path.exists(input_path):
                                    torchvision.utils.save_image(input, input_path)

                            elapsed.update(time.time() - start_time)
                            cbrs.update(CBR)
                            snrs.update(SNR_actual)
                            if mse.item() > 0:
                                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                                psnrs.update(psnr.item())
                                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                                msssims.update(msssim)

                            with torch.no_grad():
                                recon = recon_image.clamp(0.0, 1.0)
                                ssim_v = ssim_from_images(input, recon, CalcuSSIM_1)
                                batch_metrics = {
                                    "loss": loss_G.detach().mean(),
                                    "psnr": psnr_from_images(input, recon),
                                    "ssim": ssim_v,
                                    "ed": ed_from_images(input, recon),
                                }
                                if not args.metrics_disable_lpips:
                                    batch_metrics["lpips"] = lpips_from_images(input, recon, backends)
                                if not args.metrics_disable_clip:
                                    batch_metrics["clip_score"] = clip_score_from_images(input, recon, backends)
                                group_agg.update(batch_metrics, n=input.shape[0])
                                snr_agg.update(batch_metrics, n=input.shape[0])

                            log = (' | '.join([
                                f'Time {elapsed.val:.3f}',
                                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                                f'SNR {snrs.val:.1f}',
                                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                                f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                                f'ED {group_agg.as_dict().get("ed", float("nan")):.3f}',
                            ]))
                            logger.info(log)
                    else:
                        for batch_idx, batch in enumerate(test_loader):
                            input, names = batch
                            start_time = time.time()
                            input = input.to(device)

                            if adversarial_testing:
                                recon_image, CBR, SNR_actual, mse, loss_G = test_net(
                                    input, SNR, rate,
                                    adversarial_testing=True,
                                    fgsm_epsilon=test_epsilon,
                                    attack_method=test_attack,
                                    attack_mode=args.attack_mode,
                                    jamming_power=args.jamming_power,
                                    snr_drop=args.snr_drop,
                                    is_bob=is_bob,
                                    is_eve=is_eve,
                                )
                            else:
                                recon_image, CBR, SNR_actual, mse, loss_G = test_net(
                                    input,
                                    SNR,
                                    rate,
                                    is_bob=is_bob,
                                    is_eve=is_eve,
                                )

                            if save_recon:
                                mode_str = "adversarial" if adversarial_testing else "clean"
                                recon_dir = f"./data/recon/{mode_str}/snr_{SNR}dB_rate_{rate}/"
                                os.makedirs(recon_dir, exist_ok=True)
                                recon_path = os.path.join(recon_dir, f"{names[0]}")
                                torchvision.utils.save_image(recon_image, recon_path)

                                input_dir = f"./data/recon/input/"
                                os.makedirs(input_dir, exist_ok=True)
                                input_path = os.path.join(input_dir, f"{names[0]}")
                                if not os.path.exists(input_path):
                                    torchvision.utils.save_image(input, input_path)

                            elapsed.update(time.time() - start_time)
                            cbrs.update(CBR)
                            snrs.update(SNR_actual)

                            msssim_db = float("nan")
                            if mse.item() > 0:
                                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                                psnrs.update(psnr.item())
                                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                                msssims.update(msssim)
                                msssim_db = -10 * np.log10(max(1e-10, 1 - msssim))

                            with torch.no_grad():
                                recon = recon_image.clamp(0.0, 1.0)
                                ssim_v = ssim_from_images(input, recon, CalcuSSIM_1)
                                batch_metrics = {
                                    "loss": loss_G.detach().mean(),
                                    "psnr": psnr_from_images(input, recon),
                                    "ssim": ssim_v,
                                    "msssim_db": msssim_db,
                                    "ed": ed_from_images(input, recon),
                                }
                                if not args.metrics_disable_lpips:
                                    batch_metrics["lpips"] = lpips_from_images(input, recon, backends)
                                if not args.metrics_disable_clip:
                                    batch_metrics["clip_score"] = clip_score_from_images(input, recon, backends)
                                group_agg.update(batch_metrics, n=input.shape[0])
                                snr_agg.update(batch_metrics, n=input.shape[0])

                            log = (' | '.join([
                                f'{"[ADV]" if adversarial_testing else "[CLEAN]"} Time {elapsed.val:.3f}',
                                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                                f'SNR {snrs.val:.1f}',
                                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                                f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                                f'MS-dB {msssim_db:.2f}',
                                f'ED {group_agg.as_dict().get("ed", float("nan")):.3f}',
                            ]))
                            logger.info(log)

                group_vals = group_agg.as_dict()
                lpips = group_vals.get("lpips")
                lpips_db = 10 * math.log10(lpips) if lpips is not None and lpips > 0 else None
                clip_score = group_vals.get("clip_score")
                clip_score_db = 10 * math.log10(clip_score) if clip_score is not None and clip_score > 0 else None
                test_logger.write_row({
                    "split": "test",
                    "testing_mode": testing_mode,
                    "party": party,
                    "enable_an": bool(args.enable_an),
                    "sigma_an": float(args.sigma_an) if args.enable_an else 0.0,
                    "shared_seed": int(args.shared_seed) if args.enable_an else None,
                    "attack_method": test_attack if adversarial_testing else "none",
                    "epsilon": test_epsilon if adversarial_testing else None,
                    "pgd_steps": args.pgd_steps if (adversarial_testing and test_attack == 'pgd') else None,
                    "pgd_step_size": args.pgd_step_size if (adversarial_testing and test_attack == 'pgd') else None,
                    "pgd_random_start": args.pgd_random_start if (adversarial_testing and test_attack == 'pgd') else None,
                    "attack_mode": args.attack_mode if adversarial_testing else "none",
                    "testset": args.testset,
                    "trainset": args.trainset,
                    "model": args.model,
                    "channel_type": args.channel_type,
                    "snr": int(SNR),
                    "rate": int(rate),
                    "loss": group_vals.get("loss"),
                    "psnr": group_vals.get("psnr"),
                    "ssim": group_vals.get("ssim"),
                    "msssim_db": group_vals.get("msssim_db"),
                    "ed": group_vals.get("ed"),
                    "lpips": lpips,
                    "lpips_db": lpips_db,
                    "clip_score": clip_score,
                    "clip_score_db": clip_score_db,
                })

                logger.info(
                    f"Completed testing for SNR={SNR}dB, Rate={rate}: "
                    f"PSNR={group_vals.get('psnr', float('nan')):.3f}, "
                    f"SSIM={group_vals.get('ssim', float('nan')):.4f}, "
                    f"ED={group_vals.get('ed', float('nan')):.3f}"
                )

                for t in metrics:
                    t.clear()

            snr_vals = snr_agg.as_dict()
            lpips = snr_vals.get("lpips")
            snr_vals["lpips_db"] = 10 * math.log10(lpips) if lpips is not None and lpips > 0 else None
            clip_score = snr_vals.get("clip_score")
            snr_vals["clip_score_db"] = 10 * math.log10(clip_score) if clip_score is not None and clip_score > 0 else None
            test_logger.write_row({
                "split": "test",
                "testing_mode": testing_mode,
                "party": party,
                "enable_an": bool(args.enable_an),
                "sigma_an": float(args.sigma_an) if args.enable_an else 0.0,
                "shared_seed": int(args.shared_seed) if args.enable_an else None,
                "attack_method": test_attack if adversarial_testing else "none",
                "epsilon": test_epsilon if adversarial_testing else None,
                "pgd_steps": args.pgd_steps if (adversarial_testing and test_attack == 'pgd') else None,
                "pgd_step_size": args.pgd_step_size if (adversarial_testing and test_attack == 'pgd') else None,
                "pgd_random_start": args.pgd_random_start if (adversarial_testing and test_attack == 'pgd') else None,
                "attack_mode": args.attack_mode if adversarial_testing else "none",
                "testset": args.testset,
                "trainset": args.trainset,
                "model": args.model,
                "channel_type": args.channel_type,
                "snr": int(SNR),
                "rate": "all",
                "loss": snr_vals.get("loss"),
                "psnr": snr_vals.get("psnr"),
                "ssim": snr_vals.get("ssim"),
                "msssim_db": snr_vals.get("msssim_db"),
                "ed": snr_vals.get("ed"),
                "lpips": lpips,
                "lpips_db": snr_vals["lpips_db"],
                "clip_score": clip_score,
                "clip_score_db": snr_vals["clip_score_db"],
            })

    logger.info(f"Finish Test! Metrics written to {test_csv}")


if __name__ == '__main__':
    seed_torch()

    # If resuming from a checkpoint inside a prior run folder, keep writing logs/models there.
    if args.resume and args.training:
        inferred = infer_workdir_from_checkpoint(args.resume)
        if inferred:
            config.workdir = inferred
            config.log = os.path.join(config.workdir, 'train.log')
            config.samples = os.path.join(config.workdir, 'samples')
            config.models = os.path.join(config.workdir, 'models')

    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    
    # Validate argument combinations
    if args.enable_adversarial:
        logger.info(
            f"Adversarial training enabled with attack={args.adv_attack}, epsilon={args.fgsm_epsilon}, alpha={args.adversarial_alpha}"
        )
        logger.info("Will use clean evaluation during training and testing")

    # Ensure data/recon directory exists for test function
    try:
        makedirs("./data/recon/")
    except Exception as e:
        logger.error(f"Failed to create data/recon directory: {e}")

    torch.manual_seed(seed=config.seed)
    import random
    random.seed(config.seed)
    np.random.seed(config.seed)
    # Initialize adversarial-capable network if adversarial training OR adversarial eval is enabled
    if args.enable_adversarial or (hasattr(args, 'adv_eval') and args.adv_eval):
        net = SwinJSCC_adv(args, config)
        logger.info("Initialized SwinJSCC with adversarial training capabilities")
    else:
        net = SwinJSCC(args, config)
        logger.info("Initialized standard SwinJSCC network")
    
    # Load pretrained weights for fine-tuning if specified
    if args.training and args.resume is not None:
        if args.model_path is not None:
            logger.warning("--resume is set; ignoring --model-path during training")
        if args.pretrained_path and os.path.exists(args.pretrained_path):
            logger.info("--resume is set; skipping --pretrained-path loading")
    elif args.pretrained_path and os.path.exists(args.pretrained_path) and args.training:
        logger.info(f"Loading pretrained weights from: {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint
        
        model_dict = net.state_dict()
        
        # Load matching layers
        new_dict = OrderedDict()
        loaded = 0
        skipped = 0
        for k in pretrained_dict:
            if k in model_dict and pretrained_dict[k].size() == model_dict[k].size():
                new_dict[k] = pretrained_dict[k]
                loaded += 1
            else:
                skipped += 1
        
        if loaded > 0:
            model_dict.update(new_dict)
            net.load_state_dict(model_dict)
            logger.info(f"✅ Loaded {loaded} layers from pretrained model")
            logger.info(f"⚠️  Skipped {skipped} incompatible layers")
        else:
            logger.warning(f"⚠️  No compatible layers found! Architecture mismatch.")
            logger.warning(f"   Pretrained has {len(pretrained_dict)} layers, model has {len(model_dict)} layers")
            logger.warning(f"   Training from scratch instead.")
    
    # Load model checkpoint if specified (testing, or warm-start training when --resume is not used)
    elif args.model_path is not None:
        load_weights(args.model_path)
        logger.info(f"Loaded model from: {args.model_path}")

    net = net.cuda()

    # Measure model efficiency (unified logger)
    log_model_efficiency(
        model=net,
        model_type='swinjscc',
        device='cuda',
        logger=logger,
        snr=config.multiple_snr[0] if hasattr(config, 'multiple_snr') and config.multiple_snr else 10.0,
        config=config,
    )

    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    train_loader, test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0
    start_epoch = 0

    # Resume training if requested
    if args.training and args.resume is not None:
        start_epoch, loaded_global_step = resume_from_checkpoint(
            ckpt_path=args.resume,
            net=net,
            optimizer=optimizer,
            device=config.device,
            logger=logger,
        )
        if loaded_global_step > 0:
            global_step = loaded_global_step
        else:
            global_step = start_epoch * train_loader.__len__()
        try:
            cur_lr = float(optimizer.param_groups[0].get('lr', cur_lr))
        except Exception:
            pass

    steps_epoch = start_epoch

    if args.training:
        for epoch in range(steps_epoch, config.tot_epoch):
            logger.info(f'====== Current epoch {epoch} ======')
            logger.info(f"Learning rate: {cur_lr}")

            train_one_epoch(args, epoch, optimizer)

            # Clear CUDA cache to prevent memory accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (epoch + 1) % config.save_model_freq == 0:
                save_name = f"model_epoch_{epoch + 1}.pth"
                save_path = os.path.join(config.models, save_name)
                logger.info(f"Saving model checkpoint to: {save_path}")
                if args.enable_adversarial:
                    logger.info(f"  - Adversarial training enabled (epsilon={args.fgsm_epsilon}, alpha={args.adversarial_alpha})")
                extra_state = {
                    # Save the NEXT epoch to run for easy resume
                    'epoch': int(epoch + 1),
                    'global_step': int(global_step),
                    'optimizer': optimizer.state_dict(),
                    'args': vars(args),
                }
                save_model(net, save_path=save_path, extra_state=extra_state)

                # Also maintain a stable "last" checkpoint for auto-resume.
                last_path = os.path.join(config.models, "checkpoint_last.pth")
                try:
                    save_model(net, save_path=last_path, extra_state=extra_state)
                except Exception as e:
                    logger.warning(f"Failed to write checkpoint_last.pth ({e})")
                
                # Double check the file was created and log its size
                if os.path.exists(save_path):
                    logger.info(f"Successfully saved {save_name} ({os.path.getsize(save_path) / 1024 / 1024:.2f} MB)")
                else:
                    logger.error(f"Failed to save model at {save_path}!")

        logger.info(f"Training completed! Metrics written to {os.path.join(args.metrics_dir, 'train_metrics.csv')}")
    else:
        # Testing mode - use --multiple-snr for SNR values
        # The test() function will use args.multiple_snr by default
        if args.model_path is not None:
            # Test with specific model
            test(save_recon=args.save_recon, model_path=args.model_path, logger=logger, device=config.device)
        else:
            # Use default testing (no custom model path)
            test(save_recon=args.save_recon)
        logger.info("Testing completed!")

