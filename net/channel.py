import torch.nn as nn
import numpy as np
import os
import torch
import time


class Channel(nn.Module):
    """
    Currently the channel model is either error free, erasure channel,
    rayleigh channel or the AWGN channel.
    """

    def __init__(self, args, config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = args.channel_type
        self.device = config.device
        self.h = torch.sqrt(torch.randn(1) ** 2
                            + torch.randn(1) ** 2) / 1.414
        if config.logger:
            config.logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
                args.channel_type, args.multiple_snr))

    def gaussian_noise_layer(self, input_layer, std, name=None, deterministic=False, seed=42):
        if deterministic:
            # Use local generator instead of global seed
            device = input_layer.device if hasattr(input_layer, 'device') else input_layer.get_device()
            noise_shape = input_layer.shape
            real_dtype = input_layer.real.dtype if torch.is_complex(input_layer) else input_layer.dtype
            
            # Create local generator to avoid polluting global random state
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            noise_real = torch.randn(noise_shape, dtype=real_dtype, device=device, generator=generator) * std
            
            # Use different seed for imaginary part to ensure proper complex noise
            generator.manual_seed(int(seed) + 1)
            noise_imag = torch.randn(noise_shape, dtype=real_dtype, device=device, generator=generator) * std
        else:
            device = input_layer.device if hasattr(input_layer, 'device') else input_layer.get_device()
            noise_real = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
            noise_imag = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std, name=None, deterministic=False, return_h: bool = False, seed=42):
        if deterministic:
            # Use local generator instead of global seed
            device = input_layer.device if hasattr(input_layer, 'device') else input_layer.get_device()
            real_dtype = input_layer.real.dtype if torch.is_complex(input_layer) else input_layer.dtype
            shape = input_layer.real.shape if torch.is_complex(input_layer) else input_layer.shape
            
            # Create local generator to avoid polluting global random state
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            h_real = torch.randn(shape, dtype=real_dtype, device=device, generator=generator)
            
            generator.manual_seed(int(seed) + 1)
            h_imag = torch.randn(shape, dtype=real_dtype, device=device, generator=generator)
            h = torch.sqrt(h_real ** 2 + h_imag ** 2) / torch.sqrt(torch.tensor(2.0))
            
            generator.manual_seed(int(seed) + 2)
            noise_real = torch.randn(shape, dtype=real_dtype, device=device, generator=generator) * std
            generator.manual_seed(int(seed) + 3)
            noise_imag = torch.randn(shape, dtype=real_dtype, device=device, generator=generator) * std
        else:
            noise_real = torch.normal(mean=0.0, std=std, size=input_layer.shape)
            noise_imag = torch.normal(mean=0.0, std=std, size=input_layer.shape)
            h = torch.sqrt(torch.normal(mean=0.0, std=1, size=input_layer.shape) ** 2
                           + torch.normal(mean=0.0, std=1, size=input_layer.shape) ** 2) / np.sqrt(2)
        if not deterministic:
            if hasattr(input_layer, 'device'):
                device = input_layer.device
            else:
                device = input_layer.get_device()
            noise = noise_real.to(device) + 1j * noise_imag.to(device)
            h = h.to(device)
        else:
            noise = noise_real + 1j * noise_imag

        out = input_layer * h + noise
        if return_h:
            return out, h
        return out


    def complex_normalize(self, x, power):
        pwr = torch.mean(x ** 2) * 2
        out = torch.sqrt(torch.tensor(power, dtype=x.dtype, device=x.device)) * x / torch.sqrt(pwr)
        return out, pwr

    def forward_post_normalize(self, normalized_input, pwr, chan_param,
                               preserve_gradients=False, deterministic=False,
                               deterministic_seed=42):
        """Channel forward starting from an already-normalized feature.

        Skips the complex_normalize step so that perturbations applied in
        normalized space are not absorbed by re-normalization.

        Pipeline: normalized_input → complex split → channel noise → denormalize

        Args:
            normalized_input: Power-normalized feature (same shape as encoder output).
            pwr: Power scalar returned by complex_normalize (for denormalization).
            chan_param: Channel SNR in dB.
            preserve_gradients: If True, keep gradient flow through channel noise.
            deterministic: Use seeded noise for reproducibility.
            deterministic_seed: Seed for deterministic noise.

        Returns:
            Denormalized noisy feature (same shape as input).
        """
        channel_tx = normalized_input
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j
        channel_output = self.complex_forward(channel_in, chan_param, deterministic, deterministic_seed=deterministic_seed)
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_shape)
        if self.chan_type == 1 or self.chan_type == 'awgn':
            noise = (channel_output - channel_tx)
            if not preserve_gradients:
                noise = noise.detach()
                noise.requires_grad = False
            channel_tx = channel_tx + noise
            return channel_tx * torch.sqrt(pwr)
        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            noise = (channel_output - channel_tx)
            if not preserve_gradients:
                noise = noise.detach()
                noise.requires_grad = False
            channel_tx = channel_tx + noise
            return channel_tx * torch.sqrt(pwr)


    def forward(self, input, chan_param, avg_pwr=False, preserve_gradients=False, deterministic=False, deterministic_seed=42):
        if avg_pwr:
            power = 1
            channel_tx = torch.sqrt(torch.tensor(power, dtype=input.dtype, device=input.device)) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1)
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j
        channel_output = self.complex_forward(channel_in, chan_param, deterministic, deterministic_seed=deterministic_seed)
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_shape)
        if self.chan_type == 1 or self.chan_type == 'awgn':
            noise = (channel_output - channel_tx)
            # Only detach noise if not doing adversarial training
            if not preserve_gradients:
                noise = noise.detach()
                noise.requires_grad = False
            channel_tx = channel_tx + noise
            if avg_pwr:
                return channel_tx * torch.sqrt(avg_pwr * 2)
            else:
                return channel_tx * torch.sqrt(pwr)
        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            noise = (channel_output - channel_tx)
            if not preserve_gradients:
                noise = noise.detach()
                noise.requires_grad = False
            channel_tx = channel_tx + noise
            if avg_pwr:
                return channel_tx * torch.sqrt(avg_pwr * 2)
            else:
                return channel_tx * torch.sqrt(pwr)

    def complex_forward(self, channel_in, chan_param, deterministic=False, return_h: bool = False, deterministic_seed=42):
        if self.chan_type == 0 or self.chan_type == 'none':
            if return_h:
                return channel_in, None
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            channel_tx = channel_in
            sigma = torch.sqrt(torch.tensor(1.0 / (2 * 10 ** (chan_param / 10)), dtype=channel_in.dtype, device=channel_in.device))
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="awgn_chan_noise",
                                                    deterministic=deterministic,
                                                    seed=deterministic_seed)
            if return_h:
                return chan_output, None
            return chan_output

        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            channel_tx = channel_in
            sigma = torch.sqrt(torch.tensor(1.0 / (2 * 10 ** (chan_param / 10)), dtype=channel_in.dtype, device=channel_in.device))
            chan_out = self.rayleigh_noise_layer(
                channel_tx,
                std=sigma,
                name="rayleigh_chan_noise",
                deterministic=deterministic,
                return_h=return_h,
                seed=deterministic_seed,
            )
            return chan_out


    def noiseless_forward(self, channel_in):
        channel_tx = self.normalize(channel_in, power=1)
        return channel_tx

