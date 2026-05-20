from net.decoder import *
from net.encoder import *
from loss.distortion import Distortion
from net.channel import Channel
from random import choice
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional


class ANGenerator(nn.Module):
    """Complex artificial-noise generator (shared-key, reproducible).

    Notes:
    - Generates complex Gaussian AN in the complex/IQ domain.
    - Uses a nonce to avoid producing the exact same AN every forward call.
      Bob regenerates the same AN (same key + same nonce) to cancel it.
    """

    def __init__(self, shared_seed: int = 42, sigma_an: float = 0.0):
        super().__init__()
        self.shared_seed = int(shared_seed)
        self.sigma_an = float(sigma_an)

    def forward(self, shape, device: torch.device, dtype: torch.dtype, nonce: int = 0) -> torch.Tensor:
        if self.sigma_an == 0.0:
            return torch.zeros(shape, dtype=torch.complex64, device=device)

        # Keep the mapping deterministic but vary across calls via nonce.
        # Use a large odd multiplier to reduce accidental overlaps.
        base_seed = self.shared_seed + int(nonce) * 1000003
        generator = torch.Generator(device=device)

        generator.manual_seed(base_seed)
        noise_real = torch.randn(shape, generator=generator, dtype=dtype, device=device) * self.sigma_an

        generator.manual_seed(base_seed + 1)
        noise_imag = torch.randn(shape, generator=generator, dtype=dtype, device=device) * self.sigma_an

        return noise_real + 1j * noise_imag


class SwinJSCC(nn.Module):
    def __init__(self, args, config):
        super(SwinJSCC, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
        self.distortion_loss = Distortion(args)
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.channel_number = args.C.split(",")
        for i in range(len(self.channel_number)):
            self.channel_number[i] = int(self.channel_number[i])
        self.downsample = config.downsample
        self.model = args.model

        # Artificial Noise (AN) for Bob/Eve evaluation
        self.enable_an = bool(getattr(args, "enable_an", False))
        self.sigma_an = float(getattr(args, "sigma_an", 0.0))
        self.shared_seed = int(getattr(args, "shared_seed", 42))
        self._an_nonce = 0
        self._warned_an_rayleigh = False
        self.an_generator = ANGenerator(shared_seed=self.shared_seed, sigma_an=self.sigma_an) if self.enable_an else None
        if self.enable_an and config.logger is not None:
            config.logger.info(
                f"Artificial Noise enabled: sigma_an={self.sigma_an}, shared_seed={self.shared_seed}"
            )

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False, preserve_gradients=False, deterministic=False, deterministic_seed=42):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr, preserve_gradients, deterministic, deterministic_seed=deterministic_seed)
        return noisy_feature

    def feature_pass_channel_post_normalize(self, normalized_feature, pwr, chan_param,
                                            preserve_gradients=False, deterministic=False,
                                            deterministic_seed=42):
        """Channel forward starting from an already-normalized feature.
        Skips the power normalization so perturbations in normalized space survive."""
        return self.channel.forward_post_normalize(
            normalized_feature, pwr, chan_param,
            preserve_gradients=preserve_gradients,
            deterministic=deterministic,
            deterministic_seed=deterministic_seed,
        )

    def feature_pass_channel_with_an(
        self,
        feature: torch.Tensor,
        chan_param,
        *,
        is_bob: bool,
        nonce=None,
        avg_pwr=False,
        preserve_gradients: bool = False,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Transmit through channel with complex AN; Bob cancels, Eve can't."""

        chan_type = str(getattr(self.channel, "chan_type", "awgn")).lower()

        # Normalize to unit power before adding AN so channel noise sigma is well-defined.
        channel_tx, pwr = self.channel.complex_normalize(feature, power=1)

        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]
        complex_signal = channel_in[: L // 2] + channel_in[L // 2 :] * 1j

        if nonce is None:
            nonce = self._an_nonce
            self._an_nonce += 1

        if self.an_generator is None or self.sigma_an == 0.0:
            complex_an = None
            complex_signal_with_an = complex_signal
        else:
            complex_an = self.an_generator(complex_signal.shape, complex_signal.device, dtype=feature.dtype, nonce=nonce)
            complex_signal_with_an = complex_signal + complex_an

        if chan_type in {"rayleigh", "2"}:
            channel_output, h = self.channel.complex_forward(
                complex_signal_with_an,
                chan_param,
                deterministic,
                return_h=True,
            )
        else:
            channel_output = self.channel.complex_forward(complex_signal_with_an, chan_param, deterministic)
            h = None

        # Bob cancels AN when enabled.
        if self.enable_an and is_bob and complex_an is not None:
            complex_an_bob = self.an_generator(
                complex_an.shape,
                complex_an.device,
                dtype=feature.dtype,
                nonce=nonce,
            )
            if chan_type in {"rayleigh", "2"}:
                # Rayleigh model: y = h*(x+an) + n  =>  y - h*an = h*x + n
                if h is None:
                    if not self._warned_an_rayleigh and self.config.logger is not None:
                        self._warned_an_rayleigh = True
                        self.config.logger.warning(
                            "AN cancellation under Rayleigh expected fading gain h, but h was None; skipping cancellation."
                        )
                else:
                    channel_output = channel_output - (h * complex_an_bob)
            else:
                # AWGN/none: y = (x+an) + n  =>  y - an = x + n
                channel_output = channel_output - complex_an_bob

        channel_output_real = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output_real = channel_output_real.reshape(input_shape)

        if avg_pwr:
            return channel_output_real * torch.sqrt(avg_pwr * 2)
        return channel_output_real * torch.sqrt(pwr)

    def feature_pass_channel_with_an_post_normalize(
        self,
        normalized_feature: torch.Tensor,
        pwr,
        chan_param,
        *,
        is_bob: bool,
        nonce=None,
        preserve_gradients: bool = False,
        deterministic: bool = False,
        deterministic_seed: int = 42,
    ) -> torch.Tensor:
        """Like feature_pass_channel_with_an but starts from already-normalized input.
        Skips internal complex_normalize so perturbations in normalized space survive."""

        chan_type = str(getattr(self.channel, "chan_type", "awgn")).lower()

        channel_tx = normalized_feature
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]
        complex_signal = channel_in[: L // 2] + channel_in[L // 2 :] * 1j

        if nonce is None:
            nonce = self._an_nonce
            self._an_nonce += 1

        if self.an_generator is None or self.sigma_an == 0.0:
            complex_an = None
            complex_signal_with_an = complex_signal
        else:
            complex_an = self.an_generator(complex_signal.shape, complex_signal.device, dtype=normalized_feature.dtype, nonce=nonce)
            complex_signal_with_an = complex_signal + complex_an

        if chan_type in {"rayleigh", "2"}:
            channel_output, h = self.channel.complex_forward(
                complex_signal_with_an,
                chan_param,
                deterministic,
                return_h=True,
                deterministic_seed=deterministic_seed,
            )
        else:
            channel_output = self.channel.complex_forward(
                complex_signal_with_an, chan_param, deterministic,
                deterministic_seed=deterministic_seed,
            )
            h = None

        if self.enable_an and is_bob and complex_an is not None:
            complex_an_bob = self.an_generator(
                complex_an.shape,
                complex_an.device,
                dtype=normalized_feature.dtype,
                nonce=nonce,
            )
            if chan_type in {"rayleigh", "2"}:
                if h is None:
                    if not self._warned_an_rayleigh and self.config.logger is not None:
                        self._warned_an_rayleigh = True
                        self.config.logger.warning(
                            "AN cancellation under Rayleigh expected fading gain h, but h was None; skipping cancellation."
                        )
                else:
                    channel_output = channel_output - (h * complex_an_bob)
            else:
                channel_output = channel_output - complex_an_bob

        channel_output_real = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output_real = channel_output_real.reshape(input_shape)

        return channel_output_real * torch.sqrt(pwr)

    def _maybe_pass_channel_post_normalize(
        self,
        normalized_feature: torch.Tensor,
        pwr,
        chan_param,
        *,
        is_bob: bool,
        an_nonce=None,
        preserve_gradients: bool = False,
        deterministic: bool = False,
        deterministic_seed: int = 42,
    ) -> torch.Tensor:
        """Like _maybe_pass_channel but starts from already-normalized input.
        Dispatches to the appropriate post-normalize channel function."""
        if not self.pass_channel:
            return normalized_feature * torch.sqrt(pwr)
        if self.enable_an and self.an_generator is not None:
            return self.feature_pass_channel_with_an_post_normalize(
                normalized_feature,
                pwr,
                chan_param,
                is_bob=is_bob,
                nonce=an_nonce,
                preserve_gradients=preserve_gradients,
                deterministic=deterministic,
                deterministic_seed=deterministic_seed,
            )
        return self.feature_pass_channel_post_normalize(
            normalized_feature,
            pwr,
            chan_param,
            preserve_gradients=preserve_gradients,
            deterministic=deterministic,
            deterministic_seed=deterministic_seed,
        )

    def forward(self, input_image, given_SNR=None, given_rate=None, is_bob: bool = True, is_eve: bool = False):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        if given_rate is None:
            channel_number = choice(self.channel_number)
        else:
            channel_number = given_rate

        if self.model == 'SwinJSCC_w_o_SAandRA' or self.model == 'SwinJSCC_w_SA':
            feature = self.encoder(input_image, chan_param, channel_number, self.model, preserve_gradients=False)
            CBR = feature.numel() / 2 / input_image.numel()
            if self.pass_channel:
                if self.enable_an and self.an_generator is not None:
                    noisy_feature = self.feature_pass_channel_with_an(
                        feature,
                        chan_param,
                        is_bob=is_bob,
                        avg_pwr=False,
                        preserve_gradients=False,
                    )
                else:
                    noisy_feature = self.feature_pass_channel(feature, chan_param)
            else:
                noisy_feature = feature

        elif self.model == 'SwinJSCC_w_RA' or self.model == 'SwinJSCC_w_SAandRA':
            feature, mask = self.encoder(input_image, chan_param, channel_number, self.model, preserve_gradients=False)
            CBR = channel_number / (2 * 3 * 2 ** (self.downsample * 2))
            avg_pwr = torch.sum(feature ** 2) / mask.sum()
            if self.pass_channel:
                if self.enable_an and self.an_generator is not None:
                    noisy_feature = self.feature_pass_channel_with_an(
                        feature,
                        chan_param,
                        is_bob=is_bob,
                        avg_pwr=avg_pwr,
                        preserve_gradients=False,
                    )
                else:
                    noisy_feature = self.feature_pass_channel(feature, chan_param, avg_pwr)
            else:
                noisy_feature = feature
            noisy_feature = noisy_feature * mask

        recon_image = self.decoder(noisy_feature, chan_param, self.model, preserve_gradients=False)
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()


class SwinJSCC_adv(SwinJSCC):
    """
    Adversarial SwinJSCC with white-box FGSM attacks on channel input.
    Extends the original SwinJSCC with adversarial training capability.
    """
    def __init__(self, args, config):
        super(SwinJSCC_adv, self).__init__(args, config)

        # Adversarial training parameters from command line arguments
        self.epsilon = args.fgsm_epsilon  # Configurable epsilon
        self.alpha = args.adversarial_alpha     # Configurable alpha

        # Attack configuration
        self.adv_attack = getattr(args, 'adv_attack', 'fgsm')
        self.pgd_steps = int(getattr(args, 'pgd_steps', 5))
        self.pgd_step_size = getattr(args, 'pgd_step_size', None)
        self.pgd_random_start = bool(getattr(args, 'pgd_random_start', False))

        # Inner-maximization debug (sanity check for gradient masking / weak attacks)
        self.adv_debug_inner = bool(getattr(args, 'adv_debug_inner', False))
        self.adv_debug_inner_every = int(getattr(args, 'adv_debug_inner_every', 0))
        self.adv_debug_inner_max_steps = int(getattr(args, 'adv_debug_inner_max_steps', 0))
        self._adv_debug_inner_calls = 0
        
        # Initialize variables for adversarial attacks
        self._current_input = None
        self._current_mask = None
        
        # Initialize separate loss tracking for adversarial training
        self._clean_loss = None
        self._adv_loss = None
        self._total_loss = None
        # Per-batch varying seed for deterministic channel in adversarial training
        self._det_seed_counter = 0

    def _adv_debug_should_log(self) -> bool:
        if not self.adv_debug_inner:
            return False
        every = int(self.adv_debug_inner_every)
        if every > 0:
            return (int(self._adv_debug_inner_calls) % every) == 0
        # Default: print only once per run to avoid flooding
        return int(self._adv_debug_inner_calls) == 0
    
    def get_loss_components(self):
        """
        Get separate loss components from adversarial training.
        
        Returns:
            tuple: (clean_loss, adv_loss, total_loss) or (None, None, None) if not available
        """
        return self._clean_loss, self._adv_loss, self._total_loss
    
    def reset_loss_components(self):
        """Reset loss components after each batch."""
        # Properly delete tensor references to free memory
        if self._clean_loss is not None:
            del self._clean_loss
        if self._adv_loss is not None:
            del self._adv_loss
        if self._total_loss is not None:
            del self._total_loss
        
        self._clean_loss = None
        self._adv_loss = None
        self._total_loss = None
        
    def generate_fgsm_perturbation(self, normalized_feature, pwr, chan_param, mask=None, epsilon=None,
                                    deterministic_seed=42, is_bob=True, an_nonce=None):
        """
        FGSM perturbation generation in NORMALIZED signal space.

        The perturbation is optimized AFTER power normalization so that epsilon
        directly controls the over-the-air perturbation budget and is not absorbed
        by complex_normalize().

        Args:
            normalized_feature: Power-normalized channel input (requires_grad=True)
            pwr: Power scalar from complex_normalize (for denormalization)
            chan_param: Channel parameter (SNR)
            mask: Optional mask for RA models
            epsilon: Attack strength (L-inf in normalized space)
            deterministic_seed: Seed for deterministic channel noise
            is_bob: Whether receiver is Bob (cancels AN) or Eve
            an_nonce: Nonce for AN generation (must match outer pass)

        Returns:
            perturbation: FGSM perturbation in normalized space
        """
        if epsilon is None:
            epsilon = self.epsilon

        if self._current_input is None:
            raise ValueError("Current input not set for adversarial attack")

        if not normalized_feature.requires_grad:
            raise ValueError("Feature tensor must have requires_grad=True for FGSM")

        # Power-scaled epsilon: epsilon is a fraction of signal RMS
        z_rms = torch.sqrt(torch.mean(normalized_feature.detach() ** 2))
        epsilon_abs = epsilon * z_rms
        print(f"[ADV DEBUG][FGSM] z_rms={z_rms.item():.4f}, "
              f"epsilon_abs={epsilon_abs.item():.4f}, "
              f"signal_power={normalized_feature.detach().pow(2).mean().item():.4f}, "
              f"perturb_power={(epsilon_abs**2).item():.6f}")

        # Forward through post-normalize channel pipeline
        if mask is not None:
            attack_normalized = normalized_feature * mask.float()
        else:
            attack_normalized = normalized_feature

        # Use the same channel path as the outer adversarial forward pass
        noisy_feature = self._maybe_pass_channel_post_normalize(
            attack_normalized,
            pwr,
            chan_param,
            is_bob=is_bob,
            an_nonce=an_nonce,
            preserve_gradients=True,
            deterministic=True,
            deterministic_seed=deterministic_seed,
        )

        if mask is not None:
            noisy_feature = noisy_feature * mask.float()

        recon_image = self.decoder(noisy_feature, chan_param, self.model, preserve_gradients=True)
        attack_loss = self.distortion_loss_wrapper(self._current_input, recon_image.clamp(0., 1.))

        try:
            grad_feature = torch.autograd.grad(attack_loss.mean(), normalized_feature,
                                             retain_graph=False, create_graph=False)[0]
        except RuntimeError as e:
            print(f"Warning: Gradient computation failed during FGSM attack: {e}")
            print(f"Falling back to zero perturbation")
            return torch.zeros_like(normalized_feature)

        perturbation = epsilon_abs * grad_feature.sign()

        # Debug logging
        should_log = self._adv_debug_should_log()
        self._adv_debug_inner_calls += 1
        if should_log:
            clean_val = float(attack_loss.mean().detach().cpu())
            with torch.no_grad():
                adv_normalized = normalized_feature.detach() + perturbation.detach()
                if mask is not None:
                    adv_normalized = adv_normalized * mask.float()

                noisy_adv = self._maybe_pass_channel_post_normalize(
                    adv_normalized, pwr, chan_param,
                    is_bob=is_bob,
                    an_nonce=an_nonce,
                    preserve_gradients=False,
                    deterministic=True,
                    deterministic_seed=deterministic_seed,
                )

                if mask is not None:
                    noisy_adv = noisy_adv * mask.float()

                recon_adv = self.decoder(noisy_adv, chan_param, self.model, preserve_gradients=False)
                adv_val = float(self.distortion_loss_wrapper(self._current_input, recon_adv.clamp(0.0, 1.0)).mean().detach().cpu())

            delta_norm = float(perturbation.detach().norm().cpu())
            msg = (
                f"[ADV-DEBUG][FGSM] loss(clean)={clean_val:.6f} loss(adv)={adv_val:.6f} "
                f"delta_norm={delta_norm:.6f} eps_abs={float(epsilon_abs):.6f} "
            )
            if adv_val < clean_val:
                msg += "⚠️  (attack loss did NOT increase)"
            print(msg)

        if torch.rand(1).item() < 0.001:
            print(f"FGSM Debug: epsilon={epsilon}, perturbation norm={perturbation.norm().item():.6f}, normalized_feature norm={normalized_feature.norm().item():.6f}")

        return perturbation.detach()

    def generate_pgd_perturbation(
        self,
        normalized_feature,
        pwr,
        chan_param,
        mask=None,
        epsilon=None,
        step_size=None,
        num_steps=None,
        random_start=None,
        deterministic_seed=42,
        is_bob=True,
        an_nonce=None,
    ):
        """\
        L-inf PGD perturbation generation in NORMALIZED signal space.

        The perturbation is optimized AFTER power normalization so that epsilon
        directly controls the over-the-air perturbation budget and is not absorbed
        by complex_normalize().

        Args:
            normalized_feature: Power-normalized channel input (requires_grad=True)
            pwr: Power scalar from complex_normalize
            chan_param: Channel SNR
            mask: Optional mask for RA models
            epsilon: L-inf budget in normalized space
            step_size: PGD step size
            num_steps: Number of PGD iterations
            random_start: Initialize delta uniformly in [-eps, eps]
            deterministic_seed: Seed for deterministic channel noise
            is_bob: Whether receiver is Bob (cancels AN) or Eve
            an_nonce: Nonce for AN generation (must match outer pass)

        Returns:
            delta: PGD perturbation in normalized space (detached)
        """
        if epsilon is None:
            epsilon = self.epsilon
        if num_steps is None:
            num_steps = self.pgd_steps
        if random_start is None:
            random_start = self.pgd_random_start

        # Power-scaled epsilon: epsilon is a fraction of signal RMS
        z_rms = torch.sqrt(torch.mean(normalized_feature.detach() ** 2))
        epsilon_abs = float(epsilon * z_rms)
        print(f"[ADV DEBUG][PGD] z_rms={z_rms.item():.4f}, "
              f"epsilon_abs={epsilon_abs:.4f}, "
              f"signal_power={normalized_feature.detach().pow(2).mean().item():.4f}, "
              f"perturb_power={(epsilon_abs**2):.6f}")

        if step_size is None:
            if self.pgd_step_size is not None:
                step_size = float(self.pgd_step_size) * z_rms.item()
            else:
                step_size = 2.5 * epsilon_abs / max(int(num_steps), 1)

        if self._current_input is None:
            raise ValueError("Current input not set for adversarial attack")

        if not normalized_feature.requires_grad:
            raise ValueError("Feature tensor must have requires_grad=True for PGD")

        base = normalized_feature

        if random_start:
            delta = (2 * torch.rand_like(base) - 1) * epsilon_abs
        else:
            delta = torch.zeros_like(base)

        delta = delta.detach()

        should_log = self._adv_debug_should_log()
        self._adv_debug_inner_calls += 1
        loss_trace = []

        for _ in range(int(num_steps)):
            delta.requires_grad_(True)

            attack_normalized = base + delta

            if mask is not None:
                attack_normalized = attack_normalized * mask.float()

            # Use the same channel path as the outer adversarial forward pass
            # (including AN when enabled) so the gradient matches the actual loss landscape.
            noisy_feature = self._maybe_pass_channel_post_normalize(
                attack_normalized,
                pwr,
                chan_param,
                is_bob=is_bob,
                an_nonce=an_nonce,
                preserve_gradients=True,
                deterministic=True,
                deterministic_seed=deterministic_seed,
            )

            if mask is not None:
                noisy_feature = noisy_feature * mask.float()

            recon_image = self.decoder(noisy_feature, chan_param, self.model, preserve_gradients=True)
            attack_loss = self.distortion_loss_wrapper(self._current_input, recon_image.clamp(0.0, 1.0))

            if should_log:
                loss_trace.append(float(attack_loss.mean().detach().cpu()))

            grad_delta = torch.autograd.grad(attack_loss.mean(), delta, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                delta = delta + step_size * grad_delta.sign()
                delta = delta.clamp(-epsilon_abs, epsilon_abs)
            delta = delta.detach()

        if should_log:
            with torch.no_grad():
                final_normalized = base.detach() + delta.detach()
                if mask is not None:
                    final_normalized = final_normalized * mask.float()

                final_noisy = self._maybe_pass_channel_post_normalize(
                    final_normalized, pwr, chan_param,
                    is_bob=is_bob,
                    an_nonce=an_nonce,
                    preserve_gradients=False,
                    deterministic=True,
                    deterministic_seed=deterministic_seed,
                )

                if mask is not None:
                    final_noisy = final_noisy * mask.float()

                final_recon = self.decoder(final_noisy, chan_param, self.model, preserve_gradients=False)
                final_loss_val = float(self.distortion_loss_wrapper(self._current_input, final_recon.clamp(0.0, 1.0)).mean().detach().cpu())

            if len(loss_trace) > 0:
                first = loss_trace[0]
                last = loss_trace[-1]
                best = max(loss_trace)
                shown = loss_trace
                max_steps = int(self.adv_debug_inner_max_steps)
                if max_steps > 0:
                    shown = loss_trace[:max_steps]
                print(f"[ADV-DEBUG][PGD] step_losses={shown} (first={first:.6f}, last={last:.6f}, best={best:.6f}, final={final_loss_val:.6f})")
                if final_loss_val < first:
                    print("[ADV-DEBUG][PGD] ⚠️  final attack loss < initial loss (PGD may be ineffective or stochasticity dominates)")

        return delta.detach()

    def _maybe_pass_channel(
        self,
        feature: torch.Tensor,
        chan_param,
        *,
        is_bob: bool,
        an_nonce=None,
        avg_pwr=False,
        preserve_gradients: bool = False,
        deterministic: bool = False,
        deterministic_seed: int = 42,
    ) -> torch.Tensor:
        if not self.pass_channel:
            return feature
        if self.enable_an and self.an_generator is not None:
            return self.feature_pass_channel_with_an(
                feature,
                chan_param,
                is_bob=is_bob,
                nonce=an_nonce,
                avg_pwr=avg_pwr,
                preserve_gradients=preserve_gradients,
                deterministic=deterministic,
            )
        return self.feature_pass_channel(
            feature,
            chan_param,
            avg_pwr,
            preserve_gradients=preserve_gradients,
            deterministic=deterministic,
            deterministic_seed=deterministic_seed,
        )

    def forward(
        self,
        input_image,
        given_SNR=None,
        given_rate=None,
        adversarial_training=False,
        adversarial_testing=False,
        fgsm_epsilon=None,
        attack_method='fgsm',
        attack_mode='before_channel',
        jamming_power=0.1,
        snr_drop=5.0,
        is_bob: bool = True,
        is_eve: bool = False,
    ):
        """
        Forward pass with support for multiple adversarial attack modes.
        
        Args:
            attack_mode: Type of adversarial attack
                - 'before_channel': Attack clean feature before channel (standard)
                - 'after_channel': Attack noisy feature after channel (most realistic)
                - 'jamming': Add strong random interference (military)
                - 'snr_degradation': Amplify channel noise (stealthy)
            jamming_power: Power multiplier for jamming attack
            snr_drop: SNR degradation in dB for snr_degradation mode
        """
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        if given_rate is None:
            channel_number = choice(self.channel_number)
        else:
            channel_number = given_rate

        # Store input for adversarial attack
        self._current_input = input_image
        
        # Determine epsilon for testing
        test_epsilon = fgsm_epsilon if fgsm_epsilon is not None else self.epsilon

        # Determine attack method (testing can override model default)
        test_attack_method = attack_method if attack_method is not None else self.adv_attack

        if self.model == 'SwinJSCC_w_o_SAandRA' or self.model == 'SwinJSCC_w_SA':
            feature = self.encoder(input_image, chan_param, channel_number, self.model, preserve_gradients=False)
            CBR = feature.numel() / 2 / input_image.numel()
            mask = None  # No mask for non-RA models
            
        elif self.model == 'SwinJSCC_w_RA' or self.model == 'SwinJSCC_w_SAandRA':
            feature, mask = self.encoder(input_image, chan_param, channel_number, self.model, preserve_gradients=False)
            CBR = channel_number / (2 * 3 * 2 ** (self.downsample * 2))
            avg_pwr = torch.sum(feature ** 2) / mask.sum()
            
        # Handle adversarial training for all supported models
        if adversarial_training and self.training and self.model in ['SwinJSCC_w_o_SAandRA', 'SwinJSCC_w_SA', 'SwinJSCC_w_RA', 'SwinJSCC_w_SAandRA']:
            # If AN is enabled, use the same AN realization for the clean/adv pair.
            # This avoids confounding the adversarial objective for Eve-mode runs.
            pair_an_nonce = None
            if self.enable_an and self.an_generator is not None:
                pair_an_nonce = self._an_nonce
                self._an_nonce += 1
            if mask is not None:
                # RA model: apply mask first, then normalize
                masked_feature = feature * mask.float()
                ra_avg_pwr = torch.sum(masked_feature ** 2) / mask.float().sum()
                pwr = ra_avg_pwr * 2
                normalized = masked_feature / torch.sqrt(pwr)
            else:
                # Non-RA model: standard complex_normalize
                normalized, pwr = self.channel.complex_normalize(feature, power=1)
            
            # 1. Clean forward pass (maintain gradients for encoder training)
            clean_noisy_feature = self._maybe_pass_channel_post_normalize(
                normalized,
                pwr,
                chan_param,
                is_bob=is_bob,
                an_nonce=pair_an_nonce,
                preserve_gradients=False,
                deterministic=False,
            )
            if mask is not None:
                clean_noisy_feature = clean_noisy_feature * mask.float()

            # detach x, allowing loss gradients to flow back to the encoder.
            clean_recon = self.decoder(clean_noisy_feature, chan_param, self.model, preserve_gradients=True)
            clean_loss = self.distortion_loss_wrapper(input_image, clean_recon.clamp(0., 1.))
            
            # 2. Generate adversarial perturbation in NORMALIZED space (FGSM or PGD)
            batch_det_seed = self._det_seed_counter * 1000003 + 7
            self._det_seed_counter += 1
            # statistics match the inference-time distribution the attack targets.
            was_training = self.training
            self.eval()
            with torch.enable_grad():
                # Use a detached leaf so attack gradients flow w.r.t. delta, not model weights.
                normalized_for_attack = normalized.detach().requires_grad_(True)
                if self.adv_attack == 'pgd':
                    adv_perturbation = self.generate_pgd_perturbation(
                        normalized_for_attack, pwr, chan_param, mask,
                        deterministic_seed=batch_det_seed,
                        is_bob=is_bob,
                        an_nonce=pair_an_nonce,
                    )
                else:
                    adv_perturbation = self.generate_fgsm_perturbation(
                        normalized_for_attack, pwr, chan_param, mask,
                        deterministic_seed=batch_det_seed,
                        is_bob=is_bob,
                        an_nonce=pair_an_nonce,
                    )
            # Restore training mode for the weight-update forward pass
            if was_training:
                self.train()
            adv_perturbation = adv_perturbation.detach()
            
            # 3. Adversarial forward pass: add delta in normalized space
            adv_normalized = normalized + adv_perturbation  # Keep gradients to train encoder/decoder
            
            # Debug: Check perturbation during training
            if torch.rand(1).item() < 0.001:  # Log occasionally
                perturbation_magnitude = adv_perturbation.norm().item()
                normalized_magnitude = normalized.norm().item()
                print(f"Adversarial Training Debug: perturbation magnitude={perturbation_magnitude:.6f}, normalized magnitude={normalized_magnitude:.6f}, ratio={perturbation_magnitude/max(normalized_magnitude, 1e-8):.6f}")
            
            adv_noisy_feature = self._maybe_pass_channel_post_normalize(
                adv_normalized,
                pwr,
                chan_param,
                is_bob=is_bob,
                an_nonce=pair_an_nonce,
                preserve_gradients=False,
                deterministic=True,
                deterministic_seed=batch_det_seed,
            )
            if mask is not None:
                adv_noisy_feature = adv_noisy_feature * mask.float()

            adv_recon = self.decoder(adv_noisy_feature, chan_param, self.model, preserve_gradients=True)
            adv_loss = self.distortion_loss_wrapper(input_image, adv_recon.clamp(0., 1.))
            
            # 4. Combined loss with alpha weighting
            total_loss = self.alpha * clean_loss + (1 - self.alpha) * adv_loss
            
            # Debug: Check loss values and attack effectiveness
            clean_loss_val = clean_loss.mean().item()
            adv_loss_val = adv_loss.mean().item()
            total_loss_val = total_loss.mean().item()
            attack_effective = adv_loss_val > clean_loss_val
            loss_increase = adv_loss_val - clean_loss_val
            
            if torch.rand(1).item() < 0.001:  # Log occasionally
                print(f"[ADV-DIAG] clean_loss={clean_loss_val:.6f} adv_loss={adv_loss_val:.6f} "
                      f"total_loss={total_loss_val:.6f} attack_effective={attack_effective} "
                      f"loss_increase={loss_increase:.6f} alpha={self.alpha}")
            
            # Use adversarial reconstruction for evaluation metrics (better robustness assessment)
            recon_image = adv_recon
            loss_G = total_loss
            
            # Store separate loss components for tracking
            self._clean_loss = clean_loss.mean()
            self._adv_loss = adv_loss.mean()
            self._total_loss = total_loss.mean()
            
            # Clean up intermediate tensors to free memory
            del normalized_for_attack, adv_perturbation
            
        else:
            # Standard forward pass (non-adversarial training) or adversarial testing
            if adversarial_testing and not self.training:
                # ====================================================================
                # ADVERSARIAL TESTING MODE: Multiple attack scenarios
                # ====================================================================
                
                if attack_mode == 'before_channel':
                    # ==============================================================
                    # MODE 1: BEFORE CHANNEL (Standard)
                    # Attack in NORMALIZED space so perturbation survives power norm.
                    # ==============================================================
                    eval_det_seed = self._det_seed_counter * 1000003 + 13
                    self._det_seed_counter += 1

                    # Normalize feature ONCE (same fix as adversarial training)
                    if mask is not None:
                        masked_feature = feature * mask.float()
                        ra_avg_pwr = torch.sum(masked_feature ** 2) / mask.float().sum()
                        pwr = ra_avg_pwr * 2
                        normalized = masked_feature / torch.sqrt(pwr)
                    else:
                        normalized, pwr = self.channel.complex_normalize(feature, power=1)

                    with torch.enable_grad():
                        normalized_for_attack = normalized.clone().detach().requires_grad_(True)

                        if test_attack_method == 'pgd':
                            delta = self.generate_pgd_perturbation(
                                normalized_for_attack,
                                pwr,
                                chan_param,
                                mask,
                                epsilon=test_epsilon,
                                step_size=self.pgd_step_size,
                                num_steps=self.pgd_steps,
                                random_start=self.pgd_random_start,
                                deterministic_seed=eval_det_seed,
                                is_bob=is_bob,
                            )
                        else:
                            delta = self.generate_fgsm_perturbation(
                                normalized_for_attack, pwr, chan_param, mask,
                                epsilon=test_epsilon, deterministic_seed=eval_det_seed,
                                is_bob=is_bob,
                            )

                    # Apply perturbation in normalized space, then pass through post-normalize channel
                    adv_normalized = normalized.detach() + delta.detach()

                    noisy_feature = self._maybe_pass_channel_post_normalize(
                        adv_normalized,
                        pwr,
                        chan_param,
                        is_bob=is_bob,
                        an_nonce=None,
                        preserve_gradients=False,
                        deterministic=True,
                        deterministic_seed=eval_det_seed,
                    )
                    if mask is not None:
                        noisy_feature = noisy_feature * mask.float()
                    
                    recon_image = self.decoder(noisy_feature, chan_param, self.model, preserve_gradients=False)
                    loss_G = self.distortion_loss_wrapper(input_image, recon_image.clamp(0., 1.))
                
                elif attack_mode == 'after_channel':
                    # ==============================================================
                    # MODE 2: AFTER CHANNEL (Most Realistic)
                    # Pass through channel first, THEN attack the noisy feature
                    # ==============================================================
                    
                    # Step 1: Pass through channel first (physics happens first)
                    if mask is not None:
                        channel_feature = feature * mask.float()
                        avg_pwr = torch.sum(channel_feature ** 2) / mask.float().sum()
                        noisy_feature = self._maybe_pass_channel(
                            channel_feature,
                            chan_param,
                            is_bob=is_bob,
                            an_nonce=None,
                            avg_pwr=avg_pwr,
                            preserve_gradients=False,
                            deterministic=False,
                        )
                        noisy_feature = noisy_feature * mask.float()
                    else:
                        noisy_feature = self._maybe_pass_channel(
                            feature,
                            chan_param,
                            is_bob=is_bob,
                            an_nonce=None,
                            avg_pwr=False,
                            preserve_gradients=False,
                            deterministic=False,
                        )
                    
                    # Step 2: Generate FGSM/PGD attack on the NOISY feature
                    with torch.enable_grad():
                        noisy_base = noisy_feature.clone().detach()

                        if test_attack_method == 'pgd':
                            if self.pgd_random_start:
                                delta = (2 * torch.rand_like(noisy_base) - 1) * float(test_epsilon)
                            else:
                                delta = torch.zeros_like(noisy_base)
                            delta = delta.detach()
                            step_size = float(self.pgd_step_size) if self.pgd_step_size is not None else float(test_epsilon) / max(int(self.pgd_steps), 1)

                            for _ in range(int(self.pgd_steps)):
                                delta.requires_grad_(True)
                                noisy_for_attack = noisy_base + delta
                                temp_recon = self.decoder(noisy_for_attack, chan_param, self.model, preserve_gradients=True)
                                attack_loss = self.distortion_loss_wrapper(input_image, temp_recon.clamp(0.0, 1.0))
                                grad_delta = torch.autograd.grad(attack_loss.mean(), delta, retain_graph=False, create_graph=False)[0]
                                with torch.no_grad():
                                    delta = delta + step_size * grad_delta.sign()
                                    delta = delta.clamp(-float(test_epsilon), float(test_epsilon))
                                delta = delta.detach()
                        else:
                            noisy_for_attack = noisy_base.requires_grad_(True)
                            temp_recon = self.decoder(noisy_for_attack, chan_param, self.model, preserve_gradients=True)
                            attack_loss = self.distortion_loss_wrapper(input_image, temp_recon.clamp(0.0, 1.0))
                            grad_noisy = torch.autograd.grad(attack_loss.mean(), noisy_for_attack, retain_graph=False, create_graph=False)[0]
                            delta = float(test_epsilon) * grad_noisy.sign()

                    # Step 3: Add perturbation to noisy feature
                    noisy_feature = noisy_feature + delta.detach()
                    
                    # Step 4: Decode
                    recon_image = self.decoder(noisy_feature, chan_param, self.model, preserve_gradients=False)
                    loss_G = self.distortion_loss_wrapper(input_image, recon_image.clamp(0., 1.))
                
                elif attack_mode == 'jamming':
                    # ==============================================================
                    # MODE 3: JAMMING ATTACK (Military/Security)
                    # Add strong random interference
                    # ==============================================================
                    
                    if mask is not None:
                        jamming_noise = torch.randn_like(feature) * jamming_power * mask
                        jammed_feature = feature + jamming_noise
                        avg_pwr_jammed = torch.sum(jammed_feature ** 2) / mask.sum()
                        noisy_feature = self._maybe_pass_channel(
                            jammed_feature,
                            chan_param,
                            is_bob=is_bob,
                            an_nonce=None,
                            avg_pwr=avg_pwr_jammed,
                            preserve_gradients=False,
                            deterministic=False,
                        )
                        noisy_feature = noisy_feature * mask
                    else:
                        jamming_noise = torch.randn_like(feature) * jamming_power
                        jammed_feature = feature + jamming_noise
                        noisy_feature = self._maybe_pass_channel(
                            jammed_feature,
                            chan_param,
                            is_bob=is_bob,
                            an_nonce=None,
                            avg_pwr=False,
                            preserve_gradients=False,
                            deterministic=False,
                        )
                    
                    # Decode
                    recon_image = self.decoder(noisy_feature, chan_param, self.model, preserve_gradients=False)
                    loss_G = self.distortion_loss_wrapper(input_image, recon_image.clamp(0., 1.))
                
                elif attack_mode == 'snr_degradation':
                    # ==============================================================
                    # MODE 4: SNR DEGRADATION (Stealthy)
                    # Amplify channel noise by reducing effective SNR
                    # ==============================================================
                    
                    # Calculate effective SNR (degraded)
                    effective_snr = chan_param - snr_drop
                    
                    if mask is not None:
                        avg_pwr = torch.sum(feature ** 2) / mask.sum()
                        if self.pass_channel:
                            noisy_feature = self._maybe_pass_channel(
                                feature,
                                effective_snr,
                                is_bob=is_bob,
                                an_nonce=None,
                                avg_pwr=avg_pwr,
                                preserve_gradients=False,
                                deterministic=False,
                            )
                        else:
                            # If channel is disabled, simulate SNR degradation manually
                            noise_std = 10 ** (-effective_snr / 20.0)
                            noise = torch.randn_like(feature) * noise_std
                            noisy_feature = feature + noise
                        noisy_feature = noisy_feature * mask
                    else:
                        if self.pass_channel:
                            noisy_feature = self._maybe_pass_channel(
                                feature,
                                effective_snr,
                                is_bob=is_bob,
                                an_nonce=None,
                                avg_pwr=False,
                                preserve_gradients=False,
                                deterministic=False,
                            )
                        else:
                            # If channel is disabled, simulate SNR degradation manually
                            noise_std = 10 ** (-effective_snr / 20.0)
                            noise = torch.randn_like(feature) * noise_std
                            noisy_feature = feature + noise
                    
                    # Decode
                    recon_image = self.decoder(noisy_feature, chan_param, self.model, preserve_gradients=False)
                    loss_G = self.distortion_loss_wrapper(input_image, recon_image.clamp(0., 1.))
                
                else:
                    raise ValueError(f"Unknown attack mode: {attack_mode}")
                    
            else:
                # Standard forward pass (clean training/testing)
                if mask is not None:
                    noisy_feature = self._maybe_pass_channel(
                        feature,
                        chan_param,
                        is_bob=is_bob,
                        an_nonce=None,
                        avg_pwr=avg_pwr,
                        preserve_gradients=False,
                        deterministic=False,
                    )
                    noisy_feature = noisy_feature * mask
                else:
                    noisy_feature = self._maybe_pass_channel(
                        feature,
                        chan_param,
                        is_bob=is_bob,
                        an_nonce=None,
                        avg_pwr=False,
                        preserve_gradients=False,
                        deterministic=False,
                    )

                recon_image = self.decoder(noisy_feature, chan_param, self.model, preserve_gradients=False)
                mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
                loss_G = self.distortion_loss_wrapper(input_image, recon_image.clamp(0., 1.)).mean()

        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()

