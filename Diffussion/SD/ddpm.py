import torch
import numpy as np


class DDPMSampler:

    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):
        # scaled linear schedule
        self.betas = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32
            )
            ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alpha_cum = torch.cumprod(self.alphas, 0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timestamps = torch.arange(num_training_steps - 1, -1, -1, dtype=torch.int)

    def set_inference_timesteps(self, num_inferance_steps=50):
        self.num_inferance_steps = num_inferance_steps
        step_ratio = self.num_training_steps // self.num_training_steps
        self.timestamps = np.linspace(
            0, self.num_training_steps - step_ratio, self.num_training_steps, dtype=int
        )[::-1]

    def add_noise(self, orig_sample: torch.FloatTensor, timestamps: torch.IntTensor):
        alpha_cumprod = self.alpha_cum.to(
            device=orig_sample.device, dtype=orig_sample.dtype
        )
        timestamps = timestamps.to(orig_sample.device)  # type: ignore

        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[timestamps])
        one_minus_sqrt_alpha_cumprod = torch.sqrt(1 - alpha_cumprod[timestamps])

        expand_dims = (-1,) + (1,) * (orig_sample.dim() - 1)
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(expand_dims)
        one_minus_sqrt_alpha_cumprod = one_minus_sqrt_alpha_cumprod.view(expand_dims)

        noise = torch.randn(
            orig_sample.shape,
            generator=self.generator,
            device=orig_sample.device,
            dtype=orig_sample.dtype,
        )

        noisy_sample = (sqrt_alpha_cumprod * orig_sample) + (
            one_minus_sqrt_alpha_cumprod * noise
        )

        return noisy_sample

    def _get_previous_timestep(self, timestamp: int):
        prev_t = timestamp - self.num_training_steps // self.num_inferance_steps
        return prev_t
    
    
    def set_strength(self, strength=1):
        start_step = self.num_inferance_steps - int(self.num_inferance_steps * strength)
        self.timestamps = self.timestamps[start_step:]
        self.start_step = start_step
        

    def step(self, timestep: int, latent: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_cum[t]
        alpha_prod_t_prev = (
            self.alpha_cum[prev_t]
            if prev_t >= 0
            else torch.tensor(1.0, device=self.betas.device)
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        pred_origin_sample = (
            torch.sqrt(latent - beta_prod_t) * model_output
        ) / torch.sqrt(alpha_prod_t)

        pred_orig_coeff = (torch.sqrt(alpha_prod_t_prev) * current_beta_t) / beta_prod_t
        current_sample_coeff = (
            torch.sqrt(current_alpha_t) * (beta_prod_t_prev) / beta_prod_t
        )

        pred_prev_sample = (
            pred_origin_sample * pred_orig_coeff + current_sample_coeff * latent
        )

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)

        if t > 0:
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            variance = torch.sqrt(variance) * noise
        
        pred_prev_sample = pred_prev_sample + variance
            
        return pred_prev_sample