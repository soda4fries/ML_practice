import torch

import numpy as np
import tqdm as tqdm
from Diffussion.SD.util import get_time_embedding, rescale
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(
    prompt: str,
    uncond_prompt: str,
    input_image=None,
    strength: float = 0.8,
    cfg: bool = True,
    guidance_scale: float = 7.5,
    sampler: str = "ddqm",
    n_inference_steps: int = 50,
    models={},
    seed: int = None,
    tokenizer=None,
    device: torch.device = None,
    idle_device: torch.device = None,
):

    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be between 0 and 1")
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed:
            generator.manual_seed(seed)
        else:
            generator.seed()

        clip = models["clip"]
        clip.to(device)

        if cfg:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_context)

            # (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])

        else:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            # (1,77,768)
            context = clip(cond_tokens)
        to_idle(clip)

        if sampler == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown Sampler: {sampler}")
        latent_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)

            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            input_image_tensor = input_image_tensor.unsqueeze(0).permute(0, 3, 1, 2)

            encoder_noise = torch.randn(
                latent_shape, generator=generator, device=device
            )

            latent = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latent = sampler.add_noise(latent, sampler.timesteps[0])
            to_idle(encoder)
        else:
            latent = torch.randn(latent_shape, generator=generator, device=device)

    diffusion = models["diffusion"]
    diffusion.to(device)

    timestamps = tqdm(sampler.timesteps)

    for i, timestamp in enumerate(timestamps):
        time_embedding = get_time_embedding(timestamp).to(device)

        model_input = latent

        if cfg:
            model_input = model_input.repeat(2, 1, 1, 1)

        model_output = diffusion(model_input, context, time_embedding)

        if cfg:
            output_cond, output_uncond = model_output.chuck(2)
            model_output = (
                guidance_scale * (output_cond - output_uncond) + output_uncond
            )

        latent = sampler.step(timestamp, latent, model_output)
    to_idle(diffusion)

    decoder = models["decoder"]
    decoder.to(device)

    output_image = decoder(latent)

    output_image = rescale(output_image, (-1, 1), (0, 255), clamp=True)

    output_image = output_image.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
    return output_image[0]
