import torch
import numpy as np
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from typing import Any, Dict, List, Optional, Union
from PIL import Image

# Constants
BASE_SEQ_LEN = 256
MAX_SEQ_LEN = 4096
BASE_SHIFT = 0.5
MAX_SHIFT = 1.2

def calculate_timestep_shift(image_seq_len: int) -> float:
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN)
    b = BASE_SHIFT - m * BASE_SEQ_LEN
    mu = image_seq_len * m + b
    return mu

def prepare_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    mu: Optional[float] = None,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)

    timesteps = scheduler.timesteps
    num_inference_steps = len(timesteps)
    return timesteps, num_inference_steps

class FluxWithCFGPipeline(FluxPipeline):
    @torch.inference_mode()
    def generate_images(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 3.5,
        output_type: Optional[str] = "pil",
    ):
        device = self._execution_device
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            device=device,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, _ = self.prepare_latents(
            1,  # batch_size
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_timestep_shift(image_seq_len)
        timesteps, num_inference_steps = prepare_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        for t in timesteps:
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                return_dict=False,
            )[0]
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            torch.cuda.empty_cache()

        return self._decode_latents_to_image(latents, height, width, output_type)

    def _decode_latents_to_image(self, latents, height, width, output_type):
        vae = self.vae
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]
        return self.image_processor.postprocess(image, output_type=output_type)[0]
