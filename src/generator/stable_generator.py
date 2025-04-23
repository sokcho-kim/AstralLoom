from diffusers import StableDiffusionPipeline
from generator.base import BaseTextureGenerator
import torch

class StableTextureGenerator(BaseTextureGenerator):
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    def generate(self, prompt: str, **kwargs):
        image = self.pipe(prompt).images[0]
        return { "albedo": image }      #(단일 albedo)
