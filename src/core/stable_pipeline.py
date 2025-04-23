import torch
from diffusers import StableDiffusionPipeline

class StableTextureGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to(device)
        self.device = device

    def generate(self, prompt: str, height: int = 512, width: int = 512, steps: int = 25):
        result = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps
        )
        return result.images[0]  # PIL Image

if __name__ == "__main__":
    generator = StableTextureGenerator()
    prompt = "stylized stone tiles with lava and fire, game texture"
    image = generator.generate(prompt)

    image.save("../assets/outputs/stone_tiles3.png")
    print("✅ 생성 완료!")
