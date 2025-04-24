import os
from typing import Dict, List
from PIL import Image
from diffusers import StableDiffusionPipeline
from controlnet_aux.normal import NormalPreprocessor

class TexturePipeline:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype="auto"
        ).to("cuda")
        self.normal_preprocessor = NormalPreprocessor.from_pretrained("lllyasviel/Annotators")

    def generate_texture(self, prompts: List[str], output_dir="assets/outputs") -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        result_paths = {}

        for i, prompt in enumerate(prompts):
            result = self.pipe(prompt)
            image = result.images[0]

            texture_path = os.path.join(output_dir, f"texture_q{i+1}.png")
            image.save(texture_path)
            result_paths[f"albedo_q{i+1}"] = texture_path

            # Normal map 생성
            normal_map = self.normal_preprocessor(image)
            normal_path = os.path.join(output_dir, f"normal_q{i+1}.png")
            normal_map.save(normal_path)
            result_paths[f"normal_q{i+1}"] = normal_path

        return result_paths

if __name__ == "__main__":
    prompts = [
        "wood texture, top view, stylized",
        "wood texture variant 1, stylized",
        "wood texture variant 2, stylized",
        "wood texture, side view, seamless"
    ]

    pipeline = TexturePipeline()
    output = pipeline.generate_texture(prompts)
    print("Generated files:", output)
