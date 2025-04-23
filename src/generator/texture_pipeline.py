from abc import ABC, abstractmethod
from typing import Dict
from PIL import Image

# 공통 인터페이스
class BaseTextureGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Image.Image]:
        pass

# Stable Diffusion 기반 Albedo 생성기
class StableTextureGenerator(BaseTextureGenerator):
    def __init__(self):
        # Stable Diffusion 모델 로딩 
        from diffusers import StableDiffusionPipeline
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

    def generate(self, prompt: str, **kwargs):
        image = self.pipe(prompt).images[0]
        return {"albedo": image}

# ControlNet 기반 Normal 생성기 
class ControlNetNormalGenerator(BaseTextureGenerator):
    def __init__(self):
        from controlnet_aux.normal import NormalPreprocessor
        self.preprocessor = NormalPreprocessor.from_pretrained("lllyasviel/ControlNet")

    def generate(self, prompt: str, **kwargs):
        albedo = kwargs.get("albedo")
        if albedo is None:
            raise ValueError("ControlNetNormalGenerator requires 'albedo' as input.")
        normal_map = self.preprocessor(albedo)
        return {"normal": normal_map}

# MaterialGAN 기반 Normal 생성기 (dummy)
class MaterialGANGenerator(BaseTextureGenerator):
    def __init__(self):
        pass  # GAN 모델 로딩 예정

    def generate(self, prompt: str, **kwargs):
        dummy_image = Image.new("RGB", (512, 512), color=(150, 150, 255))  # Dummy normal map
        return {"normal": dummy_image}

# 파이프라인 본체
# class TexturePipeline:
#     def __init__(self, base_generator, normal_generator=None):
#         self.base_generator = base_generator
#         self.normal_generator = normal_generator

#     def generate(self, prompt: str):
#         result = {}
#         # Albedo 생성
#         result.update(self.base_generator.generate(prompt))

#         # Normal map 생성 (ControlNet or MaterialGAN)
#         if self.normal_generator:
#             result.update(self.normal_generator.generate(prompt, albedo=result["albedo"]))

#         return result

class TexturePipeline:
    def __init__(self, base_generator: BaseTextureGenerator, normal_generator: BaseTextureGenerator = None):
        self.base_generator = base_generator               # Albedo generator
        self.normal_generator = normal_generator           # Normal generator (optional)

    def generate(self, prompt: str):
        result = {}

        # Step 1: Albedo 생성
        result.update(self.base_generator.generate(prompt))

        # Step 2: Normal map 생성 (ControlNet or others)
        if self.normal_generator:
            result.update(self.normal_generator.generate(prompt, albedo=result["albedo"]))

        return result
