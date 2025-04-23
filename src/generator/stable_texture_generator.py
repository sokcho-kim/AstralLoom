from abc import ABC, abstractmethod
from typing import Dict
from PIL import Image
import os
from diffusers import StableDiffusionPipeline

# 공통 인터페이스
class BaseTextureGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Image.Image]:
        pass

# Stable Diffusion 기반 Albedo 생성기
class StableTextureGenerator(BaseTextureGenerator):
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype="auto"
        ).to("cuda")

    def generate(self, prompt: str, **kwargs):
        result = self.pipe(prompt)
        image = result.images[0]
        return {"albedo": image}

# ControlNet Normal Preprocessor (backup용)
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

# TexFusion Generator (보류 중)
class TexFusionGenerator(BaseTextureGenerator):
    def __init__(self, model_path="path/to/texfusion/weights.pth"):
        import texfusion
        self.model = texfusion.load_model(model_path)

    def generate(self, prompt: str, **kwargs):
        albedo, normal = self.model.infer(prompt)
        return {"albedo": albedo, "normal": normal}

# DreamTexture Generator (보류 중)
class DreamTextureGenerator(BaseTextureGenerator):
    def __init__(self, model_path="path/to/dreamtexture/weights.pth"):
        import dreamtexture
        self.model = dreamtexture.load_model(model_path)

    def generate(self, prompt: str, **kwargs):
        albedo, normal = self.model.infer(prompt)
        return {"albedo": albedo, "normal": normal}
