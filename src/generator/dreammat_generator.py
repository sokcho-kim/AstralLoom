from generator.base import BaseTextureGenerator
from PIL import Image
import torch

# Dummy 모델 정의 (DreamMat 모델 자리)
def load_dreammat_model():
    class DummyDreamMatModel:
        def generate_all_maps(self, prompt):
            dummy_image = Image.new("RGB", (512, 512), color=(200, 200, 200))  # 회색 dummy image
            return {
                "albedo": dummy_image,
                "normal": dummy_image,
                "roughness": dummy_image
            }
    return DummyDreamMatModel()

class DreamMatGenerator(BaseTextureGenerator):
    def __init__(self):
        # DreamMat 모델 로딩 (multi-output)
        self.model = load_dreammat_model().to("cuda")

    def generate(self, prompt: str, **kwargs):
        outputs = self.model.generate_all_maps(prompt)
        return {
            "albedo": outputs["albedo"],
            "normal": outputs["normal"],
            "roughness": outputs["roughness"]
        }