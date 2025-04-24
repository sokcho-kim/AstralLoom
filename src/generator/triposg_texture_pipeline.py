import os
from typing import Dict, List
from PIL import Image
import torch
# Triposg model import (예시)
# from triposg import TriposgTextureGenerator  # <- Triposg 실제 import 경로에 맞게 수정 필요
from diffusers import DiffusionPipeline
from diffusers import RectifiedFlowScheduler

class TexturePipeline:
    def __init__(self, model_id="VAST-AI/TripoSG"):
        # self.pipe = DiffusionPipeline.from_pretrained(
        #     model_id,
        #     torch_dtype="auto"
        # ).to("cuda")

        # self.pipe = DiffusionPipeline.from_pretrained("VAST-AI/TripoSG").to("cuda")
        # self.pipe = DiffusionPipeline.from_pretrained(
        #     model_id,
        #     scheduler=None,
        #     torch_dtype=torch.float16
        # ).to("cuda")

        scheduler = RectifiedFlowScheduler.from_pretrained("VAST-AI/TripoSG", subfolder="scheduler")
        self.pipe = DiffusionPipeline.from_pretrained("VAST-AI/TripoSG", scheduler=scheduler).to("cuda")

        # prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
        # image = pipe(prompt).images[0]

    def generate_texture(self, prompts: List[str], output_dir="./assets/outputs") -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        result_paths = {}

        for i, prompt in enumerate(prompts):
            output = self.generator.generate(prompt)
            albedo = output["albedo"]
            normal = output.get("normal")
            roughness = output.get("roughness")
            ao = output.get("ao")

            # 저장 경로 설정
            albedo_path = os.path.join(output_dir, f"albedo_q{i+1}.png")
            normal_path = os.path.join(output_dir, f"normal_q{i+1}.png") if normal else None
            roughness_path = os.path.join(output_dir, f"roughness_q{i+1}.png") if roughness else None
            ao_path = os.path.join(output_dir, f"ao_q{i+1}.png") if ao else None

            # 이미지 저장
            albedo.save(albedo_path)
            if normal: normal.save(normal_path)
            if roughness: roughness.save(roughness_path)
            if ao: ao.save(ao_path)

            # 결과 경로 기록
            result_paths[f"albedo_q{i+1}"] = albedo_path
            if normal: result_paths[f"normal_q{i+1}"] = normal_path
            if roughness: result_paths[f"roughness_q{i+1}"] = roughness_path
            if ao: result_paths[f"ao_q{i+1}"] = ao_path

        return result_paths

if __name__ == "__main__":
    prompts = [
        "mossy stone tile, top view",
        "mossy stone tile, variation 1",
        "mossy stone tile, variation 2",
        "mossy stone tile, side view, seamless"
    ]

    pipeline = TexturePipeline()
    output = pipeline.generate_texture(prompts)
    print("Generated files:", output)
