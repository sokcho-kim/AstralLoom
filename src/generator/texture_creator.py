# from generator.flux_pipeline import FluxWithCFGPipeline
# import torch
# import os

# # C:\AstralLoom\assets\models\j_cute3d_flux.safetensors
# def load_flux_model(model_path="C:/AstralLoom/assets/models/j_cute3d_flux.safetensors"):
#     pipe = FluxWithCFGPipeline.from_single_file(
#         model_path,
#         torch_dtype=torch.float16,
#         safety_checker=None,
#         variant="fp16",
#     ).to("cuda")

#     return pipe

# def generate_texture(prompt, output_dir="assets/outputs/"):
#     pipe = load_flux_model()

#     image = pipe.generate_images(
#         prompt=prompt,
#         height=512,
#         width=512,
#         num_inference_steps=20,
#         guidance_scale=3.5,
#     )

#     os.makedirs(output_dir, exist_ok=True)
#     save_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}.png")
#     image.save(save_path)

#     return save_path

# from diffusers import FluxPipeline, AutoencoderKL
# from transformers import CLIPTextModel, CLIPTokenizer
# from generator.flux_pipeline import FluxWithCFGPipeline
# import torch
# import os
# from PIL import Image

# def load_flux_model(model_path="C:/AstralLoom/assets/models/lowpoly_flux.safetensors"):
#     # 1. 텍스트 인코더와 토크나이저 로드
#     text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
#     tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

#     # 2. VAE 로드 (Stable Diffusion V1.5용)
#     vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)

#     # 3. Flux 모델 로드
#     pipe = FluxWithCFGPipeline.from_single_file(
#         model_path,
#         text_encoder=text_encoder,
#         # tokenizer=tokenizer,
#         vae=vae,
#         torch_dtype=torch.float16,
#     ).to("cuda")

#     return pipe

# from generator.flux_pipeline import FluxWithCFGPipeline
# from transformers import CLIPTextModel, CLIPTokenizer
# from diffusers import AutoencoderKL
# from safetensors.torch import load_file
# import torch

# def load_flux_model(model_path="C:/AstralLoom/assets/models/lowpoly_flux.safetensors"):
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # 1. 텍스트 인코더 + 토크나이저
#     text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
#     tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

#     # 2. VAE
#     vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(device)

#     # 3. Flux transformer (커스텀 유닛)
#     state_dict = load_file(model_path)

#     # 4. 파이프라인 인스턴스 만들기
#     pipe = FluxWithCFGPipeline(
#         vae=vae,
#         text_encoder=text_encoder,
#         tokenizer=tokenizer,
#         # torch_dtype=torch.float16,
#     ).to(device).to(torch.float16)

#     # 5. 로드한 가중치 주입
#     pipe.load_state_dict(state_dict, strict=False)

#     return pipe

# def generate_texture(prompt, output_dir="assets/outputs/"):
#     pipe = load_flux_model()

#     image = pipe.generate_images(
#         prompt=prompt,
#         height=512,
#         width=512,
#         num_inference_steps=20,
#         guidance_scale=3.5,
#     )

#     # os.makedirs(output_dir, exist_ok=True)
#     # save_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}.png")
#     # image.save(save_path)

#     # 이미지 타입 처리
#     if isinstance(image, list):
#         image = image[0]
#     elif isinstance(image, dict) and "images" in image:
#         image = image["images"][0]

#     os.makedirs(output_dir, exist_ok=True)
#     filename = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt)
#     save_path = os.path.join(output_dir, f"{filename}.png")
#     image.save(save_path)

#     return save_path

import os
import re
from PIL import Image
import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel
from generator.flux_pipeline import FluxWithCFGPipeline

def load_flux_model(model_path="C:/AstralLoom/assets/models/lowpoly_flux.safetensors"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 구성 요소 수동 로딩
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(device)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    # Flux pipeline 로드 (커스텀 클래스 사용)
    pipe = FluxWithCFGPipeline.from_single_file(
        model_path,
        vae=vae,
        text_encoder=text_encoder,
        torch_dtype=torch.float16,
    ).to(device)

    return pipe

def generate_texture(prompt, output_dir="assets/outputs/"):
    pipe = load_flux_model()

    image = pipe.generate_images(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=20,
        guidance_scale=3.5,
    )

    # 저장 경로 처리
    os.makedirs(output_dir, exist_ok=True)
    filename = re.sub(r"[^a-zA-Z0-9_-]", "_", prompt)
    save_path = os.path.join(output_dir, f"{filename}.png")
    image.save(save_path)

    return save_path
