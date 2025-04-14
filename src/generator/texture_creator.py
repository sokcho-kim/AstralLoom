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

from diffusers import FluxPipeline, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from generator.flux_pipeline import FluxWithCFGPipeline
import torch
import os

def load_flux_model(model_path="C:/AstralLoom/assets/models/lowpoly_flux.safetensors"):
    # 1. 텍스트 인코더와 토크나이저 로드
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # 2. VAE 로드 (Stable Diffusion V1.5용)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)

    # 3. Flux 모델 로드
    pipe = FluxWithCFGPipeline.from_single_file(
        model_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        torch_dtype=torch.float16,
    ).to("cuda")

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

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}.png")
    image.save(save_path)

    return save_path

