from diffusers import StableDiffusionPipeline
import torch
import os

def load_model():
    # model_id = "luodian/OpenTexture-Diffusion"  # OpenTexture-Diffusion 모델
    # model_id = "GeorgeQi/Paint3d_UVPos_Control"  # Paint3d_UVPos_Control 모델
    # model_id = "dream-textures/texture-diffusion"  # dream-textures 모델
    model_id = "KingNish/Realtime-FLUX"  # Realtime-FLUX 모델
    # 방법 1. https://huggingface.co/dream-textures/texture-diffusion 이 모델을 학습 시킨다 
    # 방법 2. 1차로 2d생성 하게 하고 https://huggingface.co/spaces/stabilityai/stable-fast-3d 이거를 거치게 한다  
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,    # 너는 GPU 있으니까 float16
        safety_checker=None
    ).to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"xformers not available: {e}")

    return pipe

def generate_texture(prompt, output_dir="assets/outputs/"):
    pipe = load_model()

    # 텍스처 생성
    with torch.autocast("cuda"):
        image = pipe(
            prompt,
            num_inference_steps=30,   # 좀 더 디테일 높여서 생성
            guidance_scale=7.5
        ).images[0]

    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}.png")
    image.save(save_path)

    return save_path
