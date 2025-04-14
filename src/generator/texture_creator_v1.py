# from diffusers import StableDiffusionPipeline
# import torch
# import os

# # 모델 로딩 함수
# def load_model(model_path="assets/models/lowpoly_flux.safetensors"):
#     # StableDiffusionPipeline에 로컬 모델 불러오기
#     pipe = StableDiffusionPipeline.from_single_file(
#         model_path,
#         torch_dtype=torch.float16,    # GPU 메모리 절약 (필요하면 float32로 바꿔도 됨)
#         safety_checker=None           # 안전 필터 끔 (텍스쳐 작업용이니까)
#     ).to("cuda" if torch.cuda.is_available() else "cpu")
    
#     pipe.enable_xformers_memory_efficient_attention()  # (선택) 메모리 최적화
#     return pipe

# # 텍스쳐 생성 함수
# def generate_texture(prompt, output_dir="assets/outputs/"):
#     # 모델 경로
#     model_path = "assets/models/lowpoly_flux.safetensors"  
    
#     # 모델 로드
#     pipe = load_model(model_path)

#     # 프롬프트로 이미지 생성
#     with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
#         image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]

#     # 저장 폴더 만들기
#     os.makedirs(output_dir, exist_ok=True)

#     # 저장 경로
#     save_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}.png")
#     image.save(save_path)

#     return save_path

###  이건 되는거 ###
from diffusers import StableDiffusionPipeline
import torch
import os

# def load_model():
#     model_id = "Lykon/dreamshaper-8"
#     pipe = StableDiffusionPipeline.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16,   # 너 GPU 있다고 했으니까 무조건 float16
#         safety_checker=None
#     ).to("cuda")

#     # xformers 사용하려고 하는데 없으면 그냥 넘어가게 (에러 막기)
#     try:
#         pipe.enable_xformers_memory_efficient_attention()
#     except Exception as e:
#         print(f"xformers not available: {e}")

#     return pipe


# def load_model():
#     model_id = "runwayml/stable-diffusion-v1-5"  # 텍스처 드리머 모델
#     pipe = StableDiffusionPipeline.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16,     # GPU니까 무조건 float16
#         safety_checker=None
#     ).to("cuda")

#     try:
#         pipe.enable_xformers_memory_efficient_attention()
#     except Exception as e:
#         print(f"xformers not available: {e}")

#     return pipe


def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"  # Stable Diffusion 1.5
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,    # 너는 GPU 있으니까 무조건 float16
        safety_checker=None
    ).to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"xformers not available: {e}")

    return pipe



# 텍스쳐 생성
def generate_texture(prompt, output_dir="assets/outputs/"):
    # 모델 로딩
    pipe = load_model()

    # 텍스쳐 생성
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]

    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}.png")
    image.save(save_path)

    return save_path

