from diffusers import StableDiffusionPipeline
import torch

# 파일 경로
model_path = "C:/stable-diffusion-webui-1/models/Stable-diffusion/lowpoly_flux.safetensors"
save_path = "C:/stable-diffusion-webui-1/models/diffusers_flux/"

# 모델 로드 (GPU 자동 매핑 추가!!)
pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"  # 이거 추가!
)

# 저장 (Diffusers 포맷)
pipe.save_pretrained(save_path)
print("모델 변환 완료!")
