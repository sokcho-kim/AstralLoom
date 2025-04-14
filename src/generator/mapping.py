import torch
from safetensors.torch import load_file
from src.generator.flux_transformer_unet import FluxTransformerUNet

# 1. 모델과 state_dict 로드
model = FluxTransformerUNet()
state_dict = load_file("C:/AstralLoom/assets/models/lowpoly_flux.safetensors")

# 2. state_dict 키 확인
model_keys = model.state_dict().keys()

# 3. 매핑 시작
mapped_dict = {}

for model_key in model_keys:
    # safetensors에서 비슷한 키 찾기
    matched_keys = [k for k in state_dict.keys() if model_key.split(".")[-1] in k]

    if matched_keys:
        matched_key = matched_keys[0]  # 가장 비슷한 거 하나 고르기
        mapped_dict[model_key] = state_dict[matched_key]
    else:
        print(f"매칭 실패: {model_key}")

# 4. 모델에 가중치 로드
model.load_state_dict(mapped_dict, strict=False)

print("✅ FluxTransformerUNet 매핑 완료")
