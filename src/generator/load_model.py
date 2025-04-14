from safetensors.torch import load_file

# 1. Flux 모델 파일 경로
model_path = "C:/AstralLoom/assets/models/lowpoly_flux.safetensors"

# 2. safetensors 파일 로드
state_dict = load_file(model_path)

# 3. 키 목록을 파일로 저장
with open("flux_model_keys.txt", "w") as f:
    for key in state_dict.keys():
        f.write(key + "\n")

print("키 리스트 저장 완료: flux_model_keys.txt")
