import sys
import os
from PIL import Image
from generator.texture_creator import generate_texture

def generate_4quadrant_texture(prompts, output_dir="assets/outputs/", final_name="merged_texture.png"):
    assert len(prompts) == 4, "프롬프트는 반드시 4개여야 합니다."

    os.makedirs(output_dir, exist_ok=True)
    img_paths = []

    # 1. 4개 텍스처 생성
    for i, prompt in enumerate(prompts):
        path = generate_texture(prompt, output_dir)
        img_paths.append(path)

    # 2. 4개 이미지 로드
    imgs = [Image.open(p) for p in img_paths]

    # 3. 빈 1024x1024 캔버스 생성
    canvas = Image.new('RGB', (1024, 1024))

    # 4. 위치별로 붙이기
    canvas.paste(imgs[0], (0, 0))       # 1사분면
    canvas.paste(imgs[1], (512, 0))     # 2사분면
    canvas.paste(imgs[2], (0, 512))     # 3사분면
    canvas.paste(imgs[3], (512, 512))   # 4사분면

    # 5. 저장
    save_path = os.path.join(output_dir, final_name)
    canvas.save(save_path)
    print(f"✅ 합쳐진 텍스처 저장 완료: {save_path}")

    return save_path
