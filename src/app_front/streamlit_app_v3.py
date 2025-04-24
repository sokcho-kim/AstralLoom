# version 3. change models  
import streamlit as st
import sys
import os

# 현재 파일 기준으로 상위 src 경로를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from generator.stable_generator import StableTextureGenerator
from generator.dreammat_generator import DreamMatGenerator

# 클래스 형식으로 메뉴에 따라 모델 선택
MODEL_OPTIONS = {
    "Stable Diffusion v1.5": StableTextureGenerator,
    "DreamMat": DreamMatGenerator
}

st.set_page_config(page_title="AstralLoom Texture Generator", layout="centered")
st.title("텍스쳐 생성 프로토타입")
st.markdown("\nStable Diffusion 및 DreamMat 기반의 모델을 사용하여 텍스쳐를 생성합니다.\n")

# 모델 선택에 따라 각 사부명에 대해 프롬프트 입력하기
model_label = st.selectbox("모델 선택", list(MODEL_OPTIONS.keys()))
ModelClass = MODEL_OPTIONS[model_label]

# 4개 프롬프트 입력 포맷
with st.form("prompt_form"):
    prompts = []
    for i in range(4):
        prompts.append(st.text_input(f"{i+1}사분면 프롬프트", f"Example prompt {i+1}"))
    submitted = st.form_submit_button("생성 시작")

# 생성 발동
if submitted:
    generator = ModelClass()
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../assets/outputs"))
    os.makedirs(output_dir, exist_ok=True)

    st.write("### 생성 결과")
    cols = st.columns(4)

    for i, prompt in enumerate(prompts):
        result = generator.generate(prompt)
        for key, img in result.items():
            filename = f"{key}_q{i+1}.png"
            save_path = os.path.join(output_dir, filename)
            img.save(save_path)

            with cols[i]:
                st.image(img, caption=f"Q{i+1}: {key}")
                with open(save_path, "rb") as f:
                    st.download_button(
                        label=f"Download {key}",
                        data=f,
                        file_name=filename,
                        mime="image/png"
                    )
