import streamlit as st
import sys
import os

# 현재 파일 기준으로 상위 src 경로를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from generator.texfusion_dreamtexture_generator import (
    StableTextureGenerator,
    ControlNetNormalGenerator
)
from generator.texture_pipeline import TexturePipeline

st.set_page_config(page_title="AstralLoom Texture Generator", layout="centered")
st.title("AstralLoom 텍스쳐 생성기")
st.markdown("텍스트 프롬프트를 입력하여 Albedo 및 Normal 맵을 생성합니다.")

# 모델 선택 (TexFusion, DreamTexture 비활성화)
ALBEDO_GENERATORS = {
    "Stable Diffusion (사용 가능)": StableTextureGenerator
}

NORMAL_GENERATORS = {
    "없음": None,
    "ControlNet Normal": ControlNetNormalGenerator
}

albedo_model_label = st.selectbox("Albedo Generator 선택", list(ALBEDO_GENERATORS.keys()))
normal_model_label = st.selectbox("Normal Generator 선택", list(NORMAL_GENERATORS.keys()))

ModelClass_Albedo = ALBEDO_GENERATORS[albedo_model_label]
ModelClass_Normal = NORMAL_GENERATORS[normal_model_label]

# 프롬프트 입력
prompt = st.text_input("프롬프트 입력", "mossy stone tiles")
if st.button("생성 시작"):
    base_generator = ModelClass_Albedo()
    normal_generator = ModelClass_Normal() if ModelClass_Normal else None
    pipeline = TexturePipeline(base_generator, normal_generator)

    result = pipeline.generate(prompt)

    st.write("### 생성 결과")
    for key, img in result.items():
        st.image(img, caption=f"{key.capitalize()} Map")
        with open(f"{key}_output.png", "wb") as f:
            img.save(f, format="PNG")
        with open(f"{key}_output.png", "rb") as f:
            st.download_button(
                label=f"{key.capitalize()} 다운로드",
                data=f,
                file_name=f"{key}_output.png",
                mime="image/png"
            )
