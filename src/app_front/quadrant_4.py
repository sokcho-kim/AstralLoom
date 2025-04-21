import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generator.generate_4quadrant import generate_4quadrant_texture


st.set_page_config(page_title="AstralLoom Texture Generator", layout="centered")

st.title("🧵 AstralLoom Texture Generator")
st.markdown("로우폴리 게임 텍스처 4사분면 이미지 생성기입니다.")

# 1. 사용자 입력 받기
with st.form("prompt_form"):
    prompt1 = st.text_input("1사분면 (정면+윗면)", "wooden planks, top view")
    prompt2 = st.text_input("2사분면 (변형1)", "wooden planks, mossy texture")
    prompt3 = st.text_input("3사분면 (변형2)", "wooden planks, cracked paint")
    prompt4 = st.text_input("4사분면 (옆면, seamless)", "wood side texture, seamless")

    submitted = st.form_submit_button("🧪 Generate Texture")

if submitted:
    with st.spinner("텍스처 생성 중... 잠시만 기다려주세요 🌀"):
        prompts = [prompt1, prompt2, prompt3, prompt4]
        image_path = generate_4quadrant_texture(prompts)
    
    st.success("생성 완료!")
    st.image(image_path, caption="🖼️ Generated 4-Quadrant Texture", use_column_width=True)

    # 다운로드 버튼
    with open(image_path, "rb") as f:
        st.download_button(
            label="📥 이미지 다운로드",
            data=f,
            file_name=os.path.basename(image_path),
            mime="image/png"
        )
