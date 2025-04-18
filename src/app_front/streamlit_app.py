import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64

st.title("AstralLoom - Game Asset Texture Generator")

# 입력창
prompt = st.text_input("Enter your prompt:", "a grain of wood skin texture, seamless, UV unwrapped, flat lighting, lowpoly style")

if st.button("Generate Texture"):
    if prompt:
        with st.spinner('Generating texture...'):
            # FastAPI 서버에 요청
            response = requests.post(
                "http://localhost:8000/generate",
                json={"prompt": prompt}
            )

            if response.status_code == 200:
                data = response.json()
                image_base64 = data["image_base64"]

                # base64 디코딩
                image_data = base64.b64decode(image_base64)
                image = Image.open(BytesIO(image_data))

                # 이미지 보여주기
                st.image(image, caption="Generated Texture", use_container_width=True)

                # 다운로드 버튼
                st.download_button(
                    label="Download Texture",
                    data=image_data,
                    file_name="texture.png",
                    mime="image/png"
                )
            else:
                st.error(f"Failed to generate texture: {response.text}")
