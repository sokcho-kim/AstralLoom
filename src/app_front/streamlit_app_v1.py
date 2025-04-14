import base64
import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title("AstralLoom Texture Generator")

prompt = st.text_input("Enter your prompt:", "cute frog UV map")

if st.button("Generate Texture"):
    if prompt:
        with st.spinner('Generating...'):
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

                # 다운로드 버튼 추가
                btn = st.download_button(
                    label="Download Texture",
                    data=image_data,
                    file_name="texture.png",
                    mime="image/png"
                )
            else:
                st.error(f"Failed to generate texture: {response.text}")
