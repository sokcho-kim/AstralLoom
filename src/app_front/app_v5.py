import streamlit as st
from PIL import Image
# from triposg import TriposgTextureGenerator  # 실제 Triposg 경로에 맞게 수정 필요
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generator.triposg_texture_pipeline import TexturePipeline

# Generator Class
class TriposgTexturePipeline:
    def __init__(self):
        self.generator = TexturePipeline()

    def generate_texture(self, prompt: str, output_dir="assets/outputs"):
        os.makedirs(output_dir, exist_ok=True)
        output = self.generator.generate(prompt)

        paths = {}
        for key in ["albedo", "normal", "roughness", "ao"]:
            if key in output:
                path = os.path.join(output_dir, f"{key}.png")
                output[key].save(path)
                paths[key] = path
        return paths

# Streamlit UI
st.set_page_config(page_title="AstralLoom Triposg Texture Generator", layout="centered")
st.title("Triposg 기반 PBR 텍스쳐 생성기")

prompt = st.text_input("텍스처 생성 프롬프트", "mossy stone tiles")
if st.button("생성 시작"):
    pipeline = TriposgTexturePipeline()
    result_paths = pipeline.generate_texture(prompt)

    st.write("### 생성된 텍스처")
    for key, path in result_paths.items():
        st.image(Image.open(path), caption=f"{key.capitalize()} Map")
        with open(path, "rb") as f:
            st.download_button(
                label=f"{key.capitalize()} 다운로드",
                data=f,
                file_name=os.path.basename(path),
                mime="image/png"
            )
