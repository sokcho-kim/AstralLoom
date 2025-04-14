from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from generator.texture_creator import generate_texture
from generator.texture_creator_v1 import generate_texture
import uvicorn
import os
import base64

app = FastAPI()

# 요청 받을 데이터 구조 정의
class GenerateRequest(BaseModel):
    prompt: str

# 생성 완료된 이미지 리턴 구조
class GenerateResponse(BaseModel):
    image_path: str

# # 생성 엔드포인트
# @app.post("/generate", response_model=GenerateResponse)
# async def generate(request: GenerateRequest):
#     try:
#         prompt = request.prompt
#         # 텍스쳐 생성
#         output_path = generate_texture(prompt)

#         # 파일 존재 여부 확인
#         if not os.path.exists(output_path):
#             raise HTTPException(status_code=500, detail="Failed to generate texture")

#         return GenerateResponse(image_path=output_path)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=dict)
async def generate(request: GenerateRequest):
    try:
        prompt = request.prompt
        output_path = generate_texture(prompt)

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Failed to generate texture")

        # 파일을 base64로 읽어서 전송
        with open(output_path, "rb") as img_file:
            img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return {"image_base64": img_base64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 서버 실행 (로컬용)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
