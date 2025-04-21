import torch
from diffusers import FluxPipeline
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # diffusers 내부 로깅 사용

class FluxWithCFGPipeline(FluxPipeline):
    @classmethod
    def from_single_file(cls, pretrained_model_path, vae=None, text_encoder=None, **kwargs):
        # 기존 FluxPipeline 로드
        pipe = FluxPipeline.from_single_file(
            pretrained_model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            variant="fp16",
            **kwargs
        )

        # 클래스 변경
        pipe.__class__ = cls

        # 수동으로 필요한 컴포넌트 추가
        if vae:
            pipe.vae = vae
        if text_encoder:
            pipe.text_encoder = text_encoder

        logger.info("FluxWithCFGPipeline loaded successfully.")
        return pipe

    @torch.inference_mode()
    def generate_images(self, prompt, height=512, width=512, num_inference_steps=20, guidance_scale=3.5):
        # pipeline 기본 generate 방식 사용
        images = self(prompt=prompt,
                      height=height,
                      width=width,
                      num_inference_steps=num_inference_steps,
                      guidance_scale=guidance_scale).images
        return images[0]  # 첫 번째 이미지 반환
