import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
from PIL import Image
import numpy as np

class FluxCustomPipeline:
    def __init__(self, 
                 flux_model_path, 
                 text_encoder_name="openai/clip-vit-large-patch14", 
                 vae_name="stabilityai/sd-vae-ft-mse"):
        
        # 1. 모델 로드
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load transformer (UNet-like)
        self.transformer_state = load_file(flux_model_path)
        self.transformer = self._build_transformer()
        self.transformer.load_state_dict(self.transformer_state, strict=False)
        self.transformer.eval().to(self.device)

        # Load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_name)

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(vae_name, torch_dtype=torch.float16).to(self.device)

        # Scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.scheduler.set_timesteps(30, device=self.device)

    def _build_transformer(self):
        # 임시로 Transformer 구조 빌드
        # 실제 Flux 모델에 맞는 구조를 알아야 함
        raise NotImplementedError("Flux 모델의 Transformer 아키텍처를 구성해야 합니다.")

    def generate(self, prompt, height=512, width=512, num_inference_steps=30, guidance_scale=3.5):
        # 1. 프롬프트 인코딩
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_embeds = self.text_encoder(**inputs).last_hidden_state

        # 2. Latent 초기화
        batch_size = 1
        latents = torch.randn(
            (batch_size, self.transformer.config.in_channels // 4, height // 8, width // 8),
            device=self.device,
            dtype=torch.float16
        )

        # 3. 디노이징 루프
        for t in self.scheduler.timesteps:
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = self.transformer(
                latents,
                timestep / 1000,
                encoder_hidden_states=prompt_embeds
            )[0]

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            torch.cuda.empty_cache()

        # 4. VAE 디코딩
        with torch.no_grad():
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        
        image = self._postprocess(image)
        return image

    def _postprocess(self, image_tensor):
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image = image_tensor.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).round().astype("uint8")
        return Image.fromarray(image)

