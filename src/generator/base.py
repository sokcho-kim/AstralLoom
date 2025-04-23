from abc import ABC, abstractmethod
from typing import Dict
from PIL import Image

class BaseTextureGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Image.Image]:
        """
        공통 인터페이스: 모델은 딕셔너리 형태로 결과 반환
        예: { "albedo": Image, "normal": Image, "roughness": Image }
        """
        @abstractmethod
        def generate(self, prompt: str, **kwargs) -> Dict[str, Image.Image]:
            pass
