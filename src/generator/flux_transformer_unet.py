import torch
import torch.nn as nn

class FluxTransformerUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 블록 배열
        self.double_blocks = nn.ModuleList([
            self._make_double_block() for _ in range(8)  # 예시: 8개 블록
        ])

        self.single_blocks = nn.ModuleList([
            self._make_single_block() for _ in range(4)  # 예시: 4개 블록
        ])

        # 최종 출력 projection
        self.final_proj = nn.Conv2d(320, 4, kernel_size=1)

    def _make_double_block(self):
        # Double Attention + MLP Block
        return nn.Sequential(
            nn.LayerNorm(320),
            nn.MultiheadAttention(embed_dim=320, num_heads=8, batch_first=True),
            nn.LayerNorm(320),
            nn.Sequential(
                nn.Linear(320, 1280),
                nn.GELU(),
                nn.Linear(1280, 320),
            )
        )

    def _make_single_block(self):
        # Single MLP Block
        return nn.Sequential(
            nn.LayerNorm(320),
            nn.Linear(320, 1280),
            nn.GELU(),
            nn.Linear(1280, 320)
        )

    def forward(self, x, timestep=None, encoder_hidden_states=None):
        # x: [batch, channels, height, width]
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  # (B, H*W, C)

        # 통과
        for block in self.double_blocks:
            x = block(x)

        for block in self.single_blocks:
            x = block(x)

        # 다시 이미지 형태로 변환
        x = x.permute(0, 2, 1).view(b, c, h, w)

        # 최종 Projection
        x = self.final_proj(x)

        return x
