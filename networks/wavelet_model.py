"""小波分支 + 注意力融合（WGA）。

主分支沿用 TransUNet 的 R50+ViT，小波分支是轻量 CNN（约 0.9M 参数），
在 ViT 输出（14x14 token 网格）处做跨注意力融合，再由原解码器上采样。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.vit_seg_modeling import VisionTransformer


def _block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class WaveletBranch(nn.Module):
    """输入细节能量图 C3，输出与 ViT 同分辨率的特征"""

    def __init__(self, dim=768):
        super().__init__()
        self.net = nn.Sequential(
            _block(1, 32, stride=2),    # 112
            _block(32, 64, stride=2),   # 56
            _block(64, 128, stride=2),  # 28
            _block(128, dim, stride=2),  # 14
        )

    def forward(self, x):
        return self.net(x)


class WaveletGuidedAttention(nn.Module):
    """Q 来自主分支，K/V 来自小波分支；残差 + LayerNorm"""

    def __init__(self, dim=768, heads=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_out = nn.GroupNorm(1, dim)  # 特征图布局下按通道+空间归一化

    def forward(self, f_main, f_wave):
        q = self.norm_q(f_main.flatten(2).transpose(1, 2))
        kv = self.norm_kv(f_wave.flatten(2).transpose(1, 2))
        out, _ = self.attn(q, kv, kv)
        out = f_main + out.transpose(1, 2).reshape(f_main.shape)
        return self.norm_out(out)


class SEBlock(nn.Module):
    """轻量对照：通道注意力作用在主分支特征上"""

    def __init__(self, dim=768, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = x.mean(dim=(2, 3))
        return x * self.fc(w).unsqueeze(-1).unsqueeze(-1)


class WaveletTransUNet(VisionTransformer):
    """fusion: none / se / cross"""

    def __init__(self, config, img_size=224, num_classes=2, fusion="cross", heads=4):
        super().__init__(config, img_size, num_classes)
        self.fusion = fusion
        if fusion == "cross":
            self.wave_branch = WaveletBranch(config.hidden_size)
            self.wga = WaveletGuidedAttention(config.hidden_size, heads)
        elif fusion == "se":
            self.se = SEBlock(config.hidden_size)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoded, attn_weights, features = self.transformer(x)

        if self.fusion != "none":
            b, n_patch, hidden = encoded.size()
            h = int(n_patch ** 0.5)
            fmap = encoded.permute(0, 2, 1).reshape(b, hidden, h, h)
            if self.fusion == "cross":
                guide = x[:, 2:3]
                f_wave = self.wave_branch(guide)
                if f_wave.shape[-1] != h:
                    f_wave = F.interpolate(f_wave, size=(h, h), mode="bilinear", align_corners=False)
                fmap = self.wga(fmap, f_wave)
            else:
                fmap = self.se(fmap)
            encoded = fmap.reshape(b, hidden, n_patch).permute(0, 2, 1)

        x = self.decoder(encoded, features)
        return self.segmentation_head(x)
