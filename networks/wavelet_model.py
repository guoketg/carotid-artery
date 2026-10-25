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

        # encoded 来自 transformer，形状 (B, n_patch, hidden) = (B, 196, 768)
        if self.fusion != "none":
            b, n_patch, hidden = encoded.size()
            h = int(n_patch ** 0.5)                              # 14：token 网格边长，196 = 14×14
            # 把 token 序列还原成二维特征图，供卷积/注意力操作：(B, 196, 768) → (B, 768, 14, 14)
            fmap = encoded.permute(0, 2, 1).reshape(b, hidden, h, h)
            if self.fusion == "cross":
                # 侧支输入只取第 3 通道 C3（高频能量图），形状 (B, 1, 224, 224)
                # 注意：C2 不参与这里，它只在输入端作为第 2 通道进过 stem
                guide = x[:, 2:3]
                # WaveletBranch：4 次 stride=2 卷积，224→112→56→28→14，通道 1→32→64→128→768
                # 输出 (B, 768, 14, 14)，与主分支 fmap 同分辨率同通道数
                f_wave = self.wave_branch(guide)
                # 兜底对齐（正常情况下已是 14×14，不会触发），避免尺寸不匹配报错
                if f_wave.shape[-1] != h:
                    f_wave = F.interpolate(f_wave, size=(h, h), mode="bilinear", align_corners=False)
                # WGA：Q 取自 fmap，K/V 取自 f_wave，4 头跨注意力后残差相加 + GroupNorm
                # 形状始终是 (B, 768, 14, 14)，只重加权已有的 196 个 token，不新增位置
                fmap = self.wga(fmap, f_wave)
            else:
                # 对照实验：SE 通道注意力只作用于主分支自身特征，不引入侧支
                fmap = self.se(fmap)
            # 变回 transformer 期望的序列格式：(B, 768, 14, 14) → (B, 196, 768)
            encoded = fmap.reshape(b, hidden, n_patch).permute(0, 2, 1)

        x = self.decoder(encoded, features)
        return self.segmentation_head(x)
