"""方案1：局部支（纯 R50 CNN）与全局支（纯 ViT-B/16）彻底分离，14² 单点融合。

与 TransUNet 的区别：TransUNet 的 ViT 吃的是 R50 的 14²/1024 特征，局部与全局
串在同一条主路上（用户指出「两个都用了」）；这里 ViT 用 16×16 patch 直接从原图
取 token，全程不接触任何 CNN 特征，两条支路只在 14² 瓶颈处融合一次：

- 局部支：R50（PreAct + GN）→ 112²/64、56²/256、28²/512、14²/1024 → 1×1 到 768；
- 全局支：ViT-B/16（12 层、768 维、196 token），输入 = 原图 3 通道；
- 融合：14² 单点 `make_fusion(fuse_op)`，γ 零初始化 → 第 0 步两路各自独立；
- 解码：`DecoderCup` 输入 = 融合后的 ViT token，skip = [28²/512, 56²/256, 112²/64]
  **全部来自 CNN 局部支**（全局支不提供 multi-scale skip）。

单点融合下局部瓶颈（B）没有下游消费者，故融合按 `bidir=False` 只回写 A；
`B` 的信息仍通过 Freqout 里的 `Fre_R` 与空间注意力进入解码器。
"""
import copy
import logging
import os

import numpy as np
import torch
import torch.nn as nn

from networks.dual_encoder_model import make_fusion
from networks.vit_seg_configs import get_b16_config
from networks.vit_seg_modeling import DecoderCup, Embeddings, Encoder, SegmentationHead
from networks.vit_seg_modeling_resnet_skip import ResNetV2, np2th

logger = logging.getLogger(__name__)

VIT_B16_NPZ = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'


class SplitLocalGlobalTransUNet(nn.Module):
    """局部（R50）与全局（ViT-B/16）并联，只在 14² 做一次融合。

    config 传入 R50-ViT-B_16 的配置：`resnet` 段给局部支，`hidden_size` /
    `decoder_channels` / `skip_channels` / `n_skip` 给融合与解码器。
    """

    def __init__(self, config, img_size=224, num_classes=2, fuse_op='frc',
                 window=8, dk_ratio=2, bidir=False, vit_pretrained=VIT_B16_NPZ):
        super().__init__()
        assert img_size == 224, 'split 依赖 16×16 patch → 14×14 token 网格'
        self.img_size = img_size
        self.fuse_op = fuse_op
        self.vit_pretrained = vit_pretrained

        width = int(64 * config.resnet.width_factor)
        hidden = config.hidden_size

        # ---- 局部支：纯 R50
        self.local = ResNetV2(block_units=config.resnet.num_layers,
                              width_factor=config.resnet.width_factor)
        self.local_bottleneck = nn.Conv2d(width * 16, hidden, 1)

        # ---- 全局支：纯 ViT-B/16（独立 patch embedding，不吃 CNN 特征）
        vit_cfg = get_b16_config()
        vit_cfg.transformer.num_layers = int(config.transformer.num_layers)
        self.vit_emb = Embeddings(vit_cfg, img_size=img_size, in_channels=3)
        self.vit_enc = Encoder(vit_cfg, vis=False)

        # ---- 14² 单点融合
        self.fuse = make_fusion(fuse_op, hidden, 8, dk_ratio, window, bidir)

        # ---- 解码器：skip 全来自局部支（DecoderCup 会原地改 skip_channels，必须深拷贝）
        dec_cfg = copy.deepcopy(config)
        self.decoder = DecoderCup(dec_cfg)
        self.segmentation_head = SegmentationHead(
            in_channels=dec_cfg['decoder_channels'][-1],
            out_channels=num_classes,
            kernel_size=3,
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        raw = x[:, :1] if x.size(1) > 1 else x
        img = raw.repeat(1, 3, 1, 1)

        # 局部支：14²/1024 + 三级 skip（28²/512、56²/256、112²/64）
        deep, features = self.local(img)
        b = self.local_bottleneck(deep)

        # 全局支：ViT 直接从原图取 196 token
        t, _ = self.vit_emb(img)
        t, _ = self.vit_enc(t)

        # 14² 单点融合（frc 严格按 Syn-Net 原文）
        bs, n_tok, ch = t.shape
        h = int(n_tok ** 0.5)
        a = t.transpose(1, 2).reshape(bs, ch, h, h)
        a, _ = self.fuse(a, b)
        t = a.flatten(2).transpose(1, 2).contiguous()

        out = self.decoder(t, features)
        return self.segmentation_head(out)

    # ------------------------------------------------------------ 权重

    def load_from(self, weights):
        """局部支取 R50 权重；全局支取 ViT-B/16 权重（encoder 与 R50+ViT 同源）。"""
        with torch.no_grad():
            self.local.root.conv.weight.copy_(np2th(weights['conv_root/kernel'], conv=True))
            self.local.root.gn.weight.copy_(np2th(weights['gn_root/scale']).view(-1))
            self.local.root.gn.bias.copy_(np2th(weights['gn_root/bias']).view(-1))
            for bname, block in self.local.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=bname, n_unit=uname)

            self.vit_enc.encoder_norm.weight.copy_(
                np2th(weights['Transformer/encoder_norm/scale']))
            self.vit_enc.encoder_norm.bias.copy_(
                np2th(weights['Transformer/encoder_norm/bias']))
            posemb = np2th(weights['Transformer/posembed_input/pos_embedding'])
            if posemb.size(1) - 1 == self.vit_emb.position_embeddings.size(1):
                posemb = posemb[:, 1:]          # 去掉 cls token
            self.vit_emb.position_embeddings.copy_(posemb)
            for bname, block in self.vit_enc.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            # 16×16 patch embedding 只在纯 ViT-B/16 权重里有（R50+ViT 是 1×1）
            if os.path.isfile(self.vit_pretrained):
                vw = np.load(self.vit_pretrained)
                self.vit_emb.patch_embeddings.weight.copy_(
                    np2th(vw['embedding/kernel'], conv=True))
                self.vit_emb.patch_embeddings.bias.copy_(np2th(vw['embedding/bias']))
            else:
                logger.warning('未找到 %s：全局支 patch embedding 保持随机初始化',
                               self.vit_pretrained)

    def extra_repr(self):
        return f'fuse_op={self.fuse_op}'
