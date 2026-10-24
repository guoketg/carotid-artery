# 技术设计：小波预处理 + 注意力融合

## 0. 前提

- 模态为 **B 型二维超声**：无 HU 窗宽窗位；斑点噪声乘性；有深度衰减、声影、钙化强反射、探头增益差异。
- 目标不是方法新颖性，而是**在参数量/计算量受控下**验证"一个预处理 + 一个注意力融合"能否带来显著提升。

## D1 — 预处理主选 DWT（Haar/db1，1~2 级）

- 不用 FFT 直接当输入通道：全局频谱无空间定位能力，相位不易对齐，对"边界在哪"帮助有限。
- Haar 1~2 级足够；3 级以上在超声中基本是噪声。
- 边界用对称延拓（symmetric padding），避免振铃。
- DWT 用固定 2×2 卷积核手写实现：无参数、可 GPU、可逆，不引入第三方依赖。

## D2 — 三通道构造（关键）

在**对数域**分解：`log(1+x)` → DWT → 细节子带软阈值 → IDWT。

| 通道 | 内容 | 作用 |
|---|---|---|
| C1 | 原图（归一化后） | 保真，保留全部回声与边界 |
| C2 | `IDWT(LL, softT(LH/HL/HH))` | 低斑点噪声的结构先验 |
| C3 | `sqrt(LH²+HL²+HH²)` → 高斯平滑 | 边界 / 钙化面 / 表面强反射显著性 |

- VisuShrink 阈值：`T = σ·sqrt(2·logN)`，`σ = median(|detail|)/0.6745`。
- 三通道**共用**原图的 mean/std，不要各自归一化。
- 8bit 图像先按 1%~99% 分位截断再拉伸，抑制设备增益差异。

## D3 — 编码器：非对称并行，不做双 ViT

- 双 ViT 显存翻倍、训练不稳；R50 patch embedding 本身 16×16 感受野已能吸收部分小波信息。
- 主分支 = 原 TransUNet（R50 + ViT-B）；小波分支 = 4 stage 轻量 CNN（宽度约 1/4），从头训练。
- 只在 stage1/2/3 与主分支 skip 融合，显存增量约 10%~20%，融合发生在有语义层级的尺度。

## D4 — 融合主方法：交叉注意力 WGA

- `Q` = 主分支 `F_o`；`K/V` = 小波分支 `F_w`（下采样对齐）；heads 4~8；残差 + LayerNorm。
- 放在 stage2/3（56×56 / 28×28），控制 token 数与开销。
- 对照：SE / CBAM 作用于 concat 特征（轻量融合基线）。
- 结论口径：主方法需同时优于"无融合"和"轻量融合"，才能说明注意力形式本身有贡献。

## D4 实现说明（2026-10-23）

实际把小波分支的融合放在 **ViT 输出瓶颈（14×14，196 token）**，不是 D3 描述的 stage2/3 skip：

- `WaveletBranch` 输入 C3（细节能量图），4 次 stride-2 卷积到 14×14 / 768 通道（约 0.9M 参数）；
- `WaveletGuidedAttention`：Q 取主分支 token，K/V 取小波分支 token，4 头，残差 + GroupNorm；
- 解码器与 skip 连接保持官方实现不变。

理由：瓶颈 token 少、注意力开销低，改动集中在 `WaveletTransUNet.forward`，baseline 可完全复现。代价是没用上 stage2/3 的高分辨率边界信息，后续可作为改进点。

## D5 — 傅里叶的落点

- 若需响应"傅里叶"：FFT → 可学习复权重（1×1，global filter 思路）→ IFFT，放在 bottleneck 或 skip，参数极少。
- 不作为输入预处理通道，不做频域替换原图。
- 可选对照：DoG / 高通残差作为第三通道替代 C3。

## D6 — 消融矩阵

| 编号 | 配置 | 验证 |
|---|---|---|
| A0 | 原图单通道 TransUNet | baseline |
| A1 | 3 通道早融合（C1/C2/C3），无注意力 | 预处理是否有效 |
| A2 | 原图复制 3 通道 + WGA（伪分支） | 融合结构是否有效 |
| A3 | 3 通道 + 并行分支 + WGA | 完整方法 |

附加：小波 1 级 vs 2 级；阈值开/关；融合位置 stage2 / stage3 / 两处；CLAHE 替代 C2。

## D7 — 预训练权重适配

- 输入通道改 3 后不能直接加载 R50 `conv1`：新权重各输入通道取原 3 通道均值 ×3（保持激活量级），或用灰度复制。
- ViT 部分正常加载；轻量小波分支从头训练。

## D8 — 评估协议（摘要，详见 experiments.md）

- 5-fold 病例级划分，mean ± std，paired t-test 或 Wilcoxon，p<0.05。
- 指标：Dice、IoU、HD95、ASSD、Sensitivity、Specificity、Precision。
- 必报 Params / FLOPs / 单张时延；斑块为小目标，损失用 Dice+BCE，重点看 HD95。
- 失败案例：低回声斑块、声影区、钙化伴声影、增益过曝。

## D9 — 工程兼容记录

- 权重：`f:\code\MyPython\model\vit_checkpoint\imagenet21k\R50+ViT-B_16.npz`（461,217,452 字节，358 个数组，R50 混合版）。
- 已修复 TransUNet 老 bug：`pjoin` 在 Windows 下用 `os.path.join` 拼 npz 键名会产生反斜杠，改为 `"/".join(args)`。
