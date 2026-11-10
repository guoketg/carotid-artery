# FGR（频率门控改写器）是怎么融合的

配套代码：[fgr.py](file:///e:/mkl/code/python_code/carotid-artery/networks/fgr.py)；概念背景：[fusion-operators-explained.md](fusion-operators-explained.md)、[edge-fusion-explained.md](edge-fusion-explained.md)。

## 0. 一页版

FGR = Frequency-Gated Rewriter（频率门控改写器，2026-11-09 自研）。一句话：**在解码器每级输出点上，把特征拆成低频/高频两条频带支，用「通道的 DCT 谱描述子」算出逐通道开关门，门只调频带支，1×1 卷积改写后加回原特征（残差）**。核心公式一行：

```
y = x + Fuse1x1( σ(Linear(DCT描述子([x, hf, lf]))) ⊙ [hf, lf] )
```

最重要的数字：弱骨干 UNet 上 **ΔDice +1.703 / ΔHD −0.168，双门槛全过**（全项目最强单模块正信号）；R50 强骨干上 +0.305 / −0.009，三项同向但不过线（单 seed）。

## 1. 输入是怎么处理的

FGR 的输入不是图像，是解码器特征图 `x (B, C, H, W)`。进入门控前先做频带拆解（[fgr.py#L65-L69](file:///e:/mkl/code/python_code/carotid-artery/networks/fgr.py#L65-L69)）：

- **hf（高频带）**：对 x 逐通道做 4 邻域 Laplacian 卷积取幅值、逐样本 z-score、2×2 均值池化到半分辨率，再双线性上采样回 H×W。三个细节槽位共用同一幅幅值图（方向无关的拉普拉斯响应，不做方向伪造）。运算在 [wavelet.py#L280-L302](file:///e:/mkl/code/python_code/carotid-artery/networks/wavelet.py#L280-L302) `laplacian_split_channels`。
- **lf（低频带）**：Haar 低通 LL（stride-2 均值池化），上采样回 H×W。
- 两者都是 C 通道，拼成 2C 的「频带支」。拆解零参数、无第三方小波库。
- 注意消歧：这里的 hf/lf 是**模块内部对特征图**的频带拆解；输入侧老师定的「边缘图/小波高频图」是对**原始图像**做对数域 Haar、LL 置零 IDWT 重建（见 [wavelet-input-routes.md](wavelet-input-routes.md)），两者不是同一个东西。

## 2. 模块内部数据流与融合细节

| 步骤 | 运算 | 输出形状 | 作用 |
|---|---|---|---|
| 拆解 | laplacian_split + 上采样 | hf/lf 各 (B,C,H,W) | 供给频带内容 |
| 谱描述子 | z_c = ⟨cat[x,hf,lf]_c, base_c⟩ | z (B,3C) | 把每个通道压成一个「频谱指纹」标量 |
| 门 | w = σ(Linear(3C→2C)) | w (B,2C,1,1) | 逐通道标量开关，门只看谱不看空间 |
| 改写 | Fuse = Conv1x1(2C→C) | (B,C,H,W) | 把门控后的频带支混合回 C 通道 |
| 残差 | y = x + Fuse(...) | (B,C,H,W) | 同形输出，恒等起步 |

**特征融合模块逐项**（最常被追问的部分）：

- **融合位置**：解码器四级输出点，DDBE 之后（次序：ABF→ACAM→concat→DBB→DDBE→FGR，[unet.py#L292-L303](file:///e:/mkl/code/python_code/carotid-artery/networks/unet.py#L292-L303)）。四级通道/分辨率 512²@28²、256²@56²、128²@112²、64²@224²，输入输出同形 (B,C,H,W)。
- **融合轴向**：**通道轴 + 逐位置改写**，不是空间轴注意力。1×1 卷积在每个像素位置内部混合通道，门 w 是逐通道标量、对整张特征图广播——门不看任何空间位置关系（对比：QKV 跨位置注意力是空间轴，两者在本项目里严格区分）。三条信息走的路：原图信息在 x 主干**原样通过**；高频/低频内容走频带支被门调幅后改写回主干。
- **算子公式**（[fgr.py#L70-L73](file:///e:/mkl/code/python_code/carotid-artery/networks/fgr.py#L70-L73)）：
  - `z_c = ⟨cat[x,hf,lf]_c, base_c⟩`：FcaNet（ICCV 2021）式固定 DCT 基，4×4 频率网格共 16 个基循环分配给 3C 个通道，每个基是一张固定的 H×W 余弦图，与特征图逐元素乘再求和 = 一个数。基是 `register_buffer(persistent=False)`，零参数不进 state_dict。
  - `w = σ(Linear(z))`：谱描述子 → 2C 个门值。
  - `y = x + Fuse1x1(w ⊙ [hf, lf])`：门只乘频带支（2C），**不乘 x 主干**。
- **初始化与残差**：gate.weight=0、gate.bias=2.0 → 初始 w=σ(2)≈0.88 接近全开（故意非零）；fuse 权重和偏置全零 → 第 0 步输出逐位等于 x（恒等起步，冒烟脚本 4 臂 IDENTITY 验证过）。纪律是**「一条链上只能有一个零」**：唯一零点是 fuse。梯度有两条路——fuse 自身直连（零权重照样有梯度）；gate 经 fuse 传导，但 w≈0.88 非零所以梯度不断。若 gate 也零初始化，w⊙fuse 两个零相乘梯度消失，模块会永久关闭（项目教训：零初始化可能导致模块实际被关闭）。
- **参数增量**：每级 gate 3C×2C + fuse 2C×C = 8C²+3C，四级合计 **+2.79M**（UNet 60.3M→63.1M，+4.6%）；R50 解码器通道更小，+0.69M（20.22M→20.91M）。对照 DDBE 单挂 +9.26M 的注意力机器，FGR 是极轻实现。
- **对照档位**：
  - **vs DDBE**（文献 BDFR-Net 模块）：同挂解码器四级输出、同为改写式。DDBE=重机器（SE/CBAM/转置卷积三支拼接），FGR=轻门控（谱描述子+线性门+1×1）。R50 上 FGR +0.305 与 DDBE-Haar +0.435 同级——轻机器拿到同量级信号。
  - **vs ddbectl/plain**（results.md §5(12)）：把 DDBE 三支换成普通特征 → +0.060≈噪声；FGR 反向检验「只有频带内容+轻门控」→ +0.305。两个对照夹逼出结论：**增益来自频率内容本身，不是拼接改写结构**。
  - **vs 组合档**：R50 上 DDBE+FGR 同点串联比 FGR 单挂 **−0.164**（互覆：两者供给同样的频率内容，信息冗余，9.26M 注意力机器反而压掉轻门控）；弱骨干上 FGR+DDBE 却比仅 FGR 再 +0.138 Dice / HD −0.104（小幅互补）。方向相反 → 改写机器的冗余度取决于骨干特征统计留下的空间。
  - 空间轴（QKV 注意力）与注入式（FreqFusion/FEM/R-GEF/MREF）在 R50 上全部判负——通道轴逐位置改写是目前唯一稳定正信号的轴向，但这是跨实验格局归纳，不是同变量判别对照。

- **损失与训练配置**：ce_dice（2 通道 logits 的交叉熵 + Dice），SGD lr=0.01，max_epochs=200、batch_size=4、grad_accum=2、每 5 epoch 评估取 epoch150、seed 1234。这么选：全项目统一协议，保证与基线成对可比（硬约束）。

## 3. 指标与结果

Dice=分割交并比（越大越好，受阈值漂移影响）；ASSD=表面平均对称距离；HD=medpy 口径 Hausdorff 距离（最大表面距离，越小越好，对离群点敏感）。

R50 强骨干（基线 90.4903 / 0.14466 / HD 0.70630，单 seed 1234）：

| 臂 | Dice% | ASSD | HD | ΔDice | ΔHD |
|---|---|---|---|---|---|
| FGR 单挂（+0.69M） | 90.7955 | 0.13912 | 0.69732 | +0.305 | −0.009 |
| DDBE4+FGR（+9.95M） | 90.6318 | 0.14104 | 0.66315 | +0.142 | −0.043 |

UNet 弱骨干（基线 86.5455 / 0.22500 / HD 1.13618，同协议）：

| 臂 | Dice% | ASSD | HD | ΔDice | ΔHD |
|---|---|---|---|---|---|
| unet+fgr（+2.79M→63.1M） | 88.2484 | 0.19369 | 0.96818 | **+1.703** | **−0.168** |
| unet+ddbe4+fgr（+40.14M→100.4M） | 88.3863 | 0.17687 | 0.86406 | +1.841 | −0.272 |

**怎么读**：R50 两行是方向证据——FGR 三项同向但单 seed 不过门槛（0.5%/0.1mm，噪声带 ±0.3），组合行判负（互覆）。UNet 两行是证据——FGR 双门槛全过；组合行的 +0.138 Dice 差在噪声带内、HD −0.104 刚过线，只能算「弱互补信号」。出处：results.md §5(16)(17)、§7(4)。

## 4. 代码实现在哪里

- 模块本体：[fgr.py](file:///e:/mkl/code/python_code/carotid-artery/networks/fgr.py)（DCT 基 L30-47；FGR 类 L50-73）
- UNet 挂载：[unet.py#L268-L270](file:///e:/mkl/code/python_code/carotid-artery/networks/unet.py#L268-L270)（init_kaiming 之后创建，RNG 对齐纪律：同 seed 基础权重与不挂载版逐位一致）；解码串联 [unet.py#L292-L303](file:///e:/mkl/code/python_code/carotid-artery/networks/unet.py#L292-L303)
- 频带拆解：[wavelet.py#L280-L302](file:///e:/mkl/code/python_code/carotid-artery/networks/wavelet.py#L280-L302)
- 队列/复现：[queue_fgr_ddbe_combo.ps1](file:///e:/mkl/code/python_code/carotid-artery/scripts/queue_fgr_ddbe_combo.ps1)；冒烟：[smoke_fgr_ddbe_combo.py](file:///e:/mkl/code/python_code/carotid-artery/scripts/smoke_fgr_ddbe_combo.py)

## 5. 结论边界与其他

| 能说 | 不能说 |
|---|---|
| 恒等起步成立（冒烟 4 臂 IDENTITY，同 seed 权重逐位对齐） | 多 seed 稳健性——全部数字是单 seed 1234 |
| 弱骨干上 +1.703/−0.168 双门槛全过，是全项目最强单模块正信号 | R50 上 FGR 是否真有效——+0.305 在噪声带内，待 s2024/s3407 复核 |
| R50 上 DDBE+FGR 同点串联互覆（信息冗余） | 弱骨干上 DDBE 对 FGR 的互补是否稳定——Dice 差 +0.138 不过线 |

已知局限：单 seed；R50 强骨干特征饱和、频带补不进空间（骨干强弱依赖，与 §6 K2+拉普拉斯 +0.978 一致）；hf 三槽位共享同一幅值图是有意为之的极简口径，方向信息依赖 x 主干自带。

复现（PowerShell，弱骨干 FGR 单臂）：

```powershell
E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe train.py --dataset Carotid --subset SPARC --seed 1234 --batch_size 4 --grad_accum 2 --max_epochs 200 --eval_test_every 5 --arch unet --fgr 1
```
