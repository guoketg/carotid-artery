# 双骨干交错编码 + 四级双向融合（D11）实现说明

> 对应设计条目 [technical-design.md](technical-design.md) D11。本文只描述**已实现并实测**的内容，代码在
> [networks/dual_encoder_model.py](../networks/dual_encoder_model.py)，训练开关在 [train.py](../train.py)。
> 所有 shape、参数量、显存、耗时均为本机实测（RTX 5060 Laptop 8GB），非估算；估算值单独标注。

## 0. 一页版

```
                    ┌──────────────── 主路（R50-ViT-B/16，ImageNet21k）────────────────┐
x (B,1,224,224) ──┬─► repeat 3ch ─► root ─►★F1─► pool ─► block1 ─►★F2─► block2 ─►★F3─► block3
                  │                 112²/64      55²/256      28²/512      14²/1024
                  │                   │            │            │            │
                  │                   │            │            │        patch_emb 1×1
                  │                   │            │            │            ▼
                  │                   │            │            │      196×768 ─► 12×ViT ─►★F4
                  │                   │            │            │                      14²/768
                  └─► Haar(LH|HL|HH) ─► root ─►★F1─► pool ─► block1 ─►★F2─► block2 ─►★F3─► block3
                     上采样回 224²             112²/64      55²/256      28²/512      14²/1024 ─► 1×1 ─► 14²/768
                                                                                              │
                                                                                            ★F4
                                                                                              ▼
                                        DecoderCup(skip = [28²/512, 56²/256, 112²/64]) ─► head
```

三个要点：

1. **主路只吃 C1**（复制成 3 通道），侧路只吃 1 级 Haar 的三个方向子带，两路输入**无重叠**。
2. **两级骨干按 stage 交错执行**，不是各自跑完再拼。★F1 的融合结果会进入主路 `block1`，
   所以侧路信息真正进入了主路主干，而不是只挂在 skip 上。
3. **四个接点全部双向**：`A' = A + γ_A·GN(Attn(Q=A, K=V=B))`、`B' = B + γ_B·GN(Attn(Q=B, K=V=A))`。
   `γ` 与输出投影均零初始化 → 第 0 步与 baseline 逐元素等价（实测 `max|Δ| = 0.000e+00`）。

---

## 1. 输入：两条路各自的张量

数据集 `CarotidDataset` 在 `--arch dual` 时被强制 `use_wavelet=0`，因此每个样本只给**单通道**归一化灰度图
`x ∈ R^{B×1×224×224}`（[train.py#L114-L117](../train.py#L114-L117)）。

| 路 | 构造 | 输出 | 代码 |
|---|---|---|---|
| 主路 | `x.repeat(1,3,1,1)`，三通道完全相同（满足 ViT `conv1` 的 3 通道契约） | `(B,3,224,224)` | [dual_encoder_model.py#L328-L336](../networks/dual_encoder_model.py#L328-L336) |
| 侧路 | 对数域 → Haar DWT → 取 LH/HL/HH（各 112²）→ 双线性上采样回 224² → 逐通道 z-score | `(B,3,224,224)` | [dual_encoder_model.py#L32-L46](../networks/dual_encoder_model.py#L32-L46) |

侧路构造的四个细节，缺一个都会掉点：

- **对数域**：`log1p(x - min + 1e-3)`。超声动态范围大，线性域的子带被少数强回声主导。
- **只取三方向子带，不取 LL**。LL 是 2×2 均值，与主路 C1 高度重合，喂进去等于两路吃同一份信息
  （这正是 A3 失败的第一条原因）。
- **上采样回 224² 是硬要求**：Haar 1 级输出是 112²，若直接喂 R50，`conv_root`（stride 2）+ `maxpool`（stride 2）
  会把它降到 28²，整条侧路比主路低两个层级，四个接点全部对不上。
- **逐通道 z-score**：三个子带的动态范围差一个数量级（HH 最弱），不标准化会让侧路 `gn_root` 只对 LH 敏感。

---

## 2. 编码：交错执行的数据流（含 55→56 的坑）

主路 = `VisionTransformer.transformer.embeddings.hybrid_model`（`ResNetV2`），侧路 = 新建的 `side_res`（同构 `ResNetV2`）。
两者**不是**各跑一遍 `ResNetV2.forward`，而是把 stage 拆开逐步调用
（[dual_encoder_model.py#L273-L326](../networks/dual_encoder_model.py#L273-L326)）：

```python
# ★F1
a = mres.root(main_in);  b = sres.root(side_in)      # 112²/64 两侧
a, b = fuse[1](a, b);  f1 = a                        # f1 → skip

# 下采样 + block1
a, b = pool(a), pool(b)                              # 112 → 55（MaxPool2d(3,2,0)，不是 56！）
a, b = mres.body[0](a), sres.body[0](b)              # 55²/256

# ★F2 必须在 56² 上做
a_f = pad(a, 56);  b_f = pad(b, 56)                  # 右下零填充，复刻 ResNetV2.forward 的 skip 处理
a_f, b_f = fuse[2](a_f, b_f);  f2 = a_f              # f2 → skip（56²，喂 decoder）
a = a_f[:, :, :55, :55];  b = b_f[:, :, :55, :55]    # 裁回 55² 继续走主干

# ★F3
a, b = mres.body[1](a), sres.body[1](b)              # 28²/512
a, b = fuse[3](a, b);  f3 = a                        # f3 → skip

# 侧路对齐 + 主路进 ViT
a = mres.body[2](a)                                  # 14²/1024
b = side_proj(sres.body[2](b))                       # 14²/1024 → 14²/768
t = patch_emb(a).flatten(2).transpose(1,2) + pos_emb # 196×768
t = encoder_norm(12 × ViT block(t))

# ★F4
ta = t^T → (B,768,14,14);  ta, b = fuse[4](ta, b);  t = ta.flatten → 196×768
return t, [f3, f2, f1]
```

### 2.1 为什么必须复刻 55→56

`ResNetV2.forward` 里 `self.maxpool = MaxPool2d(3, 2, 0)`，224→112→**55**（不是 56）。
而 `DecoderCup` 的上采样是 14→28→56→112，skip 必须匹配 56²。
原实现的做法是：**把 55² 零填充到 56² 作为 skip，但主干继续用未填充的 55²**。
`DualEncoderTransNet` 必须逐字复刻这个行为，否则 ★F2 之后的 28² 尺寸会与 baseline 不同（55→27 vs 28），
decoder 的 `cat` 直接报错。

代价：★F2 在 56² 上融合，最后一行/列（含 pad 的零）被裁掉。
这是**有意为之**——几何上与 baseline 完全一致，融合模块不改变任何分辨率。

### 2.2 shape 契约（实测）

| 节点 | 主路 A | 侧路 B | 下一步 |
|---|---|---|---|
| input | 2×1×224×224 | 2×3×224×224（子带） | — |
| root | 2×64×112×112 | 2×64×112×112 | **★F1** |
| pool+block1 | 2×256×55×55 | 2×256×55×55 | pad→56 |
| pad | 2×256×56×56 | 2×256×56×56 | **★F2** |
| block2 | 2×512×28×28 | 2×512×28×28 | **★F3** |
| block3 | 2×1024×14×14 | 2×1024×14×14 → 1×1 → 2×768×14×14 | — |
| patch embed | 2×196×768 | — | 12×ViT |
| ViT out | 2×196×768 → 2×768×14×14 | 2×768×14×14 | **★F4** |
| decoder | logits 2×2×224×224 | — | — |

侧路 block3 输出是 1024 通道（`width_factor=1 → 64×16`），必须用 `side_proj`（1×1，1024→768，0.79M）
降到 ViT 的 `hidden_size`，否则 ★F4 的两侧通道对不上。

---

## 3. 融合算子

### 3.1 主算子：双向跨支注意力

代码 [dual_encoder_model.py#L69-L137](../networks/dual_encoder_model.py#L69-L137)。

设接点特征 `A, B ∈ R^{B×C×H×W}`，`d_k = C / dk_ratio`（默认 2，取 4 的倍数）：

```
Q_A = W_q^A(A),  K_A = W_k^A(A),  V_A = W_v^A(A)     # 1×1 conv，C → 3·d_k
Q_B = W_q^B(B),  K_B = W_k^B(B),  V_B = W_v^B(B)

A' = A + γ_A · GN( W_o^A( Attn(Q_A, K_B, V_B) ) )
B' = B + γ_B · GN( W_o^B( Attn(Q_B, K_A, V_A) ) )
```

**为什么这样设计（四条，都对应 A3 的失败原因）**：

| 设计 | 解决什么 |
|---|---|
| 两个方向都要算 | A3 只让侧路给主路供 K/V，侧路本身不被改写 → 单向注入，信息只走一次 |
| `Q` 来自本支、`K/V` 来自对方 | 这才叫跨支交互；各支自己算注意力（CBAM/SE）本质上仍是支内操作 |
| `GN` 放在残差**支路内** | 写成 `GN(A + Δ)` 会在 `γ=0` 时仍归一化 A，破坏第 0 步等价性（A3 的第 4 条错） |
| `γ` 标量零初始化 + `W_o` 零初始化 | 双保险：起点严格等于 baseline，融合模块只会从"不改变"开始学 |

`dk_ratio=2` 是参数量与显存的折中。若用 `d_k = C`（标准 MHA），融合参数会从 3.70M 涨到约 12.6M；
若用 `C/4`，降到约 1.0M 但每个头只有 16 维（C=64 时），表达能力不足。

### 3.2 两个 mode：窗内 / 全局

| mode | 用法 | token 数/次 | 实测单次耗时（batch 4，前向） |
|---|---|---|---|
| `window` | 按 `ws×ws` 切窗，窗内做注意力 | F1: 196 窗 × 64 token；F2: 64 窗 × 49 token | F1 6.0 ms / F2 5.7 ms |
| `global` | 全图 flatten 后做注意力 | F3: 784 token；F4: 196 token | F3 5.7 ms / F4 2.3 ms |

`auto` 的分配规则（[dual_encoder_model.py#L236-L238](../networks/dual_encoder_model.py#L236-L238)）：

```python
kind = "window" if res >= 56 else "global"   # 112²/56² 窗内，28²/14² 全局
```

**这条规则曾经写反过**（`res <= 56`），导致 ★F1（112²）用了全局注意力：12544 token × 双向，
单次 265 ms，batch 8 单步从 1.6 s 涨到 6.8 s。修正后 ★F1 降到 6.0 ms（44 倍）。
留档在 [status.md](status.md) 已知问题。

数值依据（估算，非实测）：★F1 全局注意力的 logits 为 `heads × 12544² × 4 byte ≈ 3.1 GB/方向`，
batch 4 双向直接 OOM；窗内 8×8 把每窗 token 降到 64，logits 降到约 77 MB。

### 3.3 三个对照算子

| 类 | 档位 | 公式 | 用于 |
|---|---|---|---|
| `BidirGate` | L1 | `A' = A + γ_A·(A ⊙ σ(Conv7×7([mean(B); max(B)])))`，B 对称 | R2：只调幅度、不重组内容 |
| `BidirConcat` | L0 | `A' = A + γ_A·Conv1×1([A; B])`，B 对称 | R3：区分"跨支交互"与"多一路特征" |
| `bidir=False` | — | 只算 `A'`，`B` 原样返回 | R7：单向对照，复现 A3 的病 |

三者都保留 `γ` 零初始化，因此与主算子一样满足"第 0 步等价 baseline"。

`BidirConcat` 的 1×1 也零初始化。注意它的参数量最大（2·C·2C），
在 C=768 时单个接点就有 2.36M，比 `BidirCrossAttention` 还大——**参数量不是"融合强弱"的判据**，
能不能重组内容是。

### 3.4 强融合新增算子（r4 / r5 / r6，2026-10-24 落地）

针对"A3 融合太弱"的三个根因（固定网格、只融一点、侧路太窄），再补三种结构上不同归因的算子。
三者同样保持 `γ / 输出投影 / offset 卷积` 零初始化，第 0 步与 baseline 逐元素相同。

| 档位 | 类 / 开关 | 公式 | 针对的根因 | 实测参数量（四级合计） |
|---|---|---|---|---|
| r4 `--fuse_op deform` | `BidirDeform`（D-LKA 式） | 一支 `1×1→d` 后用 `3×3` 预测 `2k²` 偏移，在另一支的 `d` 维特征上做 `deform_conv2d(k=5)`；`A' = A + γ_A·GN(proj(y))` | **采样网格固定**：让采样点跟着解剖边界走 | 3.69M（F1 0.089 / F2 1.068 / F3 1.200 / F4 1.333） |
| r5 `--fuse_op aff` | `BidirAFF`（AFF, CVPR 2021, arXiv 2009.14082） | `m = MS-CAM(A+B)`；`A' = A + γ_A·(A⊙m)`，`B' = B + γ_B·(B⊙(1−m))` | **只用加性/门控，不做内容重组**：MS-CAM 同时给局部与全局权重 | 0.47M（F1 0.002 / F2 0.034 / F3 0.133 / F4 0.298） |
| r6 `--cross_scale 1` | `cross_scale`（BiFPN 式 top-down） | `14²→28²→56²→112²` 逐级回流到 skip：`f3 += γ₀·up(cs43(ta))`，`f2 += γ₁·up(cs32(f3))`，`f1 += γ₂·up(cs21(f2))` | **只融一点**：把瓶颈信息在推理前多级广播到三条 skip | +44K（三个 1×1） |

- r4 的 `d = max(32, min(128, dim // dk_ratio))`，两方向共用降维投影，因此比"双向各建一套"省一半。
- r5 把 AFF 原文的 BN 全部换成 GroupNorm（batch 4 下 BN 统计不稳）；`γ` 挂在残差上而不是乘在 `A⊙m` 里，保持零初始化等价。
- r6 与 `--fuse_op` 正交，可叠加（`--fuse_op auto --cross_scale 1` 即 r6）。

零初始化等价性实测（`probe_dual.py`，γ=0）：

```text
deform : total 121.65M  max|baseline-dual| = 0.000e+00  PASS
aff    : total 118.43M  max|baseline-dual| = 0.000e+00  PASS
auto+cs: total 122.20M  max|baseline-dual| = 0.000e+00  PASS
```

> 备注：以上三行的显存/单步耗时探针是在队列 1 正在训练时跑的，GPU 争用导致计时失真
> （实测 0.96–11.3 s/step，正常应为 0.3 s 量级），因此**只采信等价性与参数量**；
> 真实耗时等 queue 2 的 `logs/queue/queue_dual2_status.csv` 出数后再补。

---

## 4. 跳跃连接：三条 skip 全部被融合结果替换

`DecoderCup` 的接口没变（[vit_seg_modeling.py#L311-L317](../networks/vit_seg_modeling.py#L311-L317)、
[#L365](../networks/vit_seg_modeling.py#L365)）：

```python
features = [f3, f2, f1]          # 顺序：28² / 56² / 112²
out = self.decoder(encoded, features)
```

| skip | 分辨率/通道 | 来源 | 与 baseline 的差异 |
|---|---|---|---|
| `features[0]` | 28²/512 | ★F3 输出（双向融合后的主路特征） | 含侧路信息 |
| `features[1]` | 56²/256 | ★F2 输出（在 56² 上融合，含 pad） | 含侧路信息 |
| `features[2]` | 112²/64 | ★F1 输出 | 含侧路信息 |
| `encoded` | 196×768 | ★F4 输出 | 含侧路信息 |

关键点：**侧路信息有三条通道进入解码器**，而不是只在瓶颈灌一次。

1. 沿主路主干向下（★F1 → block1 → ★F2 → …），影响后续所有 stage 的特征；
2. 三条 skip 直接进 `DecoderCup` 的三个 `cat`；
3. 瓶颈 ★F4 进 `DecoderCup` 的输入。

A3 只有第 3 条，且侧路容量只有 0.9M——这是"融合特征太弱"的结构性原因。

`DecoderCup` 本身与 baseline 完全一致：
`upsample(2×) → concat(skip) → DoubleConv`，三层，输出 224²×16 通道，接 `SegmentationHead`（1×1 → 2 类）。

---

## 5. 参数、显存、时间（实测）

### 5.1 参数量

| 部分 | 参数 | 占比 |
|---|---|---|
| 主路（R50-ViT + decoder + head） | 105.28 M | 86.5% |
| 侧路（R50 11.89M + 1×1 投影 0.79M） | 12.68 M | 10.4% |
| 融合模块 | 3.70 M | 3.0% |
| **合计** | **121.66 M** | +16.4% vs baseline |

融合模块明细：

| 接点 | 类型 | 参数 | 说明 |
|---|---|---|---|
| ★F1 | BidirCrossAttention（窗内） | 0.017 M | C=64，d_k=32，4 头 |
| ★F2 | BidirCrossAttention（窗内） | 0.264 M | C=256，d_k=128，8 头 |
| ★F3 | BidirCrossAttention（全局） | 1.053 M | C=512，d_k=256，8 头 |
| ★F4 | BidirCrossAttention（全局） | 2.366 M | C=768，d_k=384，8 头 |

侧路 12.68M 而不是 D11.3 里估的 8.5M，原因是 `config.resnet.num_layers = (3,4,9)`
（不是早期假设的 3,4,6），block3 有 9 个 bottleneck。

### 5.2 显存与单步耗时（batch sweep，30 步，含 momentum SGD 状态 + CE + Dice）

| batch | 峰值显存 | 单步耗时 | 结论 |
|---|---|---|---|
| 2 | 2.13 GB | 267.9 ms | |
| 4 | 3.32 GB | 319.8 ms | **训练选用** |
| 6 | 4.21 GB | 416.3 ms | 安全上限 |
| 8 | 5.01 GB | **1101.6 ms** | 进入 WDDM 换页区，2.6× 惩罚 |

**batch 8 不能用**。8GB 的 RTX 5060 Laptop 在 Windows 下，当显存占用逼近专用显存上限时，
驱动会把一部分页换到系统内存（共享显存），带宽掉一个数量级。峰值 5.01 GB 看着"没满"，
但 PyTorch 的 caching allocator 还持有已释放的缓存块，实际提交量已接近阈值。
batch 8 的耗时是 batch 6 的 2.65 倍，而计算量只多 33%——这个比值就是换页的直接证据。

因此训练配置固定为 **batch 4 + `--grad_accum 2`**（等效 batch 8），
实现见 [trainer.py#L83-L107](../trainer.py#L83-L107)。

### 5.3 逐段耗时（batch 4，前向，`no_grad`）

| 段 | 耗时 | 备注 |
|---|---|---|
| Haar 子带构造 | < 1 ms | 固定核，无参数 |
| main root 112² | 4.9 ms | |
| side root 112² | 4.7 ms | |
| ★F1 窗内注意力 | 6.0 ms | |
| main block1 | 12.7 ms | |
| side block1 | 12.1 ms | |
| ★F2 窗内注意力 | 5.7 ms | |
| main block2 | 18.6 ms | |
| side block2 | 17.9 ms | |
| ★F3 全局注意力 | 5.7 ms | |
| main block3 | 32.0 ms | |
| side block3 + proj | 30.7 ms | |
| 12 × ViT block | 33.6 ms | |
| ★F4 全局注意力 | 2.3 ms | |
| decoder + head | 15.6 ms | |
| **整步 fwd+bwd（dual）** | **335.9 ms** | batch 4 |
| 整步 fwd+bwd（baseline） | 204.8 ms | 比值 1.64× |

四个融合模块合计 19.7 ms，只占整步的 5.9%。**融合不是瓶颈，第二套 R50 才是。**

### 5.4 120 epoch 时间预估（按实测 0.64 s/optimizer-step）

| 子集 | 训练样本 | optimizer step/epoch | 训练/epoch | 120 epoch（+每 10 epoch 测测试集） |
|---|---|---|---|---|
| SPARC | 428 | 54（= 107 micro-batch / 2） | 34.6 s | **≈ 72 min** |
| Zhongnan | 518 | 65 | 41.6 s | **≈ 87 min** |

对照 baseline（batch 8，无累积）：50–57 min。**dual 约为 1.4×**。
1 epoch 冒烟实测总耗时 57 s（含 82 张测试集推理 + CUDA 初始化）。

---

## 6. 正确性验证（R0 探针）

脚本：[scripts/probe_dual.py](../scripts/probe_dual.py)。

### 6.1 零初始化等价性 ✅

同一 `torch.manual_seed(1234)` 下先建 baseline 再建 dual（`super().__init__` 消耗同一段 RNG 流，
保证主路初始化逐位相同），两边 `load_from` 同一份 npz，同一输入：

```
[equiv] gamma=0  max|baseline-dual| = 0.000e+00  (rel 0.000e+00)  -> PASS
```

**逐元素完全相同**，不是"小于 1e-5"。这意味着训练第 0 步的预测与 baseline 一致，
任何后续差异都只能来自融合模块学到的内容，归因干净。

### 6.2 shape 契约 ✅

见 2.2 表，全部与 baseline 对齐；`DecoderCup` 的 `cat([up, skip])` 不报错。

### 6.3 权重加载 ✅

侧路复用主路同一份 ImageNet21k R50 权重（`conv_root/kernel`、`gn_root/scale|bias`、各 block unit），
见 [dual_encoder_model.py#L345-L356](../networks/dual_encoder_model.py#L345-L356)。
侧路**不需要** BN→GN 改造：`ResNetV2` 的 `PreActBottleneck` 内部本来就用 `GroupNorm`
（`gn1/gn2/gn3/gn_proj`），batch 4 的统计量不会像 BN 那样抖。

### 6.4 冒烟（1 epoch，SPARC）✅

```
107 iterations per epoch. 54 max iterations
...
epoch 1 test curve: {'epoch': 1, 'dice': 0.7948, 'assd': 9.92, 'hd95': 24.50, 'abs_delta_area': 4028.4}
test summary: n_cases=82
```

链路完整：数据集单通道 → 子带构造 → 交错编码 → 四级融合 → decoder → 四项指标 → `metrics.csv`。

---

## 7. 与 baseline 的完整差异清单

| 项 | baseline（A0） | dual（D11） |
|---|---|---|
| 输入 | 1 通道 repeat 3 | 主路 1 通道 repeat 3；侧路 3 通道子带 |
| 编码器 | R50-ViT ×1 | R50-ViT ×1 + R50 ×1 |
| 编码方式 | `ResNetV2.forward` 整体 | 拆 stage 交错，四级融合后继续往下 |
| skip | 3 条，纯主路 | 3 条，全部含侧路信息 |
| 融合参数 | 0 | 3.70 M |
| 总参数 | 105.28 M | 121.66 M |
| 第 0 步输出 | — | 与 baseline 逐元素相同 |
| 单步（batch 4） | 204.8 ms | 335.9 ms |
| 训练配置 | batch 8 | batch 4 + accum 2 |
| 输出目录 | `..._SPARC` | `..._bs4_ga2_SPARC_dual_L1234_auto` |

目录后缀保证不覆盖 baseline 权重（CLAUDE.md 第 2 条）。

---

## 8. 已知风险与局限

1. **★F2 的 56²→55² 裁剪**：融合结果最后一行/列被丢。影响可忽略（3.5% 像素），
   但意味着 ★F2 的窗内注意力有一部分输出是白算的（pad 区的零）。
2. **侧路的不对称性**：侧路 block3 出 1024 通道后被 1×1 压到 768，而主路在 ★F4 之前是 1024 通道进
   patch_embedding。侧路的 1024 通道全部来自 R50，没有 ViT。这是有意的（全局语义已由 12 层 ViT 负责），
   但答辩时不能说"为了省显存"，要说"职责分工"。
3. **`auto` 档位对分辨率敏感**：判据是 `res >= 56 → window`。若以后改 `img_size`（比如 256），
   112/128 两个接点仍会走窗内（因为都 >= 56），★F3 的 32² 会走全局，逻辑仍然成立；
   但如果把 ★F1 的窗调大（`window=16`），每窗 token 变 256，cost 涨 16 倍，需要重新测。
4. **融合模块只有 3.7M，未做容量扫描**。R2（gate，0.24M）/ R3（concat，7.5M）两个对照会回答
   "是不是参数越多越好"——R3 参数比主方案多一倍，如果它不更好，说明增益来自**内容重组**而不是容量。
5. **未接 AMP**（CLAUDE.md 第 3 条要求，但 trainer 目前没有）。当前 fp32 + batch 4 已满足显存约束，
   AMP 留作加速手段。**注意**：开 AMP 后再测一次 batch 6/8，显存占用会变，WDDM 阈值也可能变。
6. **`_pad_skip` 在 `a.size(2)==56` 时直接返回原张量**（不 copy）。若某天输入尺寸使 block1 输出正好 56，
   融合会原地改 skip，需留意。当前 img_size=224 不会触发。

---

## 9. 运行方式

```powershell
# 主方案（R1）
.\venv\Scripts\python.exe train.py --dataset Carotid --subset SPARC --arch dual `
    --fuse_op auto --levels 1,2,3,4 --batch_size 4 --grad_accum 2 `
    --max_epochs 120 --eval_test_every 10

# 对照
# R2 门控:   --fuse_op gate
# R3 concat: --fuse_op concat
# R7 单向:   --fuse_op auto --bidir 0

# 探针（不写权重）
.\venv\Scripts\python.exe scripts\probe_dual.py --batch_sizes 4,6,8 --steps 30

# 夜间队列（R1 → R2 → R3 → R7，两个子集，串行）
powershell -ExecutionPolicy Bypass -File scripts\night_queue_dual.ps1
```
