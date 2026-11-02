# 多中心联合训练（MC_union）：只改训练名单，把 Zhongnan 448 张并进 SPARC 的训练集

配套代码：[trainer.py](../../trainer.py)、[dataset_carotid.py](../../datasets/dataset_carotid.py)、[queue_c2d_edge_union.ps1](../../scripts/queue_c2d_edge_union.ps1)、[eval_cross_center.py](../../eval_cross_center.py)；口径背景：[experiments.md](../experiments.md) §0 / §1.7。

## 0. 一页版

**一句话**：网络、损失、优化器、训练长度、预训练权重全部与 A0 完全一致，唯一改动是训练列表从 SPARC 350 张换成 SPARC 350 + Zhongnan 448 = 798 张；测试仍然只在 SPARC test 的 160 张上算。

| 项 | Dice % | ASSD mm | HD mm | \|ΔArea\| mm² |
|---|---|---|---|---|
| A0（SPARC 350 训练 → SPARC test 160） | 90.32 | 0.146 | 0.64 | 2.11 |
| **MC_union（SPARC+ZN 798 训练 → SPARC test 160）** | **91.06** | **0.134** | **0.60** | 2.0 |
| A0（只训 SPARC）→ Zhongnan test 187（**真零样本跨中心**） | 90.75 | 0.1405 | 0.5978 | 1.8447 |
| MC_union（训练已含 ZN 448 张）→ Zhongnan test 187（**同中心留出，不是跨中心**） | 92.28 | 0.1169 | 0.5233 | 1.4720 |

相对 A0：本中心 **+0.74**（过 §0 的 \|ΔDice\|≥0.5 门槛），HD 同向降 0.04 mm，且**不增加任何参数**。但必须说清楚：**这条不能当作论文卖点**——训练数据量从 350 张涨到 798 张（2.3 倍），测试集没变，涨点是数据红利的预期结果，不是方法贡献；而且 MC_union 的 92.28 是在**训练过的同一中心**的留出测试集上得到的，**不是跨中心泛化**（真正的零样本跨中心对照只有 A0→ZN 90.75 与 D1→ZN 91.18）。

## 1. 输入是怎么处理的

### 1.1 三个列表文件（唯一改动点）

训练脚本按 `<list_dir>/<subset>/train.txt` 找名单（[trainer.py L65-73](file:///e:/mkl/code/python_code/carotid-artery/trainer.py#L65-L73)），所以联合训练只需要一个带 `SPARC/` 子层的目录：

| 文件 | 行数 | 内容 |
|---|---|---|
| [lists/lists_Carotid2D_union/SPARC/train.txt](file:///e:/mkl/code/python_code/carotid-artery/lists/lists_Carotid2D_union/SPARC/train.txt) | 798 | SPARC 训练 350 + Zhongnan 训练 448，每行一个 `.nii.gz` 相对路径 |
| [lists/lists_Carotid2D_union/SPARC/test.txt](file:///e:/mkl/code/python_code/carotid-artery/lists/lists_Carotid2D_union/SPARC/test.txt) | 160 | 就是原 SPARC test，逐行未动 |

行首 `./` 在 [read_list](file:///e:/mkl/code/python_code/carotid-artery/datasets/dataset_carotid.py#L17-L21) 里被去掉，标签路径由 [_label 命名规则](file:///e:/mkl/code/python_code/carotid-artery/datasets/dataset_carotid.py#L33-L34) 自动推出，所以两个中心的目录命名差异（`PlaqueDataNii_SPARC_S/` 与 `PlaqueDataNii_Zhongnan_S/`）不影响读取。

**不泄漏的前提**：Zhongnan 只有 train 切片进训练；SPARC test 160 张从未出现在 train.txt 里；评测用的 Zhongnan test 是从它自己的 `test.txt` 独立读取的（见 4 节）。

### 1.2 单张切片的读取链

[__getitem__ L88-101](file:///e:/mkl/code/python_code/carotid-artery/datasets/dataset_carotid.py#L88-L101) 依次做：

1. `nib.load` 读图像与 `_label` 标签（float32）；若意外是 3D 数据取中间层；
2. 标签 `> 0.5` 二值化；
3. **仅训练集**做增强：[rot90 / 左右翻转 / 上下翻转 / ±15° 旋转](file:///e:/mkl/code/python_code/carotid-artery/datasets/dataset_carotid.py#L74-L86)，图像用双三次（order=3）、标签用最近邻（order=0），保证标签不被插值出灰边；
4. `normalize_image`：按 **1%~99% 分位截断**后做逐张 z-score（[L37-46](file:///e:/mkl/code/python_code/carotid-artery/datasets/dataset_carotid.py#L37-L46)）——截断是为了压掉超声的亮点伪影，不让单个极值拖偏均值方差；
5. `resize_pair` 到 224×224（[L49-55](file:///e:/mkl/code/python_code/carotid-artery/datasets/dataset_carotid.py#L49-L55)），图像 order=3、标签 order=0 再 `>0.5` 二值化；
6. 返回 `image: [1,224,224]` 单通道 + `label: [224,224] long` + 原始相对路径 `case`（评测回原网格用）。

两个中心的原始切片分辨率/网格不同，但都先在自己原分辨率上做标准化再统一 resize，因此网络看到的分布口径一致；差异只体现在形态与设备纹理上。

## 2. 模型与损失

### 2.1 模型

就是 A0：TransUNet，R50 + ViT-B/16 串联，解码器接 R50 skip，输入 224×224，输出 2 通道 logits，加载 ImageNet21k 预训练（`--arch baseline`，构造分支在 [train.py L206 起](file:///e:/mkl/code/python_code/carotid-artery/train.py#L206)）。**参数量、FLOPs 与 A0 完全相同**，没有新增任何模块，也没有第 4 输入通道（`edge_in=0`）。

### 2.2 损失

`0.5·CE + 0.5·Dice`，与 A0 逐字一致（[trainer.py L143-L144](file:///e:/mkl/code/python_code/carotid-artery/trainer.py#L143-L144)）：CE 作用在 2 通道 logits 上，Dice 对 softmax 后的前景概率算；无深监督、无辅助损失、无领域损失（`--domain_loss 0`）。

### 2.3 训练配置与为什么

| 项 | 值 | 位置 |
|---|---|---|
| 优化器 | SGD，lr 0.01，momentum 0.9，wd 1e-4 | [L145](file:///e:/mkl/code/python_code/carotid-artery/trainer.py#L145) |
| 调度 | poly：`lr = 0.01·(1 − iter/max_iter)^0.9`，每次 optimizer step 更新 | [L193-L195](file:///e:/mkl/code/python_code/carotid-artery/trainer.py#L193-L195) |
| batch | `--batch_size 4 --grad_accum 2`（等效 bs8），梯度累积每 2 个 micro-batch 更新一次 | [L186-L192](file:///e:/mkl/code/python_code/carotid-artery/trainer.py#L186-L192) |
| 长度 | 200 epoch × 200 iter/epoch = 20000 step（poly 总长绑死） | [L150-L152](file:///e:/mkl/code/python_code/carotid-artery/trainer.py#L150-L152) |
| seed | 1234（`--seed` 默认） | [train.py L64-L65](file:///e:/mkl/code/python_code/carotid-artery/train.py#L64-L65) |
| 读数 | 终点 ep200（`epoch_199.pth`），`--eval_test_every 5` 每 5 epoch 记一次曲线 | [trainer.py L222-L239](file:///e:/mkl/code/python_code/carotid-artery/trainer.py#L222-L239) |

为什么全部不动：这是一个**单变量对照**。A0 与 MC_union 之间只差 `--list_dir` 一个参数（[队列脚本 L50-58](file:///e:/mkl/code/python_code/carotid-artery/scripts/queue_c2d_edge_union.ps1#L50-L58)），所以两者的 Dice 差只能归因于训练数据组成，不能归因于网络、损失或训练长度。

## 3. 指标与结果

- **Dice**：前景重合度，对预测/GT 面积失衡敏感（预测整张全前景也能拿高分，所以要配 ΔArea 一起看）。
- **ASSD**：预测表面到 GT 表面的平均距离（mm），越小越好。
- **HD**：最大表面距离（mm），只报这一个，不用 hd95；它被单个漂浮小岛/断裂主导，正对应临床最怕的远处假阳性。
- **|ΔArea|**：预测面积减 GT 面积的绝对值（mm²），逐切片平均。

全部在**原分辨率网格**上用 NIfTI 头里的 spacing 计算（[metrics_carotid.py](file:///e:/mkl/code/python_code/carotid-artery/metrics_carotid.py#L15-L55)），逐切片平均、不做病例聚合。

**怎么读**：

1. SPARC test 上 +0.74 越过 0.5 门槛，且 ASSD −0.012、HD −0.04 同向 → 边界指标同向改善，不是面积漂移（|ΔArea| 反而从 2.11 降到 2.0）。但训练数据量同时从 350 涨到 798，**这个涨点是数据量红利，不是方法增益**。
2. Zhongnan 的 92.28 只能读作「同中心留出测试」：训练已用 ZN 的 448 张训练切片，测试用 ZN 另外 187 张，两者病例交集为 0（已核验 348 个训练病例 vs 148 个测试病例，无重叠），所以不是数据泄漏；但**它不是跨中心泛化**——模型训练时已经见过 Zhongnan 这个域的图像。真正的零样本跨中心对照是 A0→ZN 90.75、D1→ZN 91.18。
3. 这是**单 seed**；§1.1 的 seed 噪声带是 ±0.3，+0.74 超出，但仍需 `UNION_s2024`（已入队、11-02 09:02 启动）复跑确认 → **待验证**。
4. 同一轮里 A0+sobel / A0+laplacian 只有 +0.14 / +0.13（噪声内），可作为「同一批队列、同一口径」的负对照：数据口径带来的读数更大，但这正好说明它来自数据量，而不是任何输入/结构改动。

## 4. 代码实现在哪里

| 环节 | 位置 |
|---|---|
| 队列入口（3 个 job 串行，MC_union 排第一） | [queue_c2d_edge_union.ps1 L50-58](file:///e:/mkl/code/python_code/carotid-artery/scripts/queue_c2d_edge_union.ps1#L50-L58) |
| 名单读取 + 训练/测试 DataLoader | [trainer.py L65-L84](file:///e:/mkl/code/python_code/carotid-artery/trainer.py#L65-L84) |
| 数据构造与增强 | [dataset_carotid.py L58-L109](file:///e:/mkl/code/python_code/carotid-artery/datasets/dataset_carotid.py#L58-L109) |
| 损失与优化器 | [trainer.py L90-L145](file:///e:/mkl/code/python_code/carotid-artery/trainer.py#L90-L145) |
| 训练内 ep200 读数（`test_curve.csv` / `test summary`） | [trainer.py L222-L239](file:///e:/mkl/code/python_code/carotid-artery/trainer.py#L222-L239)、[L280](file:///e:/mkl/code/python_code/carotid-artery/trainer.py#L280) |
| 跨中心零样本评测 | [eval_cross_center.py L81-L135](file:///e:/mkl/code/python_code/carotid-artery/eval_cross_center.py#L81-L135) |

复现命令（PowerShell，工作目录 `e:\mkl\code\python_code\carotid-artery`）：

```powershell
E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe train.py --subset SPARC --batch_size 4 --grad_accum 2 --max_epochs 200 --eval_test_every 5 --arch baseline --list_dir ./lists/lists_Carotid2D_union

E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe eval_cross_center.py --ckpt E:\mkl\code\python_code\model\TU_Carotid224\TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_mcUnion_c2d --arch baseline --train_subset union --test_subset Zhongnan --epoch 199
```

## 5. 结论边界与其他

| 能说 | 不能说 |
|---|---|
| 在完全相同的网络/损失/超参下，训练集并入 Zhongnan 448 张后，SPARC 测试 Dice +0.74、HD −0.04 | 这是方法贡献——它只是训练数据口径（training setup），且涨点与数据量翻倍混在一起，**不能当论文卖点** |
| Zhongnan test 187 的同中心留出读数为 92.28（训练/测试病例交集为 0，无泄漏） | 92.28 是跨中心泛化：训练已用 ZN 训练切片，它只是同中心留出测试 |
| 增益伴随 ASSD、HD、|ΔArea| 三项同向，且参数量为 0 增量 | 增益来自「跨中心多样性」（见下面的关键歧义） |

**关键歧义（必须写进 limitation）**：A0 用 350 张、MC_union 用 798 张，训练数据量翻了 2.3 倍。+0.74 里有多少来自「数据变多」、多少来自「跨中心多样性」，现有实验**分不开**。要区分至少需要一项等数据量对照：把 SPARC 自身的 350 张过采样/重复到 798 张再训一次（只变数据量、不变中心），或做数据量-性能曲线。这两项**尚未做**，所以 +0.74 不能记在「跨中心」账上，更不能当成方法创新。

其他局限：

- 单 seed 结论，`UNION_s2024`（seed 2024）正在跑，`UNION_dl`（再叠领域损失）在其后 → 两者都待验证；
- Zhongnan 侧只有一次同中心留出读数（187 例），没有第三中心、也没有反向对照（只训 Zhongnan、测 SPARC test）；
- 训练内读数与留出/跨中心读数都是在原分辨率网格上逐切片平均，不做病例聚合、不做显著性检验（§0 明确约定）；
- 训练时长随数据量从约 55 min 涨到约 115 min（同机同口径），这是数据量翻倍的直接代价。
