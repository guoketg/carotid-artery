# 双骨干交错四级融合：训练命令速查

本文只讲「怎么把训练跑起来」，方案原理见 [dual-encoder-design.md](../dual-encoder-design.md)，消融矩阵见 [experiments.md](../experiments.md)。

所有命令都要在项目根目录 `F:\code\MyPython\TransUNet` 下执行。`train.py` 里的 `../model` 和 `./lists/lists_Carotid` 都是相对路径，换了目录会把权重写到别处、或找不到划分文件。

## 0. 跑之前必须确认：哪些融合档位现在学不动

当前代码把融合支路的输出投影和缩放系数 `gamma` **都**初始化成了 0。两者相乘导致梯度恒为 0，支路永远停在恒等映射上：

| 融合档位 | 初始等价于原图基线 | 能否学习 |
|---|---|---|
| `window` / `global`（跨支注意力） | 是 | 否，缩放系数、输出投影、qkv 梯度全为 0 |
| `concat`（拼接后 1×1 回投影） | 是 | 否，同上 |
| `deform`（可变形大核跨支聚合） | 是 | 否，输出投影与偏移卷积都为零 |
| `gate`（逐像素门控） | 是 | 能，门控来自 `sigmoid(0)=0.5`，非零 |
| `aff`（MS-CAM 幅度调制） | 是 | 能，注意力分支正常初始化 |
| `auto` + `--cross_scale 1` | 是 | 注意力支路不能，但跨尺度回流能 |

因此**要验证双骨干的增益，现在只能选 `aff`、`gate`，或 `auto` 配 `--cross_scale 1`**。选 `auto`、`deform`、`concat` 跑出来的结果在数值上等于单通道原图基线，不能用来支撑结论。

## 1. 单次训练（前台，实时看日志）

```powershell
cd F:\code\MyPython\TransUNet

.\venv\Scripts\python.exe train.py `
  --dataset Carotid --subset SPARC `
  --arch dual --levels 1,2,3,4 --fuse_op aff `
  --batch_size 4 --grad_accum 2 `
  --max_epochs 120 --eval_test_every 10 `
  --tag rerun1
```

`--tag rerun1` 会拼进输出目录名。每次重训都换一个 tag，否则会覆盖上一次的曲线和权重。

## 2. 后台运行（关掉终端也不中断）

```powershell
New-Item -ItemType Directory -Force logs | Out-Null

Start-Process -FilePath ".\venv\Scripts\python.exe" `
  -ArgumentList "train.py","--dataset","Carotid","--subset","SPARC","--arch","dual","--levels","1,2,3,4","--fuse_op","aff","--batch_size","4","--grad_accum","2","--max_epochs","120","--eval_test_every","10","--tag","rerun1" `
  -RedirectStandardOutput "logs\rerun1.out.log" `
  -RedirectStandardError "logs\rerun1.err.log" `
  -NoNewWindow
```

注意 `Start-Process` 的参数是用逗号分隔的字符串数组，不能直接粘 PowerShell 的续行写法。

## 3. 冒烟测试（先确认能跑通）

```powershell
.\venv\Scripts\python.exe train.py `
  --dataset Carotid --subset SPARC `
  --arch dual --levels 1,2,3,4 --fuse_op aff `
  --batch_size 2 --grad_accum 1 `
  --max_epochs 1 --eval_test_every 0 `
  --tag smoke
```

`--eval_test_every 0` 关闭测试集评估，只为确认前向、反传、显存都没问题。

## 4. 复用夜间队列脚本

脚本里已经内置了「等 GPU 空闲 10 分钟」「旧输出目录自动改名归档」「每个 job 单独写日志并追加状态 CSV」：

```powershell
# 只跑可变形大核跨支聚合那一档
powershell -ExecutionPolicy Bypass -File scripts\night_queue_dual2.ps1 -Only r4_sparc

# 先看它会执行什么
powershell -ExecutionPolicy Bypass -File scripts\night_queue_dual2.ps1 -DryRun

# 跳过不需要的 job
powershell -ExecutionPolicy Bypass -File scripts\night_queue_dual2.ps1 -Skip r4_sparc,r5_sparc
```

队列脚本里的 job 名只是槽位标识（`r4_sparc` 等），配置由脚本第 43–50 行的 `--fuse_op` 参数决定，改配置直接改那一行。

## 5. 参数速查

| 参数 | 该给什么 | 说明 |
|---|---|---|
| `--dataset Carotid` | 必给 | 自动设 `root_path=F:\code\datasets\2D`、`list_dir=./lists/lists_Carotid`、`num_classes=2` |
| `--subset SPARC\|Zhongnan` | 必给 | 决定读 `lists/lists_Carotid/<subset>/train.txt` |
| `--arch dual` | 双骨干必给 | 同时会把 `use_wavelet` 强制置 0，侧路自己从原图算方向子带 |
| `--levels 1,2,3,4` | 四个接点全开 | 1=112²/64，2=56²/256，3=28²/512，4=14²/768 |
| `--fuse_op` | `aff` / `gate` / `auto` / `deform` / `concat` / `window` / `global` | 见第 0 节，死锁档位不要拿来下结论 |
| `--batch_size 4 --grad_accum 2` | 固定 | 等效 batch 8；直接给 batch 8 会触发 WDDM 换页，单步慢 2.6 倍 |
| `--max_epochs 120` | 与历史实验一致 | 超过 60 才会每 50 epoch 存一次中间权重 |
| `--eval_test_every 10` | 建议 10 | 每 10 epoch 在测试集上算 Dice、ASSD、HD95、DeltaArea，写入 `test_curve.csv` |
| `--tag` | 每次换一个 | **只在 `--arch dual` 时生效** |
| `--bidir 0` | 单向对照才给 | 融合只改写主路，侧路原样传出 |
| `--side_pretrained 0` | 侧路权重消融才给 | 侧路不加载 ImageNet21k R50 |
| `--num_workers` | 保持默认 1 | 本机内存紧张，调大会被系统杀进程 |
| `--seed` | 默认 1234 | 改种子会加进目录名，可用于复现性检查 |

超参别动：`base_lr=0.01`、`img_size=224`、`n_skip=3`、`vit_name=R50-ViT-B_16` 都是历史实验的固定口径，改了就不能和已有结果对比。

## 6. 输出位置与文件

目录名由 `train.py` 按参数自动拼接，例如上面的命令得到：

```text
F:\code\MyPython\model\TU_Carotid224\
  TU_pretrain_R50-ViT-B_16_skip3_epo120_bs4_ga2_224_SPARC_dual_L1234_aff_rerun1\
    log.txt           每 epoch 的 loss、学习率、耗时
    log\              tensorboard 事件文件
    test_curve.csv    每 10 epoch 的测试集四指标
    metrics.csv       最后一个 epoch 的逐病例指标
    epoch_*.pth       权重
```

启动后先看两行输出确认没走偏：

- `The length of train set is: ...` —— SPARC 全量应是千级切片；
- 首个 epoch 耗时约 75 秒（约 0.70 秒/迭代，107 迭代/epoch）。

`num_workers=1` 是瓶颈，GPU 利用率只有六成多，这是已知情况，不是配置错误。

## 7. 跑完验证融合支路是否真的在学

读最后一个权重里的缩放系数：

```powershell
.\venv\Scripts\python.exe -c "import torch,glob; p=sorted(glob.glob(r'F:\code\MyPython\model\TU_Carotid224\*rerun1*\epoch_*.pth'))[-1]; sd=torch.load(p,map_location='cpu'); print({k:round(v.abs().max().item(),6) for k,v in sd.items() if 'gamma' in k})"
```

四个接点的 `gamma` 全部为 0，说明这一档融合没有生效，结果等于单通道原图基线；非 0 才说明侧路信息确实进入了主路。

## 8. 一次跑多档位

把要跑的档位写成数组循环，避免手敲重复命令：

```powershell
cd F:\code\MyPython\TransUNet

$ops = @("aff", "gate")   # 只放第 0 节里「能学」的档位

foreach ($op in $ops) {
    $tag = "rerun_$op"
    Write-Host "===== start $tag ====="
    .\venv\Scripts\python.exe train.py `
      --dataset Carotid --subset SPARC `
      --arch dual --levels 1,2,3,4 --fuse_op $op `
      --batch_size 4 --grad_accum 2 `
      --max_epochs 120 --eval_test_every 10 `
      --tag $tag *> "logs\$tag.log"
    Write-Host "===== end $tag exit=$LASTEXITCODE ====="
}
```

单 job 约 2.5 小时，三个 job 需要连续占卡。跑之前确认没有别的 `train.py` 在跑。
