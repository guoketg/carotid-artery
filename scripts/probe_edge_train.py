"""边缘特征融合·输入级 的真实数据训练步自检（跑 3 个 optimizer step）。

验证：数据集 → 模型 → loss → backward → optimizer.step 全链路在 edge_in=1 下通畅，
且显存占用在预期内（stem 多一个通道，理论上与 baseline 几乎相同）。
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_carotid import CarotidDataset
from networks.vit_seg_modeling import CONFIGS, VisionTransformer
from torch.utils.data import DataLoader

def main():
    ROOT = r"E:\mkl\code\datasets\Carotid2D"
    LIST = os.path.join("E:\\mkl\\code\\python_code\\carotid-artery",
                        "lists", "lists_Carotid2D", "SPARC", "train.txt")

    ds = CarotidDataset(root=ROOT, list_file=LIST, split="train", img_size=224,
                        use_wavelet=False)
    print(f"[data] train 张数 = {len(ds)}，单样本 shape = {tuple(ds[0]['image'].shape)}",
          flush=True)

    config = copy.deepcopy(CONFIGS["R50-ViT-B_16"])
    config.n_classes = 2
    config.n_skip = 3
    config.patches.grid = (14, 14)

    net = VisionTransformer(config, img_size=224, num_classes=2, edge_in=True).cuda()
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    ce = torch.nn.CrossEntropyLoss()

    # Windows 下 DataLoader 多进程需要 __main__ 保护；这里只用 num_workers=0
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    torch.cuda.reset_peak_memory_stats()
    it = 0
    for batch in dl:
        img = batch["image"].cuda()
        lab = batch["label"].cuda()
        out = net(img)
        loss = ce(out, lab.long())
        loss.backward()
        opt.step()
        opt.zero_grad()
        it += 1
        print(f"[step {it}] img={tuple(img.shape)} out={tuple(out.shape)} "
              f"loss={loss.item():.4f} peak_mem={torch.cuda.max_memory_allocated()/2**30:.2f}GB",
              flush=True)
        if it >= 3:
            break
    print("[ok] edge_in=1 训练链路通畅", flush=True)


if __name__ == "__main__":
    main()
