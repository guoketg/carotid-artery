---
name: paper-search
description: Find and rank medical image segmentation papers by venue priority, recent top conferences first. Use when asked to collect papers, update the reading list, or refresh docs/venue-survey.md. Do not use for non-medical-imaging surveys.
---

# Paper Search

按固定优先级检索并筛选论文，结果写入 `docs/venue-survey.md`，不新建文件。

## 1. 优先级（硬性）

| 档 | 范围 | 处理 |
|---|---|---|
| 1 | 2025–2026 顶会：CVPR、ICCV、ECCV、NeurIPS、ICLR、ICML、AAAI | 最优先，占每批 ≥ 50% |
| 2 | CCF B 会议：MICCAI、IPMI、BIBM、ICME、ICASSP、ICMR | 次优先 |
| 3 | 期刊：TMI、MedIA、UMB、CMPB、Pattern Recognition、IEEE TIP、JBHI | 再次 |
| 4 | 高价值 arXiv | 仅限 2024 年后、方法可直接落地、且尚无会议版本 |
| 剔除 | 2022 年前的 arXiv、无会议版本的老预印本、非 CCF 小会、低质量预印本 | 不收录 |

- 2021–2022 年的经典工作（U-Net 系）只在「基线/对照」语境下保留，不进主推清单。
- 每条必须写明年份与出处；不确定就标「待核」，不得省略。

## 2. 检索

DBLP 已被反爬拦截，用 OpenAlex：

```text
https://api.openalex.org/works?search=<query>&per-page=8&filter=from_publication_date:2024-01-01&select=title,publication_year,primary_location
```

- `per-page` 不超过 10：WebFetch 返回的正文约 16k 字符会截断，超了 JSON 解析失败。
- WebFetch 结果是包装 JSON，真实内容在 `references[0].content`，需要二次 `JSON.parse`。
- 会议名优先取 `primary_location.source.display_name`；取不到就改用 WebSearch 用精确标题核对，禁止猜。
- 检索词必须锚定模态与部位，不要只搜 `medical image segmentation`：
  `carotid` / `plaque` / `ultrasound` / `intima media` / `speckle` / `B-mode`，
  再叠加方法词 `decoder` / `boundary loss` / `uncertainty` / `semi-supervised` / `domain generalization`。
- 每批 10–20 条起步，先在对话里给结论，再落表。

## 3. 记录格式

每条一行，五列：

| # | 论文 | 出处（年份+会议） | 核心做法 | 可迁移点（对应本项目哪条线） |

追加批次时在文末新开「§N 补充清单」，注明日期与筛选口径，不改动既有编号。

## 4. 判据

本项目约束：2D 颈动脉超声、两个中心、单卡 RTX 3060 12GB、无文本提示、本科生投稿。
只收录满足「能涨点 + 3060 能训 + 有可发表的创新点」的工作，三者缺一不写进主推清单。

## 5. 交付

- 汇总写在 `docs/venue-survey.md`，按上述优先级排序，而非按主题平铺。
- 回答时先给「本批新增几篇、其中 25–26 顶会几篇」，再给最值得先做的 2–3 条。
- 剔除旧条目时说明剔除了哪几条及理由，不静默删除。
