# SV-Recall V4: 基于 LightGCN 的短视频协同过滤召回通道

V4 在 V3 多模态内容召回链路之外，新增一条基于用户-短视频交互图的协同过滤召回通道。它不使用 text / image / video 内容特征，而是将 MicroLens-100K 的隐式反馈交互构造成 user-item 二部图，使用 LightGCN 学习 user / item ID embedding，并通过 BPR 优化用户对已交互 item 的偏好排序。

V4 的定位是补充 V3 的内容语义召回：V3 侧重多模态内容理解与语义泛化，V4 侧重用户行为图中的协同过滤信号。两者后续可作为多路召回通道进行融合。

## 1. 项目目标

- 复用 V3 已处理好的 MicroLens-100K 行为样本；
- 基于 train split 构建 user-item 二部图；
- 训练 LightGCN 协同过滤召回模型；
- 使用 BPR loss 优化正负 item 排序；
- 使用 full-catalog ranking 评估 Recall@K、HitRate@K、NDCG@K、MRR@K；
- 为后续多路召回融合提供协同过滤候选与排序分数。

## 2. 数据集

V4 使用 V3 的 MicroLens-100K 处理结果：

```text
data/processed/v3/microlens_100k/behavior_samples_train.csv
data/processed/v3/microlens_100k/behavior_samples_val.csv
data/processed/v3/microlens_100k/behavior_samples_test.csv
data/processed/v3/microlens_100k/item_ids.csv
```

数据规模：

| split | samples |
|---|---:|
| train | 507,529 |
| val | 105,938 |
| test | 105,938 |

图构建口径：

- train split 的正反馈边用于构造 LightGCN 图；
- 原始 `user_id` 映射为连续 `user_index`；
- `item_index` 沿用 V3 的 `item_ids.csv`，保证 V3 / V4 item 编号一致；
- val / test split 用于 sampled validation 和 full-catalog 评估，不参与 train 图构建。

## 3. 技术方案

### 3.1 LightGCN

V4 的 LightGCN 保留 ID embedding 和图传播：

- user embedding；
- item embedding；
- 归一化邻接矩阵 `D^-1/2 A D^-1/2`；
- 多层 user-item 图传播；
- 对第 0 层到第 L 层 embedding 求平均作为最终表示。

当前训练配置：

| config | value |
|---|---:|
| embedding dim | 128 |
| layers | 3 |
| loss | BPR |
| batch size | 4096 |
| epochs | 50 |
| learning rate | 0.001 |
| weight decay | 0.0001 |

当前模型规模：

| module | parameters |
|---|---:|
| user embedding | 12,800,000 |
| item embedding | 2,526,464 |
| total | 15,326,464 |

### 3.2 BPR 训练

每个训练样本包含：

- `user_index`
- `positive_item_index`
- `negative_item_index`

负样本从该用户未交互 item 中采样。BPR 目标是让：

```text
score(user, positive_item) > score(user, negative_item)
```

训练过程中的 `accuracy` 是 pairwise 排序正确率，即正样本分数高于采样负样本分数的比例。

## 4. 项目结构

```text
short-video-mmrec/
├── configs/
│   └── v4/
│       └── microlens_lightgcn.yaml
├── scripts/
│   └── v4/
│       ├── 01_train_lightgcn.py
│       └── 02_eval_full_recall.py
├── src/
│   └── v4/
│       ├── data/
│       │   └── microlens_graph_dataset.py
│       ├── models/
│       │   └── lightgcn.py
│       ├── losses/
│       │   └── bpr_loss.py
│       └── train/
│           └── train_lightgcn.py
└── outputs/
    └── v4/
        └── microlens_100k/
            └── lightgcn_bpr_e50/
```

## 5. V4 运行流程

### Step 1: 训练 LightGCN

```bash
python scripts/v4/01_train_lightgcn.py \
  --config configs/v4/microlens_lightgcn.yaml \
  --output-dir outputs/v4/microlens_100k/lightgcn_bpr_e50 \
  --epochs 50
```

输出：

```text
outputs/v4/microlens_100k/lightgcn_bpr_e50/lightgcn_best.pt
outputs/v4/microlens_100k/lightgcn_bpr_e50/lightgcn_latest.pt
outputs/v4/microlens_100k/lightgcn_bpr_e50/train_config.json
```

当前最佳 checkpoint：

```text
outputs/v4/microlens_100k/lightgcn_bpr_e50/lightgcn_best.pt
```

最佳 sampled validation 出现在 epoch 43：

| split | loss | accuracy | pos_score | neg_score | score_margin |
|---|---:|---:|---:|---:|---:|
| train | 0.0214 | 99.54% | 6.7889 | 0.1580 | 6.6309 |
| val | 0.2996 | 87.36% | 3.4259 | 0.1260 | 3.2999 |

### Step 2: full-catalog 评估

```bash
python scripts/v4/02_eval_full_recall.py \
  --checkpoint outputs/v4/microlens_100k/lightgcn_bpr_e50/lightgcn_best.pt \
  --eval-split test
```

输出：

```text
outputs/v4/microlens_100k/lightgcn_bpr_e50/full_recall_test_metrics.json
```

评估口径：

- 使用 train split 构造 LightGCN 图；
- 编码所有 train 用户和全量 item；
- 对每个 eval 用户打分全量 19,738 个候选 item；
- 过滤 train 中已交互 item；
- 使用 val/test split 的观测交互作为相关 item；
- 计算 Recall@K、HitRate@K、NDCG@K、MRR@K。

## 6. 评估结果

当前结果来自：

```text
outputs/v4/microlens_100k/lightgcn_bpr_e50/full_recall_test_metrics.json
```

评估用户数为 100,000，候选 item 数为 19,738。

| K | Recall@K | HitRate@K | NDCG@K | MRR@K |
|---:|---:|---:|---:|---:|
| 10 | 2.08% | 2.14% | 0.98% | 0.66% |
| 20 | 3.80% | 3.91% | 1.42% | 0.79% |
| 50 | 7.27% | 7.53% | 2.11% | 0.90% |
| 100 | 11.69% | 12.04% | 2.84% | 0.96% |

## 7. 与 V3 对比

V3 最佳多模态召回结果来自 10 epochs、batch size 1024 的多模态双塔训练版本。

| K | V4 LightGCN Recall@K | V3 Multimodal Recall@K |
|---:|---:|---:|
| 10 | 2.08% | 2.86% |
| 20 | 3.80% | 5.22% |
| 50 | 7.27% | 9.57% |
| 100 | 11.69% | 14.57% |

V4 的单路召回弱于 V3 最佳多模态召回，但与 V3 形成互补：V3 依赖多模态内容特征进行语义泛化，V4 依赖用户行为图建模协同过滤关系。后续多路召回融合可以同时利用内容语义候选和协同过滤候选。

## 8. 当前版本定位

V4 已完成：

- MicroLens train graph 构造；
- LightGCN user / item embedding 与图传播；
- BPR 正负样本训练；
- sampled validation；
- full-catalog ranking 评估；
- 与 V3 多模态召回的指标对比。

V4 尚未包含：

- V3 多模态召回与 V4 协同过滤召回的分数融合；
- LightGCN item embedding 与多模态 item embedding 的联合训练；
- popularity / V3 / V4 多路召回统一对比报告；
- reranker 或融合排序模型。

## 9. 后续计划

- 将 V3 多模态召回和 V4 LightGCN 召回做 score-level 或 candidate-level fusion；
- 引入 popularity baseline 与多路召回对照表；
- 尝试把 V3 item content embedding 作为 LightGCN item embedding 初始化或正则；
- 在融合候选上加入轻量 reranker，结合内容特征、协同过滤分数和用户历史特征优化 TopK。
