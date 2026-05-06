# SV-Recall V3: 基于多模态内容理解的短视频 Item 表征与语义召回

V3 是“面向短视频搜索推荐的多模态内容理解与语义召回系统”的多模态 item 表征版本，在 V2 行为监督双塔召回基础上，引入 MicroLens-100K 官方预提取的文本、图像和视频特征，构建图文视频融合的短视频 item encoder，并接入语义召回训练与 full-catalog 评估流程。

V3 的核心目标是验证多模态内容特征能否以工程化方式接入短视频召回系统，使 item 表征从纯文本语义扩展为 text / image / video 融合表示。

## 1. 项目目标

本项目面向短视频多模态内容理解与召回场景，完成以下能力：

- 基于 MicroLens-100K 构造短视频隐式反馈召回样本；
- 对齐 text / image / video 三类官方预提取多模态特征；
- 构建多模态 Item Encoder：文本特征 + 图像特征 + 视频特征 + item ID embedding；
- 构建基于历史行为的 User Tower；
- 使用 sampled softmax 训练多模态双塔召回模型；
- 使用 full-catalog ranking 进行 Recall@K、HitRate@K、NDCG@K、MRR@K 离线评估。

## 2. 技术栈

- Python 3.10
- PyTorch
- pandas / numpy
- MicroLens-100K
- BGE-M3 text feature
- CLIP-RN50 image feature
- VideoMAE video feature
- Two-Tower Retrieval
- Sampled Softmax
- Full-Catalog Ranking

## 3. 数据集

本项目使用 MicroLens-100K 数据集：

- 用户数：100,000
- item 数：19,738
- 交互数：719,405
- 反馈类型：隐式反馈，观测交互均保存为 `label=1`

处理后的数据位于：

```text
data/processed/v3/microlens_100k/
```

当前样本规模：

| split | samples |
|---|---:|
| train | 507,529 |
| val | 105,938 |
| test | 105,938 |

MicroLens-100K 官方预提取特征：

| modality | backbone | dim | shape |
|---|---|---:|---|
| text | BGE-M3 | 1024 | 19738 x 1024 |
| image | CLIP-RN50 | 1024 | 19738 x 1024 |
| video | VideoMAE | 768 | 19738 x 768 |

## 4. 项目结构

```text
short-video-mmrec/
├── configs/
│   └── v3/
│       └── microlens_mvp.yaml
├── scripts/
│   └── v3/
│       ├── 00_check_microlens_data.py
│       ├── 01_prepare_microlens_samples.py
│       ├── 02_train_multimodal_retrieval.py
│       └── 03_eval_full_recall.py
├── src/
│   └── v3/
│       ├── data/
│       │   └── microlens_retrieval_dataset.py
│       ├── models/
│       │   ├── multimodal_item_encoder.py
│       │   └── user_tower.py
│       ├── losses/
│       │   └── sampled_softmax_loss.py
│       ├── eval/
│       │   └── retrieval_metrics.py
│       └── train/
│           └── train_retrieval.py
├── data/
│   └── processed/
│       └── v3/
│           └── microlens_100k/
└── outputs/
    └── v3/
        └── microlens_100k/
            └── retrieval_mvp_bs1024_e10/
```

## 5. V3 运行流程

### Step 1: 检查 MicroLens 原始数据

```bash
python scripts/v3/00_check_microlens_data.py
```

输出：

```text
data/processed/v3/microlens_100k/raw_check_summary.json
```

### Step 2: 构造召回训练样本

```bash
python scripts/v3/01_prepare_microlens_samples.py
```

输出：

```text
data/processed/v3/microlens_100k/behavior_samples_train.csv
data/processed/v3/microlens_100k/behavior_samples_val.csv
data/processed/v3/microlens_100k/behavior_samples_test.csv
data/processed/v3/microlens_100k/item_ids.csv
data/processed/v3/microlens_100k/feature_config.json
data/processed/v3/microlens_100k/prepare_summary.json
```

### Step 3: 训练多模态双塔召回模型

```bash
python scripts/v3/02_train_multimodal_retrieval.py \
  --config configs/v3/microlens_mvp.yaml \
  --output-dir outputs/v3/microlens_100k/retrieval_mvp_bs1024_e10 \
  --batch-size 1024 \
  --epochs 10
```

输出：

```text
outputs/v3/microlens_100k/retrieval_mvp_bs1024_e10/retrieval_best.pt
outputs/v3/microlens_100k/retrieval_mvp_bs1024_e10/retrieval_latest.pt
outputs/v3/microlens_100k/retrieval_mvp_bs1024_e10/train_config.json
```

当前模型规模：

| module | parameters |
|---|---:|
| item tower | 2,676,608 |
| user tower | 526,336 |
| total | 3,202,944 |

### Step 4: 全量召回评估

```bash
python scripts/v3/03_eval_full_recall.py \
  --checkpoint outputs/v3/microlens_100k/retrieval_mvp_bs1024_e10/retrieval_best.pt \
  --eval-split test
```

输出：

```text
outputs/v3/microlens_100k/retrieval_mvp_bs1024_e10/full_recall_test_metrics.json
```

评估口径：

- 使用 train split 构造固定用户历史；
- 在全量 19,738 个 item 上打分召回；
- 过滤 train 中已交互 item；
- 使用 test split 中的观测交互作为相关 item；
- 计算 Recall@K、HitRate@K、NDCG@K、MRR@K。

## 6. 核心方法

### 6.1 多模态 Item Encoder

V3 的 item 侧从 V2 的文本语义 item tower 扩展为多模态 item encoder。

Item Encoder 输入包括：

- BGE-M3 标题文本特征：提供文本语义；
- CLIP-RN50 图像特征：提供封面 / 图像视觉语义；
- VideoMAE 视频特征：提供视频内容动态语义；
- item ID embedding：补充 item 级记忆信号。

三类模态特征分别经过 projection 降维，再与 item ID embedding concat 后输入 MLP，输出 L2 normalize 后的 item embedding。

### 6.2 历史行为 User Tower

MicroLens-100K 是隐式反馈数据，不包含 KuaiRec 中的 `watch_ratio`。因此 V3 当前使用较轻量的用户建模方式：

- 取用户最近历史 item；
- 使用多模态 item encoder 编码历史 item；
- 对有效历史进行 masked mean pooling；
- 经过 MLP 得到最终 user embedding。

### 6.3 Sampled Softmax 召回训练

V3 使用 sampled softmax 训练多模态双塔召回模型。每个正样本配合多个采样负样本，在同一 batch 中优化 user embedding 与正 item embedding 的相似度，并拉开负 item。

当前训练配置：

- batch size：1024
- epochs：10
- negatives per positive：32
- output dim：512
- temperature：0.07

## 7. 评估结果

当前结果来自：

```text
outputs/v3/microlens_100k/retrieval_mvp_bs1024_e10/full_recall_test_metrics.json
```

评估用户数为 100,000，候选 item 数为 19,738。

| K | Recall@K | HitRate@K | NDCG@K | MRR@K |
|---:|---:|---:|---:|---:|
| 10 | 2.86% | 3.03% | 1.35% | 0.92% |
| 20 | 5.22% | 5.47% | 1.95% | 1.09% |
| 50 | 9.57% | 9.96% | 2.82% | 1.23% |
| 100 | 14.57% | 15.09% | 3.64% | 1.30% |

由于 MicroLens-100K 的候选池更大、每个用户 test 正反馈较少，且缺少显式负反馈，V3 当前结果主要用于验证多模态内容特征接入召回链路的可行性，而不是作为最终排序效果上限。

## 8. 当前版本定位

V3 是多模态 item 表征与语义召回验证版本，重点完成 text / image / video 内容特征对齐、融合建模、训练和全量召回评估闭环。

V3 已完成：

- MicroLens-100K 原始数据检查与样本构造；
- text / image / video 三模态特征对齐；
- BGE-M3 + CLIP-RN50 + VideoMAE 多模态 Item Encoder；
- 历史行为 User Tower；
- sampled softmax 双塔召回训练；
- 10 万用户、1.97 万候选 item 的 full-catalog 评估。

当前 V3 尚未包含：

- 更强的多模态融合结构，如 gated fusion、attention fusion；
- 基于 watch time / like / comment 等更细粒度行为强度的用户建模；
- LightGCN 协同过滤召回通道；
- 多路召回融合与 reranker。

## 9. 后续计划

- 引入 gated fusion 或 attention fusion，提升不同模态间的信息融合能力；
- 将 V2 的行为监督 user tower 与 V3 的多模态 item encoder 进行融合；
- 引入 LightGCN 协同过滤召回通道，补充交互图信号；
- 在多模态召回结果上加入轻量 reranker，结合内容、协同过滤和行为特征优化 TopK 排序。
