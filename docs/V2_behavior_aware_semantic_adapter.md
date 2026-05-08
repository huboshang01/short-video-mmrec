# SV-Recall V2: 基于强文本表征与行为监督的短视频语义召回模型

V2 是“面向短视频搜索推荐的多模态内容理解与语义召回系统”的行为监督语义召回版本，在 V1 文本语义召回 baseline 基础上，引入 KuaiRec 2.0 的用户观看行为、item ID 与类目特征，训练面向短视频推荐任务的双塔召回模型。

V2 的核心目标是把 V1 中冻结的静态文本 embedding，升级为经过用户行为监督适配的 behavior-aware semantic representation，使召回结果不仅具备内容语义相关性，也能反映用户观看偏好。

## 1. 项目目标

本项目面向短视频推荐召回场景，完成以下能力：

- 基于 KuaiRec 2.0 构造带时序信息的用户行为样本；
- 使用 `BAAI/bge-small-zh-v1.5` 构建 item 文本 embedding cache；
- 构建多特征融合 Item Tower：文本语义 + item ID 协同过滤记忆 + 类目先验；
- 构建时序感知 User Tower：最近行为 + 行为强度 + 时间衰减；
- 使用 pointwise BCE 训练 user-item 双塔召回模型；
- 使用 full-catalog ranking 进行 Recall@K、HitRate@K、NDCG@K、MRR@K 离线评估。

## 2. 技术栈

- Python 3.10
- PyTorch
- pandas / numpy
- sentence-transformers
- BGE text embedding
- CUDA
- KuaiRec 2.0
- Two-Tower Retrieval
- Pointwise BCE

## 3. 数据集

本项目使用 KuaiRec 2.0 数据集中的：

- `small_matrix.csv`：用户-视频交互数据，包含 `watch_ratio` 等行为信号；
- V1 生成的 `video_text.csv`：短视频文本语义字段；
- `item_categories.csv`：短视频类目信息。

V2 处理后的数据位于：

```text
data/processed/v2/
```

当前样本规模：

| split | samples |
|---|---:|
| train | 3,741,264 |
| val | 467,653 |
| test | 467,653 |

总计 4,676,570 条行为样本，覆盖 1411 名用户和 3327 个短视频 item。

行为标签规则：

```text
watch_ratio >= 1.0  -> label = 1，正反馈
watch_ratio < 0.7   -> label = 0，显式负反馈
其他区间             -> label = -1，中性样本
```

## 4. 项目结构

```text
short-video-mmrec/
├── scripts/
│   └── v2/
│       ├── 01_prepare_behavior_samples.py
│       ├── 02_encode_item_text_bge.py
│       ├── 03_train_retrieval_bce.py
│       └── 04_eval_full_recall.py
├── src/
│   └── v2/
│       ├── data/
│       │   └── time_aware_retrieval_dataset.py
│       ├── models/
│       │   ├── retrieval_item_tower.py
│       │   └── retrieval_user_tower.py
│       ├── losses/
│       │   └── retrieval_loss.py
│       ├── eval/
│       │   └── retrieval_metrics.py
│       └── train/
│           └── train_retrieval.py
├── data/
│   └── processed/
│       └── v2/
└── outputs/
    └── v2/
        ├── embeddings/
        └── retrieval_bce_full_e12_bs4096_temp01/
```

## 5. V2 运行流程

### Step 1: 构造行为样本

```bash
python scripts/v2/01_prepare_behavior_samples.py
```

输出：

```text
data/processed/v2/behavior_samples_train.csv
data/processed/v2/behavior_samples_val.csv
data/processed/v2/behavior_samples_test.csv
data/processed/v2/item_text_v2.csv
data/processed/v2/behavior_samples_summary.json
```

### Step 2: 构建 item 文本向量缓存

```bash
python scripts/v2/02_encode_item_text_bge.py \
  --model-name BAAI/bge-small-zh-v1.5 \
  --batch-size 64 \
  --device auto \
  --normalize
```

输出：

```text
outputs/v2/embeddings/item_text_embeddings.npy
outputs/v2/embeddings/item_ids.npy
outputs/v2/embeddings/item_text_embedding_meta.csv
outputs/v2/embeddings/embedding_config.json
```

当前 embedding 配置：

- 文本模型：`BAAI/bge-small-zh-v1.5`
- item 数量：3327
- embedding 维度：512
- max sequence length：256
- 归一化：开启

### Step 3: 训练双塔召回模型

```bash
python scripts/v2/03_train_retrieval_bce.py \
  --embedding-config outputs/v2/embeddings/embedding_config.json \
  --train-samples data/processed/v2/behavior_samples_train.csv \
  --val-samples data/processed/v2/behavior_samples_val.csv \
  --output-dir outputs/v2/retrieval_bce_full_e12_bs4096_temp01 \
  --batch-size 4096 \
  --epochs 12 \
  --temperature 0.1 \
  --max-train-samples -1 \
  --max-val-samples -1 \
  --num-workers 4 \
  --log-every 50
```

输出：

```text
outputs/v2/retrieval_bce_full_e12_bs4096_temp01/retrieval_bce_best.pt
outputs/v2/retrieval_bce_full_e12_bs4096_temp01/retrieval_bce_latest.pt
outputs/v2/retrieval_bce_full_e12_bs4096_temp01/retrieval_bce_train_config.json
```

### Step 4: 全量召回评估

```bash
python scripts/v2/04_eval_full_recall.py \
  --checkpoint outputs/v2/retrieval_bce_full_e12_bs4096_temp01/retrieval_bce_best.pt \
  --train-samples data/processed/v2/behavior_samples_train.csv \
  --eval-samples data/processed/v2/behavior_samples_val.csv \
  --ks 10,20,50,100 \
  --output outputs/v2/retrieval_bce_full_e12_bs4096_temp01/full_recall_val_metrics.json
```

评估口径：

- 使用 train split 构造固定用户历史；
- 用户历史只使用 `watch_ratio >= 1.0` 的最近正反馈行为；
- 在全量 3327 个 item 上打分召回；
- 过滤 train 中已正反馈 item；
- 使用 val split 中的正反馈 item 计算 Recall@K、HitRate@K、NDCG@K、MRR@K。

## 6. 核心方法

### 6.1 多特征融合 Item Tower

V2 的 item 侧从“纯文本语义 adapter”升级为“文本语义 + 协同过滤记忆 + 类目先验”的融合 Item Tower。

Item Tower 输入包括：

- BGE item text embedding：提供内容语义和冷启动泛化能力；
- item ID embedding：学习高密度交互数据中的协同过滤记忆；
- category embedding：提供粗粒度类目先验。

三类特征经过 concat 后进入 MLP，输出 L2 normalize 后的 item embedding，可直接用于 dot / cosine 召回。

### 6.2 时序感知 User Tower

V2 的 user 侧采用工业推荐里常见的“时序感知 + 最近行为 + 行为强度 + 时间衰减”用户表示。

User Tower 使用用户最近历史 item 构建兴趣向量：

- 最近行为：每个用户最多使用最近 50 条历史正反馈；
- 行为强度：使用 `watch_ratio` 表示兴趣强度，并对极端值截断；
- 时间衰减：越新的行为权重越高；
- 共享 item tower：历史 item 和 target item 使用同一套 item 表征空间。

### 6.3 Pointwise BCE 召回训练

V2 当前使用 pointwise BCE 训练：

```text
score(user, item) = user_embedding · item_embedding
```

相比旧版 in-batch InfoNCE，pointwise BCE 可以直接利用 KuaiRec 中由低 `watch_ratio` 构造的显式负反馈，也允许同一个用户拥有多个正反馈 item，更符合短视频推荐召回场景。

## 7. 评估结果

当前最佳结果来自：

```text
outputs/v2/retrieval_bce_full_e12_bs4096_temp01/full_recall_val_metrics.json
```

评估用户数为 1411，候选 item 数为 3327。

| K | Recall@K | HitRate@K | NDCG@K | MRR@K |
|---:|---:|---:|---:|---:|
| 10 | 3.51% | 93.20% | 30.98% | 48.26% |
| 20 | 5.75% | 97.17% | 27.15% | 48.54% |
| 50 | 11.37% | 99.72% | 23.11% | 48.63% |
| 100 | 19.16% | 100.00% | 22.64% | 48.64% |

与 V1 纯文本 semantic 推荐相比，V2 引入行为监督后，召回结果更能反映用户观看偏好。相比同口径热门 / 随机召回基线，V2 的 Recall@10 约提升 6.69x / 8.57x，Recall@100 约提升 4.63x / 4.66x。

## 8. 当前版本定位

V2 是行为监督语义召回版本，主要完成推荐系统召回阶段的 user-item 表征学习和全量召回评估。

V2 已完成：

- KuaiRec 行为样本构造与时序切分；
- BGE item embedding cache；
- 文本语义 + item ID + 类目的多特征 Item Tower；
- 最近行为 + watch_ratio + 时间衰减的 User Tower；
- pointwise BCE 双塔训练；
- full-catalog ranking 离线评估。

当前 V2 尚未包含：

- LightGCN 等图协同过滤召回通道；
- Transformer / GRU 序列兴趣编码器；
- 多路召回融合与重排；
- 线上服务化和实时更新。

## 9. 后续计划

- 引入 LightGCN 协同过滤召回通道，补充高阶 user-item 交互图信号；
- 引入 hard negative sampling、BPR 或 sampled softmax 优化排序学习目标；
- 引入 GRU / Transformer 用户序列编码器，增强短期兴趣建模；
- 与 V3 多模态 item 表征融合，扩展图文视频多模态语义召回能力。
