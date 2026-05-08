# Short-Video MMRec: 面向短视频搜索推荐的多模态内容理解与语义召回系统

本项目面向短视频搜索推荐场景，围绕 **内容语义表征、用户行为建模、多模态内容理解与语义召回** 构建分阶段演进的离线推荐系统。

项目从 KuaiRec 2.0 文本语义召回 baseline 出发，逐步引入用户观看行为监督、多特征双塔召回、多模态 item 表征和 full-catalog 召回评估，目标是验证短视频内容语义、用户兴趣行为和图文视频多模态特征在召回阶段的建模方式。

## 项目路线图

| Stage | Module | Dataset | Core |
|---|---|---|---|
| V1 | Text Semantic Recall Baseline | KuaiRec 2.0 | BGE 文本向量 + FAISS 语义索引 + I2I / Q2I / U2I 召回 |
| V2 | Behavior-Aware Semantic Retrieval | KuaiRec 2.0 | 多特征 Item Tower + 时序感知 User Tower + pointwise BCE 双塔训练 |
| V3 | Multimodal Item Representation | MicroLens-100K | BGE-M3 + CLIP-RN50 + VideoMAE 多模态 Item Encoder + sampled softmax |

详细文档：

- [SV-Recall V1: 基于 KuaiRec 的短视频内容语义召回与用户兴趣推荐最小闭环](docs/V1_minimal_semantic_recall.md)
- [SV-Recall V2: 基于强文本表征与行为监督的短视频语义召回模型](docs/V2_behavior_aware_semantic_adapter.md)
- [SV-Recall V3: 基于多模态内容理解的短视频 Item 表征与语义召回](docs/V3_multimodal_item_recall.md)

## V1: 文本语义召回最小闭环

V1 基于 KuaiRec 2.0 构造短视频文本语义字段，融合封面文字、caption、topic tag 和多级类目信息，使用 `BAAI/bge-small-zh-v1.5` 编码 3327 个短视频 item，并通过 FAISS 构建 cosine 语义索引。

已完成能力：

- item-to-item 相似视频召回；
- query-to-item 自然语言文本检索；
- user-to-item 基于高 `watch_ratio` 历史视频的用户兴趣召回；
- semantic / popularity / random baseline 对比评估。

V1 证明了文本语义召回链路可以跑通，但纯文本 semantic 用户推荐在 KuaiRec 行为评估中弱于 popularity baseline，因此引出 V2 的行为监督建模。

## V2: 行为监督语义召回模型

V2 在 V1 静态文本语义基础上，引入 KuaiRec 2.0 的用户观看行为、item ID 和类目特征，训练 behavior-aware 双塔召回模型。

核心设计：

- 多特征融合 Item Tower：BGE 文本语义 + item ID 协同过滤记忆 + category 类目先验；
- 时序感知 User Tower：最近行为 + `watch_ratio` 行为强度 + 时间衰减；
- pointwise BCE：利用高 `watch_ratio` 正反馈和低 `watch_ratio` 显式负反馈进行 user-item 对齐；
- full-catalog ranking：在全量候选 item 上评估召回效果。

数据规模：

| split | samples |
|---|---:|
| train | 3,741,264 |
| val | 467,653 |
| test | 467,653 |

当前最佳 V2 评估结果（`retrieval_bce_full_e12_bs4096_temp01`），基于 1411 名验证用户、3327 个候选短视频 item：

| K | Recall@K | HitRate@K | NDCG@K | MRR@K |
|---:|---:|---:|---:|---:|
| 10 | 3.51% | 93.20% | 30.98% | 48.26% |
| 20 | 5.75% | 97.17% | 27.15% | 48.54% |
| 50 | 11.37% | 99.72% | 23.11% | 48.63% |
| 100 | 19.16% | 100.00% | 22.64% | 48.64% |

## V3: 多模态 Item 表征与语义召回

V3 基于 MicroLens-100K，引入官方预提取的 text / image / video 多模态特征，构建短视频多模态 item encoder，并接入双塔召回训练与 full-catalog 评估。

多模态特征：

| modality | backbone | dim | shape |
|---|---|---:|---|
| text | BGE-M3 | 1024 | 19738 x 1024 |
| image | CLIP-RN50 | 1024 | 19738 x 1024 |
| video | VideoMAE | 768 | 19738 x 768 |

数据规模：

| split | samples |
|---|---:|
| train | 507,529 |
| val | 105,938 |
| test | 105,938 |

当前 V3 模型参数量约 320 万，在 100,000 名测试用户、19,738 个候选 item 上完成 full-catalog 评估：

| K | Recall@K | HitRate@K | NDCG@K | MRR@K |
|---:|---:|---:|---:|---:|
| 10 | 2.86% | 3.03% | 1.35% | 0.92% |
| 20 | 5.22% | 5.47% | 1.95% | 1.09% |
| 50 | 9.57% | 9.96% | 2.82% | 1.23% |
| 100 | 14.57% | 15.09% | 3.64% | 1.30% |

V3 当前定位是多模态内容特征接入召回链路的 MVP 验证版本。由于 MicroLens-100K 候选池更大、每个用户 test 正反馈较少，且缺少显式负反馈，当前指标主要用于验证多模态 item 表征和召回训练闭环。

## Project Structure

```text
short-video-mmrec/
├── configs/
│   ├── v2/
│   └── v3/
├── data/
│   ├── raw/
│   └── processed/
│       ├── v1/
│       ├── v2/
│       └── v3/
├── docs/
│   ├── V1_minimal_semantic_recall.md
│   ├── V2_behavior_aware_semantic_adapter.md
│   └── V3_multimodal_item_recall.md
├── outputs/
│   ├── v1/
│   ├── v2/
│   └── v3/
├── scripts/
│   ├── v1/
│   ├── v2/
│   └── v3/
└── src/
    ├── v2/
    └── v3/
```

## Environment

V1 / V2 / V3 共用 Python + PyTorch 生态，主要依赖：

- PyTorch
- pandas / numpy
- sentence-transformers
- FAISS
- scikit-learn
- tqdm
- pyyaml

V1 依赖可参考：

```bash
pip install -r requirements-v1.txt
```

V2 依赖可参考：

```bash
pip install -r requirements-v2.txt
```

## 项目定位

本项目重点不在于复现完整工业推荐系统中的召回、粗排、精排、重排和在线 A/B 流程，而是聚焦于短视频推荐场景中的 **多模态内容理解与语义召回**。

当前已完成：

- 文本语义召回 baseline；
- 行为监督双塔召回模型；
- 多模态 item 表征与召回训练；
- full-catalog 离线召回评估。

后续可继续扩展：

- LightGCN 协同过滤召回通道；
- GRU / Transformer 用户序列兴趣建模；
- 多路召回融合与轻量 reranker；
- API 服务化与可视化 Demo。
