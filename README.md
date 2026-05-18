# Short-Video MMRec: 面向短视频搜索推荐的多模态内容理解与语义召回系统

本项目面向短视频搜索推荐场景，围绕 **内容语义表征、用户行为建模、多模态内容理解与语义召回** 构建分阶段演进的离线推荐系统。

项目从 KuaiRec 2.0 文本语义召回 baseline 出发，逐步引入用户观看行为监督、多特征双塔召回、多模态 item 表征和 full-catalog 召回评估，目标是验证短视频内容语义、用户兴趣行为和图文视频多模态特征在召回阶段的建模方式。

## 项目路线图

| Stage | Module | Dataset | Core |
|---|---|---|---|
| V1 | Text Semantic Recall Baseline | KuaiRec 2.0 | BGE 文本向量 + FAISS 语义索引 + I2I / Q2I / U2I 召回 |
| V2 | Behavior-Aware Semantic Retrieval | KuaiRec 2.0 | 多特征 Item Tower + 时序感知 User Tower + pointwise BCE 双塔训练 |
| V3 | Multimodal Item Representation | MicroLens-100K | BGE-M3 + CLIP-RN50 + VideoMAE 多模态 Item Encoder + sampled softmax |
| V4 | LightGCN Collaborative Recall | MicroLens-100K | 用户-短视频交互图 + LightGCN 协同过滤召回 + BPR 训练 |
| V5 | MLLM Semantic Profile Retrieval | MicroLens-100K | 5 frames + Qwen2.5-VL 语义 Profile + LoRA 推荐对齐 + Profile Recall / Fusion |

详细文档：

- [SV-Recall V1: 基于 KuaiRec 的短视频内容语义召回与用户兴趣推荐最小闭环](docs/V1_minimal_semantic_recall.md)
- [SV-Recall V2: 基于强文本表征与行为监督的短视频语义召回模型](docs/V2_behavior_aware_semantic_adapter.md)
- [SV-Recall V3: 基于多模态内容理解的短视频 Item 表征与语义召回](docs/V3_multimodal_item_recall.md)
- [SV-Recall V4: 基于 LightGCN 的短视频协同过滤召回通道](docs/V4_lightgcn_collaborative_recall.md)
- [SV-Recall V5: 基于 Qwen2.5-VL 语义 Profile 与推荐对齐 LoRA 的短视频召回增强](docs/V5_lora_vlm_instruction_tuning.md)

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

V3 当前定位是多模态内容特征接入召回链路的完整训练与评估版本。由于 MicroLens-100K 候选池更大、每个用户 test 正反馈较少，且缺少显式负反馈，当前指标主要用于衡量多模态 item 表征和召回训练闭环的离线效果。

## V4: LightGCN 协同过滤召回通道

V4 复用 V3 的 MicroLens-100K 行为样本，在 train split 上构造用户-短视频二部图，训练 LightGCN 协同过滤召回模型。当前最佳版本为 50 epochs 训练结果，best checkpoint 出现在 epoch 43。

核心设计：

- 纯 ID 图协同过滤：user embedding + item embedding + LightGCN 图传播；
- BPR 训练目标：对每个正反馈采样未交互负 item，优化正负排序间隔；
- full-catalog ranking：在全量候选 item 上评估协同过滤召回效果；
- 与 V3 互补：V3 提供内容语义召回，V4 提供交互图召回。

当前 V4 在 100,000 名测试用户、19,738 个候选 item 上完成 full-catalog 评估：

| K | Recall@K | HitRate@K | NDCG@K | MRR@K |
|---:|---:|---:|---:|---:|
| 10 | 2.08% | 2.14% | 0.98% | 0.66% |
| 20 | 3.80% | 3.91% | 1.42% | 0.79% |
| 50 | 7.27% | 7.53% | 2.11% | 0.90% |
| 100 | 11.69% | 12.04% | 2.84% | 0.96% |

V4 是协同过滤召回通道的完整离线版本，后续可继续扩展为 V3 + V4 多路召回融合。

## V5: 多模态大模型语义 Profile 与推荐对齐 LoRA

V5 面向短视频内容理解与语义召回增强，基于 MicroLens-100K 的 5 frames 多图输入和 item 标题，使用 Qwen2.5-VL 生成结构化语义 Profile，并通过 LLaMA-Factory LoRA 指令微调将 Profile 从解释性描述进一步对齐到推荐召回语义。最终将 LoRA Profile 文本编码为 BGE-M3 embedding，构建独立的 Profile 召回通道，并与 title、官方多模态特征和 fusion 表征进行 full-catalog 对比评估。

核心设计：

- MLLM semantic profile：基于 5 frames + title 生成 summary、topic、visual object、scene、style、emotion 等结构化字段；
- retrieval-aware teacher：保留标题强信号，强化 `search_queries / main_topics / interest_tags` 等推荐召回字段；
- Qwen2.5-VL LoRA SFT：使用 LLaMA-Factory 对 VLM 进行轻量指令微调，稳定生成推荐对齐 Profile；
- compact profile_text：使用 `title + title_keywords + search_queries + interest_tags + main_topics` 构造召回文本；
- profile recall / fusion：将 Profile embedding 接入 FAISS full-catalog 召回，并与 title / multimodal / fusion 统一评估。

V5 在 100,000 名测试用户、19,738 个候选 item 上完成 full-catalog 评估：

| Method | Recall@100 | 说明 |
|---|---:|---|
| V5 title baseline | 4.88% | 仅使用 item 标题 |
| V5 zero-shot profile | 3.27% | 原始 MLLM Profile 召回 |
| V5-LoRA profile | 5.28% | 推荐对齐 LoRA Profile |
| V5 fusion | 5.51% | 原 V5 融合表征 |
| V5-LoRA fusion | 6.01% | 加入 LoRA Profile 后的融合表征 |
| V5 multimodal | 7.62% | 官方 text / image / video 多模态表征 |

其中，LoRA Profile 相比 zero-shot Profile 的 Recall@100 相对提升 61.8%，并超过 title baseline；LoRA fusion 相比 V5 fusion 相对提升 9.1%。这说明 V5 将多模态大模型生成的内容 Profile 从解释性语义描述，进一步转化为可用于召回增强的推荐语义特征。

V5 当前新增脚本：

```text
scripts/v5/
├── download_microlens_100k_frames.sh
├── 00_check_frames.py
├── 01_build_profile_inputs.py
├── 02_generate_profiles_qwen_vl.py
├── 03_clean_profiles.py
├── 04_encode_profile_embeddings.py
├── 05_build_profile_faiss.py
├── 06_eval_profile_recall.py
└── 07_case_study.py

scripts/v5_lora/
├── 01_run_profile_text_ablation.py
├── 02_build_retrieval_aware_teacher.py
├── 03_build_llamafactory_sft_dataset.py
├── 04_train_lora.sh
├── 05_generate_lora_profiles.py
├── 06_eval_lora_profiles.py
└── 07_case_study_lora.py
```

## Project Structure

```text
short-video-mmrec/
├── configs/
│   ├── v2/
│   ├── v3/
│   ├── v4/
│   ├── v5/
│   └── v5_lora/
├── data/
│   ├── raw/
│   └── processed/
│       ├── v1/
│       ├── v2/
│       ├── v3/
│       ├── v5/
│       └── v5_lora/
├── docs/
│   ├── V1_minimal_semantic_recall.md
│   ├── V2_behavior_aware_semantic_adapter.md
│   ├── V3_multimodal_item_recall.md
│   ├── V4_lightgcn_collaborative_recall.md
│   └── V5_lora_vlm_instruction_tuning.md
├── outputs/
│   ├── v1/
│   ├── v2/
│   ├── v3/
│   ├── v4/
│   ├── v5/
│   └── v5_lora/
├── scripts/
│   ├── v1/
│   ├── v2/
│   ├── v3/
│   ├── v4/
│   ├── v5/
│   └── v5_lora/
└── src/
    ├── v2/
    ├── v3/
    ├── v4/
    ├── v5/
    └── v5_lora/
```

## Environment

V1 / V2 / V3 / V4 共用 Python + PyTorch 生态，主要依赖：

- PyTorch
- pandas / numpy
- sentence-transformers
- FAISS
- scikit-learn
- tqdm
- pyyaml

V5 的 Qwen2.5-VL Profile 生成阶段额外需要：

- transformers
- qwen-vl-utils
- accelerate

V5-LoRA 指令微调阶段使用 LLaMA-Factory，额外依赖：

- peft
- trl
- datasets
- LLaMA-Factory

V1 依赖可参考：

```bash
pip install -r requirements-v1.txt
```

V2 依赖可参考：

```bash
pip install -r requirements-v2.txt
```

V5 依赖可参考：

```bash
pip install -r requirements-v5.txt
pip install -r requirements-v5_lora.txt
```

## 项目定位

本项目重点不在于复现完整工业推荐系统中的召回、粗排、精排、重排和在线 A/B 流程，而是聚焦于短视频推荐场景中的 **多模态内容理解与语义召回**。

当前已完成：

- 文本语义召回 baseline；
- 行为监督双塔召回模型；
- 多模态 item 表征与召回训练；
- LightGCN 协同过滤召回通道；
- 多模态大模型语义 Profile 与推荐对齐 LoRA 召回增强工程链路；
- full-catalog 离线召回评估。

后续可继续扩展：

- GRU / Transformer 用户序列兴趣建模；
- V3 多模态召回 + V4 协同过滤召回的多路融合与轻量 reranker；
- API 服务化与可视化 Demo。
