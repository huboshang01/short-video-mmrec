# SV-Recall V5: 基于 Qwen2.5-VL 语义 Profile 与推荐对齐 LoRA 的短视频召回增强

V5 是“面向短视频搜索推荐的多模态内容理解与语义召回系统”的多模态大模型语义 Profile 版本。本文中的 V5 默认包含基础 Profile 生成链路和 `v5_lora` 推荐对齐模块：先使用 Qwen2.5-VL 基于 5 frames + title 生成结构化短视频 Profile，再通过 retrieval-aware teacher profile 与 LLaMA-Factory LoRA 指令微调，使 Profile 更适合推荐召回场景。

V5 的核心目标是验证多模态大模型生成的内容 Profile 能否从解释性语义描述，进一步转化为可用于 full-catalog 召回增强的语义特征。

## 1. 项目目标

- 基于 MicroLens-100K 的短视频标题与 5 帧视觉输入生成结构化内容 Profile；
- 保留 summary、topic、visual object、scene、style、emotion 等可解释字段；
- 构造推荐对齐的 `title_keywords / search_queries / interest_tags / main_topics` 字段；
- 使用 LLaMA-Factory 对 Qwen2.5-VL 进行 LoRA 指令微调；
- 将 LoRA Profile 文本编码为 BGE-M3 embedding，并构建 FAISS 召回索引；
- 在 100,000 名测试用户和 19,738 个候选 item 上进行 full-catalog Recall@K 评估；
- 与 title、原始 Profile、官方多模态特征和 fusion 表征进行统一对比。

## 2. 数据集

V5 使用 MicroLens-100K：

```text
data/processed/v3/microlens_100k/item_meta.csv
data/processed/v3/microlens_100k/item_ids.csv
data/processed/v3/microlens_100k/behavior_samples_train.csv
data/processed/v3/microlens_100k/behavior_samples_val.csv
data/processed/v3/microlens_100k/behavior_samples_test.csv
data/raw/microlens_100k/frames/MicroLens-100k_frames_interval_1_number_5/
```

数据规模：

| split | samples |
|---|---:|
| train | 507,529 |
| val | 105,938 |
| test | 105,938 |

Profile 与 LoRA SFT 数据规模：

| data | count |
|---|---:|
| item profiles | 19,738 |
| SFT train | 17,764 |
| SFT val | 986 |
| SFT test | 988 |

评估口径：

- candidate items: 19,738；
- evaluated users: 100,000；
- train split 用于构造用户历史；
- test split 用于 full-catalog 召回评估。

## 3. 技术方案

### 3.1 MLLM Semantic Profile

基础 V5 使用 Qwen2.5-VL 对每个短视频 item 生成结构化 Profile。输入包括：

```text
5 frames + title
```

Profile schema：

```json
{
  "summary": "",
  "main_topics": [],
  "visual_objects": [],
  "scene": "",
  "content_type": "",
  "style": "",
  "emotion": "",
  "target_audience": "",
  "search_queries": []
}
```

该阶段提供可解释的短视频内容语义表示，并将 Profile 接入 embedding、FAISS index 和 full-catalog recall 评估链路。

### 3.2 Retrieval-Aware Teacher Profile

`v5_lora` 在基础 Profile 上构造推荐对齐 teacher profile，显式保留标题强信号，并新增召回友好字段：

```json
{
  "title_keywords": [],
  "interest_tags": [],
  "summary": "",
  "main_topics": [],
  "visual_objects": [],
  "scene": "",
  "content_type": "",
  "style": "",
  "emotion": "",
  "target_audience": "",
  "search_queries": []
}
```

用于召回 embedding 的 compact profile_text：

```text
标题：{title}
标题关键词：{title_keywords}
推荐检索词：{search_queries}
兴趣标签：{interest_tags}
主题：{main_topics}
```

该模板将 Profile 的文本化方式从完整描述转向推荐检索语义，减少低权重解释字段对召回 embedding 的干扰，同时保留完整 JSON Profile 供内容理解与案例分析使用。

### 3.3 Qwen2.5-VL LoRA SFT

LoRA SFT 使用 LLaMA-Factory 完成。训练样本输入为 5 frames + title，输出为 retrieval-aware teacher profile JSON。

当前训练配置：

| config | value |
|---|---|
| base model | Qwen2.5-VL-7B-Instruct |
| framework | LLaMA-Factory |
| template | qwen2_vl |
| finetuning type | LoRA |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA target | q_proj, v_proj |
| epochs | 1 |
| precision | bf16 |

训练产物：

```text
outputs/v5_lora/microlens_100k/checkpoints/qwen2_5_vl_lora_sft/
```

其中最外层 `adapter_model.safetensors` 为最终 LoRA adapter。

### 3.4 Profile Recall 与 Fusion

06 评估阶段执行以下流程：

```text
LoRA raw profile
-> profile cleaning
-> compact profile_text
-> BGE-M3 embedding
-> FAISS index
-> full-catalog user-to-item recall
```

评估方法包括：

- `title`: 仅使用 item title embedding；
- `profile`: 使用 LoRA Profile compact embedding；
- `multimodal`: 使用 MicroLens 官方 text / image / video 多模态特征；
- `fusion`: 融合 title、profile、image、video 表征。

## 4. 项目结构

```text
short-video-mmrec/
├── configs/
│   ├── v5/
│   │   ├── profile_generation.yaml
│   │   ├── profile_embedding.yaml
│   │   └── profile_retrieval.yaml
│   └── v5_lora/
│       ├── profile_text_ablation.yaml
│       ├── teacher_profile.yaml
│       ├── sft_dataset.yaml
│       ├── qwen2_5_vl_lora_sft.yaml
│       ├── profile_generation_lora.yaml
│       └── profile_retrieval_lora.yaml
├── scripts/
│   ├── v5/
│   │   ├── 00_check_frames.py
│   │   ├── 01_build_profile_inputs.py
│   │   ├── 02_generate_profiles_qwen_vl.py
│   │   ├── 03_clean_profiles.py
│   │   ├── 04_encode_profile_embeddings.py
│   │   ├── 05_build_profile_faiss.py
│   │   ├── 06_eval_profile_recall.py
│   │   └── 07_case_study.py
│   └── v5_lora/
│       ├── 01_run_profile_text_ablation.py
│       ├── 02_build_retrieval_aware_teacher.py
│       ├── 03_build_llamafactory_sft_dataset.py
│       ├── 04_train_lora.sh
│       ├── 05_generate_lora_profiles.py
│       ├── 06_eval_lora_profiles.py
│       └── 07_case_study_lora.py
├── src/
│   ├── v5/
│   └── v5_lora/
├── data/
│   └── processed/
│       ├── v5/
│       └── v5_lora/
└── outputs/
    ├── v5/
    └── v5_lora/
```

## 5. V5 运行流程

### Step 1: 构建基础 Profile 输入

```bash
python scripts/v5/00_check_frames.py
python scripts/v5/01_build_profile_inputs.py
```

输出：

```text
data/processed/v5/microlens_100k/profile_inputs.jsonl
```

### Step 2: 生成并评估基础 V5 Profile

```bash
python scripts/v5/02_generate_profiles_qwen_vl.py \
  --backend qwen2_5_vl \
  --resume
python scripts/v5/03_clean_profiles.py
python scripts/v5/04_encode_profile_embeddings.py
python scripts/v5/05_build_profile_faiss.py
python scripts/v5/06_eval_profile_recall.py
```

核心输出：

```text
data/processed/v5/microlens_100k/profiles_raw.jsonl
data/processed/v5/microlens_100k/profiles_clean.jsonl
outputs/v5/microlens_100k/reports/profile_recall_test_metrics.json
```

### Step 3: 构造推荐对齐 SFT 数据

```bash
python scripts/v5_lora/01_run_profile_text_ablation.py
python scripts/v5_lora/02_build_retrieval_aware_teacher.py
python scripts/v5_lora/03_build_llamafactory_sft_dataset.py
```

输出：

```text
data/processed/v5_lora/microlens_100k/retrieval_aware_teacher_profiles.jsonl
data/processed/v5_lora/microlens_100k/sft_train.json
data/processed/v5_lora/microlens_100k/sft_val.json
data/processed/v5_lora/microlens_100k/sft_test.json
data/processed/v5_lora/microlens_100k/dataset_info.json
```

### Step 4: LoRA 指令微调

```bash
bash scripts/v5_lora/04_train_lora.sh \
  configs/v5_lora/qwen2_5_vl_lora_sft.yaml
```

单机多卡示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 NPROC_PER_NODE=4 \
bash scripts/v5_lora/04_train_lora.sh \
  configs/v5_lora/qwen2_5_vl_lora_sft.yaml
```

输出：

```text
outputs/v5_lora/microlens_100k/checkpoints/qwen2_5_vl_lora_sft/
```

### Step 5: 生成 LoRA Profile

```bash
python scripts/v5_lora/05_generate_lora_profiles.py \
  --config configs/v5_lora/profile_generation_lora.yaml \
  --backend qwen2_5_vl_lora \
  --resume
```

输出：

```text
data/processed/v5_lora/microlens_100k/profiles_lora_raw.jsonl
```

### Step 6: 评估 LoRA Profile

```bash
python scripts/v5_lora/06_eval_lora_profiles.py \
  --config configs/v5_lora/profile_retrieval_lora.yaml
```

输出：

```text
data/processed/v5_lora/microlens_100k/profiles_lora_clean.jsonl
data/processed/v5_lora/microlens_100k/profile_text_lora.csv
data/processed/v5_lora/microlens_100k/profile_quality_report.json
outputs/v5_lora/microlens_100k/reports/profile_recall_test_metrics.json
```

案例分析：

```bash
python scripts/v5_lora/07_case_study_lora.py \
  --config configs/v5_lora/profile_retrieval_lora.yaml \
  --item-ids 1,25,100,1000 \
  --topk 10
```

输出：

```text
outputs/v5_lora/microlens_100k/reports/profile_case_study.json
```

## 6. 评估结果

最终结果来自：

```text
outputs/v5_lora/microlens_100k/reports/profile_recall_test_metrics.json
```

评估用户数为 100,000，候选 item 数为 19,738。

### 6.1 V5-LoRA Full-Catalog Recall

| Method | Recall@10 | Recall@20 | Recall@50 | Recall@100 |
|---|---:|---:|---:|---:|
| title | 1.53% | 2.42% | 3.63% | 4.88% |
| LoRA profile | 1.57% | 2.47% | 3.78% | 5.28% |
| multimodal | 2.34% | 3.52% | 5.46% | 7.62% |
| fusion | 1.77% | 2.83% | 4.33% | 6.01% |

### 6.2 与基础 V5 对比

| Method | Recall@100 |
|---|---:|
| V5 title baseline | 4.88% |
| V5 zero-shot profile | 3.27% |
| V5-LoRA profile | 5.28% |
| V5 fusion | 5.51% |
| V5-LoRA fusion | 6.01% |
| V5 multimodal | 7.62% |

关键结果：

- V5-LoRA Profile 的 Recall@100 从 3.27% 提升到 5.28%，相对提升 61.8%；
- V5-LoRA Profile 超过 title baseline，说明推荐对齐后的 Profile 具备独立召回价值；
- V5-LoRA fusion 的 Recall@100 从 5.51% 提升到 6.01%，相对提升 9.1%；
- 官方 multimodal 表征仍提供最强单路内容召回，LoRA Profile 作为可解释语义通道提供补充增益。

Profile 质量统计：

| metric | value |
|---|---:|
| total profiles | 19,738 |
| valid profiles | 19,724 |
| valid rate | 99.93% |

## 7. 当前版本定位

V5 已完成：

- MicroLens-100K 5 frames 输入检查与 Profile 输入构造；
- Qwen2.5-VL 结构化短视频 Profile 生成；
- Profile 清洗、文本化、embedding 与 FAISS index；
- retrieval-aware teacher profile 构造；
- LLaMA-Factory VLM LoRA 指令微调；
- LoRA Profile 批量生成与清洗；
- title / profile / multimodal / fusion full-catalog 召回评估；
- item-to-item 与 query-to-item case study。

V5 的主要结论是：多模态大模型 Profile 不仅可以提供可解释内容理解结果，在经过推荐对齐 LoRA 后，也可以作为有效的短视频召回语义特征接入离线推荐系统。

## 8. 后续计划

- 对 LoRA Profile 与官方 multimodal 表征进行更细粒度的 score-level fusion；
- 引入行为邻居或共观看正样本构造更强 teacher profile；
- 探索 field-level 权重、query expansion 和轻量 reranker；
- 将 Profile case study 扩展为可视化检索 Demo。
