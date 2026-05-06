# SV-Recall V1: 基于 KuaiRec 的短视频内容语义召回与用户兴趣推荐最小闭环

本项目是“面向短视频搜索推荐的多模态内容理解与语义召回系统”的 V1 版本，基于 KuaiRec 2.0 数据集完成短视频内容语义召回的最小闭环实现。

V1 重点不在于构建完整工业推荐系统，而是完成从视频文本字段构造、文本向量编码、FAISS 语义索引、相似视频召回、文本 Query 检索、用户兴趣推荐到基础指标评估的完整流程。

## 1. 项目目标

本项目面向短视频搜索推荐场景，完成以下能力：

- 基于封面文字、caption、topic_tag、category 等字段构造视频语义文本；
- 使用 `BAAI/bge-small-zh-v1.5` 编码短视频内容语义；
- 使用 FAISS 构建视频语义向量索引；
- 实现 item-to-item 相似视频召回；
- 实现 text-to-video 文本 Query 检索；
- 实现基于用户高 watch_ratio 历史视频的用户兴趣推荐；
- 使用 Recall@K、HitRate@K、NDCG@K 进行基础离线评估。

## 2. 技术栈

- Python 3.10
- PyTorch
- pandas / numpy / scikit-learn
- sentence-transformers
- FAISS
- KuaiRec 2.0
- WSL2 Ubuntu

## 3. 数据集

本项目使用 KuaiRec 2.0 数据集中的：

- `small_matrix.csv`：用户-视频交互数据，包含 `watch_ratio` 等行为信号；
- `kuairec_caption_category.csv`：视频 caption、topic_tag 和多级类目信息；
- `item_categories.csv`：视频标签信息，可用于后续扩展。

当前 V1 已生成 3327 个短视频 item 的文本语义数据：

```text
data/processed/v1/kuairec/video_text.csv
```

由于数据集和中间向量文件较大，本仓库不直接上传原始数据和生成的 embedding/index 文件。请用户自行下载 KuaiRec 2.0 数据集，并放置到：

```text
data/raw/
```

本项目默认会在 `data/raw/` 下自动查找所需 CSV 文件。

## 4. 项目结构

```text
short-video-mmrec/
├── README.md
├── requirements-v1.txt
├── .gitignore
├── configs/
├── scripts/
│   ├── v1/
│   │   ├── 00_check_env.py
│   │   ├── 01_check_data.py
│   │   ├── 02_build_video_text.py
│   │   ├── 03_encode_video_text.py
│   │   ├── 04_build_faiss_index.py
│   │   ├── 05_similar_video.py
│   │   ├── 06_text_query_search.py
│   │   ├── 07_user_interest_recommend.py
│   │   └── 08_eval_recall_metrics.py
│   ├── v2/
│   └── v3/
├── data/
│   ├── raw/
│   └── processed/
│       └── v1/
│           └── kuairec/
│               └── video_text.csv
└── outputs/
    └── v1/
        ├── embeddings/
        ├── indexes/
        └── reports/
```

## 5. 环境安装

```bash
conda create -n kuairec_v1 python=3.10 -y
conda activate kuairec_v1
pip install -r requirements-v1.txt
```

检查环境：

```bash
python scripts/v1/00_check_env.py
```

## 6. V1 运行流程

### Step 1: 检查数据

```bash
python scripts/v1/01_check_data.py
```

### Step 2: 构造视频文本字段

```bash
python scripts/v1/02_build_video_text.py
```

输出：

```text
data/processed/v1/kuairec/video_text.csv
```

### Step 3: 文本向量编码

```bash
python scripts/v1/03_encode_video_text.py \
  --model-name BAAI/bge-small-zh-v1.5 \
  --batch-size 64 \
  --device auto \
  --normalize
```

输出：

```text
outputs/v1/embeddings/video_text_embeddings.npy
outputs/v1/embeddings/video_ids.npy
outputs/v1/embeddings/video_text_meta.csv
outputs/v1/embeddings/embedding_config.json
```

当前已完成的编码配置：

- 文本模型：`BAAI/bge-small-zh-v1.5`
- 向量维度：512
- 视频数量：3327
- 归一化：开启，供 cosine / inner product 检索使用

### Step 4: 构建 FAISS 语义索引

```bash
python scripts/v1/04_build_faiss_index.py --metric cosine
```

输出：

```text
outputs/v1/indexes/video_text_faiss.index
outputs/v1/indexes/faiss_index_config.json
```

当前索引使用 `IndexIDMap2 + IndexFlatIP`，在 L2 normalize 后以 cosine 相似度进行精确向量检索，并保留原始 `video_id` 作为 FAISS 返回 ID。

### Step 5: 相似视频召回

```bash
python scripts/v1/05_similar_video.py --video-id 109 --topk 10 --save
```

该步骤实现 item-to-item semantic recall，即输入一个视频 ID，召回语义相似的视频。

### Step 6: 文本 Query 检索

```bash
python scripts/v1/06_text_query_search.py --query "篮球投篮教学" --topk 10 --save
```

该步骤实现 text-to-video semantic retrieval，即输入用户搜索词，召回语义相关的视频。

### Step 7: 用户兴趣推荐

```bash
python scripts/v1/07_user_interest_recommend.py \
  --user-id 14 \
  --topk 10 \
  --pos-threshold 1.0 \
  --max-history 50 \
  --exclude-mode profile \
  --save
```

该步骤实现 user-to-item semantic recommendation，即基于用户高 watch_ratio 历史视频构建用户兴趣向量，并召回相关视频。

### Step 8: 基础指标评估

```bash
python scripts/v1/08_eval_recall_metrics.py \
  --topk 10 \
  --pos-threshold 1.0 \
  --test-size 1 \
  --max-history 50
```

评估指标包括：

- HitRate@K
- Recall@K
- NDCG@K

输出文件：

```text
outputs/v1/reports/eval_detail_top10_thr1.0.csv
outputs/v1/reports/eval_summary_top10_thr1.0.csv
```

并对比以下 baseline：

- semantic：基于内容语义的用户兴趣推荐；
- popularity：热门视频推荐；
- random：随机推荐。

当前已完成 Top10 / Top100 / Top500 三组评估，评估用户数均为 1411。

| TopK | method | HitRate@K | Recall@K | NDCG@K |
|---:|---|---:|---:|---:|
| 10 | popularity | 1.98% | 1.98% | 1.19% |
| 10 | random | 0.35% | 0.35% | 0.16% |
| 10 | semantic | 0.35% | 0.35% | 0.15% |
| 100 | popularity | 50.11% | 9.60% | 5.51% |
| 100 | random | 27.07% | 3.10% | 1.40% |
| 100 | semantic | 23.67% | 2.64% | 1.32% |
| 500 | popularity | 94.40% | 36.12% | 12.81% |
| 500 | random | 82.28% | 15.50% | 4.82% |
| 500 | semantic | 75.62% | 13.33% | 4.21% |

从当前结果看，V1 证明了文本语义召回链路可以跑通，但纯文本 semantic 用户推荐在 KuaiRec 行为评估上不如 popularity baseline。这说明仅依赖冻结文本 embedding 和简单用户兴趣平均，难以充分刻画短视频用户偏好，也直接引出 V2 的行为监督语义召回建模。

## 7. 核心方法

### 7.1 相似视频召回：item-to-item

输入一个 `video_id`，系统读取该视频的文本语义向量，并在 FAISS 视频语义索引中召回 TopK 相似视频。

### 7.2 文本 Query 检索：query-to-item

输入一句文本 Query，系统使用同一个文本 embedding 模型将 Query 编码为向量，并在视频语义向量库中检索相关视频。

### 7.3 用户兴趣推荐：user-to-item

系统从 `small_matrix.csv` 中读取用户高 `watch_ratio` 的历史视频，使用这些视频的语义向量按 `watch_ratio` 加权平均得到用户兴趣向量，再通过 FAISS 召回 TopK 推荐视频。当前默认使用 `watch_ratio >= 1.0` 作为正反馈历史，最多取 50 条历史视频，并支持过滤画像视频、过滤全部已看视频或不过滤三种模式。

## 8. 当前版本定位

V1 是一个短视频内容语义召回最小闭环，主要完成推荐系统中的召回阶段。

V1 已完成：

- 3327 个短视频的文本语义构造与 BGE 编码；
- FAISS cosine 语义索引构建；
- I2I 相似视频召回；
- Q2I 自然语言文本检索；
- U2I 基于历史高 watch_ratio 行为的用户兴趣召回；
- semantic / popularity / random baseline 离线评估。

当前 V1 不是完整推荐系统，尚未包含：

- 多路召回；
- CTR / watch_ratio 排序模型；
- 粗排、精排、重排；
- 实时在线服务；
- 视频帧级多模态内容理解。

## 9. 当前不足

当前 V1 主要存在以下不足：

- 只使用文本字段，没有使用视频帧、封面图等视觉内容；
- 使用通用文本 embedding 模型，没有基于 KuaiRec 行为数据微调；
- 只完成召回阶段，没有排序模型和重排策略；
- Query 检索没有结合 BM25、关键词匹配和 Query 改写；
- 用户兴趣推荐只使用简单的历史视频向量加权平均；
- FAISS 当前使用精确检索，尚未扩展到大规模 ANN 检索；
- 尚未完成 API 服务化和可视化 Demo。

## 10. 后续计划

当前项目已经在 V1 基础上继续演进：

- V2：引入 BGE 强文本表征、item_id / category 特征和 watch_ratio 行为监督，构建 behavior-aware 双塔语义召回模型；
- V3：基于 MicroLens-100K 接入 BGE-M3 文本特征、CLIP-RN50 图像特征和 VideoMAE 视频特征，构建图文视频多模态 item 表征与召回模型；
- 后续：可继续扩展 LightGCN 协同过滤召回通道、序列兴趣建模、轻量 reranker、API 服务化与可视化 Demo。

## 11. License

This project is for research and learning purposes.
