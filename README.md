# SV-Recall V1: 基于 KuaiRec 的短视频内容语义召回与用户兴趣推荐最小闭环

本项目是“面向短视频搜索推荐的多模态内容理解与语义召回系统”的 V1 版本，基于 KuaiRec 数据集完成短视频内容语义召回的最小闭环实现。

V1 重点不在于构建完整工业推荐系统，而是完成从视频文本字段构造、文本向量编码、FAISS 语义索引、相似视频召回、文本 Query 检索、用户兴趣推荐到基础指标评估的完整流程。

## 1. 项目目标

本项目面向短视频搜索推荐场景，完成以下能力：

- 基于视频 caption、topic_tag、category 等字段构造视频语义文本；
- 使用文本 embedding 模型编码短视频内容语义；
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

本项目使用 KuaiRec 数据集中的：

- `small_matrix.csv`：用户-视频交互数据；
- `kuairec_caption_category.csv`：视频 caption、topic_tag 和多级类目信息；
- `item_categories.csv`：视频标签信息，可用于后续扩展。

由于数据集和中间向量文件较大，本仓库不直接上传原始数据和生成的 embedding/index 文件。请用户自行下载 KuaiRec 数据集，并放置到：

```text
data/raw/
```

本项目默认会在 `data/raw/` 下自动查找所需 CSV 文件。

## 4. 项目结构

```text
short-video-mmrec-v1/
├── README.md
├── requirements-v1.txt
├── .gitignore
├── configs/
├── scripts/
│   ├── 00_check_env.py
│   ├── 01_check_data.py
│   ├── 02_build_video_text.py
│   ├── 03_encode_video_text.py
│   ├── 04_build_faiss_index.py
│   ├── 05_similar_video.py
│   ├── 06_text_query_search.py
│   ├── 07_user_interest_recommend.py
│   └── 08_eval_recall_metrics.py
├── data/
│   ├── raw/
│   └── processed/
└── outputs/
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
python scripts/00_check_env.py
```

## 6. V1 运行流程

### Step 1: 检查数据

```bash
python scripts/01_check_data.py
```

### Step 2: 构造视频文本字段

```bash
python scripts/02_build_video_text.py
```

输出：

```text
data/processed/video_text.csv
```

### Step 3: 文本向量编码

```bash
python scripts/03_encode_video_text.py \
  --model-name BAAI/bge-small-zh-v1.5 \
  --batch-size 64 \
  --device auto \
  --normalize
```

输出：

```text
outputs/embeddings/video_text_embeddings.npy
outputs/embeddings/video_ids.npy
outputs/embeddings/video_text_meta.csv
outputs/embeddings/embedding_config.json
```

### Step 4: 构建 FAISS 语义索引

```bash
python scripts/04_build_faiss_index.py --metric cosine
```

输出：

```text
outputs/indexes/video_text_faiss.index
outputs/indexes/faiss_index_config.json
```

### Step 5: 相似视频召回

```bash
python scripts/05_similar_video.py --video-id 0 --topk 10
```

该步骤实现 item-to-item semantic recall，即输入一个视频 ID，召回语义相似的视频。

### Step 6: 文本 Query 检索

```bash
python scripts/06_text_query_search.py --query "篮球教学" --topk 10
```

该步骤实现 text-to-video semantic retrieval，即输入用户搜索词，召回语义相关的视频。

### Step 7: 用户兴趣推荐

```bash
python scripts/07_user_interest_recommend.py \
  --user-id 0 \
  --topk 10 \
  --pos-threshold 1.0 \
  --max-history 50 \
  --exclude-mode profile
```

该步骤实现 user-to-item semantic recommendation，即基于用户高 watch_ratio 历史视频构建用户兴趣向量，并召回相关视频。

### Step 8: 基础指标评估

```bash
python scripts/08_eval_recall_metrics.py \
  --topk 10 \
  --pos-threshold 1.0 \
  --test-size 1 \
  --max-history 50
```

评估指标包括：

- HitRate@K
- Recall@K
- NDCG@K

并对比以下 baseline：

- semantic：基于内容语义的用户兴趣推荐；
- popularity：热门视频推荐；
- random：随机推荐。

## 7. 核心方法

### 7.1 相似视频召回：item-to-item

输入一个 `video_id`，系统读取该视频的文本语义向量，并在 FAISS 视频语义索引中召回 TopK 相似视频。

### 7.2 文本 Query 检索：query-to-item

输入一句文本 Query，系统使用同一个文本 embedding 模型将 Query 编码为向量，并在视频语义向量库中检索相关视频。

### 7.3 用户兴趣推荐：user-to-item

系统从 `small_matrix.csv` 中读取用户高 `watch_ratio` 的历史视频，使用这些视频的语义向量按 `watch_ratio` 加权平均得到用户兴趣向量，再通过 FAISS 召回 TopK 推荐视频。

## 8. 当前版本定位

V1 是一个短视频内容语义召回最小闭环，主要完成推荐系统中的召回阶段。

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

- V2：引入视频帧 / 封面图特征，增强内容理解；
- V3：构建图文多模态语义召回；
- V4：加入用户行为建模和排序模型；
- V5：完成 API / Demo / 工程化部署。

## 11. License

This project is for research and learning purposes.
