# Short-Video MMRec: 面向短视频搜索推荐的多模态内容理解与语义召回系统

本项目面向短视频搜索推荐业务，围绕 **内容语义表征、用户行为建模、多模态内容理解与语义召回** 等核心问题，构建一个从静态文本语义召回 baseline，逐步演进到行为监督召回模型与图文视频多模态 item 表征的工程化系统。

项目以 KuaiRec 等公开短视频推荐数据集为基础，首先完成短视频内容语义召回的最小闭环；随后引入用户观看行为监督，训练 Behavior-Aware Semantic Adapter，实现 item 内容语义与用户兴趣行为的对齐；进一步扩展视频帧视觉编码模块，将文本语义、视频帧视觉特征与用户行为信号融合，用于短视频搜索推荐场景下的语义召回与个性化推荐。

## 项目路线图

当前项目主干分为三个阶段：

### V1: Text-based Semantic Recall Baseline

基于 KuaiRec small matrix 构建短视频内容语义召回最小闭环。

核心内容包括：

- 文本字段清洗与拼接
- 短视频文本语义编码
- FAISS 向量索引构建
- 相似视频检索
- 基于用户历史行为的兴趣向量推荐
- semantic / popularity / random baseline 对比
- Recall@K、HitRate@K 等基础指标评估

V1 的目标是完成从 item 内容字段编码、向量检索、相似视频召回到用户兴趣推荐的完整 baseline，为后续行为监督训练和多模态扩展提供工程基础。

---

### V2: Behavior-Aware Semantic Adapter

在 V1 静态文本语义召回基础上，引入更强文本语义表征和用户行为监督，训练面向短视频推荐任务的语义召回模型。

核心内容包括：

- BGE / sentence-transformers 文本语义表征
- embedding cache 构建与复用
- item encoder / semantic adapter
- user encoder
- watch_ratio 加权用户兴趣建模
- InfoNCE + MSE + CE 多任务训练
- 行为监督下的 user-item 语义对齐
- Recall@K、NDCG@K、Avg WatchRatio@K、Category Hit@K 等召回评估

V2 的目标是将 V1 中冻结的静态文本 embedding，升级为经过用户行为监督适配的 behavior-aware semantic representation，使召回结果不仅具备内容语义相关性，也能够反映用户观看偏好。

---

### V3: Multimodal Video-Text Item Representation

在 V2 行为监督语义召回基础上，进一步引入视频帧视觉特征，构建图文视频融合的多模态 item 表征。

核心内容包括：

- 视频抽帧 / 封面图处理
- CLIP image encoder 提取视觉语义特征
- 帧级特征聚合
- 文本语义特征与视觉帧特征融合
- multimodal item representation 构建
- 多模态表征接入语义召回流程

V3 的目标是将 item 表征从纯文本语义扩展为图文视频多模态语义表示，进一步增强短视频内容理解能力，使系统能够同时利用文本字段和视觉内容进行语义召回。

> 说明：由于 KuaiRec 等公开推荐数据集通常不直接提供原始视频流或完整视频帧，V3 中的视频帧理解模块将以外部公开视频数据集、样例视频或可插拔多模态模块的方式进行验证，并作为后续真实短视频业务场景中的 item encoder 扩展组件。

---

## 后续优化规划

在完成 V1-V3 主干后，项目可继续扩展以下方向：

- 引入 Qwen2.5-VL / InternVL 等多模态大模型，对视频封面或关键帧生成视觉描述，并将生成描述接入语义召回系统；
- 构建 query-to-video 的自然语言检索模块，支持用户通过自然语言表达兴趣需求；
- 在 FAISS 召回结果上加入轻量级 reranker，结合 user embedding、item embedding、watch_ratio、category、popularity 等特征进行重排序；
- 构建面向短视频推荐的 conversational recommender demo，实现自然语言意图理解、语义召回和推荐理由生成。

## 项目定位

本项目重点不在于复现完整工业推荐系统中的召回、粗排、精排、重排和在线 A/B 流程，而是聚焦于短视频推荐场景中的 **多模态内容理解与语义召回**。

项目核心关注：

- 如何从短视频文本字段中构建内容语义 embedding；
- 如何利用用户观看行为监督优化 item / user 表征；
- 如何将文本语义、视频视觉特征和用户行为信号统一到语义召回框架中；
