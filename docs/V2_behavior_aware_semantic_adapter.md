# V2: Behavior-Aware Semantic Adapter

V2 目标是基于 BGE / sentence-transformers 等强文本表征，并结合 watch_ratio、category 等用户行为监督信号，训练面向短视频语义召回任务的 item encoder、user encoder 和 behavior-aware semantic adapter。

后续内容包括：

- BGE / sentence-transformers 文本表征
- embedding cache
- item encoder / semantic adapter
- user encoder
- watch_ratio 加权兴趣建模
- InfoNCE + MSE + CE 多任务训练
- 行为监督下的语义召回评估
