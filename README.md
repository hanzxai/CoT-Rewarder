# 🧠 CoT-Rewarder

Chain-of-Thought Reward Framework (based on open-r1)

> Exploring fine-grained reasoning alignment at the step level.

这个项目是在 **[open-r1](https://github.com/huggingface/open-r1)** 的基础上改动的，
主要目的是探索更细粒度的 **step-level reasoning 优化与奖励建模**。

---

## 🚀 改动点

- 增加 **step-level 打分与筛选逻辑**，支持对每一步推理过程进行建模
- 兼容 **DPO / GRPO / RLHF** 等不同训练方式
- 调整代码结构，更轻量，便于快速实验和复现
