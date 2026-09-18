# reference/ — 论文数学定义的参考实现

本目录提供论文中式(2)–(4)与算法 1 的独立参考实现（仅依赖 numpy）：

| 函数/类 | 对应 | 说明 |
|---|---|---|
| `clip_3sigma` | 式(2)(4) 的 clip_{3σ} | 逐维截断至训练集均值 ±3σ（必要不充分条件，见 §4.3） |
| `graft` | 式(2) 嫁接 | 双父本欧氏线性插值（非测地线，§7.2） |
| `hybridize_sample` | 式(3) 杂交 | 条件后验采样；OOD 回退在生产系统处理 |
| `CultivateGP` | 算法 1 | Matérn 5/2 GP 奖励模型，历史截断 100 |
| `cultivate_step` | 式(4) 培育 | 一阶差分梯度（利用）+ Thompson 式探索项 |

默认超参数与论文表A1 一致：η=0.05、λ=0.3。

## 运行测试

```bash
cd reference
python test_botanica_ops.py        # 或 python -m pytest
```

## 边界声明

这是数学定义的参考实现，用于复核论文公式与教学演示，**不是** Botanica-OS 系统
本体：完整系统（React + TypeScript 前端、FastAPI 后端、PyTorch VAE/DDSP 管线）
按论文 §8.5 的计划另行发布。本实现未经性能优化（朴素 GP 预测 O(t²)，生产系统
通过历史截断与增量更新控制延迟，见 §4.3、§4.7）。
