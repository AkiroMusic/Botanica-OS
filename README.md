# Botanica-OS

神经潜在空间的「有机化」操作：AI 辅助 Botanica/IDM 音色设计的交互范式与美学边界

本仓库是配套论文的开放材料库。论文提出「有机化操作」（Organic Operations）设计框架，
将神经潜在空间交互重组为三种植物学隐喻范式，并在统一的可微 DSP 实时管线上实现原型系统
Botanica-OS，通过 N=12 位 Botanica/IDM 制作人与 ML 研究者的混合方法用户研究量化三范式差异。

## 三种有机化操作

| 范式 | 输入 | 时间结构 | 数学形式 | 一句话描述 |
|---|---|---|---|---|
| 嫁接 Graft | 双父本 z_a, z_b + 混合系数 α | 单步、线性 | 式(2) | 两个音色父本之间的连续过渡 |
| 杂交 Hybridize | 语义条件向量 c（CLAP 文本嵌入） | 单步、概率 | 式(3) | 在条件语义点周围按意图变异 |
| 培育 Cultivate | 交互历史 H_t + 当前 z_t | 多步、轨迹 | 式(4)/算法1 | 基于偏好反馈的 bandit 式轨迹优化 |

## 关键结果（N=12；培育条件 N=11）

- 嫁接：总体 SUS 最高（M=78.3，SD=5.5），SUS 可学性子量表亦最高（75.5；vs 杂交 Holm 校正 p=0.048）
- 培育：能动性维度全场最高（M=5.9/7）
- 杂交：五个主观维度全部最低，实测可控性低于其概念层定位（本研究的核心设计发现之一）
- H1 未获支持：同一 N=6 子样本内，嫁接 SUS 78.2 vs RAVE 坐标编辑基线 78.1；语义层操作未提升
  整体可用性，但显著改变任务层体验结构（示能性 5.4 vs 2.8，Wilcoxon p=0.031，探索性比较）
- 端到端延迟 47±3 ms（16 kHz、128 样本块、CPU 推理），满足 <100 ms 的演奏式交互要求（DR5）

完整数字以论文表7（结果总表，见 `results/`）为唯一口径。所有结论限于 N=12 样本、
Botanica/IDM 单一风格与 4 周随访窗口。

## 仓库结构

```
analysis/      定量再分析脚本 reanalysis.py 与数据模板 study_data_template.csv
reference/     论文式(2)–(4)与算法1的参考实现（便于复核数学定义；完整系统代码另行发布）
results/       延迟分解（表4）、结果总表（表7）、基线公平性（表8）、超参数（表A1）
study/         用户研究材料：任务协议、SUS/Likert 问卷原卷、访谈提纲、知情同意书模板、数据管理计划
supplementary/ 拉丁方分配设计、社区话语分析语料清单、目标音色簇、质性编码手册
seeds.yaml     随机种子登记（模型训练 / UMAP / Bootstrap / 任务顺序）
```

## 复现环境

- 后端：Python 3.11、PyTorch 2.1、librosa 0.10、FastAPI；推理 MacBook Pro M2（仅 CPU），训练单张 A100
- 前端：React 18 + TypeScript 5，WebSocket 与后端通信
- 条件分支：CLAP 文本嵌入（`laion/clap-htsat-fused`，LAION-CLAP，无修改）
- 潜在维 d=16；UMAP（50 邻居、min_dist 0.1）启动时预计算，每 100 次交互增量更新
- Cultivate：GP 回归（Matérn 5/2），历史截断至最近 100 次交互，η=0.05、λ=0.3

完整超参数见 `results/tableA1_hyperparameters.csv`。

## 数据与材料获取

- **直接获取**：本仓库中的分析脚本、结果汇总表、研究协议与问卷原卷、参考实现。
- **申请制访问**：个体级去标识问卷数据与访谈转录涉及被试隐私，签署数据使用协议后提供。
  联系：akiromusic@qq.com
- **不分发**：受版权保护的训练音频（Bandcamp 作品、Freesound CC-BY-NC 条目、作者私人收藏）
  不随本仓库或开源权重分发；仓库提供训练集元数据与自动提取脚本以支持复现。
- **退出通道**：训练数据涉及艺术家可随时通过 akiromusic@qq.com 申请退出；已设公开 opt-out 机制。

## 许可

- 本仓库代码（含 `reference/`、`analysis/`）以 **GPL-3.0** 许可开源，见 [LICENSE](LICENSE)。
- 预训练权重的公开发布以训练数据许可复核的最终结论为前提；若复核不支持直接发布，
  将发布移除未授权素材后重训的版本。见论文 §4.5、§8.5。
- 第三方组件的许可兼容性逐项说明见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。

## 引用

如引用本工作，请使用论文（见 [CITATION.cff](CITATION.cff)）：

> 章乃驰（AkiroMusic）. 神经潜在空间的「有机化」操作：AI 辅助 Botanica/IDM 音色设计的交互范式与美学边界. 2026.
