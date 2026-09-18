# Botanica-OS

**Organic Operations on Neural Latent Spaces: Interaction Paradigms and Aesthetic Boundaries of AI-Assisted Botanica/IDM Timbre Design**

English | [中文](#中文)

---

## Overview

This is the open-materials repository accompanying our paper. The paper proposes the
framework of **organic operations**, which reorganizes neural latent-space interaction
into three botanically-metaphor paradigms, instantiates them in a prototype system
(**Botanica-OS**) built on a unified differentiable-DSP real-time pipeline, and quantifies
the differences between the three paradigms in a mixed-methods user study with N = 12
Botanica/IDM producers and ML researchers.

## The Three Organic Operations

| Paradigm | Input | Time structure | Formula | Description |
|---|---|---|---|---|
| **Graft** | Dual parents z_a, z_b + mixing coefficient α | Single-step, linear | Eq. (2) | A continuous transition between two timbre parents |
| **Hybridize** | Semantic condition vector c (CLAP text embedding) | Single-step, probabilistic | Eq. (3) | Intention-guided variation around a semantic point |
| **Cultivate** | Interaction history H_t + current z_t | Multi-step, trajectory | Eq. (4) / Algorithm 1 | Bandit-style trajectory optimization driven by preference feedback |

## Key Results (N = 12; Cultivate condition N = 11)

- **Graft** yields the highest overall SUS (M = 78.3, SD = 5.5) and the highest SUS
  learnability subscale (75.5; vs. Hybridize, Holm-corrected p = 0.048).
- **Cultivate** yields the highest agency rating (M = 5.9/7).
- **Hybridize** scores lowest on all five subjective dimensions, and its experienced
  controllability falls below its conceptual placement — one of the paper's core design
  findings.
- **H1 was not supported**: within the same N = 6 subsample, Graft SUS 78.2 vs. the RAVE
  coordinate-editing baseline 78.1. Semantic-level operations did not raise overall
  usability, but substantially restructured task-level experience (affordance 5.4 vs. 2.8,
  Wilcoxon p = 0.031; exploratory comparison).
- End-to-end latency 47 ± 3 ms (16 kHz, 128-sample blocks, CPU inference), meeting the
  <100 ms performative-interaction requirement (DR5).

All numbers follow Table 7 of the paper as the single source of truth (see `results/`).
All conclusions are bounded by the sample (N = 12), a single style (Botanica/IDM), and a
four-week follow-up window.

## Repository Structure

```
analysis/       Quantitative re-analysis script (reanalysis.py) and data template (study_data_template.csv)
reference/      Reference implementation of Eqs. (2)–(4) and Algorithm 1 (numpy only, unit-tested)
results/        Result tables: latency breakdown (Table 4), summary of results (Table 7),
                baseline fairness controls (Table 8), hyperparameters (Table A1)
study/          User-study materials: task protocol, SUS/Likert questionnaires, interview guide,
                consent-form and recruitment templates, data management plan
supplementary/  Latin-square design, discourse-corpus schema, target-timbre clusters, coding manual
docs/           Annotations, reading notes, and other auxiliary materials (extensible; see below)
seeds.yaml      Random-seed registry (model training / UMAP / bootstrap / task-order randomization)
```

## Extending This Repository

New auxiliary materials (annotations, reading notes, extended derivations, replications)
are welcome and should follow these conventions:

- **Where**: put scholarly annotations and extended notes in `docs/<topic>/`; new
  study-protocol material follows the existing `study/` and `supplementary/` layout.
- **Naming**: `YYYY-MM-DD_<topic>_<type>.md` for dated notes (e.g., `2026-10-01_gp-derivation_note.md`).
- **Language**: English first, with a Chinese version in the same file (or a `*.zh.md`
  sibling), matching this README.
- **Provenance**: every claim that comes from the paper must cite its section number;
  new empirical material must state its scope limits as the paper does.
- **License**: new code joins the repository's GPL-3.0; new prose joins CC-BY 4.0.

## Reproduction Environment

- Backend: Python 3.11, PyTorch 2.1, librosa 0.10, FastAPI; inference on a MacBook Pro M2
  (CPU only), training on a single A100.
- Frontend: React 18 + TypeScript 5, communicating with the backend over WebSocket.
- Conditioning branch: CLAP text embeddings (`laion/clap-htsat-fused`, LAION-CLAP, unmodified).
- Latent dimension d = 16; UMAP projection (50 neighbors, min_dist 0.1) precomputed at
  startup and incrementally updated every 100 interactions.
- Cultivate: GP regression with a Matérn 5/2 kernel, history truncated to the most recent
  100 interactions, η = 0.05, λ = 0.3.

Full hyperparameters: `results/tableA1_hyperparameters.csv`.

## Data and Materials Access

- **Direct download**: analysis scripts, result tables, study protocols and questionnaires,
  and the reference implementation in this repository.
- **Application-based access**: individual-level de-identified questionnaire data and
  interview transcripts involve participant privacy and are provided after a signed data-use
  agreement. Contact: akiromusic@qq.com
- **Not distributed**: copyright-protected training audio (Bandcamp releases, Freesound
  CC-BY-NC entries, the author's private collection) is not distributed with this repository
  or with open weights; training-set metadata and extraction scripts are provided instead.
- **Opt-out**: artists whose work is involved in the training data may request removal at
  any time via akiromusic@qq.com.

## License

- Code in this repository (`reference/`, `analysis/`) is released under **GPL-3.0**
  (see [LICENSE](LICENSE)).
- Release of pretrained weights is conditional on the final review of training-data
  licensing; if direct release is not supported, a retrained version with unauthorized
  material removed will be released instead (paper §4.5, §8.5).
- Third-party component compatibility is documented item by item in
  [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md).

## Citation

To cite this work, use the accompanying paper (see [CITATION.cff](CITATION.cff)):

> AkiroMusic. Organic Operations on Neural Latent Spaces: Interaction Paradigms and
> Aesthetic Boundaries of AI-Assisted Botanica/IDM Timbre Design. 2026.

---

## 中文

### 概述

本仓库是配套论文的开放材料库。论文提出**「有机化操作」**（Organic Operations）设计框架，
将神经潜在空间交互重组为三种植物学隐喻范式，在统一的可微 DSP 实时管线上实现原型系统
**Botanica-OS**，并通过 N=12 位 Botanica/IDM 制作人与 ML 研究者的混合方法用户研究
量化三范式差异。

### 三种有机化操作

| 范式 | 输入 | 时间结构 | 数学形式 | 一句话描述 |
|---|---|---|---|---|
| **嫁接 Graft** | 双父本 z_a, z_b + 混合系数 α | 单步、线性 | 式(2) | 两个音色父本之间的连续过渡 |
| **杂交 Hybridize** | 语义条件向量 c（CLAP 文本嵌入） | 单步、概率 | 式(3) | 在条件语义点周围按意图变异 |
| **培育 Cultivate** | 交互历史 H_t + 当前 z_t | 多步、轨迹 | 式(4)/算法1 | 基于偏好反馈的 bandit 式轨迹优化 |

### 关键结果（N=12；培育条件 N=11）

- **嫁接**：总体 SUS 最高（M=78.3，SD=5.5），SUS 可学性子量表亦最高（75.5；
  vs 杂交，Holm 校正 p=0.048）。
- **培育**：能动性维度全场最高（M=5.9/7）。
- **杂交**：五个主观维度全部最低，实测可控性低于其概念层定位——本研究的核心设计发现之一。
- **H1 未获支持**：同一 N=6 子样本内，嫁接 SUS 78.2 vs RAVE 坐标编辑基线 78.1。
  语义层操作未提升整体可用性，但显著改变了任务层体验的结构（示能性 5.4 vs 2.8，
  Wilcoxon p=0.031，探索性比较）。
- 端到端延迟 47±3 ms（16 kHz、128 样本块、CPU 推理），满足 <100 ms 的演奏式交互要求（DR5）。

全文数字以论文表7（结果总表，见 `results/`）为唯一口径。所有结论限于 N=12 样本、
Botanica/IDM 单一风格与 4 周随访窗口。

### 仓库结构

```
analysis/       定量再分析脚本 reanalysis.py 与数据模板 study_data_template.csv
reference/      论文式(2)–(4)与算法1的参考实现（仅依赖 numpy，含单元测试）
results/        结果汇总表：延迟分解（表4）、结果总表（表7）、基线公平性（表8）、超参数（表A1）
study/          用户研究材料：任务协议、SUS/Likert 问卷原卷、访谈提纲、知情同意书与招募模板、数据管理计划
supplementary/  拉丁方分配设计、社区话语分析语料 schema、目标音色簇、质性编码手册
docs/           注解、读书笔记与其他辅助材料（可扩展，约定见上）
seeds.yaml      随机种子登记（模型训练 / UMAP / Bootstrap / 任务顺序）
```

### 仓库扩展约定

欢迎添加论文辅助材料（注解、读书笔记、推导扩展、复现尝试），约定如下：

- **位置**：学术注解与扩展笔记放 `docs/<主题>/`；新的研究协议材料沿用现有
  `study/` 与 `supplementary/` 的目录结构。
- **命名**：带日期的笔记用 `YYYY-MM-DD_<主题>_<类型>.md`（如 `2026-10-01_gp-derivation_note.md`）。
- **语言**：英文在前，同一文件内附中文版本（或 `*.zh.md` 姊妹文件），与本 README 一致。
- **出处**：凡来自论文的论断须标注章节号；新增实证材料须像论文一样声明适用边界。
- **许可**：新增代码遵循本仓库 GPL-3.0；新增文字遵循 CC-BY 4.0。

### 复现环境

- 后端：Python 3.11、PyTorch 2.1、librosa 0.10、FastAPI；推理 MacBook Pro M2（仅 CPU），
  训练单张 A100。
- 前端：React 18 + TypeScript 5，WebSocket 与后端通信。
- 条件分支：CLAP 文本嵌入（`laion/clap-htsat-fused`，LAION-CLAP，无修改）。
- 潜在维 d=16；UMAP（50 邻居、min_dist 0.1）启动时预计算，每 100 次交互增量更新。
- 培育：GP 回归（Matérn 5/2），历史截断至最近 100 次交互，η=0.05、λ=0.3。

完整超参数见 `results/tableA1_hyperparameters.csv`。

### 数据与材料获取

- **直接获取**：本仓库中的分析脚本、结果汇总表、研究协议与问卷原卷、参考实现。
- **申请制访问**：个体级去标识问卷数据与访谈转录涉及被试隐私，签署数据使用协议后提供。
  联系：akiromusic@qq.com
- **不分发**：受版权保护的训练音频（Bandcamp 作品、Freesound CC-BY-NC 条目、作者私人收藏）
  不随本仓库或开源权重分发；仓库提供训练集元数据与自动提取脚本以支持复现。
- **退出通道**：训练数据涉及艺术家可随时通过 akiromusic@qq.com 申请退出。

### 许可

- 本仓库代码（`reference/`、`analysis/`）以 **GPL-3.0** 许可开源，见 [LICENSE](LICENSE)。
- 预训练权重的公开发布以训练数据许可复核的最终结论为前提；若复核不支持直接发布，
  将发布移除未授权素材后重训的版本（论文 §4.5、§8.5）。
- 第三方组件的许可兼容性逐项说明见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。

### 引用

如引用本工作，请使用论文（见 [CITATION.cff](CITATION.cff)）：

> AkiroMusic. Organic Operations on Neural Latent Spaces: Interaction Paradigms and
> Aesthetic Boundaries of AI-Assisted Botanica/IDM Timbre Design. 2026.
> （中文题目：神经潜在空间的「有机化」操作：AI 辅助 Botanica/IDM 音色设计的交互范式与美学边界）
