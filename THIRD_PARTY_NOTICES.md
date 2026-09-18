# 第三方组件与数据许可说明（THIRD_PARTY_NOTICES）

本仓库及其随附开源材料依赖以下第三方组件与数据源。各条目仅陈述本工作使用方式与许可状态；
许可全文以各上游项目发布版本为准。

## 模型与代码组件

| 组件 | 用途 | 许可 | 说明 |
|---|---|---|---|
| CLAP（LAION-CLAP）代码 | 杂交模式的文本条件嵌入 | MIT | 论文附录 B：`laion/clap-htsat-fused` 权重，无修改使用 |
| `laion/clap-htsat-fused` 模型权重 | 同上 | CC-BY 4.0 | 按 Hugging Face 模型卡标注条款再分发 |
| PyTorch | 训练与推理框架 | BSD-3-Clause | — |
| librosa | 音频特征（Mel 频谱等） | ISC | — |
| DDSP（differentiable DSP） | 可微合成器后端（谐波加噪声） | Apache-2.0 | 论文引用 Engel 等 2019；使用其 Python 实现 |
| umap-learn | 潜在空间 2D 投影 | BSD-3-Clause | 50 邻居、min_dist 0.1 |
| scikit-learn / SciPy / NumPy | 统计分析与数值计算 | BSD-3-Clause | — |
| FastAPI / React / TypeScript | 服务与前端 | MIT | — |

## 训练数据来源

| 数据源 | 许可 | 本工作中的使用 |
|---|---|---|
| NSynth 数据库子集（12 小时） | CC-BY 4.0 | 可直接使用与再分发 |
| Freesound 生态录音（4 小时：鸟鸣、水流、风、虫鸣） | 逐条 CC0 / CC-BY / CC-BY-NC | CC-BY 条目在开源权重发布时附带署名清单；经逐条核查，CC-BY-NC 条目仅用于本地实验与论文配图，**不进入**随开源权重分发的训练集 |
| Bandcamp 购买的 Botanica/IDM 作品片段（8 小时的一部分） | 私人听用授权 | 训练用途已逐一联系艺术家并取得书面许可（覆盖涉及的 12 位艺术家中的 11 位；剩余 1 位未回应，其作品已从训练集移除）；**音频不随任何仓库分发**，仅提供元数据与提取脚本 |
| 作者私人收藏中非本人创作的部分 | — | 按上条同样流程处理 |

## 本仓库材料的许可层级

1. 代码（`reference/`、`analysis/`）：GPL-3.0。
2. 研究协议、问卷、结果汇总表（`study/`、`results/`、`supplementary/`）：CC-BY 4.0
   （SUS 量表条目转述自 Brooke 1996 章节中译，用于研究复现目的）。
3. 预训练模型权重：**暂不分发**。发布以训练数据许可复核最终结论为前提；
   若复核不支持直接发布，将发布移除未授权素材后重训的版本（论文 §4.5、§8.5）。
4. 个体级去标识被试数据与访谈转录：**不在本仓库**，申请制访问（akiromusic@qq.com）。

## 退出通道（opt-out）

训练数据涉及的艺术家或权利人可随时通过 akiromusic@qq.com 申请将其作品移出训练集及后续重训版本。
