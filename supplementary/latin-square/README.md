# 拉丁方任务顺序分配（对应论文 §6.1、§6.3）

## 设计

被试内设计，三种范式任务顺序按 3×3 拉丁方平衡：

| 顺序代码 | 任务序列 |
|---|---|
| GHC | Graft → Hybridize → Cultivate |
| HCG | Hybridize → Cultivate → Graft |
| CGH | Cultivate → Graft → Hybridize |

每种顺序分配 4 名被试（N=12）。顺序效应与疲劳效应的检验结果见论文 §6.11
（Kruskal-Wallis H(2)=1.4，p=0.50；首末位置比较 p=0.38，均不显著）。

## 文件

- `assignments_template.csv`：分配表模板（subject_id, group, order）。
  个体级分配属于申请制数据包，不随本仓库公开；本表供新研究复用该设计。

## 复用说明

新研究可直接复用此设计：被试数需为 3 的倍数；将 subject_id 随机分配至三种顺序并
在各顺序内平衡子组（producer / researcher）。任务顺序随机化的随机种子登记于仓库
`seeds.yaml` 的 `task_order_randomization` 字段。
