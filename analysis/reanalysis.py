# -*- coding: utf-8 -*-
"""
Botanica-OS 用户研究 · 再分析脚本（修改稿 v2）
====================================================

用途
----
按修改稿 §6.4 / §6.11 的统计方案，对原始数据重新计算结果总表（表7）中
标注【待算】的全部单元格，包括：

  1. 描述统计（M、SD）与 10,000 次 Bootstrap 95% CI
  2. 跨三范式的 Friedman 检验（逐指标）
  3. 事后两两 Wilcoxon 符号秩检验 × Holm 校正（全部配对，逐指标）
  4. 效应量：Cliff's δ、Cohen's d_z
  5. 贝叶斯因子 BF10（成对，JZS 先验 r=0.707，需安装 pingouin；缺失时跳过）
  6. 顺序/疲劳效应检验（拉丁方位置趋势、首末位置对比）
  7. ITT 敏感性分析（Cultivate 缺失保守插补后重跑主检验）
  8. 线性混合模型敏感性分析（score ~ condition*group + order + (1|subject)，
     需安装 statsmodels；缺失时跳过）
  9. 子组探索性比较（制作人 vs ML 研究者，N=6/组，Wilcoxon 秩和）

用法
----
  1. 将原始数据整理为 long 格式 CSV（模板见同目录 study_data_template.csv）：
       subject_id, group, order, position, condition,
       sus_overall, sus_learnability,
       affordance, agency, control, pleasure, aesthetic_fit
     —— 每行 = 一位被试在一个范式条件下的全部评分。
     condition 取值：Graft / Hybridize / Cultivate；
     baseline 数据（RAVE/AudioLDM，N=6）单独放在 baseline 数据文件或
     同文件的 condition=RAVE_bare / AudioLDM_t2a 行（position 留空）。
  2. python reanalysis.py --data study_data_template.csv --outdir results
  3. 脚本输出 results/ 下的 CSV 与 markdown 汇总表，直接回填论文表7。

依赖：pandas, numpy, scipy（必需）；pingouin（BF10，可选）；statsmodels（LMM，可选）
随机种子：SEED = 20240927（与仓库 seeds.yaml 一致【作者核对】）
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20240927
RNG = np.random.default_rng(SEED)

METRICS = ["sus_overall", "sus_learnability",
           "affordance", "agency", "control", "pleasure", "aesthetic_fit"]
CONDITIONS = ["Graft", "Hybridize", "Cultivate"]
PAIRS = list(itertools.combinations(CONDITIONS, 2))


# --------------------------------------------------------------------------
# 效应量
# --------------------------------------------------------------------------
def cliffs_delta(x, y):
    """Cliff's δ：组间优于关系的非参数效应量，范围 (-1, 1)。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def cohens_dz(a, b):
    """组内设计的 Cohen's d_z = mean(diff)/sd(diff)。"""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return np.nan
    return d.mean() / d.std(ddof=1)


def bootstrap_ci(x, n_boot=10_000, ci=95):
    """Bootstrap 均值 95% CI（percentile 法）。"""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    means = RNG.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return (lo, hi)


# --------------------------------------------------------------------------
# 校正
# --------------------------------------------------------------------------
def holm(pvals):
    """Holm-Bonferroni 逐步校正，返回与输入同序的校正后 p 值。"""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * (n - np.arange(n))
    adj = np.maximum.accumulate(adj)
    adj = np.minimum(adj, 1.0)
    out = np.empty(n)
    out[order] = adj
    return out


def bf10_pair_t(x, y, r=0.707):
    """成对样本 JZS Bayes factor BF10（Rouder et al. 2009）。
    依赖 pingouin；未安装则返回 None。"""
    try:
        import pingouin as pg  # noqa
    except ImportError:
        return None
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return None
    t, p = stats.ttest_1samp(d, 0)
    res = pg.ttest(d, 0, paired=False)  # 对差值做单样本 t
    return float(res["BF10"].iloc[0])


# --------------------------------------------------------------------------
# 主分析
# --------------------------------------------------------------------------
def wide_by_metric(df, metric):
    """返回 {condition: 以 subject 为索引的 Series}，仅保留三范式完整观测。"""
    out = {}
    for c in CONDITIONS:
        sub = df[df.condition == c].set_index("subject_id")[metric].dropna()
        out[c] = sub
    return out


def friedman_and_posthoc(df, metric):
    """Friedman + Holm 校正事后 Wilcoxon + 效应量 + CI + BF10。
    只用三条件均无缺失的被试（within-subject 完整样本）。"""
    w = {c: df[df.condition == c].set_index("subject_id")[metric] for c in CONDITIONS}
    complete = pd.concat(w, axis=1).dropna()  # 完整被试
    complete.columns = CONDITIONS
    res = {"metric": metric, "n_complete": len(complete)}

    if len(complete) >= 3 and complete.nunique().min() > 1:
        stat, p = stats.friedmanchisquare(*[complete[c] for c in CONDITIONS])
        res["friedman_chi2"], res["friedman_p"] = stat, p
    else:
        res["friedman_chi2"], res["friedman_p"] = np.nan, np.nan

    # 描述统计 + CI（per-condition，允许 N 不等）
    for c in CONDITIONS:
        s = w[c].dropna()
        lo, hi = bootstrap_ci(s)
        res[f"{c}_M"], res[f"{c}_SD"] = s.mean(), s.std(ddof=1)
        res[f"{c}_N"] = len(s)
        res[f"{c}_CI"] = f"[{lo:.1f}, {hi:.1f}]"

    # 事后：全部配对 + Holm
    raw_p, pair_info = [], {}
    for a, b in PAIRS:
        joined = pd.concat([w[a], w[b]], axis=1, keys=[a, b]).dropna()
        if len(joined) < 3:
            pair_info[(a, b)] = {"wilcoxon_W": np.nan, "wilcoxon_p_raw": np.nan,
                                 "cliffs_delta": np.nan, "cohens_dz": np.nan,
                                 "n": len(joined), "bf10": None}
            raw_p.append(np.nan)
            continue
        stat_w, p_w = stats.wilcoxon(joined[a], joined[b], zero_method="wilcox")
        raw_p.append(p_w)
        pair_info[(a, b)] = {
            "wilcoxon_W": stat_w, "wilcoxon_p_raw": p_w,
            "cliffs_delta": cliffs_delta(joined[a], joined[b]),
            "cohens_dz": cohens_dz(joined[a], joined[b]),
            "n": len(joined),
            "bf10": bf10_pair_t(joined[a], joined[b]),
        }
    adj = holm([p if not np.isnan(p) else 1.0 for p in raw_p])
    for (pair, p_adj) in zip(PAIRS, adj):
        pair_info[pair]["wilcoxon_p_holm"] = p_adj
    res["pairs"] = pair_info
    return res


def order_fatigue(df, metric="sus_overall"):
    """顺序与疲劳效应（修改稿 §6.11）。
    - 位置趋势：Kruskal-Wallis across position 1/2/3（组内标准化的分）
    - 疲劳对比：每被试 position 3 vs position 1 的 Wilcoxon（组内，跨条件标准化）
    """
    d = df[df.condition.isin(CONDITIONS)].copy()
    d["z"] = d.groupby("subject_id")[metric].transform(
        lambda s: (s - s.mean()) / s.std(ddof=1) if s.notna().sum() > 1 and s.std(ddof=1) else np.nan)
    d = d.dropna(subset=["z"])
    groups = [g["z"].dropna() for _, g in d.groupby("position")]
    if all(len(g) > 1 for g in groups) and len(groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups)
    else:
        kw_stat, kw_p = np.nan, np.nan
    wide = d.pivot_table(index="subject_id", columns="position", values="z")
    if {1, 3}.issubset(wide.columns):
        both = wide[[1, 3]].dropna()
        fat_stat, fat_p = stats.wilcoxon(both[3], both[1], zero_method="wilcox")
    else:
        both = pd.DataFrame()
        fat_stat, fat_p = np.nan, np.nan
    return {"kw_chi2": kw_stat, "kw_p": kw_p,
            "fatigue_W": fat_stat, "fatigue_p": fat_p,
            "n_fatigue_pairs": len(both)}


def itt_sensitivity(df, metric="sus_overall"):
    """ITT 敏感性：Cultivate 缺失按三范式最低观测值保守插补后重跑 Friedman。"""
    w = wide_by_metric(df, metric)
    complete = pd.concat(w, axis=1)
    n_missing_before = int(complete["Cultivate"].isna().sum())
    vals = complete[CONDITIONS].values
    if np.isfinite(vals.astype(float)).sum() == 0:
        return {"n_imputed_cultivate": n_missing_before, "floor_value": np.nan,
                "friedman_chi2": np.nan, "friedman_p": np.nan, "n": 0}
    floor = np.nanmin(vals)
    complete["Cultivate"] = complete["Cultivate"].fillna(floor)
    complete = complete.dropna()
    if len(complete) < 3 or complete[CONDITIONS].nunique().min() < 2:
        return {"n_imputed_cultivate": n_missing_before, "floor_value": floor,
                "friedman_chi2": np.nan, "friedman_p": np.nan, "n": len(complete)}
    stat, p = stats.friedmanchisquare(*[complete[c] for c in CONDITIONS])
    return {"n_imputed_cultivate": n_missing_before, "floor_value": floor,
            "friedman_chi2": stat, "friedman_p": p, "n": len(complete)}


def lmm_sensitivity(df, metric):
    """线性混合模型敏感性分析：score ~ C(condition)*C(group) + C(order) + (1|subject)。
    需要 statsmodels；未安装则返回 None。"""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    d = df[df.condition.isin(CONDITIONS)].copy()
    d = d[[metric, "condition", "group", "order", "subject_id"]].dropna()
    if d[metric].nunique() < 2 or d["subject_id"].nunique() < 5:
        return None
    md = smf.mixedlm(f"{metric} ~ C(condition) * C(group) + C(order)",
                     d, groups=d["subject_id"])
    try:
        fit = md.fit(reml=True, method="lbfgs")
    except Exception as e:  # 收敛失败时不阻塞
        return {"error": str(e)}
    return fit.summary().as_text()


def subgroup_exploratory(df):
    """子组（制作人 vs 研究者，各 N=6）探索性比较：Wilcoxon 秩和（Mann-Whitney）。"""
    rows = []
    for c in CONDITIONS:
        for m in METRICS:
            a = df[(df.condition == c) & (df.group == "producer")][m].dropna()
            b = df[(df.condition == c) & (df.group == "researcher")][m].dropna()
            if len(a) >= 3 and len(b) >= 3:
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            else:
                u, p = np.nan, np.nan
            rows.append({"condition": c, "metric": m,
                         "producer_M": a.mean(), "researcher_M": b.mean(),
                         "U": u, "p_raw": p,
                         "note": "exploratory, N=6/group, uncorrected"})
    return pd.DataFrame(rows)


def baseline_compare(df_base):
    """基线（RAVE_bare / AudioLDM_t2a，N=6）描述统计 + 与主研究条件的探索性 Wilcoxon。
    仅对同时完成主条件与基线的被试做配对比较。"""
    if df_base.empty:
        return pd.DataFrame()
    rows = []
    for c in ["Graft", "Cultivate"]:
        for m in METRICS:
            for b in ["RAVE_bare", "AudioLDM_t2a"]:
                main = df_base[df_base.condition == c].set_index("subject_id")[m]
                base = df_base[df_base.condition == b].set_index("subject_id")[m]
                joined = pd.concat([main, base], axis=1, keys=["main", "base"]).dropna()
                if len(joined) >= 3:
                    w, p = stats.wilcoxon(joined["main"], joined["base"], zero_method="wilcox")
                    delta = cliffs_delta(joined["main"], joined["base"])
                else:
                    w, p, delta = np.nan, np.nan, np.nan
                rows.append({"main_cond": c, "baseline": b, "metric": m,
                             "main_M": main.mean(), "base_M": base.mean(),
                             "n_paired": len(joined),
                             "wilcoxon_p_raw": p, "cliffs_delta": delta,
                             "note": "exploratory, uncorrected"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
def fmt_res(r):
    lines = [f"### {r['metric']}  (完整样本 N={r['n_complete']})",
             f"- Friedman: chi2={r['friedman_chi2']:.3f}, p={r['friedman_p']:.4f}"]
    for c in CONDITIONS:
        lines.append(f"- {c}: M={r[f'{c}_M']:.2f}, SD={r[f'{c}_SD']:.2f}, "
                     f"N={r[f'{c}_N']}, 95%CI {r[f'{c}_CI']}")
    for (a, b), info in r["pairs"].items():
        bf = f", BF10={info['bf10']:.2f}" if info["bf10"] else ""
        lines.append(f"- {a} vs {b}: W={info['wilcoxon_W']:.1f}, "
                     f"p_holm={info['wilcoxon_p_holm']:.4f}, "
                     f"delta={info['cliffs_delta']:.2f}, dz={info['cohens_dz']:.2f}, "
                     f"N={info['n']}{bf}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="study_data_template.csv")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(exist_ok=True)

    df = pd.read_csv(args.data)
    df["condition"] = df["condition"].str.strip()
    main_df = df[df.condition.isin(CONDITIONS)].copy()
    base_df = df[df.condition.isin(["RAVE_bare", "AudioLDM_t2a"])].copy()

    report = ["# Botanica-OS 再分析结果（修改稿 v2 · 表7 回填来源）",
              f"# 随机种子 SEED={SEED}（Bootstrap）；数据：{args.data}", ""]

    # 1) 主分析
    all_rows = []
    for m in METRICS:
        r = friedman_and_posthoc(main_df, m)
        report.append(fmt_res(r)); report.append("")
        for (a, b), info in r["pairs"].items():
            all_rows.append({"metric": m, "comparison": f"{a} vs {b}",
                             **{k: v for k, v in info.items()}})
        all_rows.append({"metric": m, "comparison": "Friedman(3 conditions)",
                         "friedman_chi2": r["friedman_chi2"], "friedman_p": r["friedman_p"],
                         **{f"{c}_M": r[f"{c}_M"] for c in CONDITIONS},
                         **{f"{c}_CI": r[f"{c}_CI"] for c in CONDITIONS}})
    pd.DataFrame(all_rows).to_csv(out / "table7_backfill.csv", index=False, encoding="utf-8-sig")

    # 2) 顺序/疲劳
    of = order_fatigue(main_df)
    report.append("## 顺序与疲劳效应\n" + "\n".join(f"- {k}: {v}" for k, v in of.items()) + "")
    pd.DataFrame([of]).to_csv(out / "order_fatigue.csv", index=False)

    # 3) ITT
    itt = itt_sensitivity(main_df)
    report.append("## ITT 敏感性（Cultivate 保守插补）\n" +
                  "\n".join(f"- {k}: {v}" for k, v in itt.items()) + "")
    pd.DataFrame([itt]).to_csv(out / "itt_sensitivity.csv", index=False)

    # 4) LMM（可选）
    for m in ["sus_overall", "agency"]:
        s = lmm_sensitivity(main_df, m)
        if s is not None:
            report.append(f"## LMM 敏感性（{m}）\n```\n{s}\n```")

    # 5) 子组与基线（探索性）
    subgroup_exploratory(main_df).to_csv(out / "subgroup_exploratory.csv", index=False, encoding="utf-8-sig")
    if not base_df.empty:
        baseline_compare(pd.concat([main_df, base_df])).to_csv(
            out / "baseline_exploratory.csv", index=False, encoding="utf-8-sig")

    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\n[ok] 结果已写入 {out.resolve()}")


if __name__ == "__main__":
    main()
