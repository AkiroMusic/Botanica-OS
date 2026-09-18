# -*- coding: utf-8 -*-
"""Botanica-OS 三种有机化操作的参考实现。

对应论文：
    式(2)  嫁接 Graft      z_g = clip_{3σ}( α·z_a + (1−α)·z_b )
    式(3)  杂交 Hybridize  z_h = μ_ϕ(c) + σ_ϕ(c)⊙ε,  ε∼N(0, I)
    式(4)  培育 Cultivate  z_{t+1} = clip_{3σ}( z_t + η·∇_z μ̂|_{z_t} + λ·σ̂(z_t;H_t)⊙ε )
    算法 1                GP 预测（Matérn 5/2）+ 一阶差分梯度 + Thompson 式探索项

说明：
- 本文件是论文数学定义的参考实现（仅依赖 numpy），便于复核与教学；
  完整 Botanica-OS 系统（React/FastAPI/PyTorch）另行发布。
- clip_{3σ} 为逐维截断至训练集统计均值 ±3σ（论文 §4.3）。
- 默认超参数与论文表A1 一致：η=0.05、λ=0.3、GP 核 Matérn 5/2、历史截断 100。
"""
from __future__ import annotations

import numpy as np

ETA_DEFAULT = 0.05     # Cultivate 学习率 η
LAMBDA_DEFAULT = 0.3   # Cultivate 探索系数 λ
HISTORY_MAX = 100      # GP 历史截断（论文 §4.3：控制在 20 ms 以内的截断策略）
GRAD_EPS = 1e-3        # 一阶差分步长（算法 1 注：梯度用一阶差分近似）


def clip_3sigma(z: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """逐维截断至训练集统计均值 ±3σ 范围（论文式(2)(4) 的 clip_{3σ}）。

    注意（论文 §4.3）：该截断阻止越界外推，但不保证样本落在训练分布的有效
    支撑上——3σ 是必要不充分条件。
    """
    return np.clip(z, mu - 3.0 * sigma, mu + 3.0 * sigma)


def graft(z_a: np.ndarray, z_b: np.ndarray, alpha: float,
          train_mu: np.ndarray, train_sigma: np.ndarray) -> np.ndarray:
    """式(2)：双父本线性插值 + 3σ 截断。

    alpha∈[0,1]：1.0 返回父本 a，0.0 返回父本 b。插值是父本连线上的欧氏线性
    插值，不是训练分布流形上的测地线（论文 §4.3、§7.2）。
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return clip_3sigma(alpha * z_a + (1.0 - alpha) * z_b, train_mu, train_sigma)


def hybridize_sample(mu_phi: np.ndarray, sigma_phi: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """式(3)：从条件后验 N(μ_ϕ(c), σ_ϕ(c)²I) 采样。

    mu_phi / sigma_phi 由 CLAP 文本嵌入经条件分支 MLP 得到；生产系统中调用方
    需自行处理 OOD 条件回退（论文 §4.3：σ_ϕ(c) 任一分量超过训练集条件后验
    标准差第 99 百分位时判定 OOD——该机制晚于被评估版本，见 §8.1 版本偏差）。
    """
    return mu_phi + sigma_phi * rng.standard_normal(mu_phi.shape)


def matern_5_2(r: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
    """Matérn 5/2 核（论文表A1：GP 核函数）。r 为成对距离。"""
    c = np.sqrt(5.0) * r / length_scale
    return (1.0 + c + c * c / 3.0) * np.exp(-c)


class CultivateGP:
    """基于交互历史 H_t = {(z_i, a_i, r_i)} 的 GP 奖励模型（算法 1）。

    - 目标 r 为归一化后的奖励（论文 §4.3：跨评分者各自均值-标准差标准化后按
      ICC(2,1)=0.71 归一化加权平均；本参考实现假定输入已完成该预处理）。
    - 历史截断至最近 HISTORY_MAX=100 次交互。
    """

    def __init__(self, length_scale: float = 1.0, obs_noise: float = 1e-2):
        self.length_scale = length_scale
        self.obs_noise = obs_noise
        self._z: list[np.ndarray] = []
        self._r: list[float] = []

    def add(self, z: np.ndarray, reward: float) -> None:
        """追加一次交互 (z_i, r_i)，超出截断长度时丢弃最旧记录。"""
        self._z.append(np.asarray(z, dtype=float))
        self._r.append(float(reward))
        if len(self._z) > HISTORY_MAX:
            self._z, self._r = self._z[-HISTORY_MAX:], self._r[-HISTORY_MAX:]

    def __len__(self) -> int:
        return len(self._z)

    def predict(self, z: np.ndarray) -> tuple[float, float]:
        """GP 后验 (μ̂(z;H_t), σ̂(z;H_t))。历史为空时返回 (0, 1)。"""
        if not self._z:
            return 0.0, 1.0
        zs = np.stack(self._z)
        rs = np.asarray(self._r)
        k = matern_5_2(np.linalg.norm(zs - z, axis=1), self.length_scale)
        K = matern_5_2(np.linalg.norm(zs[:, None, :] - zs[None, :, :], axis=-1),
                       self.length_scale)
        K += self.obs_noise * np.eye(len(zs))
        w = np.linalg.solve(K, k)
        mu = float(w @ rs)
        sigma = float(np.sqrt(max(matern_5_2(0.0, self.length_scale) - k @ w,
                                  1e-12)))
        return mu, sigma


def cultivate_step(z_t: np.ndarray, gp: CultivateGP,
                   train_mu: np.ndarray, train_sigma: np.ndarray,
                   rng: np.random.Generator,
                   eta: float = ETA_DEFAULT, lam: float = LAMBDA_DEFAULT,
                   grad_eps: float = GRAD_EPS) -> np.ndarray:
    """式(4)/算法 1：培育一步更新。

    梯度项 η·∇μ̂ 是利用，λ·σ̂⊙ε 是探索：奖励模型不确定（σ̂ 大）的区域主动
    探索，确定（σ̂ 小）的区域利用已知梯度（论文 §5.6）。
    """
    z_t = np.asarray(z_t, dtype=float)
    mu0, sigma_t = gp.predict(z_t)

    # 一阶差分近似 ∇_z μ̂（算法 1 第 2 行）
    grad = np.empty_like(z_t)
    for i in range(z_t.size):
        zp = z_t.copy(); zp[i] += grad_eps
        zm = z_t.copy(); zm[i] -= grad_eps
        grad[i] = (gp.predict(zp)[0] - gp.predict(zm)[0]) / (2.0 * grad_eps)

    z_next = z_t + eta * grad + lam * sigma_t * rng.standard_normal(z_t.shape)
    return clip_3sigma(z_next, train_mu, train_sigma)
