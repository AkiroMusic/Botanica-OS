# -*- coding: utf-8 -*-
"""botanica_ops 参考实现的单元测试：run with `python -m pytest reference/` 或直接运行。"""
import numpy as np

from botanica_ops import (CultivateGP, HISTORY_MAX, clip_3sigma, cultivate_step,
                          graft, hybridize_sample, matern_5_2)

D = 16
MU = np.zeros(D)
SIGMA = np.ones(D)
rng = np.random.default_rng(42)


def test_clip_3sigma_bounds():
    z = MU + 10.0 * SIGMA
    out = clip_3sigma(z, MU, SIGMA)
    assert np.all(out == 3.0 * SIGMA)
    assert np.all(np.abs(out) <= 3.0 * SIGMA + 1e-12)


def test_graft_endpoints_and_midpoint():
    za, zb = rng.normal(size=D), rng.normal(size=D)
    assert np.allclose(graft(za, zb, 1.0, MU, SIGMA), clip_3sigma(za, MU, SIGMA))
    assert np.allclose(graft(za, zb, 0.0, MU, SIGMA), clip_3sigma(zb, MU, SIGMA))
    mid = graft(za, zb, 0.5, MU, SIGMA)
    expected = clip_3sigma(0.5 * za + 0.5 * zb, MU, SIGMA)
    assert np.allclose(mid, expected)


def test_graft_alpha_range():
    za, zb = rng.normal(size=D), rng.normal(size=D)
    try:
        graft(za, zb, 1.5, MU, SIGMA)
        raise AssertionError("alpha>1 should raise")
    except ValueError:
        pass


def test_hybridize_sample_shape_and_seed():
    mu = rng.normal(size=D)
    sig = np.full(D, 0.5)
    a = hybridize_sample(mu, sig, np.random.default_rng(0))
    b = hybridize_sample(mu, sig, np.random.default_rng(0))
    assert a.shape == (D,)
    assert np.allclose(a, b)  # 同种子可复现


def test_matern_5_2_properties():
    assert np.isclose(matern_5_2(np.array(0.0)), 1.0)
    r = np.linspace(0, 5, 50)
    k = matern_5_2(r)
    assert np.all(k >= 0) and np.all(np.diff(k) <= 1e-12)  # 非负且单调衰减


def test_cultivate_gp_truncation():
    gp = CultivateGP()
    for i in range(HISTORY_MAX + 25):
        gp.add(rng.normal(size=D), float(i % 7) / 6.0)
    assert len(gp) == HISTORY_MAX  # 截断至最近 100 次


def test_cultivate_gp_empty_history():
    gp = CultivateGP()
    mu, sigma = gp.predict(np.zeros(D))
    assert mu == 0.0 and sigma == 1.0


def test_cultivate_step_respects_bounds_and_deterministic_when_certain():
    gp = CultivateGP()
    # 完全确定的奖励模型（σ̂→极小）时，探索项消失，同种子结果可复现
    for i in range(50):
        z = np.full(D, -0.5 + 0.02 * i)
        gp.add(z, float(np.sum(np.abs(z))))
    z_t = np.zeros(D)
    a = cultivate_step(z_t, gp, MU, SIGMA, np.random.default_rng(0))
    b = cultivate_step(z_t, gp, MU, SIGMA, np.random.default_rng(0))
    assert a.shape == (D,)
    assert np.all(np.abs(a) <= 3.0 * SIGMA + 1e-12)
    assert np.allclose(a, b)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
