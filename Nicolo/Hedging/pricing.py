from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from vol import black76

Fix = namedtuple("Fix", "R theta leg w")   # realized sum, years to future fixings, their leg, their weight


def fixing_state(st, fut, t):
    """Split the fixings into realized (t_k <= t) and future ones."""
    d, leg = st.fix_dates, st.fix_leg
    w, past = 1 / len(d), d <= t
    # If the settlement is missing on a fixing date the last available one is used (asof)
    R = sum((w * fut[st.legs[l]].settle.asof(k) for k, l in zip(d[past], leg[past])), 0.0)
    return Fix(R, (d[~past] - t).days.values / 365, leg[~past], np.full((~past).sum(), w))


def _vols(smiles, m):
    return np.array([s(m) for s in smiles])


def _cov(vol, rho, fx):
    s = vol[fx.leg]
    same = np.equal.outer(fx.leg, fx.leg)
    return np.outer(s, s) * np.where(same, 1.0, rho) * np.minimum.outer(fx.theta, fx.theta)


def _basket_inputs(K, f, smiles, rho, fx):
    Ke, a = K - fx.R, fx.w * f[fx.leg]
    vol = _vols(smiles, max(Ke, 1e-8) / a.sum())
    return Ke, a, vol, _cov(vol, rho, fx)


def mm_price(K, call, f, smiles, rho, fx):
    """Moment matching: lognormal fit of the future part of the average + Black-76."""
    # No discount D(t,T): futures-style margined premia (not paid upfront); D is common to call and put, so K_put is almost unchanged
    # With fixings already realized the smile is read at moneyness (K-R)/M1_future (generalises eq. 5; in the note when no fixing has occurred)
    if fx.theta.size == 0:
        return max(fx.R - K, 0.0) if call else max(K - fx.R, 0.0)
    Ke, a, _, S = _basket_inputs(K, f, smiles, rho, fx)
    M1 = a.sum()
    var = a @ np.expm1(S) @ a                        # eq. (14)-(18) in matrix form
    return black76(M1, Ke, np.sqrt(np.log1p(var / M1**2)), call)


def zero_cost_put(k_call, f, smiles, rho, fx):
    c = mm_price(k_call, True, f, smiles, rho, fx)
    h = lambda K: mm_price(K, False, f, smiles, rho, fx) - c
    return brentq(h, 1e-3, 2 * max((fx.w * f[fx.leg]).sum(), k_call), xtol=1e-8)


def simulate(K, f, smiles, rho, fx, M, seed):
    """Exact bivariate lognormal sampling at the fixing dates + geometric-average control variate."""
    Ke, a, vol, S = _basket_inputs(K, f, smiles, rho, fx)
    mu = -0.5 * vol[fx.leg]**2 * fx.theta
    L = np.linalg.cholesky(S + 1e-14 * np.eye(len(mu)))
    eta = mu + np.random.default_rng(seed).standard_normal((M, len(mu))) @ L.T
    A = (a * np.exp(eta)).sum(1)
    W = fx.w.sum()
    g = fx.w / W
    lnG = np.log(W) + g @ np.log(f[fx.leg]) + eta @ g
    assert np.all(A >= np.exp(lnG) - 1e-9), "AM-GM violated"
    mG = np.log(W) + g @ (np.log(f[fx.leg]) + mu)
    vG2 = g @ S @ g
    return A, np.exp(lnG), np.exp(mG + vG2 / 2), np.sqrt(vG2), Ke

def mc_estimate(sim, call, control=True):
    A, G, Fgeo, vG, Ke = sim
    pay = (lambda X: np.maximum(X - Ke, 0)) if call else (lambda X: np.maximum(Ke - X, 0))
    x = pay(A) - pay(G) if control else pay(A)
    return x.mean() + (black76(Fgeo, Ke, vG, call) if control else 0.0), x.std(ddof=1) / np.sqrt(len(x))


def mc_price(K, call, f, smiles, rho, fx, M, seed=0, control=True):
    return mc_estimate(simulate(K, f, smiles, rho, fx, M, seed), call, control)


def validate_black(M, seed=0, F=60.0, sigma=0.35, K=62.0):
    """Single fixing at T on one leg: MM and MC must collapse to Black-76."""
    rows, flat = [], (lambda m: sigma,) * 2
    for T in (0.1, 0.25, 0.5, 1.0):
        fx = Fix(0.0, np.array([T]), np.array([0]), np.array([1.0]))
        for call in (True, False):
            bl = black76(F, K, sigma * np.sqrt(T), call)
            mc, se = mc_price(K, call, np.array([F, F]), flat, 0.99, fx, M, seed, control=False)
            mm = mm_price(K, call, np.array([F, F]), flat, 0.99, fx)
            rows.append(dict(T=T, call=call, black=bl, mm=mm, mc=mc, se=se, z=(mc - bl) / se))
    return pd.DataFrame(rows)


def mc_convergence(K, call, f, smiles, rho, fx, M, seed=0):
    """Running MC estimate and its standard error as paths accumulate."""
    A, G, Fgeo, vG, Ke = simulate(K, f, smiles, rho, fx, M, seed)
    pay = (lambda X: np.maximum(X - Ke, 0)) if call else (lambda X: np.maximum(Ke - X, 0))
    x = pay(A) - pay(G)
    n = np.unique(np.geomspace(100, M, 60).astype(int))
    mean = np.cumsum(x)[n - 1] / n
    se = np.array([x[:k].std(ddof=1) / np.sqrt(k) for k in n])
    return n, mean + black76(Fgeo, Ke, vG, call), se
