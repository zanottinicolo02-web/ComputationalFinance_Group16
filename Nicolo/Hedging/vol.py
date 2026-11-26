import dataclasses

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.stats import norm


def black76(F, K, v, call=True):
    """Undiscounted Black-76 with total vol v = sigma*sqrt(T); intrinsic value if v=0 or K<=0."""
    F, K, v = np.broadcast_arrays(*(np.asarray(x, float) for x in (F, K, v)))
    ok = (v > 1e-12) & (K > 0) & (F > 0)
    with np.errstate(all="ignore"):
        d1 = (np.log(F / K) + 0.5 * v**2) / v
        d2 = d1 - v
        px = F * norm.cdf(d1) - K * norm.cdf(d2) if call else K * norm.cdf(-d2) - F * norm.cdf(-d1)
    intrinsic = np.maximum(F - K, 0) if call else np.maximum(K - F, 0)
    out = np.where(ok, px, intrinsic)
    return out if out.ndim else float(out)


def implied_vol(price, F, K, T, call):
    """Asay (1982): invert undiscounted Black-76 (futures-style margining, r = 0).
    No early-exercise premium: with margined premium and r=0, early exercise is never optimal (convex payoff, martingale underlying), so American = European."""
    f = lambda s: black76(F, K, s * np.sqrt(T), call) - price
    return brentq(f, 1e-4, 5.0, xtol=1e-10) if f(1e-4) < 0 < f(5.0) else np.nan


def quote_ok(df, p):
    rel = (df.ask - df.bid) / df.mid
    return (df.bid >= p.min_bid) & (df.ask >= df.bid) & (df.mid > 0) & (rel <= p.max_rel_spread) & (df.days >= p.min_days)


def fit_smile(x, vol, kind):
    """Smile as a function of moneyness m = K/F, flat outside the calibrated range."""
    s = pd.Series(vol, index=x).groupby(level=0).mean()
    x, vol = s.index.values, s.values
    if len(x) == 1:
        return lambda m: np.full_like(m, vol[0], dtype=float) if np.ndim(m) else float(vol[0])
    if kind == "pchip":
        g = PchipInterpolator(x, vol)
    else:
        c = np.polyfit(x, vol, 2)
        g = lambda z: np.polyval(c, z)
    lo, hi = x[0], x[-1]
    return lambda m: g(np.clip(np.log(m), lo, hi))


def _implied_otm(opts, legs, mask):
    otm = np.where(opts.cp == "C", opts.strike >= opts.F, opts.strike < opts.F)
    o = opts[opts.leg.isin(legs) & mask & otm].copy()
    # IV from the bid/ask mid (settlement ~ mid in the data); OTM options only (puts below F, calls above)
    o["iv"] = [implied_vol(r.mid, r.F, r.strike, r.tau, r.cp == "C") for r in o.itertuples()]
    return o.dropna(subset=["iv"])


def build_smiles(opts, legs, p):
    """Daily smile per leg from liquid OTM options. Missing days: last valid smile carried forward;
    leading days without a valid smile: relaxed liquidity filter, then back-fill from the first valid one."""
    grid = (opts.strike % p.strike_step == 0) if p.strike_step else True
    o = _implied_otm(opts, legs, quote_ok(opts, p) & grid)
    relaxed = dataclasses.replace(p, min_bid=p.min_bid / 5, max_rel_spread=2 * p.max_rel_spread)
    o_rel = _implied_otm(opts, legs, quote_ok(opts, relaxed) & grid)
    groups = {k: g for k, g in o.groupby(["date", "leg"])}
    groups_rel = {k: g for k, g in o_rel.groupby(["date", "leg"])}
    dates = sorted(opts.date.unique())
    fits = {}
    for d in dates:
        for leg in legs:
            for src in (groups, groups_rel):
                g = src.get((d, leg))
                if g is not None and len(g) >= p.min_points:
                    fits[d, leg] = fit_smile(np.log(g.strike / g.F).values, g.iv.values, p.smile)
                    break
    # After the options expire (e.g. COJ21: 23 Feb 2021) the smile stays the last valid one
    by_leg = {}
    for leg in legs:
        s = pd.Series({d: fits[d, leg] for d in dates if (d, leg) in fits}, index=dates, dtype=object)
        if s.isna().all():
            raise ValueError(f"no valid smile for {leg}")
        by_leg[leg] = s.ffill().bfill()
    smiles = {d: {leg: by_leg[leg].loc[d] for leg in legs} for d in dates}
    return smiles, o


def rolling_rho(fut, legs, p, dates):
    """Rolling correlation of daily log-returns of the two futures (information up to t only)."""
    px = pd.concat([fut[l].settle for l in legs], axis=1, keys=legs)
    r = np.log(px.sort_index()).diff().dropna()
    rho = r[legs[0]].rolling(p.rho_window, min_periods=p.rho_min_obs).corr(r[legs[1]])
    return rho.reindex(dates).ffill().fillna(p.rho_init).clip(-0.9999, 0.9999)
