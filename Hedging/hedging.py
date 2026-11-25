import dataclasses

import numpy as np
import pandas as pd

from pricing import fixing_state, mm_price
from vol import black76, quote_ok


def shift(smiles, dv):
    return tuple((lambda m, s=s, d=d: s(m) + d) for s, d in zip(smiles, dv))


def gradient(fun, x, h):
    """Central finite differences of fun w.r.t. x = (f1, f2, dvol1, dvol2).
    [LIMITATION] delta/vega under a sticky-moneyness smile (consistent with the strike mapping); vega = parallel shift of each leg's smile."""
    g = np.empty(len(x))
    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = h[i]
        g[i] = (fun(x + e) - fun(x - e)) / (2 * h[i])
    return g


def structure_fun(st, k_put, smiles, rho, fx):
    """Market-maker position (long put, short call) as a function of (f1, f2, dvol1, dvol2)."""
    def fun(x):
        sm = shift(smiles, x[2:])
        return st.notional * (mm_price(k_put, False, x[:2], sm, rho, fx) - mm_price(st.k_call, True, x[:2], sm, rho, fx))
    return fun


def option_fun(i, row, smile):
    def fun(x):
        F = x[i]
        return black76(F, row.strike, (smile(row.strike / F) + x[2 + i]) * np.sqrt(max(float(row.tau), 0.0)), row.cp == "C")
    return fun


def pick_option(chain, leg, F, held, p):
    """[DECIDE - HIGH impact] vega hedge with ONE ATM call per leg (no puts/strangles); my choice.
    [LIMITATION] the vega hedge costs a lot in bid/ask (about 0.5M USD of costs over the backtest) but greatly reduces P&L variance; to switch it off, return None.
    ATM call of the leg; the held one is kept while within the moneyness band (and a looser spread limit)."""
    c = chain[(chain.leg == leg) & (chain.cp == "C")]
    if p.strike_step:                                   # same strike grid as the smile calibration
        c = c[c.strike % p.strike_step == 0]
    if held in c.index:
        h = c.loc[[held]]
        loose = dataclasses.replace(p, max_rel_spread=1.5 * p.max_rel_spread)
        if quote_ok(h, loose).iloc[0] and abs(np.log(h.strike.iloc[0] / F)) <= p.hedge_band:
            return held
    c = c[quote_ok(c, p)]
    return None if c.empty else (c.strike - F).abs().idxmin()


def mark(row, F, smile):
    """Settlement, else mid, else model price."""
    for v in (row.settle, row.mid):
        if np.isfinite(v):
            return float(v)
    return float(black76(F, row.strike, smile(row.strike / F) * np.sqrt(max(float(row.tau), 0.0)), row.cp == "C"))


def targets(gS, gO, held, tol):
    """Neutralise (delta1, delta2, vega1, vega2): options for vega (no-trade band tol), futures for the residual delta."""
    n_opt, n_fut = np.zeros(2), -gS[:2].copy()
    for i, g in enumerate(gO):
        if g is None or abs(g[2 + i]) < 1e-4:
            continue
        n = -gS[2 + i] / g[2 + i]
        n_opt[i] = held[i] if held[i] is not None and abs(n - held[i]) <= tol * abs(n) else n
        n_fut -= n_opt[i] * g[:2]
    return n_opt, n_fut


def trade_price(row, qty, fallback):
    """Buy at ask, sell at bid; at the mark if the quote is missing."""
    if row is not None and np.isfinite(row.bid) and np.isfinite(row.ask):
        return row.ask if qty > 0 else row.bid
    return fallback


def decide(st, p, k_put, d, F, sm, rho, fx, chain, pos):
    """Target hedge: options (vega) per leg, then futures (residual delta)."""
    x0, h = np.r_[F, 0, 0], [p.bump_f] * 2 + [p.bump_v] * 2
    if d == st.end:
        return {}, np.zeros(2)
    gS = gradient(structure_fun(st, k_put, sm, rho, fx), x0, h)
    chosen, gO, held_n = {}, [None, None], [None, None]
    for i, leg in enumerate(st.legs):
        held = next((tk for tk, q in pos.items() if q["leg"] == i), None)
        tk = pick_option(chain, leg, F[i], held, p)
        if tk is not None:
            chosen[i] = tk
            gO[i] = gradient(option_fun(i, chain.loc[tk], sm[i]), x0, h)
            held_n[i] = pos[tk]["n"] if tk == held else None
    n_opt, n_fut = targets(gS, gO, held_n, p.opt_tol)
    book = {tk: dict(leg=i, n=n_opt[i], K=chain.loc[tk, "strike"], cp=chain.loc[tk, "cp"],
                     exp=chain.loc[tk, "expiration"]) for i, tk in chosen.items()}
    return book, n_fut


def backtest(st, p, k_put, fut, sofr, chains, smiles, rho):
    """Daily loop: accrue -> mark -> reprice -> rebalance with bid/ask costs -> record wealth.
    [ASSUMPTION] structure valued at model price (no bid/ask on the collar); hedge options bought at ask / sold at bid and marked at settlement; futures: cost = half spread, margins ignored;
    continuous quantities (no lot rounding); everything is closed on the last fixing date paying the spread; the first-day P&L is not part of the metric."""
    legs = st.legs
    cash, nF, pos, prev, rows = 0.0, np.zeros(2), {}, None, []
    for d in [d for d in sorted(chains) if st.start <= d <= st.end]:
        F = np.array([fut[l].settle.asof(d) for l in legs])
        sm = tuple(smiles[d][l] for l in legs)
        chain, r = chains[d], rho.loc[d]
        if prev is not None:
            cash = cash * (1 + sofr.asof(prev) * (d - prev).days / 365) + nF @ (F - F_prev)
        fx = fixing_state(st, fut, d)
        V = structure_fun(st, k_put, sm, r, fx)(np.r_[F, 0, 0])

        def mark_of(tk, q):
            if tk in chain.index:
                return mark(chain.loc[tk], F[q["leg"]], sm[q["leg"]])
            if d >= q["exp"]:                                   # expired: intrinsic value
                return max(F[q["leg"]] - q["K"], 0.0) if q["cp"] == "C" else max(q["K"] - F[q["leg"]], 0.0)
            tau = max((q["exp"] - d).days, 0) / 365         # row missing in the data: model mark, never a stale price
            return float(black76(F[q["leg"]], q["K"], sm[q["leg"]](q["K"] / F[q["leg"]]) * np.sqrt(tau), q["cp"] == "C"))

        old_marks = {tk: mark_of(tk, q) for tk, q in pos.items()}
        c_opt = c_fut = 0.0
        new_pos, n_fut = decide(st, p, k_put, d, F, sm, r, fx, chain, pos)
        new_marks = {tk: mark_of(tk, q) for tk, q in new_pos.items()}
        for tk in set(pos) | set(new_pos):
            qty = new_pos.get(tk, {}).get("n", 0.0) - pos.get(tk, {}).get("n", 0.0)
            m = new_marks.get(tk, old_marks.get(tk))
            px = trade_price(chain.loc[tk] if tk in chain.index else None, qty, m)
            cash -= qty * px
            c_opt += qty * (px - m)
        for i, l in enumerate(legs):
            b = fut[l].asof(d)
            spread = b.ask - b.bid
            # [ASSUMPTION] if the future's bid/ask is missing the cost is zero
            c_fut += abs(n_fut[i] - nF[i]) * (spread / 2 if np.isfinite(spread) else 0.0)
            cash -= abs(n_fut[i] - nF[i]) * (spread / 2 if np.isfinite(spread) else 0.0)
        for tk, q in new_pos.items():
            q["last"] = new_marks[tk]
        pos, nF = new_pos, n_fut
        W = cash + V + sum(q["n"] * q["last"] for q in pos.values())
        rows.append(dict(date=d, W=W, V=V, cash=cash, nF1=nF[0], nF2=nF[1], rho=r, cost_opt=c_opt, cost_fut=c_fut,
                         nO1=sum(q["n"] for q in pos.values() if q["leg"] == 0),
                         nO2=sum(q["n"] for q in pos.values() if q["leg"] == 1)))
        prev, F_prev = d, F
    res = pd.DataFrame(rows).set_index("date")
    res["pnl"] = res.W.diff()
    return res


def ranking_metric(res):
    return float(np.mean(res.pnl.dropna() ** 2))
