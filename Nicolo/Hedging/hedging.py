import dataclasses

import numpy as np
import pandas as pd

from pricing import fixing_state, mm_price
from vol import black76, quote_ok


def shift(smiles, dv):
    return tuple((lambda m, s=s, d=d: s(m) + d) for s, d in zip(smiles, dv))


def structure_greeks(st, k_put, F, sm, rho, fx, p):
    """Delta and parallel-smile Vega of the market-maker position (long put, short call) under sticky-moneyness."""
    def val(f_vec, smiles):
        return st.notional * (
            mm_price(k_put, False, f_vec, smiles, rho, fx)
            - mm_price(st.k_call, True, f_vec, smiles, rho, fx)
        )

    delta, vega = np.empty(2), np.empty(2)
    for i in range(2):
        df = np.zeros(2); df[i] = p.bump_f
        dv = np.zeros(2); dv[i] = p.bump_v
        delta[i] = (val(F + df, sm) - val(F - df, sm)) / (2 * p.bump_f)
        vega[i] = (val(F, shift(sm, dv)) - val(F, shift(sm, -dv))) / (2 * p.bump_v)
    return delta, vega


def option_greeks(row, F_i, smile_i, p):
    """Delta and Vega of a single hedge option under sticky-moneyness."""
    tau = max(float(row.tau), 0.0)
    is_call = row.cp == "C"

    def val(f, dv):
        return black76(f, row.strike, (smile_i(row.strike / f) + dv) * np.sqrt(tau), is_call)

    delta = (val(F_i + p.bump_f, 0.0) - val(F_i - p.bump_f, 0.0)) / (2 * p.bump_f)
    vega = (val(F_i, p.bump_v) - val(F_i, -p.bump_v)) / (2 * p.bump_v)
    return delta, vega


def pick_option(chain, leg, F, held, p):
    """[DECIDE] vega hedge with ONE ATM call per leg (no puts/strangles).
    However the vega hedge costs a lot in bid/ask (about 0.5M USD of costs over the backtest) but greatly reduces P&L variance; to switch it off, return None.
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


def mark_pos(tk, q, chain, d, F, sm):
    """Mark an open option position: market quote if listed, intrinsic if expired, model price if missing."""
    i = q["leg"]
    if tk in chain.index:
        return mark(chain.loc[tk], F[i], sm[i])
    if d >= q["exp"]:                                   # expired: intrinsic value
        return max(F[i] - q["K"], 0.0) if q["cp"] == "C" else max(q["K"] - F[i], 0.0)
    tau = max((q["exp"] - d).days, 0) / 365             # row missing in the data: model mark, never a stale price
    return float(black76(F[i], q["K"], sm[i](q["K"] / F[i]) * np.sqrt(tau), q["cp"] == "C"))


def trade_price(row, qty, fallback):
    """Buy at ask, sell at bid; at the mark if the quote is missing."""
    if row is not None and np.isfinite(row.bid) and np.isfinite(row.ask):
        return row.ask if qty > 0 else row.bid
    return fallback


def decide(st, p, k_put, d, F, sm, rho, fx, chain, pos):
    """Target hedge: options (vega) per leg with no-trade band tol, then futures (residual delta)."""
    if d == st.end:
        return {}, np.zeros(2)

    delta_s, vega_s = structure_greeks(st, k_put, F, sm, rho, fx, p)
    n_fut = -delta_s
    book = {}

    for i, leg in enumerate(st.legs):
        held = next((tk for tk, q in pos.items() if q["leg"] == i), None)
        tk = pick_option(chain, leg, F[i], held, p)
        if tk is None:
            continue
        row = chain.loc[tk]
        delta_o, vega_o = option_greeks(row, F[i], sm[i], p)
        if abs(vega_o) < 1e-4:
            continue
        target_n = -vega_s[i] / vega_o
        held_n = pos[tk]["n"] if tk == held else None
        n_opt = held_n if (held_n is not None and abs(target_n - held_n) <= p.opt_tol * abs(target_n)) else target_n
        n_fut[i] -= n_opt * delta_o
        book[tk] = dict(leg=i, n=n_opt, K=row.strike, cp=row.cp, exp=row.expiration)

    return book, n_fut


def backtest(st, p, k_put, fut, sofr, chains, smiles, rho):
    """Daily loop: accrue -> mark -> reprice -> rebalance with bid/ask costs -> record wealth.
    Structure valued at model price (no bid/ask on the collar); hedge options bought at ask / sold at bid and marked at settlement; futures: cost = half spread, margins ignored;
    continuous quantities (no lot rounding); everything is closed on the last fixing date paying the spread; the first-day P&L is not part of the metric."""
    legs = st.legs
    cash, nF, pos, prev, rows = 0.0, np.zeros(2), {}, None, []
    for d in [d for d in sorted(chains) if st.start <= d <= st.end]:
        F = st.forwards(fut, d)
        sm = tuple(smiles[d][l] for l in legs)
        chain, r = chains[d], rho.loc[d]
        if prev is not None:
            cash = cash * (1 + sofr.asof(prev) * (d - prev).days / 365) + nF @ (F - F_prev)
        fx = fixing_state(st, fut, d)
        V = st.notional * (mm_price(k_put, False, F, sm, r, fx) - mm_price(st.k_call, True, F, sm, r, fx))

        old_marks = {tk: mark_pos(tk, q, chain, d, F, sm) for tk, q in pos.items()}
        new_pos, n_fut = decide(st, p, k_put, d, F, sm, r, fx, chain, pos)
        new_marks = {tk: mark_pos(tk, q, chain, d, F, sm) for tk, q in new_pos.items()}

        c_opt = c_fut = 0.0
        for tk in set(pos) | set(new_pos):
            n_old = pos[tk]["n"] if tk in pos else 0.0
            n_new = new_pos[tk]["n"] if tk in new_pos else 0.0
            qty = n_new - n_old
            m = new_marks.get(tk, old_marks.get(tk))
            px = trade_price(chain.loc[tk] if tk in chain.index else None, qty, m)
            cash -= qty * px
            c_opt += qty * (px - m)
        for i, l in enumerate(legs):
            b = fut[l].asof(d)
            spread = b.ask - b.bid
            # if the future's bid/ask is missing the cost is zero
            cost = abs(n_fut[i] - nF[i]) * (spread / 2 if np.isfinite(spread) else 0.0)
            c_fut += cost
            cash -= cost

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