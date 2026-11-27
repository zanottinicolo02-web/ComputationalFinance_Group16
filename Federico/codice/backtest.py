"""
Backtest della copertura, giorno per giorno, dall'emissione all'ultimo fixing.

Ogni giorno t (alla chiusura, sui prezzi di settlement):
  1. si ricalibrano smile e correlazione con i dati del giorno (già fatto in
     prepare(): non dipende dalla strategia);
  2. si riprezza il collar (moment matching) e se ne calcolano le greche;
  3. si aggiornano i conti: variation margin dei futures e delle opzioni
     (margining futures-style: il guadagno del giorno entra subito in cassa),
     interessi SOFR sulla cassa;
  4. si ribilancia la copertura secondo la strategia, pagando metà spread
     bid-ask su ogni contratto scambiato (si compra all'ask e si vende al bid,
     mentre le posizioni sono valutate al mid/settlement).

Valore del portafoglio:  Π_t = V_t + B_t
  V_t = valore del collar per il market maker (long put, short call) su 1M bbl
  B_t = conto cassa (i futures e le opzioni futures-style valgono zero dopo il
        regolamento giornaliero: tutto il loro P&L è già in B)
P&L del giorno:  P&L_t = Π_t - Π_{t-1}   (il giorno 0 vale -costi iniziali, perché Π_{-1} = 0)
Metrica della competizione:  (1/T) Σ_t P&L_t²

All'ultimo fixing tutte le medie sono note: V_T è il payoff, che viene pagato,
e tutte le coperture vengono chiuse.

Scomposizione del P&L (per spiegare la metrica):
  prezzo     = variazione del collar dovuta ai futures + P&L delle coperture dovuto ai futures
  volatilità = variazione del collar dovuta al nuovo smile + resto del P&L delle opzioni
  tempo      = passare del tempo e fixing che diventano noti (theta)
  interessi  = SOFR sulla cassa
  costi      = metà spread bid-ask pagato sugli scambi
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from config import CollarConfig
from corr import ewma_correlation
from data import MarketData, load_market_data
from fixing_calendar import FixingSchedule, build_fixing_schedule, year_frac
from greeks import CollarGreeks, collar_greeks, option_delta, option_delta_smile, option_vega_point
from pricer import MarketState, build_state, collar_value, solve_k_put
from strategy import HedgeStrategy
from vol import black76_price, calibrate_smiles, compute_implied_vols


# ------------------------------------------------------------------ preparazione
@dataclass
class Context:
    """Tutto ciò che non dipende dalla strategia: dati, smile, Kput, greche giornaliere."""
    cfg: CollarConfig
    md: MarketData
    iv: pd.DataFrame
    smiles: dict
    rho: pd.Series
    sched: FixingSchedule
    days: pd.DatetimeIndex
    k_put: float
    call_price: float               # prezzo per barile della call all'emissione (= put)
    states: list = field(default_factory=list)
    greeks: list = field(default_factory=list)
    dV_price: np.ndarray = None     # componenti della variazione del collar (in $)
    dV_vol: np.ndarray = None
    fut_settle: pd.DataFrame = None
    fut_spread: pd.DataFrame = None


def prepare(cfg: CollarConfig, md: MarketData | None = None) -> Context:
    md = md or load_market_data(cfg)
    iv = compute_implied_vols(md, cfg)
    smiles = calibrate_smiles(iv, cfg)
    rho = ewma_correlation(md, cfg)
    sched = build_fixing_schedule(cfg, md.option_expiry(cfg.front_contract))
    days = md.dates[(md.dates >= cfg.emission_date) & (md.dates <= sched.last_fixing)]

    st0 = build_state(days[0], md, sched, smiles, rho, cfg)
    k_put, c0 = solve_k_put(st0, cfg.k_call)

    ctx = Context(cfg, md, iv, smiles, rho, sched, days, k_put, c0)
    ctx.fut_settle = md.futures_wide("settle")[list(cfg.legs)]
    ctx.fut_spread = md.futures_wide("spread")[list(cfg.legs)]
    N = cfg.notional_bbl
    for d in days:
        st = build_state(d, md, sched, smiles, rho, cfg)
        ctx.states.append(st)
        ctx.greeks.append(collar_greeks(st, k_put, cfg.k_call, N))

    # scomposizione della variazione giornaliera del collar
    n = len(days)
    ctx.dV_price, ctx.dV_vol = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        s0, s1 = ctx.states[t - 1], ctx.states[t]
        df = [(b - a) if np.isfinite(a) and np.isfinite(b) else 0.0 for a, b in zip(s0.f, s1.f)]
        v_price = N * collar_value(s0.bump(df1=df[0], df2=df[1]), k_put, cfg.k_call)
        ctx.dV_price[t] = v_price - ctx.greeks[t - 1].value
        v_oldvol = N * collar_value(replace(s1, smiles=s0.smiles), k_put, cfg.k_call)
        ctx.dV_vol[t] = ctx.greeks[t].value - v_oldvol
    return ctx


# --------------------------------------------------------------- opzioni di copertura
def _hedge_option_candidates(ctx: Context, d: pd.Timestamp, strat: HedgeStrategy, leg: int = 1) -> pd.DataFrame:
    """Opzioni su una gamba negoziabili oggi (quotate, spread accettabile, scadenza lontana).

    Sono ammesse anche le ITM: un'opzione già in portafoglio che diventa ITM si
    tiene (stessa vega della gemella OTM, e cambiarla costerebbe due spread).
    Quando se ne sceglie una nuova si preferisce la OTM.
    """
    o = ctx.iv[(ctx.iv.date == d) & (ctx.iv.contract == ctx.cfg.legs[leg])]
    return o[o.bid.notna() & (o.bid > 0) & (o.rel_spread <= strat.option_max_rel_spread)
             & (o.ttm_days >= strat.option_min_ttm_days)]


def _option_quote(ctx: Context, d: pd.Timestamp, o: dict, state: MarketState):
    """(prezzo, spread, ttm, vol) dell'opzione o nel giorno d.

    Se oggi non è quotata, il prezzo si ricava dallo smile della sua gamba
    (Black-76) e lo spread resta quello dell'ultimo giorno in cui era quotata.
    """
    leg, strike, is_call = o["leg"], o["K"], o["call"]
    row = ctx.iv[(ctx.iv.date == d) & (ctx.iv.option_ticker == o["ticker"])]
    F = state.f[leg]
    smile = state.smiles[leg] if state.smiles[leg] is not None else state.smiles[1 - leg]
    if len(row) and pd.notna(row.settle.iloc[0]):
        r = row.iloc[0]
        vol = r.iv_mid if pd.notna(r.iv_mid) else float(smile.vol(strike, F))
        sp = r.spread if pd.notna(r.spread) else np.nan
        return float(r.settle), float(sp), float(r.ttm), float(vol)
    if not np.isfinite(F):
        return np.nan, np.nan, np.nan, np.nan
    if len(row):
        ttm = float(row.ttm.iloc[0])
    else:
        # l'opzione manca dal dataset quel giorno (succede per gli strike lontani, anche
        # per settimane): la si valuta comunque con lo smile, con la scadenza nota.
        # Senza questo il prezzo resterebbe fermo all'ultimo valore visto e, quando
        # l'opzione ricompare, tutto il movimento accumulato finirebbe in un giorno solo.
        exp = o.get("expiry")
        if exp is None or pd.isna(exp):
            return np.nan, np.nan, np.nan, np.nan
        ttm = year_frac(d, exp, ctx.cfg.model.weekend_weight)
        if ttm <= 0:                              # scaduta: vale il payoff
            payoff = max(F - strike, 0.0) if is_call else max(strike - F, 0.0)
            return float(payoff), np.nan, 0.0, o["vol"]
    vol = float(smile.vol(strike, F))
    return float(black76_price(F, strike, ttm, vol, is_call)), np.nan, ttm, vol


def _new_option(row, leg: int) -> dict:
    return dict(ticker=row.option_ticker, K=float(row.strike), call=row.cp == "C", leg=leg, qty=0.0,
                price=float(row.settle), spread=float(row.spread), ttm=float(row.ttm), vol=float(row.iv_mid),
                expiry=row.option_expiry)


def _static_portfolio(ctx: Context, strat: HedgeStrategy) -> list[tuple[dict, float]]:
    """Copertura statica all'emissione: il "collar opposto" fatto con opzioni quotate.

    Il market maker è long put (Kput) e short call (Kcall) sulla media: per
    compensare vende put e compra call quotate sul futures della gamba
    strat.static_leg. Gli strike esatti di solito non sono quotati: si usano i due
    strike quotati più vicini, con pesi lineari (call 78 → 60% sullo strike 80, 40%
    sul 75). Se K è fuori dagli strike quotati si usa il più vicino: all'emissione
    del training la put 40 non è quotata, quindi tutta la put va sullo strike 45.
    Con static_scale = "collar_vega" le quantità si riscalano perché la vega di
    ciascun lato sia uguale a quella della gamba corrispondente del collar.
    Restituisce [(opzione, quantità desiderata)].
    """
    d, leg, N, lot = ctx.days[0], strat.static_leg, ctx.cfg.notional_bbl, ctx.cfg.lot_bbl
    cand = _hedge_option_candidates(ctx, d, strat, leg)
    out = []
    for K, want_call, sign in ((ctx.k_put, False, -1.0), (ctx.cfg.k_call, True, +1.0)):
        c = cand[(cand.cp == ("C" if want_call else "P")) & cand.iv_mid.notna()].sort_values("strike")
        if c.empty:
            continue
        lo, hi = c[c.strike <= K], c[c.strike >= K]
        if len(lo) and len(hi) and lo.strike.iloc[-1] != hi.strike.iloc[0]:
            a, b = lo.iloc[-1], hi.iloc[0]
            w = (K - a.strike) / (b.strike - a.strike)
            parts = [(a, 1 - w), (b, w)]
        else:
            parts = [(c.iloc[(c.strike - K).abs().argsort().iloc[0]], 1.0)]
        scale = 1.0
        if strat.static_scale == "collar_vega":
            # quantità riscalate perché la vega statica sia uguale a quella della gamba del collar
            g0 = ctx.greeks[0]
            target = g0.vega_call if want_call else g0.vega_put
            Fo = ctx.states[0].f[leg]
            v = sum(sign * N * w * option_vega_point(Fo, float(r.strike), float(r.ttm), float(r.iv_mid)) for r, w in parts)
            if v != 0 and np.isfinite(v):
                scale = -strat.static_vega_mult * target / v
        for row, w in parts:
            qty = np.round(sign * N * w * scale / lot) * lot
            if qty != 0:
                out.append((_new_option(row, leg), qty))
    return out


def _adaptive_phase_days(ctx: Context, static_list: list[tuple[dict, float]],
                         k_min: int = 3, k_max: int = 12) -> int:
    """Giorni ottimali k* per costruire la copertura statica sotto penalità quadratica.

    Minimizza C_tot²/k + (k/3)·σ_vega²  =>  k* ≈ √3 · C_tot / σ_vega,
    usando solo dati noti all'emissione (spread iniziali e vol ATM del giorno 0).
    """
    c_tot = sum(abs(qty) * (o["spread"] / 2.0) for o, qty in static_list if np.isfinite(o["spread"]))
    g0, st0 = ctx.greeks[0], ctx.states[0]
    sm0 = st0.smiles[0] or st0.smiles[1]
    atm_v = sm0.atm if sm0 is not None else 0.30
    daily_vol_pts = 0.85 * (atm_v * 100.0) / np.sqrt(252.0)
    leg_vega = max(abs(g0.vega_put), abs(g0.vega_call))
    sigma_vega = max(0.5 * leg_vega * daily_vol_pts, 1.0)
    return int(np.clip(int(np.round(np.sqrt(3.0) * c_tot / sigma_vega)), k_min, k_max))


def _dynamic_targets(strat: HedgeStrategy, ctx: Context, g: CollarGreeks, static_vega: float = 0.0
                    ) -> list[tuple[str, float, float]]:
    """(nome, strike di riferimento, vega da neutralizzare) per ogni opzione dinamica di copertura."""
    if strat.vega_mode == "single":
        if strat.vega_reference == "dominant" and abs(g.vega_call) > abs(g.vega_put):
            k_ref = ctx.cfg.k_call
        else:
            k_ref = ctx.k_put
        return [("dyn", k_ref, g.vega + static_vega)]
    return []


def _opt_delta(strat: HedgeStrategy, o: dict, F: float, st: MarketState) -> float:
    """Delta dell'opzione o: coerente con lo smile (default) o semplice Black-76."""
    if strat.option_delta_smile:
        smile = st.smiles[o["leg"]] if st.smiles[o["leg"]] is not None else st.smiles[1 - o["leg"]]
        return option_delta_smile(F, o["K"], o["ttm"], o["vol"], o["call"], smile)
    return option_delta(F, o["K"], o["ttm"], o["vol"], o["call"])


# ----------------------------------------------------------------------- backtest
@dataclass
class BacktestResult:
    strategy: HedgeStrategy
    daily: pd.DataFrame
    metric: float               # (1/T) Σ P&L²  in $²
    rms_pnl: float              # sqrt(metrica), in $
    total_cost: float           # costi di transazione totali, in $
    final_value: float          # Π alla fine, in $
    n_trades_fut: int
    n_trades_opt: int


def run_backtest(ctx: Context, strat: HedgeStrategy) -> BacktestResult:
    cfg, lot = ctx.cfg, ctx.cfg.lot_bbl
    days = ctx.days
    n = len(days)
    q = np.zeros(2)                 # posizioni in futures (bbl) sulle due gambe
    opts: dict[str, dict] = {}      # opzione dinamica (vega)
    static: list[dict] = []         # opzioni della copertura statica
    B = 0.0                         # conto cassa ($)
    prev_pi = 0.0
    rows = []
    n_tf = n_to = 0
    k_phase = max(strat.static_phase_days, 1)

    def trade_option(o, new_qty, spread):
        nonlocal n_to
        if new_qty == o["qty"]:
            return 0.0
        sp = spread if np.isfinite(spread) else (o["spread"] if np.isfinite(o["spread"]) else 0.0)
        c = abs(new_qty - o["qty"]) * sp / 2
        o["qty"] = new_qty
        n_to += 1
        return c

    for t, d in enumerate(days):
        st, g = ctx.states[t], ctx.greeks[t]
        F = np.array(st.f)
        is_last = t == n - 1
        comp = dict(prezzo=0.0, volatilita=0.0, tempo=0.0, interessi=0.0, costi=0.0)
        held = list(opts.values()) + static

        # --- 1. regolamenti del giorno sulle posizioni di ieri
        if t > 0:
            d0 = days[t - 1]
            interest = B * float(ctx.md.sofr.get(d0, 0.0)) * (d - d0).days / 360.0
            F_prev = np.array(ctx.states[t - 1].f)
            dF = np.where(np.isfinite(F) & np.isfinite(F_prev), F - np.nan_to_num(F_prev), 0.0)
            vm_fut = float((q * dF).sum())
            vm_opt = opt_price_part = 0.0
            for o in held:
                p_new, sp_new, ttm_new, vol_new = _option_quote(ctx, d, o, st)
                if not np.isfinite(p_new):
                    p_new = o["price"]
                vm_opt += o["qty"] * (p_new - o["price"])
                # parte del P&L dell'opzione spiegata dal futures: si riprezza con il nuovo
                # futures, la scadenza di ieri e la vol di ieri spostata lungo lo smile di ieri
                # (stessa convenzione "sticky moneyness" usata per il collar)
                L = o["leg"]
                if np.isfinite(F[L]) and np.isfinite(F_prev[L]) and o["ttm"] > 0:
                    sm_prev = ctx.states[t - 1].smiles[L] or ctx.states[t - 1].smiles[1 - L]
                    v_shift = o["vol"]
                    if strat.option_delta_smile and sm_prev is not None:
                        v_shift = max(o["vol"] + float(sm_prev.vol(o["K"], F[L]))
                                      - float(sm_prev.vol(o["K"], F_prev[L])), 1e-4)
                    p_shift = float(black76_price(F[L], o["K"], o["ttm"], v_shift, o["call"]))
                    p_base = float(black76_price(F_prev[L], o["K"], o["ttm"], o["vol"], o["call"]))
                    opt_price_part += o["qty"] * (p_shift - p_base)
                o["price"] = p_new
                if np.isfinite(ttm_new):
                    o["ttm"], o["vol"] = ttm_new, vol_new
                if np.isfinite(sp_new):
                    o["spread"] = sp_new
            B += interest + vm_fut + vm_opt
            dV = g.value - ctx.greeks[t - 1].value
            comp["prezzo"] = ctx.dV_price[t] + vm_fut + opt_price_part
            comp["volatilita"] = ctx.dV_vol[t] + (vm_opt - opt_price_part)
            comp["tempo"] = dV - ctx.dV_price[t] - ctx.dV_vol[t]
            comp["interessi"] = interest

        cost = 0.0
        opt_delta = np.zeros(2)

        # --- 2a. copertura statica: si costruisce nei primi giorni, si riduce se sovra-coperta
        # vicino alla finestra di media, e si chiude gradualmente vicino alla scadenza.
        if t == 0 and strat.static_hedge:
            sp_list = _static_portfolio(ctx, strat)
            if strat.static_phase_days <= 0:
                k_phase = _adaptive_phase_days(ctx, sp_list)
            for o, qty in sp_list:
                o["target"] = qty
                o["letting_expire"] = False
                static.append(o)
        if t < k_phase and not is_last:
            for o in static:
                if t == 0:
                    o["sp0"] = o["spread"]
                sched = np.round(o["target"] * (t + 1) / k_phase / lot) * lot
                sp = _option_quote(ctx, d, o, st)[1] if t > 0 else o["spread"]
                if not (np.isfinite(sp) and o["ttm"] * 365 >= strat.static_close_days):
                    continue
                step = sched
                if strat.static_opportunistic and t < k_phase - 1:
                    # esecuzione opportunistica: lo spread di una stessa opzione cambia molto da un
                    # giorno all'altro (es. put 45 Apr21: da 0,40 a 0,10 nelle prime settimane). Nei
                    # giorni in cui è più stretto che al giorno 0 si compra di più (fino al doppio
                    # della rata), in quelli in cui è più largo solo quanto previsto dal piano.
                    # Non si resta mai indietro rispetto al piano e l'ultimo giorno si completa.
                    mult = float(np.clip(o["sp0"] / max(sp, 1e-6), 1.0, 2.0))
                    want = o["qty"] + o["target"] / k_phase * mult
                    lo_q, hi_q = sorted((sched, o["target"]))
                    step = np.round(np.clip(want, lo_q, hi_q) / lot) * lot
                if step != o["qty"]:
                    cost += trade_option(o, step, sp)

        # Riduzione delle ali statiche sovra-coperte negli ultimi static_trim_days giorni:
        # man mano che la finestra di media si avvicina/avanza, la vega del collar asiatico
        # scende più in fretta delle opzioni europee; ridurre la statica evita di aprire un
        # calendar spread di segno opposto sulla seconda gamba.
        if strat.static_trim_days > 0 and t >= k_phase and not is_last:
            min_excess = 0.002 * cfg.notional_bbl   # soglia materiale ($2k/pt su 1M bbl): non tocca deep-ITM con vega residua trascurabile
            for is_call_wing, collar_v in ((False, g.vega_put), (True, g.vega_call)):
                wing_opts = [o for o in static if o["call"] == is_call_wing and o["qty"] != 0
                             and 0 < o["ttm"] * 365.0 <= strat.static_trim_days and not o.get("letting_expire")]
                if not wing_opts:
                    continue
                Fo = F[wing_opts[0]["leg"]]
                wing_v = sum(o["qty"] * option_vega_point(Fo, o["K"], o["ttm"], o["vol"]) for o in wing_opts)
                excess = abs(wing_v) - abs(collar_v)
                if abs(wing_v) > 1e-6 and excess > max(0.15 * abs(collar_v), min_excess):
                    ratio = abs(collar_v) / abs(wing_v)
                    for o in wing_opts:
                        intrinsic = max(Fo - o["K"], 0.0) if o["call"] else max(o["K"] - Fo, 0.0)
                        time_val = max(o["price"] - intrinsic, 0.0)
                        sp = _option_quote(ctx, d, o, st)[1]
                        if not np.isfinite(sp) or time_val <= sp:
                            continue
                        new_q = np.round(o["qty"] * ratio / lot) * lot
                        max_step = max(np.round(abs(o["target"]) * 0.25 / lot) * lot, lot)
                        dq = np.clip(new_q - o["qty"], -max_step, max_step)
                        step_q = np.round((o["qty"] + dq) / lot) * lot
                        if abs(step_q - o["qty"]) >= lot:
                            cost += trade_option(o, step_q, sp)

        static_vega = 0.0
        exit_w = max(strat.static_exit_days, 1)
        for o in static:
            if o["qty"] == 0:
                continue
            if o["ttm"] <= 0:
                # scaduta: il valore intrinseco è già entrato nel P&L, la posizione sparisce senza costi
                o["qty"] = 0.0
                continue
            Fo = F[o["leg"]]
            intrinsic = max(Fo - o["K"], 0.0) if o["call"] else max(o["K"] - Fo, 0.0)
            time_val = max(o["price"] - intrinsic, 0.0)
            ttm_d = o["ttm"] * 365.0
            if is_last:
                cost += trade_option(o, 0.0, np.nan)
                continue
            if exit_w > 1 and ttm_d < strat.static_close_days + exit_w - 1:
                if strat.static_let_expire and time_val <= o["spread"]:
                    if ttm_d < strat.static_close_days:
                        o["letting_expire"] = True
                else:
                    rem_frac = max((ttm_d - (strat.static_close_days - 1)) / exit_w, 0.0)
                    tgt_exit = np.round(o["target"] * rem_frac / lot) * lot
                    if abs(tgt_exit) < abs(o["qty"]):
                        sp = _option_quote(ctx, d, o, st)[1]
                        cost += trade_option(o, tgt_exit, sp)
                    if o["qty"] == 0:
                        continue
            elif ttm_d < strat.static_close_days:
                # vicino alla scadenza si chiude; ma se il valore temporale è <= spread
                # (sia OTM sia deep ITM, es. call 75/80 nel 2022) chiuderla costerebbe
                # più del rischio residuo: la si lascia scadere al valore intrinseco
                if not (strat.static_let_expire and time_val <= o["spread"]):
                    cost += trade_option(o, 0.0, np.nan)
                    continue
                o["letting_expire"] = True
            if o.get("letting_expire") or (time_val <= 0.02 and ttm_d < 25):
                # opzione ormai priva di valore temporale vicino alla scadenza: vega nulla
                # (evita di coprire vega fantasma da tick minimo $0.01) e delta puramente intrinseca
                if intrinsic > 0:
                    opt_delta[o["leg"]] += o["qty"] * (1.0 if o["call"] else -1.0)
                continue
            static_vega += o["qty"] * option_vega_point(Fo, o["K"], o["ttm"], o["vol"])
            opt_delta[o["leg"]] += o["qty"] * _opt_delta(strat, o, Fo, st)

        # --- 2b. opzioni dinamiche (vega residua)
        defer_dyn = strat.static_defer_dynamic and strat.static_hedge and t < k_phase
        if not is_last and strat.vega_hedge and not defer_dyn:
            cand = _hedge_option_candidates(ctx, d, strat, 1)
            otm = cand[cand.otm]
            tradable = set(cand.option_ticker)
            for name, k_ref, vega_target in _dynamic_targets(strat, ctx, g, static_vega):
                o = opts.get(name)
                best = otm.iloc[(otm.strike - k_ref).abs().argsort().iloc[0]] if len(otm) else None
                # si cambia opzione se quella tenuta non è più negoziabile, oppure se ce n'è
                # una più vicina allo strike di riferimento di almeno option_switch_distance
                need_new = best is not None and (
                    o is None or o["ticker"] not in tradable
                    or abs(o["K"] - k_ref) - abs(best.strike - k_ref) >= strat.option_switch_distance)
                if need_new and (o is None or best.option_ticker != o["ticker"]):
                    if o is not None:
                        cost += trade_option(o, 0.0, np.nan)                # chiudo la vecchia
                    o = opts[name] = _new_option(best, 1)
                if o is None:
                    continue
                price, sp, ttm, vol = _option_quote(ctx, d, o, st)
                if np.isfinite(price):
                    o.update(price=price, ttm=ttm, vol=vol)
                vega_1 = option_vega_point(F[1], o["K"], o["ttm"], o["vol"])
                target = np.round((-vega_target / vega_1 if vega_1 > 1e-8 else 0.0) / lot) * lot
                if np.isfinite(sp) and abs(target - o["qty"]) > strat.vega_band * abs(target) + 0.5 * lot:
                    cost += trade_option(o, target, sp)
                opt_delta[1] += o["qty"] * _opt_delta(strat, o, F[1], st)
        if is_last:                                                         # chiusura finale
            for o in opts.values():
                cost += trade_option(o, 0.0, np.nan)

        # --- 3. futures (delta)
        if strat.hedge_delta and not is_last:
            target_q = -(g.delta + opt_delta)
        else:
            target_q = np.zeros(2)
        target_q = np.round(target_q / lot) * lot
        for i in (0, 1):
            if not np.isfinite(F[i]):
                target_q[i] = 0.0
            diff = target_q[i] - q[i]
            closing = target_q[i] == 0 and q[i] != 0      # chiusura: si fa sempre, banda o no
            if abs(diff) > strat.delta_band_lots * lot or closing:
                sp = ctx.fut_spread.at[d, cfg.legs[i]] if d in ctx.fut_spread.index else np.nan
                cost += abs(diff) * (float(sp) if np.isfinite(sp) else 0.0) / 2
                q[i] = target_q[i]
                n_tf += 1
        B -= cost
        comp["costi"] = -cost

        # --- 4. valore del portafoglio e P&L
        pi = g.value + B
        pnl = pi - prev_pi
        if t == 0:
            comp["tempo"] = g.value                  # ≈ 0: il collar nasce a costo zero
        prev_pi = pi
        o_dyn = opts.get("dyn") or opts.get("put")
        rows.append(dict(date=d, F1=F[0], F2=F[1], collar_value=g.value, cash=B, portfolio=pi, pnl=pnl,
                         delta1=g.delta[0], delta2=g.delta[1], vega=g.vega, vega_put=g.vega_put,
                         vega_call=g.vega_call, q1=q[0], q2=q[1],
                         opt_put_qty=o_dyn["qty"] if o_dyn else 0.0,
                         opt_put_strike=o_dyn["K"] if o_dyn else np.nan,
                         opt_call_qty=opts["call"]["qty"] if "call" in opts else 0.0,
                         opt_call_strike=opts["call"]["K"] if "call" in opts else np.nan,
                         static_qty=float(sum(abs(o["qty"]) for o in static)),
                         **{f"pnl_{k}": v for k, v in comp.items()}))

    daily = pd.DataFrame(rows).set_index("date")
    metric = float((daily.pnl ** 2).mean())
    return BacktestResult(strategy=strat, daily=daily, metric=metric, rms_pnl=float(np.sqrt(metric)),
                          total_cost=float(-daily.pnl_costi.sum()), final_value=float(daily.portfolio.iloc[-1]),
                          n_trades_fut=n_tf, n_trades_opt=n_to)


def summary_table(results: list[BacktestResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "strategia": r.strategy.label,
            "metrica ($²)": r.metric,
            "metrica per bbl ($²/bbl²)": r.metric / 1e12,
            "P&L giornaliero RMS ($)": r.rms_pnl,
            "P&L giornaliero max |.| ($)": float(r.daily.pnl.abs().max()),
            "costi totali ($)": r.total_cost,
            "valore finale Π ($)": r.final_value,
            "scambi futures": r.n_trades_fut,
            "scambi opzioni": r.n_trades_opt,
        })
    return pd.DataFrame(rows)
