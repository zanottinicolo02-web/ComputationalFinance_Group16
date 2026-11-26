"""
Esegue tutto il progetto e salva risultati e grafici in Hedging Competition/output/.

    python main.py            training (collar di febbraio 2021) + prova generale (dicembre 2020)
    python main.py --test     collar di ottobre 2022 (quando ci saranno i dati di test)

Passi:
  1. dati, vol implicite, smile, correlazione, calendario dei fixing (backtest.prepare)
  2. Kput all'emissione (moment matching) e validazione con il Monte Carlo
  3. backtest delle strategie di copertura (strategy.py)
  4. analisi di sensibilità (banda sul delta, banda sulla vega, metodo dello smile)
  5. prova generale su un collar inventato (dicembre 2020)
  6. grafici e tabelle
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from backtest import prepare, run_backtest, summary_table
from config import OUTPUT_DIR, PSEUDO_OOS, TEST, TRAIN, CollarConfig
from pricer import price_mc, price_mm
from strategy import BEST, DELTA_DAILY, MAIN_STRATEGIES, VARIANTS
import report_plots as rp


def mc_validation(ctx, dates_idx) -> pd.DataFrame:
    """Moment matching contro Monte Carlo in alcune date e per alcuni strike.

    Gli strike sono scelti in proporzione al valore atteso della media: più larghi
    lontano dalla scadenza, più stretti dentro la finestra di media (dove la parte
    ancora incerta della media è piccola).
    """
    rows = []
    m = ctx.cfg.model
    for i in dates_idx:
        st = ctx.states[i]
        fwd = (st.known_sum + np.array(st.f)[st.leg].sum()) / st.n_total
        rel = (0.95, 0.975, 1.0, 1.025, 1.05) if ctx.days[i] >= ctx.sched.dates[0] else (0.7, 0.85, 1.0, 1.15, 1.3)
        strikes = sorted(set([round(fwd * r, 1) for r in rel] + ([ctx.k_put, ctx.cfg.k_call] if i == 0 else [])))
        for K in strikes:
            is_call = K >= fwd
            mc = price_mc(st, K, is_call, n_paths=m.mc_paths, seed=m.mc_seed)
            rows.append(dict(data=ctx.days[i], strike=K, tipo="call" if is_call else "put",
                             mm=price_mm(st, K, is_call), mc=mc.price, mc_se=mc.std_error,
                             mc_senza_cv=mc.price_plain, mc_se_senza_cv=mc.std_error_plain))
    t = pd.DataFrame(rows)
    t["diff_in_se"] = np.where(t.mc_se > 0, (t.mm - t.mc) / t.mc_se.where(t.mc_se > 0), np.nan)
    return t


def _money(x: float) -> str:
    """Importo in $ leggibile: 576 k$, 3,99 M$."""
    x = 0.0 if abs(x) < 0.5 else x
    a = abs(x)
    if a >= 1e6:
        return f"{x / 1e6:.2f} M$"
    if a >= 1e3:
        return f"{x / 1e3:.0f} k$"
    return f"{x:.0f} $"


def print_summary(cfg, label, ctx, results, info, seconds, out) -> None:
    """Riepilogo leggibile a terminale."""
    line = "─" * 111
    print(line)
    print(f" {label.upper()}")
    print(line)
    s = ctx.sched
    print(f" Fixing:   {s.n1} su {cfg.legs[0]} ({s.table.fixing_date.iloc[0]:%d/%m} – "
          f"{s.table.fixing_date.iloc[s.n1 - 1]:%d/%m})  +  {s.n2} su {cfg.legs[1]}")
    print(f" Kput:     {ctx.k_put:.2f} $/bbl   (call {cfg.k_call:g} e put valgono {ctx.call_price:.3f} $/bbl "
          f"= {_money(ctx.call_price * cfg.notional_bbl)} sul nozionale)")
    print(f" Media realizzata: {info['media_realizzata']:.2f} $/bbl\n")
    print(f" {'Strategia':<61}{'Metrica ($²)':>13}{'P&L tipico':>12}{'P&L peggiore':>14}{'Costi':>11}")
    print(f" {'':<61}{'':>13}{'(RMS)':>12}{'(max |P&L|)':>14}{'':>11}")
    for r in results:
        name = f"{r.strategy.name.split('_')[0]}  {r.strategy.label}"
        print(f" {name:<61}{r.metric:>13.2e}{_money(r.rms_pnl):>12}"
              f"{_money(r.daily.pnl.abs().max()):>14}{_money(r.total_cost):>11}")
    m = {r.strategy.name: r.metric for r in results}
    best = min(results, key=lambda r: r.metric)
    print(f"\n Migliore: {best.strategy.name.split('_')[0]} ({best.strategy.label})")
    if "S0_nessuna" in m and "S1_delta" in m:
        print(f"   metrica −{(1 - best.metric / m['S0_nessuna']) * 100:.1f}% rispetto a nessuna copertura, "
              f"−{(1 - best.metric / m['S1_delta']) * 100:.1f}% rispetto al solo delta")
    if "S2_delta_vega" in m and best.strategy.name != "S2_delta_vega":
        print(f"   e −{(1 - best.metric / m['S2_delta_vega']) * 100:.1f}% rispetto a delta + vega senza copertura statica")
    print(f"\n Grafici e tabelle in: {out}   ({seconds:.0f} s)")
    print(line + "\n")


def emission_robustness(ctx_train) -> list[dict]:
    """Metrica di S4 al variare dei giorni di costruzione, per collar emessi in altre date."""
    m = ctx_train.cfg.k_call / ctx_train.states[0].f[0]
    rows = []
    for em in ("2020-03-02", "2020-04-01", "2020-06-01"):
        cfg = replace(PSEUDO_OOS, name=f"emissione_{em}", emission_date=pd.Timestamp(em))
        f0 = prepare(cfg).states[0].f[0]
        ctx = prepare(replace(cfg, k_call=float(round(m * f0))))
        for k in (10, 20, 40, 60):
            rows.append(dict(emissione=em, k_call=ctx.cfg.k_call, giorni=k,
                             metrica=run_backtest(ctx, replace(BEST, static_phase_days=k)).metric))
    return rows


def run_collar(cfg: CollarConfig, label: str, full: bool = True) -> dict:
    out = OUTPUT_DIR / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = prepare(cfg)

    results = [run_backtest(ctx, s) for s in MAIN_STRATEGIES]
    tab = summary_table(results)
    tab.to_csv(out / "sintesi_strategie.csv", index=False)
    for r in results:
        r.daily.to_csv(out / f"giornaliero_{r.strategy.name}.csv")
    best = next(r for r in results if r.strategy.name == BEST.name)
    delta = next(r for r in results if r.strategy.name == DELTA_DAILY.name)

    info = dict(collar=cfg.name, etichetta=label, k_put=ctx.k_put, k_call=cfg.k_call,
                prezzo_call_per_bbl=ctx.call_price, giorni=len(ctx.days),
                fixing=dict(n1=ctx.sched.n1, n2=ctx.sched.n2, gamba1=cfg.legs[0], gamba2=cfg.legs[1]),
                emissione=str(ctx.days[0].date()), ultimo_fixing=str(ctx.sched.last_fixing.date()),
                media_realizzata=float(np.mean([ctx.fut_settle.at[d, c] for d, c in
                                                zip(ctx.sched.table.fixing_date, ctx.sched.table.contract)])),
                strategie={r.strategy.name: dict(label=r.strategy.label, metrica=r.metric, rms=r.rms_pnl,
                                                 costi=r.total_cost, valore_finale=r.final_value,
                                                 max_pnl=float(r.daily.pnl.abs().max()))
                           for r in results})

    rp.plot_metric(results, out / "metrica.png")
    rp.plot_portfolio(results, out / "portafoglio.png")
    rp.plot_attribution(best, out / f"scomposizione_{best.strategy.name}.png")
    rp.plot_attribution(delta, out / f"scomposizione_{delta.strategy.name}.png")
    rp.plot_collar(ctx, delta, out / "collar_greche.png")
    rp.plot_market(ctx, out / "mercato.png")
    rp.plot_smile_emission(ctx, out / "smile_emissione.png")

    if full:
        # validazione del pricer
        idx = [0, len(ctx.days) // 2, int(np.searchsorted(ctx.days, ctx.sched.dates[0])) + 5]
        val = mc_validation(ctx, idx)
        val.to_csv(out / "validazione_mc.csv", index=False)
        rp.plot_mc_validation(val, out / "validazione_mc.png")
        info["validazione_mc_max_diff_in_se"] = float(val.diff_in_se.abs().max())
        info["validazione_mc_max_diff_cent"] = float((val.mm - val.mc).abs().max() * 100)

        # sensibilità: banda sul delta
        bands = []
        for b in (0, 5, 10, 25, 50, 100):
            r = run_backtest(ctx, replace(DELTA_DAILY, delta_band_lots=b))
            bands.append(dict(banda=b, metrica=r.metric, costi=r.total_cost, scambi=r.n_trades_fut))
        bands = pd.DataFrame(bands)
        bands.to_csv(out / "sensibilita_banda_delta.csv", index=False)
        rp.plot_band_sensitivity(bands, out / "sensibilita_banda_delta.png")
        # sensibilità: banda sulla vega
        vb = []
        for v in (0.0, 0.05, 0.1, 0.25, 0.5):
            r = run_backtest(ctx, replace(BEST, vega_band=v))
            vb.append(dict(banda_vega=v, metrica=r.metric, costi=r.total_cost, scambi_opzioni=r.n_trades_opt))
        pd.DataFrame(vb).to_csv(out / "sensibilita_banda_vega.csv", index=False)
        # sensibilità: quanta vega del collar si copre con la parte statica
        vm = []
        for m in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
            r = run_backtest(ctx, replace(BEST, static_vega_mult=m))
            vm.append(dict(quota_vega_statica=m, metrica=r.metric, costi=r.total_cost))
        pd.DataFrame(vm).to_csv(out / "sensibilita_vega_statica.csv", index=False)
        info["sensibilita_vega_statica"] = vm
        # sensibilità: in quanti giorni si costruisce la copertura statica
        ph = []
        for k in (1, 5, 10, 20, 40, 60):
            r = run_backtest(ctx, replace(BEST, static_phase_days=k))
            ph.append(dict(giorni=k, metrica=r.metric, costi=r.total_cost))
        pd.DataFrame(ph).to_csv(out / "sensibilita_giorni_statica.csv", index=False)
        info["sensibilita_giorni_statica"] = ph
        # robustezza: la prova generale con emissioni in altre date (call alla stessa
        # distanza percentuale dal futures del collar vero)
        info["giorni_statica_altre_emissioni"] = emission_robustness(ctx)
        # sensibilità: smile a parabola invece di PCHIP
        cq = replace(cfg, model=replace(cfg.model, smile_method="quadratic"))
        ctxq = prepare(cq)
        info["smile_parabola"] = dict(k_put=ctxq.k_put, **{
            s.name: run_backtest(ctxq, s).metric for s in (DELTA_DAILY, BEST)})
        info["sensibilita_banda_delta"] = bands.to_dict(orient="records")
        info["sensibilita_banda_vega"] = vb
        # varianti scartate
        var = [dict(variante=k, metrica=run_backtest(ctx, v).metric) for k, v in VARIANTS.items()]
        pd.DataFrame(var).to_csv(out / "varianti.csv", index=False)
        info["varianti"] = var

    with open(out / "risultati.json", "w") as fh:
        json.dump(info, fh, indent=2, default=str)
    print_summary(cfg, label, ctx, results, info, time.time() - t0, out)
    return dict(ctx=ctx, results=results, table=tab, info=info)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="esegue il collar di test (ottobre 2022)")
    args = ap.parse_args()
    if args.test:
        run_collar(TEST, "Test ottobre 2022", full=False)
        return
    train = run_collar(TRAIN, "Training (febbraio 2021)")
    oos = run_collar(PSEUDO_OOS, "Prova generale (dicembre 2020)", full=False)
    rp.plot_comparison({"training feb 2021": train["table"], "prova generale dic 2020": oos["table"]},
                       OUTPUT_DIR / "confronto_strategie.png")


if __name__ == "__main__":
    main()
