"""
Grafici del progetto. Ogni funzione salva un PNG nella cartella indicata.
Le chiama main.py; si possono anche usare da sole passando i risultati.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import BacktestResult, Context

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
LEG_COLORS = ("#2a78d6", "#eb6834")
STRAT_COLORS = {"S0_nessuna": "#898781", "S1_delta": "#2a78d6", "S2_delta_vega": "#1baf7a",
                "S3_statica_delta": "#e87ba4", "S4_statica_delta_vega": "#eb6834"}
COMP_COLORS = {"prezzo": "#2a78d6", "volatilita": "#eb6834", "tempo": "#1baf7a",
               "costi": "#e34948", "interessi": "#898781"}
COMP_LABELS = {"prezzo": "prezzo (futures)", "volatilita": "volatilità (smile)", "tempo": "tempo e fixing",
               "costi": "costi bid-ask", "interessi": "interessi"}

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "normal", "legend.frameon": False,
})


def _dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.grid(axis="x", visible=False)


def _leg_names(ctx: Context):
    from fixing_calendar import parse_contract
    names = []
    for c in ctx.cfg.legs:
        y, m = parse_contract(c)
        names.append(pd.Timestamp(year=y, month=m, day=1).strftime("%b%y"))
    return names


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ mercato
def plot_market(ctx: Context, path: Path):
    """Futures, strike del collar e finestra di media; vol ATM delle due gambe."""
    names = _leg_names(ctx)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True)
    ax = axes[0]
    px = ctx.fut_settle.loc[ctx.days]
    for i, c in enumerate(ctx.cfg.legs):
        ax.plot(px.index, px[c], color=LEG_COLORS[i], lw=2, label=names[i])
    for k, lab in [(ctx.cfg.k_call, f"Kcall = {ctx.cfg.k_call:g}"), (ctx.k_put, f"Kput = {ctx.k_put:.2f}")]:
        ax.axhline(k, color=MUTED, ls="--", lw=1)
        ax.text(px.index[0], k + 0.8, lab, color=MUTED, fontsize=9)
    ax.axvspan(ctx.sched.dates[0], ctx.sched.dates[-1] + pd.Timedelta(days=1), color="#eda100", alpha=0.15, lw=0)
    ax.set_title("Prezzo di settlement dei futures ($/bbl) e strike del collar (in giallo la finestra di media)")
    ax.legend(loc="upper right")
    ax = axes[1]
    for i, c in enumerate(ctx.cfg.legs):
        atm = [ctx.smiles[(d, c)].atm * 100 if (d, c) in ctx.smiles else np.nan for d in ctx.days]
        ax.plot(ctx.days, atm, color=LEG_COLORS[i], lw=2, label=f"ATM {names[i]}")
    ax.set_title("Volatilità implicita ATM dallo smile calibrato (%)")
    ax.legend(loc="upper right")
    _dates(ax)
    _save(fig, path)


def plot_smile_emission(ctx: Context, path: Path):
    """Vol implicite di mercato all'emissione, smile calibrato e strike mappati del collar."""
    d = ctx.days[0]
    names = _leg_names(ctx)
    st = ctx.states[0]
    m = float(np.dot([np.sum(st.leg == 0), np.sum(st.leg == 1)], st.f)) / st.n_total
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for i, c in enumerate(ctx.cfg.legs):
        g = ctx.iv[(ctx.iv.date == d) & (ctx.iv.contract == c) & ctx.iv.keep]
        F = st.f[i]
        ax.plot(g.strike, g.iv_mid * 100, "o", color=LEG_COLORS[i], ms=6, label=f"mercato {names[i]}")
        ks = np.linspace(g.strike.min() - 8, g.strike.max() + 8, 200)
        ax.plot(ks, ctx.smiles[(d, c)].vol(ks, F) * 100, color=LEG_COLORS[i], lw=1.5)
    for k, lab, off in [(ctx.k_put, "Kput", (12, -32)), (ctx.cfg.k_call, "Kcall", (8, 10))]:
        x = np.log(k / m)
        v = ctx.smiles[(d, ctx.cfg.legs[0])].vol_x(x) * 100
        ax.plot([st.f[0] * k / m], [v], "s", color=INK, ms=7)
        ax.annotate(f"{lab} = {k:.2f}\nvol {v:.1f}%", (st.f[0] * k / m, v), textcoords="offset points",
                    xytext=off, fontsize=9)
    ax.set_title(f"Smile all'emissione ({d:%d/%m/%Y}): punti di mercato, curva PCHIP, vol usate dal collar")
    ax.set_xlabel("strike ($/bbl)")
    ax.set_ylabel("vol implicita (%)")
    ax.legend()
    _save(fig, path)


def plot_mc_validation(table: pd.DataFrame, path: Path):
    """Differenza MM - MC per diversi strike e date, con banda ±2 errori standard."""
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for i, (d, g) in enumerate(table.groupby("data")):
        col = ["#2a78d6", "#eb6834", "#1baf7a"][i % 3]
        x = g.strike + (i - 1) * 0.4
        ax.errorbar(x, (g.mm - g.mc) * 100, yerr=2 * g.mc_se * 100, fmt="o", color=col, ms=5,
                    capsize=3, label=f"{pd.Timestamp(d):%d/%m/%Y}")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_title("Moment matching meno Monte Carlo (centesimi di $/bbl), barre = ±2 errori standard MC")
    ax.set_xlabel("strike ($/bbl)")
    ax.set_ylabel("MM − MC (¢/bbl)")
    ax.legend(title="data di valutazione")
    _save(fig, path)


# ------------------------------------------------------------------ collar
def plot_collar(ctx: Context, res: BacktestResult, path: Path):
    """Valore del collar, delta e vega nel tempo."""
    d = res.daily
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True, sharex=True)
    axes[0].plot(d.index, d.collar_value / 1e6, color=INK, lw=2)
    axes[0].set_title("Valore del collar per il market maker (milioni di $, long put e short call)")
    names = _leg_names(ctx)
    axes[1].plot(d.index, -d.delta1 / 1e3, color=LEG_COLORS[0], lw=2, label=f"copertura {names[0]}")
    axes[1].plot(d.index, -d.delta2 / 1e3, color=LEG_COLORS[1], lw=2, label=f"copertura {names[1]}")
    axes[1].set_title("Futures necessari per coprire il delta (migliaia di barili = contratti)")
    axes[1].legend()
    axes[2].plot(d.index, d.vega / 1e3, color=INK, lw=2, label="vega totale")
    axes[2].plot(d.index, d.vega_put / 1e3, color="#1baf7a", lw=1.2, label="dalla put (long)")
    axes[2].plot(d.index, d.vega_call / 1e3, color="#e34948", lw=1.2, label="dalla call (short)")
    axes[2].axhline(0, color=MUTED, lw=0.8)
    axes[2].set_title("Vega del collar (migliaia di $ per +1 punto di vol)")
    axes[2].legend()
    _dates(axes[2])
    _save(fig, path)


# ------------------------------------------------------------------ strategie
def plot_metric(results: list[BacktestResult], path: Path, title_suffix: str = "",
                top_names=("S0_nessuna", "S1_delta", "S4_statica_delta_vega")):
    """La metrica della competizione: P&L² giornalieri e metrica cumulata (1/t)·Σ P&L²."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True, sharex=True)
    hi, lo = 0.0, np.inf
    for r in results:
        d = r.daily
        col = STRAT_COLORS.get(r.strategy.name, INK)
        running = (d.pnl ** 2).cumsum() / np.arange(1, len(d) + 1)
        axes[1].plot(d.index, running, color=col, lw=2, label=f"{r.strategy.label}: {r.metric:.2e}")
        lo = min(lo, running[running > 0].iloc[5:].min())
        if r.strategy.name in top_names:
            axes[0].plot(d.index, d.pnl ** 2, color=col, lw=1, label=r.strategy.label)
            hi = max(hi, (d.pnl ** 2).max())
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e7, hi * 3)
    axes[0].set_title(f"P&L giornaliero al quadrato ($², scala logaritmica){title_suffix}")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set_yscale("log")
    axes[1].set_ylim(lo / 3, None)
    axes[1].set_title("Metrica cumulata (1/t)·Σ P&L² ($², scala logaritmica); in legenda il valore finale")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    _dates(axes[1])
    _save(fig, path)


def plot_portfolio(results: list[BacktestResult], path: Path):
    """Valore del portafoglio coperto Π_t = collar + cassa."""
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for r in results:
        d = r.daily
        ax.plot(d.index, d.portfolio / 1e6, color=STRAT_COLORS.get(r.strategy.name, INK), lw=2,
                label=r.strategy.label)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title("Valore del portafoglio coperto Π = collar + cassa (milioni di $)")
    ax.legend(fontsize=8)
    _dates(ax)
    _save(fig, path)


def plot_attribution(res: BacktestResult, path: Path):
    """Da dove viene la metrica: contributo cumulato di ogni componente del P&L."""
    d = res.daily
    T = len(d)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True, sharex=True)
    for c in COMP_COLORS:
        axes[0].plot(d.index, d[f"pnl_{c}"] / 1e3, color=COMP_COLORS[c], lw=1, label=COMP_LABELS[c])
        axes[1].plot(d.index, (d[f"pnl_{c}"] ** 2).cumsum() / T, color=COMP_COLORS[c], lw=2,
                     label=COMP_LABELS[c])
    axes[1].plot(d.index, (d.pnl ** 2).cumsum() / T, color=INK, lw=2, ls="--", label="P&L totale")
    axes[0].set_title(f"{res.strategy.label}: P&L giornaliero per componente (migliaia di $)")
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].set_title("Contributo cumulato alla metrica: Σ (componente)² / T  ($²)")
    axes[1].legend(fontsize=8)
    _dates(axes[1])
    _save(fig, path)


def plot_comparison(tables: dict[str, pd.DataFrame], path: Path):
    """Metrica per strategia, training e prova generale (scala logaritmica)."""
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    names = list(tables)
    strategies = tables[names[0]]["strategia"].tolist()
    w = 0.8 / len(names)
    for j, nm in enumerate(names):
        t = tables[nm]
        ax.barh(np.arange(len(strategies)) + j * w, t["metrica ($²)"], height=w * 0.9,
                color=["#2a78d6", "#eb6834"][j % 2], label=nm)
        for i, v in enumerate(t["metrica ($²)"]):
            ax.text(v * 1.08, i + j * w, f"{v:.2e}", va="center", fontsize=8, color=INK)
    ax.set_yticks(np.arange(len(strategies)) + w * (len(names) - 1) / 2)
    ax.set_yticklabels(strategies, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(right=ax.get_xlim()[1] * 4)
    ax.set_title("Metrica (1/T)·Σ P&L² per strategia ($², scala logaritmica: più corto è meglio)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_band_sensitivity(table: pd.DataFrame, path: Path):
    """Metrica e costi al variare della banda di non intervento sul delta."""
    fig, ax1 = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax1.plot(table.banda, table.metrica, "-o", color="#2a78d6", lw=2)
    ax1.set_xlabel("banda di non intervento (contratti)")
    ax1.set_ylabel("metrica ($²)", color="#2a78d6")
    ax1.set_title("Delta hedging: metrica al variare della banda (costi totali in etichetta)")
    for b, m, c in zip(table.banda, table.metrica, table.costi):
        ax1.annotate(f"{c/1e3:.0f}k$", (b, m), textcoords="offset points", xytext=(0, 8), ha="center",
                     fontsize=8, color=MUTED)
    _save(fig, path)
