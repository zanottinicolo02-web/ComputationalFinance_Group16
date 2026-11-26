import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_smiles(iv, smiles, legs, date, path):
    fig, axs = plt.subplots(1, len(legs), figsize=(10, 3.8), sharey=True)
    for ax, leg in zip(axs, legs):
        g = iv[(iv.date == date) & (iv.leg == leg)]
        if g.empty:
            ax.set(title=f"{leg} - no liquid quotes on {date:%Y-%m-%d}")
            continue
        m = np.linspace(g.strike.min() / g.F.iloc[0], g.strike.max() / g.F.iloc[0], 200)
        ax.plot(m, smiles[date][leg](m), label="fit")
        ax.scatter(g.strike / g.F, g.iv, c="k", s=14, label="market IV")
        ax.set(title=f"{leg} - {date:%Y-%m-%d}", xlabel="K / F")
    axs[0].set_ylabel("implied vol")
    axs[0].legend()
    _save(fig, path)


def plot_rho(rho, path):
    fig, ax = plt.subplots(figsize=(8, 3))
    rho.plot(ax=ax)
    ax.set(ylabel="rho", title="Rolling correlation of the two futures")
    _save(fig, path)


def plot_mc_convergence(n, price, se, ref, title, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n, price, label="MC (control variate)")
    ax.fill_between(n, price - 1.96 * se, price + 1.96 * se, alpha=0.25, label="+/-1.96 SE")
    ax.axhline(ref, c="r", ls="--", label="moment matching")
    ax.set(xscale="log", xlabel="paths", ylabel="price", title=title)
    ax.legend()
    _save(fig, path)


def plot_portfolio(res, path):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    res.W.plot(ax=ax)
    ax.set(ylabel="portfolio value (USD)", title="Hedged portfolio value")
    _save(fig, path)


def plot_pnl2(res, path):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    (res.pnl ** 2).plot(ax=ax)
    ax.set(ylabel="squared daily P&L (USD$^2$)", title="Squared daily P&L")
    _save(fig, path)


def plot_hedge(res, path):
    fig, axs = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    res[["nF1", "nF2"]].plot(ax=axs[0], title="Futures positions (bbl)")
    res[["nO1", "nO2"]].plot(ax=axs[1], title="Hedge options (bbl)")
    _save(fig, path)
