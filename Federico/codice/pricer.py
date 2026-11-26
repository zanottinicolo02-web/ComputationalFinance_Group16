"""
Prezzo del collar asiatico: moment matching (formula chiusa) e Monte Carlo.

Il sottostante
--------------
La media del contratto è  A = (1/N) Σ_k f(t_k, τ_leg(k)),  dove ogni fixing t_k
usa il futures della sua gamba (Apr21 o May21, vedi fixing_calendar.py).
Il giorno t, i fixing con t_k <= t sono già noti (il loro settlement è uscito);
gli altri sono casuali. Scriviamo

    A = S/N + U,     S = somma dei fixing già noti,
                     U = (1/N) Σ_{k futuri} f_leg(k)(t_k)

Una call su A con strike K è quindi una call su U con strike K' = K - S/N.

Dinamica (Note, sez. 4): sotto la misura risk-neutral ogni futures è un moto
browniano geometrico senza drift, f_i(t_k) = f_i(t)·exp(σ_i W_i(θ_k) - σ_i² θ_k / 2),
θ_k = t_k - t in anni, con corr(dW_1, dW_2) = ρ.

Moment matching (Note, sez. 6)
------------------------------
U non è lognormale, ma la si approssima con una lognormale che ha la stessa
media e lo stesso momento secondo:

    m     = E[U]  = (1/N) Σ_k f_leg(k)
    E[U²] = (1/N²) Σ_k Σ_l f_k f_l exp(ρ_kl σ_k σ_l min(θ_k, θ_l))

(ρ_kl = 1 se i due fixing sono sulla stessa gamba, ρ altrimenti). Con
v = ln(E[U²]/m²) la call è una formula di Black:

    Call = D · [ m N(d1) - K' N(d2) ],   d1 = (ln(m/K') + v/2)/√v,  d2 = d1 - √v

e la put si ottiene allo stesso modo. Se K' <= 0 la call è certamente in the
money (Call = D·(m - K'), Put = 0); se non restano fixing il payoff è noto.

Vol da usare: per ogni gamba si legge lo smile allo strike "mappato" che ha la
stessa moneyness del basket, K_i = f_i · K'/m (Note, sez. 6), cioè alla
log-moneyness x = ln(K'/m).

Sconto: il collar è un contratto OTC pagato alla data di pagamento T, quindi va
scontato con D(t,T) = exp(-r(t)·(T - t)), r = SOFR overnight usato come tasso
piatto (ACT/360). Le opzioni quotate invece non si scontano (r = 0, vedi vol.py).

Monte Carlo (Note, sez. 5)
--------------------------
Simulazione esatta dei due moti browniani correlati alle date di fixing,
variabili antitetiche e control variate sulla media geometrica, che ha un
prezzo esatto in forma chiusa. Serve a validare il moment matching.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from config import CollarConfig
from corr import rho_on
from data import MarketData
from fixing_calendar import FixingSchedule, payment_date


# ------------------------------------------------------------ stato del mercato
@dataclass(frozen=True)
class MarketState:
    """Tutto quello che serve per prezzare il collar in un giorno."""
    date: pd.Timestamp
    f: tuple[float, float]          # prezzi correnti delle due gambe
    smiles: tuple                   # smile delle due gambe (oggetti vol.Smile)
    rho: float
    discount: float                 # D(t, T)
    theta: np.ndarray               # anni mancanti ai fixing futuri
    leg: np.ndarray                 # gamba (0 o 1) di ciascun fixing futuro
    known_sum: float                # somma dei fixing già noti
    n_total: int                    # N, numero totale di fixing
    vol_shift: float = 0.0          # spostamento parallelo delle vol (per la vega)

    def bump(self, df1: float = 0.0, df2: float = 0.0, dvol: float = 0.0) -> "MarketState":
        return replace(self, f=(self.f[0] + df1, self.f[1] + df2), vol_shift=self.vol_shift + dvol)


def build_state(date, md: MarketData, sched: FixingSchedule, smiles: dict,
                rho: pd.Series, cfg: CollarConfig) -> MarketState:
    """Costruisce lo stato del mercato alla chiusura del giorno `date`.

    I fixing con data <= date sono considerati noti (il settlement è uscito).
    """
    t = pd.Timestamp(date)
    wide = md.futures_wide("settle")
    legs = cfg.legs
    f = tuple(float(wide.at[t, c]) if t in wide.index and pd.notna(wide.at[t, c]) else np.nan for c in legs)

    tab = sched.table
    known = tab[tab.fixing_date <= t]
    known_sum = float(sum(wide.at[d, c] for d, c in zip(known.fixing_date, known.contract)))
    fut = tab[tab.fixing_date > t]
    theta = ((fut.fixing_date - t).dt.days / 365.0).to_numpy(float)
    leg = np.where(fut.contract.to_numpy() == legs[0], 0, 1)

    sm = tuple(smiles.get((t, c)) for c in legs)
    T_pay = payment_date(cfg, sched)
    r = float(md.sofr.get(t, md.sofr[md.sofr.index <= t].iloc[-1]))
    D = float(np.exp(-r * max((T_pay - t).days, 0) / 360.0))
    return MarketState(date=t, f=f, smiles=sm, rho=rho_on(rho, t), discount=D, theta=theta,
                       leg=leg, known_sum=known_sum, n_total=sched.N)


def _leg_vols(state: MarketState, x: float) -> np.ndarray:
    """Vol delle due gambe alla log-moneyness x (letta sugli smile)."""
    out = []
    for i in (0, 1):
        s = state.smiles[i] if state.smiles[i] is not None else state.smiles[1 - i]
        out.append(float(s.vol_x(x)) + state.vol_shift)
    return np.maximum(np.array(out), 1e-4)


# ------------------------------------------------------------ moment matching
def price_mm(state: MarketState, K: float, is_call: bool) -> float:
    """Prezzo (in $/bbl, scontato) di una call o put asiatica sul basket, con moment matching."""
    N = state.n_total
    Kp = K - state.known_sum / N
    if len(state.theta) == 0:                       # tutti i fixing sono noti
        A = state.known_sum / N
        return state.discount * max(A - K, 0.0) if is_call else state.discount * max(K - A, 0.0)

    f = np.array(state.f)[state.leg]                # prezzo corrente della gamba di ogni fixing
    m = f.sum() / N
    if Kp <= 0:                                     # call certamente esercitata
        return state.discount * (m - Kp) if is_call else 0.0

    sig = _leg_vols(state, np.log(Kp / m))[state.leg]
    th = state.theta
    rho_kl = np.where(state.leg[:, None] == state.leg[None, :], 1.0, state.rho)
    EU2 = (f[:, None] * f[None, :] * np.exp(rho_kl * sig[:, None] * sig[None, :]
                                            * np.minimum(th[:, None], th[None, :]))).sum() / N**2
    v = np.log(EU2 / m**2)
    if v <= 1e-14:
        intrinsic = max(m - Kp, 0.0) if is_call else max(Kp - m, 0.0)
        return state.discount * intrinsic
    sv = np.sqrt(v)
    d1 = (np.log(m / Kp) + 0.5 * v) / sv
    d2 = d1 - sv
    if is_call:
        return state.discount * (m * norm.cdf(d1) - Kp * norm.cdf(d2))
    return state.discount * (Kp * norm.cdf(-d2) - m * norm.cdf(-d1))


def collar_value(state: MarketState, k_put: float, k_call: float) -> float:
    """Valore per barile del collar visto dal market maker: long put, short call."""
    return price_mm(state, k_put, False) - price_mm(state, k_call, True)


# ----------------------------------------------------------------- Monte Carlo
@dataclass
class MCResult:
    price: float
    std_error: float
    price_plain: float       # senza control variate
    std_error_plain: float


def price_mc(state: MarketState, K: float, is_call: bool, n_paths: int = 200_000,
             seed: int = 42, control_variate: bool = True) -> MCResult:
    """Monte Carlo esatto con variabili antitetiche e control variate geometrica."""
    N = state.n_total
    th = state.theta
    D = state.discount
    if len(th) == 0:
        p = price_mm(state, K, is_call)
        return MCResult(p, 0.0, p, 0.0)

    f = np.array(state.f)
    Kp = K - state.known_sum / N
    m = f[state.leg].sum() / N
    x = np.log(Kp / m) if Kp > 0 else 0.0
    sig_leg = _leg_vols(state, x)
    sig = sig_leg[state.leg]

    # moti browniani correlati sulle date dei fixing futuri (ordinate, possono ripetersi)
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    times = np.unique(th)
    dt = np.diff(np.concatenate([[0.0], times]))
    Z1 = rng.standard_normal((half, len(times)))
    Z2 = state.rho * Z1 + np.sqrt(1 - state.rho**2) * rng.standard_normal((half, len(times)))
    Z1, Z2 = np.vstack([Z1, -Z1]), np.vstack([Z2, -Z2])            # antitetiche
    W1, W2 = np.cumsum(Z1 * np.sqrt(dt), axis=1), np.cumsum(Z2 * np.sqrt(dt), axis=1)
    idx = np.searchsorted(times, th)
    W = np.where(state.leg == 0, W1[:, idx], W2[:, idx])          # browniano di ogni fixing
    logf = np.log(f[state.leg]) - 0.5 * sig**2 * th + sig * W
    A = (state.known_sum + np.exp(logf).sum(axis=1)) / N
    pay = np.maximum(A - K, 0) if is_call else np.maximum(K - A, 0)
    X = D * pay

    # pagamenti antitetici a coppie -> errore standard sulle medie delle coppie
    def mean_se(v):
        pairs = 0.5 * (v[:half] + v[half:])
        return float(pairs.mean()), float(pairs.std(ddof=1) / np.sqrt(half))

    p_plain, se_plain = mean_se(X)
    if not control_variate:
        return MCResult(p_plain, se_plain, p_plain, se_plain)

    # Control variate: Y = S/N + (nf/N)·Gf, con Gf = media geometrica dei fixing futuri.
    # ln Gf è normale, quindi un'opzione su Y ha prezzo esatto (formula di Black).
    # Il payoff (Y - K)+ = (nf/N)·(Gf - Kg)+ con Kg = (K - S/N)·N/nf.
    nf = len(th)
    rho_kl = np.where(state.leg[:, None] == state.leg[None, :], 1.0, state.rho)
    cov = rho_kl * sig[:, None] * sig[None, :] * np.minimum(th[:, None], th[None, :])
    mu_g = (np.log(f[state.leg]) - 0.5 * sig**2 * th).sum() / nf
    s2_g = cov.sum() / nf**2
    Gf = np.exp(logf.mean(axis=1))                                  # geometrica dei fixing futuri
    Kg = (K - state.known_sum / N) * N / nf
    if Kg <= 0:
        return MCResult(p_plain, se_plain, p_plain, se_plain)
    sg = np.sqrt(s2_g)
    d1 = (mu_g + s2_g - np.log(Kg)) / sg
    d2 = d1 - sg
    eg = np.exp(mu_g + 0.5 * s2_g)
    exact = (eg * norm.cdf(d1) - Kg * norm.cdf(d2)) if is_call else (Kg * norm.cdf(-d2) - eg * norm.cdf(-d1))
    exact *= D * nf / N
    payg = np.maximum(Gf - Kg, 0) if is_call else np.maximum(Kg - Gf, 0)
    Y = D * nf / N * payg
    c = np.cov(X, Y)
    beta = c[0, 1] / c[1, 1] if c[1, 1] > 0 else 0.0
    p_cv, se_cv = mean_se(X - beta * (Y - exact))
    return MCResult(p_cv, se_cv, p_plain, se_plain)


# ----------------------------------------------------------------------- Kput
def solve_k_put(state: MarketState, k_call: float) -> tuple[float, float]:
    """Strike della put che rende il collar a costo zero: Put(Kput) = Call(Kcall).

    Restituisce (Kput, prezzo della call). La put cresce con lo strike, quindi la
    soluzione è unica e si trova per bisezione (metodo di Brent).
    """
    c = price_mm(state, k_call, True)
    m = (state.known_sum + np.array(state.f)[state.leg].sum()) / state.n_total
    h = lambda k: price_mm(state, k, False) - c
    hi = m
    while h(hi) < 0:
        hi *= 1.5
    return brentq(h, 1e-6, hi, xtol=1e-10), c
