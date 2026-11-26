"""
Greche del collar e delle opzioni quotate.

Collar (posizione del market maker, su tutto il nozionale)
----------------------------------------------------------
Si calcolano per differenze finite centrate ("bump and reprice"): si sposta un
input di poco in su e in giù e si riprezza con il moment matching.

- delta_i = dV/df_i: di quanto cambia il valore (in $) se il futures della
  gamba i sale di 1 $/bbl. È espresso in barili: la copertura è -delta_i barili
  di futures sulla gamba i.
- gamma_i = d²V/df_i²: quanto cambia il delta quando il prezzo si muove.
- vega = dV/dσ per uno spostamento parallelo di +1 punto di vol (0,01) di
  entrambi gli smile, divisa anche nella parte della put (vega_put, positiva:
  siamo long put) e in quella della call (vega_call, negativa: siamo short call).

Lo smile è definito in log-moneyness ln(K/F), quindi quando si sposta il futures
lo smile si sposta con lui ("sticky moneyness", coerente con la mappatura degli
strike delle Note).

Opzioni quotate (Black-76, r = 0)
---------------------------------
delta call = N(d1), delta put = N(d1) - 1, vega = F·φ(d1)·√T (per unità di σ).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricer import MarketState, collar_value, price_mm


@dataclass
class CollarGreeks:
    value: float        # $ (posizione del market maker)
    delta: np.ndarray   # bbl, una per gamba
    gamma: np.ndarray   # bbl per $/bbl
    vega: float         # $ per +1 punto di vol
    vega_put: float     # parte di vega dovuta alla put (long)
    vega_call: float    # parte di vega dovuta alla call (short)


def collar_greeks(state: MarketState, k_put: float, k_call: float, notional: float,
                  rel_bump: float = 1e-3, vol_bump: float = 0.01) -> CollarGreeks:
    N = notional
    v0 = N * collar_value(state, k_put, k_call)
    delta, gamma = np.zeros(2), np.zeros(2)
    for i in (0, 1):
        f = state.f[i]
        if not np.isfinite(f) or not np.any(state.leg == i):
            continue                       # gamba senza fixing futuri: nessun rischio
        h = rel_bump * f
        up = state.bump(df1=h if i == 0 else 0.0, df2=h if i == 1 else 0.0)
        dn = state.bump(df1=-h if i == 0 else 0.0, df2=-h if i == 1 else 0.0)
        vu, vd = N * collar_value(up, k_put, k_call), N * collar_value(dn, k_put, k_call)
        delta[i] = (vu - vd) / (2 * h)
        gamma[i] = (vu - 2 * v0 + vd) / h**2
    su, sd = state.bump(dvol=vol_bump), state.bump(dvol=-vol_bump)
    vega_put = N * (price_mm(su, k_put, False) - price_mm(sd, k_put, False)) / 2
    vega_call = -N * (price_mm(su, k_call, True) - price_mm(sd, k_call, True)) / 2
    return CollarGreeks(value=v0, delta=delta, gamma=gamma, vega=vega_put + vega_call,
                        vega_put=vega_put, vega_call=vega_call)


def option_delta(F, K, T, sigma, is_call) -> float:
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0)


def option_vega_point(F, K, T, sigma) -> float:
    """Vega di un'opzione su 1 barile, per +1 punto di vol."""
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return float(F * norm.pdf(d1) * np.sqrt(T) * 0.01)


def option_delta_smile(F, K, T, sigma, is_call, smile, rel_bump: float = 1e-3) -> float:
    """Delta di un'opzione quotata coerente con lo smile del collar (sticky moneyness).

    Quando il futures si muove, lo smile si sposta con lui: la vol dello strike K
    cambia perché cambia la sua moneyness ln(K/F). La delta di Black-76 ignora
    questo effetto (sticky strike) e, con uno skew ripido, sbaglia la copertura.
    Qui si riprezza l'opzione con F ± h, spostando la vol lungo lo smile e
    ancorandola alla vol di mercato dell'opzione (sigma).
    """
    from vol import black76_price
    if smile is None or not np.isfinite(F) or T <= 0:
        return option_delta(F, K, T, sigma, is_call)
    h = rel_bump * F
    base = float(smile.vol(K, F))
    vu = max(sigma + float(smile.vol(K, F + h)) - base, 1e-4)
    vd = max(sigma + float(smile.vol(K, F - h)) - base, 1e-4)
    pu = float(black76_price(F + h, K, T, vu, is_call))
    pd_ = float(black76_price(F - h, K, T, vd, is_call))
    return (pu - pd_) / (2 * h)
