"""
Correlazione tra i due futures del collar (es. Apr21 e May21).

Si usa una media mobile esponenziale (EWMA, come RiskMetrics) dei log-rendimenti
giornalieri r_i(t) = ln(F_i(t) / F_i(t-1)):

    cov(t)   = λ·cov(t-1)   + (1-λ)·r1(t)·r2(t)
    var_i(t) = λ·var_i(t-1) + (1-λ)·r_i(t)²
    ρ(t)     = cov(t) / sqrt(var_1(t)·var_2(t))

Con λ = 0,94 i rendimenti di circa un mese fa pesano già poco: ρ si adatta in
fretta ai cambi di regime. Ogni giorno si usano solo i rendimenti fino a quel
giorno (niente informazioni dal futuro). Finché i rendimenti disponibili sono
meno di corr_min_obs si usa il valore iniziale rho_prior.

Nei dati di training ρ ≈ 0,9997: i due futures si muovono quasi identici, quindi
ρ pesa pochissimo sul prezzo del collar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CollarConfig
from data import MarketData


def ewma_correlation(md: MarketData, cfg: CollarConfig) -> pd.Series:
    """ρ(t) per ogni giorno in cui entrambe le gambe hanno un prezzo."""
    m = cfg.model
    px = md.futures_wide("settle")[list(cfg.legs)].dropna()
    r = np.log(px).diff().dropna()
    lam = m.ewma_lambda
    cov = v1 = v2 = None
    out = {px.index[0]: m.rho_prior}
    for i, (d, (a, b)) in enumerate(r.iterrows(), start=1):
        if cov is None:
            cov, v1, v2 = a * b, a * a, b * b
        else:
            cov = lam * cov + (1 - lam) * a * b
            v1 = lam * v1 + (1 - lam) * a * a
            v2 = lam * v2 + (1 - lam) * b * b
        rho = cov / np.sqrt(v1 * v2) if v1 > 0 and v2 > 0 else m.rho_prior
        out[d] = float(np.clip(rho, -0.999999, 0.999999)) if i >= m.corr_min_obs else m.rho_prior
    s = pd.Series(out, name="rho")
    s.index.name = "date"
    return s


def rho_on(rho: pd.Series, date) -> float:
    """ρ da usare in una data: l'ultimo valore disponibile fino a quella data."""
    s = rho[rho.index <= pd.Timestamp(date)]
    return float(s.iloc[-1]) if len(s) else float(rho.iloc[0])
