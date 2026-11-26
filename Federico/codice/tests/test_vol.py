"""
Test delle volatilità implicite. Esegui con:  python -m pytest -q   (dalla cartella codice/)
"""
import numpy as np
import pandas as pd
import pytest

from config import TRAIN
from data import load_market_data
from vol import atm_vol, black76_price, black76_vega, compute_implied_vols, implied_vol


@pytest.fixture(scope="module")
def iv():
    return compute_implied_vols(load_market_data(TRAIN), TRAIN)


# ----------------------------------------------------------- formula e inversione
def test_roundtrip():
    """prezzo(σ) -> vol implicita deve ridare σ, per call e put, su molti casi."""
    rng = np.random.default_rng(0)
    F = rng.uniform(30, 120, 500)
    K = F * rng.uniform(0.5, 1.6, 500)
    T = rng.uniform(0.02, 2.0, 500)
    s = rng.uniform(0.1, 1.2, 500)
    c = rng.random(500) < 0.5
    p = black76_price(F, K, T, s, c)
    ok = p > 1e-6          # opzioni con prezzo praticamente nullo non sono invertibili
    assert np.allclose(implied_vol(p[ok], F[ok], K[ok], T[ok], c[ok]), s[ok], atol=1e-7)


def test_put_call_parity():
    """Con r = 0: C - P = F - K."""
    F, K, T, s = 61.4, 70.0, 1.13, 0.25
    assert black76_price(F, K, T, s, True) - black76_price(F, K, T, s, False) == pytest.approx(F - K)


def test_vega_matches_finite_difference():
    F, K, T, s, h = 61.4, 65.0, 1.13, 0.24, 1e-5
    fd = (black76_price(F, K, T, s + h, True) - black76_price(F, K, T, s - h, True)) / (2 * h)
    assert black76_vega(F, K, T, s) == pytest.approx(fd, rel=1e-6)


def test_arbitrage_bounds_give_nan():
    # call sotto il valore intrinseco e sopra F: nessuna vol possibile
    assert np.isnan(implied_vol(5.0, 61.4, 55.0, 1.0, True))     # intrinseco = 6.4
    assert np.isnan(implied_vol(62.0, 61.4, 55.0, 1.0, True))    # > F
    assert np.isnan(implied_vol(1.0, 61.4, 55.0, 0.0, True))     # scaduta


# ------------------------------------------------------------------- dati reali
def test_emission_values(iv):
    e = iv[(iv.date == TRAIN.emission_date) & (iv.contract == "COJ21") & iv.keep].set_index(["strike", "cp"])
    assert e.loc[(65.0, "C"), "iv_mid"] == pytest.approx(0.2379, abs=1e-4)
    assert e.loc[(45.0, "P"), "iv_mid"] == pytest.approx(0.2880, abs=1e-4)


def test_bid_mid_ask_ordering(iv):
    k = iv[iv.keep & iv.iv_bid.notna() & iv.iv_ask.notna()]
    assert (k.iv_bid <= k.iv_mid + 1e-12).all() and (k.iv_mid <= k.iv_ask + 1e-12).all()


def test_filters_respected(iv):
    k = iv[iv.keep]
    f = TRAIN.iv_filters
    assert k.otm.all()
    assert (k.mid >= f.min_price).all()
    assert (k.rel_spread <= f.max_rel_spread).all()
    assert (k.ttm_days >= f.min_ttm_days).all()
    assert k.iv_mid.between(f.min_iv, f.max_iv).all()


def test_every_backtest_day_has_options(iv):
    """Ogni giorno del backtest deve avere almeno un'opzione tenuta sul May21."""
    k = iv[iv.keep & (iv.contract == "COK21")]
    days = pd.DatetimeIndex(sorted(iv.date.unique()))
    days = days[(days >= TRAIN.emission_date) & (days <= TRAIN.backtest_end)]
    assert set(days) <= set(k.date)


def test_atm_vol_emission(iv):
    a = atm_vol(iv)
    assert a.loc[TRAIN.emission_date, "COJ21"] == pytest.approx(0.249, abs=2e-3)


def test_smooth_wing_c1_continuity():
    """Con wing_decay > 0 la derivata dello smile è continua ai bordi x_min e x_max e limitata sulle ali."""
    from vol import fit_smile
    xs = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    vols = np.array([0.30, 0.27, 0.25, 0.235, 0.225])
    sm = fit_smile(xs, vols, np.ones_like(xs), method="pchip", wing_decay=0.05)
    h = 1e-5
    for edge in (sm.x_min, sm.x_max):
        s_left = (float(sm.vol_x(edge)) - float(sm.vol_x(edge - h))) / h
        s_right = (float(sm.vol_x(edge + h)) - float(sm.vol_x(edge))) / h
        assert s_left == pytest.approx(s_right, abs=1e-3)
    # sulle ali lontane la vol resta finita, positiva e vicina al bordo
    assert 0.01 < float(sm.vol_x(-5.0)) < 0.50
    assert 0.01 < float(sm.vol_x(+5.0)) < 0.50
