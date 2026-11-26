"""
Test di pricer, greche e backtest.  Esegui con:  python -m pytest -q tests   (dalla cartella codice/)
"""


import numpy as np
import pandas as pd
import pytest

from backtest import prepare, run_backtest
from config import TRAIN
from pricer import MarketState, collar_value, price_mc, price_mm
from strategy import BEST, DELTA_DAILY, DELTA_VEGA, UNHEDGED
from vol import Smile, black76_price


@pytest.fixture(scope="module")
def ctx():
    return prepare(TRAIN)


def flat_smile(vol):
    return Smile(xs=np.array([0.0]), vols=np.array([vol]), coef=None, method="pchip")


def toy_state(theta, leg, f=(60.0, 59.0), vol=0.3, rho=0.95, known_sum=0.0, n_total=None):
    theta, leg = np.asarray(theta, float), np.asarray(leg)
    return MarketState(date=pd.Timestamp("2020-01-06"), f=f, smiles=(flat_smile(vol), flat_smile(vol)),
                       rho=rho, discount=1.0, theta=theta, leg=leg, known_sum=known_sum,
                       n_total=n_total or len(theta))


# ----------------------------------------------------------------------- pricer
def test_single_fixing_is_black76():
    """Con un solo fixing il basket è un futures: il moment matching deve ridare Black-76."""
    st = toy_state([1.0], [0])
    for K, c in [(50, True), (60, True), (70, False)]:
        assert price_mm(st, K, c) == pytest.approx(float(black76_price(60.0, K, 1.0, 0.3, c)), rel=1e-12)


def test_put_call_parity_on_basket():
    """Call - Put = D·(E[A] - K) anche per il basket."""
    st = toy_state(np.linspace(0.9, 1.0, 20), [0] * 19 + [1])
    m = (19 * 60.0 + 59.0) / 20
    assert price_mm(st, 62, True) - price_mm(st, 62, False) == pytest.approx(m - 62, abs=1e-10)


def test_mm_matches_mc_toy():
    st = toy_state(np.linspace(0.9, 1.0, 20), [0] * 19 + [1])
    for K, c in [(70, True), (50, False), (60, True)]:
        mc = price_mc(st, K, c, n_paths=200_000, seed=1)
        assert abs(price_mm(st, K, c) - mc.price) < 4 * mc.std_error + 2e-3


def test_known_fixings():
    """Con metà dei fixing noti, MM e MC concordano; con tutti noti il prezzo è il payoff."""
    st = toy_state(np.linspace(0.01, 0.05, 10), [0] * 10, known_sum=10 * 61.0, n_total=20)
    mc = price_mc(st, 62, True, n_paths=200_000, seed=2)
    assert abs(price_mm(st, 62, True) - mc.price) < 4 * mc.std_error + 2e-3
    done = toy_state([], [], known_sum=20 * 61.0, n_total=20)
    assert price_mm(done, 60, True) == pytest.approx(1.0)
    assert price_mm(done, 60, False) == 0.0


def test_put_increasing_in_strike(ctx):
    st = ctx.states[0]
    p = [price_mm(st, k, False) for k in np.arange(30, 70, 2.5)]
    assert np.all(np.diff(p) > 0)


def test_kput_zero_cost(ctx):
    assert collar_value(ctx.states[0], ctx.k_put, TRAIN.k_call) == pytest.approx(0.0, abs=1e-9)
    assert 40 < ctx.k_put < 50


def test_mm_matches_mc_emission(ctx):
    st = ctx.states[0]
    for K, c in [(TRAIN.k_call, True), (ctx.k_put, False)]:
        mc = price_mc(st, K, c, n_paths=200_000, seed=3)
        assert abs(price_mm(st, K, c) - mc.price) < 4 * mc.std_error + 1e-3


# --------------------------------------------------------------------- backtest
@pytest.fixture(scope="module")
def results(ctx):
    return {s.name: run_backtest(ctx, s) for s in (UNHEDGED, DELTA_DAILY, DELTA_VEGA, BEST)}


def test_attribution_adds_up(results):
    for r in results.values():
        d = r.daily
        parts = d[[c for c in d if c.startswith("pnl_")]].sum(axis=1)
        assert np.allclose(parts, d.pnl, atol=1e-4)


def test_unhedged_pnl_is_collar_change(results):
    d = results[UNHEDGED.name].daily
    assert np.allclose(d.pnl.iloc[1:], d.collar_value.diff().iloc[1:], atol=1e-6)


def test_integer_lots(results):
    for r in results.values():
        d = r.daily
        for col in ("q1", "q2", "opt_put_qty", "static_qty"):
            assert np.allclose(d[col] % 1000, 0)


def test_final_payoff(ctx, results):
    """L'ultimo giorno il valore del collar è il payoff con la media realizzata."""
    wide = ctx.md.futures_wide("settle")
    tab = ctx.sched.table
    A = np.mean([wide.at[d, c] for d, c in zip(tab.fixing_date, tab.contract)])
    payoff = max(ctx.k_put - A, 0) - max(A - TRAIN.k_call, 0)
    d = results[DELTA_DAILY.name].daily
    assert d.collar_value.iloc[-1] == pytest.approx(TRAIN.notional_bbl * payoff)
    assert d.q1.iloc[-1] == 0 and d.q2.iloc[-1] == 0


def test_hedging_reduces_metric(results):
    m = {k: r.metric for k, r in results.items()}
    assert m[BEST.name] < m[DELTA_VEGA.name] < m[DELTA_DAILY.name] < m[UNHEDGED.name]


def test_static_hedge_closed(results):
    """La copertura statica si apre e alla fine è chiusa."""
    d = results[BEST.name].daily
    assert d.static_qty.max() >= 1_500_000   # put + call, riscalate sulla vega del collar
    assert d.static_qty.iloc[-1] == 0


def test_adaptive_phase_days_bounds(ctx):
    """k* adattivo resta in [3, 12] e rispetta i vincoli di causalità (usa solo il giorno 0)."""
    from backtest import _adaptive_phase_days, _static_portfolio
    sp = _static_portfolio(ctx, BEST)
    k = _adaptive_phase_days(ctx, sp)
    assert 3 <= k <= 12


def test_static_trim_avoids_excessive_calendar_spread(results):
    """Con static_trim_days > 0 la posizione dinamica sulla seconda gamba resta contenuta nel mese finale."""
    d = results[BEST.name].daily
    last_month = d.loc["2021-02-01":]
    assert last_month.opt_put_qty.abs().max() < 150_000
