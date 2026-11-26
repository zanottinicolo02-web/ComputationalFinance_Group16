"""
Test di dati e calendario. Esegui con:  python -m pytest -q   (dalla cartella codice/)
"""
from dataclasses import replace

import pandas as pd
import pytest

from config import TRAIN, TEST
from data import load_market_data
from fixing_calendar import (build_fixing_schedule, contract_code, front_and_next,
                             futures_expiry, parse_contract)


@pytest.fixture(scope="module")
def md():
    return load_market_data(TRAIN)


# ----------------------------------------------------------------- contratti
def test_parse_contract():
    assert parse_contract("COJ21") == (2021, 4)
    assert parse_contract("COF23") == (2023, 1)
    assert contract_code(2022, 12) == "COZ22"


@pytest.mark.parametrize("code, expiry", [
    ("COJ21", "2021-02-26"),   # Apr21
    ("COK21", "2021-03-31"),   # May21
    ("COZ22", "2022-10-31"),   # Dec22
    ("COF23", "2022-11-30"),   # Jan23
])
def test_futures_expiry(code, expiry):
    assert futures_expiry(code) == pd.Timestamp(expiry)


def test_front_and_next():
    assert front_and_next(2021, 2) == ("COJ21", "COK21")
    assert front_and_next(2022, 10) == ("COZ22", "COF23")


# -------------------------------------------------------------------- fixing
def test_training_schedule_futures_rule():
    s = build_fixing_schedule(TRAIN)
    assert s.N == 20
    assert (s.n1, s.n2) == (19, 1)          # confermato dai prof nel Q&A del 10/11
    assert s.table.fixing_date.iloc[s.n1 - 1] == pd.Timestamp("2021-02-25")


def test_training_schedule_option_rule(md):
    cfg = replace(TRAIN, roll_rule="option_expiry")
    s = build_fixing_schedule(cfg, md.option_expiry("COJ21"))
    assert md.option_expiry("COJ21") == pd.Timestamp("2021-02-23")
    assert (s.n1, s.n2) == (16, 4)


def test_test_schedule():
    s = build_fixing_schedule(TEST)
    assert s.N == 21
    assert (s.n1, s.n2) == (20, 1)          # Dec22 fino al 28/10, Jan23 il 31/10 (Q&A)
    assert s.table.fixing_date.iloc[s.n1 - 1] == pd.Timestamp("2022-10-28")


def test_year_fractions_positive():
    s = build_fixing_schedule(TRAIN)
    th = s.year_fractions(TRAIN.emission_date)
    assert (th > 0).all() and (pd.Series(th).diff().dropna() > 0).all()


# ---------------------------------------------------------------------- dati
def test_data_range(md):
    assert md.dates[0] == TRAIN.emission_date
    assert md.dates[-1] >= TRAIN.backtest_end - pd.Timedelta(days=5)


def test_emission_prices(md):
    f = md.futures_wide("settle").loc[TRAIN.emission_date]
    assert f["COJ21"] == pytest.approx(61.40)
    assert f["COK21"] == pytest.approx(61.05)
    assert md.sofr.loc[TRAIN.emission_date] == pytest.approx(0.0155)


def test_futures_bidask_complete(md):
    legs = md.futures[md.futures.contract.isin(TRAIN.legs)]
    assert legs.bid.notna().all() and legs.ask.notna().all()


def test_strike_grid(md):
    """Dataset del 31/10: griglia a passi di 5 $ in tutto il periodo.
    Lo strike 78 quindi non è quotato: all'emissione ci sono 75 e 80."""
    assert (md.options.strike % 5 == 0).all()
    chain = md.options_on(TRAIN.emission_date, "COJ21")
    calls = set(chain[chain.cp == "C"].strike)
    assert {75.0, 80.0} <= calls and TRAIN.k_call not in calls


@pytest.mark.parametrize("rule", ["futures_expiry", "option_expiry"])
def test_fixings_have_prices(md, rule):
    """Per ogni fixing deve esistere il settlement del contratto di riferimento.
    I prezzi Apr21 del 24-26/2/2021 arrivano dal file Excel dei prof."""
    cfg = replace(TRAIN, roll_rule=rule)
    s = build_fixing_schedule(cfg, md.option_expiry("COJ21"))
    wide = md.futures_wide("settle")
    missing = [(d.date(), c) for d, c in zip(s.table.fixing_date, s.table.contract)
               if d not in wide.index or pd.isna(wide.at[d, c])]
    assert not missing, f"Prezzi mancanti per i fixing: {missing}"


def test_apr21_after_option_expiry(md):
    """Dopo il 23/2 l'Apr21 non è nel parquet ma deve arrivare dall'Excel."""
    wide = md.futures_wide("settle")
    assert wide.at[pd.Timestamp("2021-02-24"), "COJ21"] == pytest.approx(67.04)
    assert wide.at[pd.Timestamp("2021-02-26"), "COJ21"] == pytest.approx(66.13)
