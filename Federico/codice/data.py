"""
Caricamento e pulizia dei dati.

Il parquet ha una riga per (data, opzione). Da qui costruiamo:
  - futures: una riga per (data, contratto) con settlement, bid, ask, mid, spread.
             I giorni in cui un futures manca dal parquet (dopo la scadenza delle
             sue opzioni) vengono presi dal file Excel delle serie storiche.
  - options: una riga per (data, opzione) con strike, tipo, scadenza, prezzi, ttm
  - sofr:    una serie giornaliera del tasso SOFR overnight (in decimali)

Convenzioni:
  - prezzi in $/bbl
  - ttm = tempo alla scadenza dell'opzione in anni = giorni / 365
    (NB: non chiamare una colonna "T": si scontra con DataFrame.T, la trasposta di pandas)
  - sofr in decimali (1,55% -> 0.0155), giorni mancanti riempiti col valore precedente
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CollarConfig


@dataclass
class MarketData:
    futures: pd.DataFrame   # colonne: date, contract, settle, bid, ask, mid, spread
    options: pd.DataFrame   # colonne: date, contract, option_expiry, strike, cp, settle, bid, ask, mid, spread, F, ttm_days, ttm
    sofr: pd.Series         # indice: date, valori in decimali
    dates: pd.DatetimeIndex # giorni di borsa presenti nel dataset

    # ------------------------------------------------------------------ utility
    def futures_wide(self, field: str = "settle") -> pd.DataFrame:
        """Tabella data × contratto con il campo richiesto (es. 'settle', 'mid')."""
        return self.futures.pivot(index="date", columns="contract", values=field)

    def option_expiry(self, contract: str) -> pd.Timestamp:
        """Scadenza delle opzioni scritte sul futures `contract`."""
        exp = self.options.loc[self.options.contract == contract, "option_expiry"].unique()
        if len(exp) != 1:
            raise KeyError(f"Scadenza opzioni non trovata (o non unica) per {contract}")
        return pd.Timestamp(exp[0])

    def options_on(self, date, contract: str, quoted_only: bool = True) -> pd.DataFrame:
        """Catena di opzioni su `contract` nel giorno `date`."""
        date = pd.Timestamp(date)
        x = self.options[(self.options.date == date) & (self.options.contract == contract)]
        if quoted_only:
            x = x[x.bid.notna()]
        return x.sort_values(["cp", "strike"])


def load_market_data(cfg: CollarConfig) -> MarketData:
    raw = pd.read_parquet(cfg.data_file)
    _check_raw(raw)

    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw["expiration"] = pd.to_datetime(raw["expiration"]).dt.normalize()

    # --- futures: una riga per (data, contratto) ---------------------------
    fut = (
        raw[["date", "future_contract_code", "underlying_settlement", "underlying_bid", "underlying_ask"]]
        .drop_duplicates(["date", "future_contract_code"])
        .rename(columns={
            "future_contract_code": "contract",
            "underlying_settlement": "settle",
            "underlying_bid": "bid",
            "underlying_ask": "ask",
        })
    )
    dates = pd.DatetimeIndex(sorted(raw["date"].unique()))
    fut = _add_futures_fixings(fut, cfg, dates)
    fut["mid"] = (fut["bid"] + fut["ask"]) / 2
    fut["spread"] = fut["ask"] - fut["bid"]
    fut = fut.sort_values(["contract", "date"]).reset_index(drop=True)

    # --- opzioni -----------------------------------------------------------
    opt = raw.rename(columns={
        "future_contract_code": "contract",
        "expiration": "option_expiry",
        "settlement_price": "settle",
        "option_bid": "bid",
        "option_ask": "ask",
        "underlying_settlement": "F",
        "time_to_expiry_days": "ttm_days",
    })
    opt["cp"] = opt["option_type"].map({"Call": "C", "Put": "P"})
    opt["mid"] = (opt["bid"] + opt["ask"]) / 2
    opt["spread"] = opt["ask"] - opt["bid"]      # NB: non usare la colonna option_spread del file
    opt["ttm_days"] = opt["ttm_days"].astype("float64")
    opt["ttm"] = opt["ttm_days"] / 365.0
    opt = opt[["date", "contract", "option_expiry", "option_ticker", "strike", "cp",
               "settle", "bid", "ask", "mid", "spread", "F", "ttm_days", "ttm"]]
    opt = opt.sort_values(["date", "contract", "cp", "strike"]).reset_index(drop=True)

    # --- SOFR ----------------------------------------------------------------
    sofr =(raw.groupby("date")["sofr_rate"].first() / 100.0).reindex(dates).ffill().bfill()
    sofr.name = "sofr"

    return MarketData(futures=fut, options=opt, sofr=sofr, dates=dates)


def load_futures_series(cfg: CollarConfig) -> pd.DataFrame:
    """Serie storica del front contract dal file Excel dei prof (foglio cfg.futures_sheet).

    Serve per i giorni in cui il futures non compare nel dataset delle opzioni,
    cioè dopo la scadenza delle sue opzioni (per Apr21: 24, 25 e 26 febbraio 2021).
    """
    s = pd.read_excel(cfg.futures_file, sheet_name=cfg.futures_sheet)
    s = s.rename(columns={"settlement": "settle"})[["date", "settle", "bid", "ask"]]
    s["date"] = pd.to_datetime(s["date"]).dt.normalize()
    s["contract"] = cfg.futures_sheet
    return s


def _add_futures_fixings(fut: pd.DataFrame, cfg: CollarConfig, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Aggiunge alla tabella dei futures i giorni presenti solo nel file Excel.

    Dove entrambe le fonti hanno un prezzo, controlla che coincidano.
    """
    ext = load_futures_series(cfg)
    ext = ext[ext.date.isin(dates)]
    both = fut.merge(ext, on=["date", "contract"], suffixes=("", "_xlsx"))
    if not np.allclose(both["settle"], both["settle_xlsx"]):
        raise ValueError(f"Prezzi diversi tra parquet ed Excel per {cfg.futures_sheet}")
    known = set(zip(fut.date, fut.contract))
    extra = ext[[(d, c) not in known for d, c in zip(ext.date, ext.contract)]]
    return pd.concat([fut, extra], ignore_index=True)


def _check_raw(raw: pd.DataFrame) -> None:
    """Controlli di coerenza: se uno fallisce, il dataset non è quello che pensiamo."""
    if raw.duplicated(["date", "option_ticker"]).any():
        raise ValueError("Righe duplicate per (data, opzione)")
    q = raw.dropna(subset=["option_bid", "option_ask"])
    if (q["option_ask"] < q["option_bid"]).any():
        raise ValueError("Opzioni con ask < bid")
    f = raw.dropna(subset=["underlying_bid", "underlying_ask"])
    if (f["underlying_ask"] < f["underlying_bid"]).any():
        raise ValueError("Futures con ask < bid")
    if raw.groupby("future_contract_code")["expiration"].nunique().max() > 1:
        raise ValueError("Un contratto ha più di una scadenza delle opzioni")


def summary(md: MarketData) -> pd.DataFrame:
    """Riepilogo mensile: giorni, copertura bid/ask, opzioni quotate."""
    o = md.options.assign(month=md.options.date.dt.to_period("M"))
    f = md.futures.assign(month=md.futures.date.dt.to_period("M"))
    return pd.DataFrame({
        "giorni": o.groupby("month").date.nunique(),
        "futures_senza_bidask_%": (f.groupby("month").bid.apply(lambda s: s.isna().mean()) * 100).round(1),
        "opzioni_quotate_%": (o.groupby("month").bid.apply(lambda s: s.notna().mean()) * 100).round(1),
        "sofr_medio_%": (md.sofr.groupby(md.sofr.index.to_period("M")).mean() * 100).round(2),
    })
