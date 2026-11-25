import numpy as np
import pandas as pd

COLS = ["date", "option_ticker", "option_type", "strike", "expiration", "settlement_price",
        "actual_underlying_ticker", "underlying_settlement", "underlying_bid", "underlying_ask",
        "option_bid", "option_ask", "sofr_rate"]


def load_options(paths):
    df = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths])      # first file has priority
    df = df.groupby(["date", "option_ticker"], sort=False).first().reset_index().rename(columns={
        "actual_underlying_ticker": "leg", "underlying_settlement": "F", "option_bid": "bid",
        "option_ask": "ask", "settlement_price": "settle", "option_type": "cp"})
    df["cp"] = df["cp"].str[0]
    df["mid"] = (df.bid + df.ask) / 2
    df["days"] = (df.expiration - df.date).dt.days
    df["tau"] = df.days / 365
    return df.sort_values(["date", "option_ticker"]).reset_index(drop=True)


def load_futures(opts, xlsx):
    """Futures settle/bid/ask per leg; the Excel series take priority and extend the parquet ones.
    [ASSUMPTION] the xlsx is the source of the COJ21 fixings on 24-26 Feb 2021 (the options no longer exist, the future still does)."""
    f = opts.groupby(["leg", "date"]).agg(settle=("F", "first"), bid=("underlying_bid", "first"),
                                          ask=("underlying_ask", "first")).reset_index()
    x = pd.concat([s.rename(columns={"settlement": "settle"}).assign(leg=k)
                   for k, s in pd.read_excel(xlsx, sheet_name=None).items()])
    f = pd.concat([x, f]).groupby(["leg", "date"], sort=True).first().reset_index().dropna(subset=["settle"])
    return {k: g.set_index("date")[["settle", "bid", "ask"]] for k, g in f.groupby("leg")}


def load_sofr(opts):
    return (opts.groupby("date").sofr_rate.first() / 100).sort_index().ffill().bfill()


def chains_by_date(opts):
    return {d: g.set_index("option_ticker") for d, g in opts.groupby("date")}
