import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import plots
from config import TEST, TRAIN, Params
from data import chains_by_date, load_futures, load_options, load_sofr
from hedging import backtest, ranking_metric
from pricing import fixing_state, mc_convergence, mc_price, mm_price, validate_black, zero_cost_put
from vol import build_smiles, rolling_rho


def main(a):
    st, p, out = (TRAIN if a.mode == "train" else TEST), Params(), Path(a.out)
    out.mkdir(exist_ok=True)

    opts = load_options(a.options)
    fut, sofr, chains = load_futures(opts, a.xlsx), load_sofr(opts), chains_by_date(opts)
    smiles, iv = build_smiles(opts, st.legs, p)
    rho = rolling_rho(fut, st.legs, p, sorted(opts.date.unique()))

    # ---- pricing at emission
    t0 = st.start
    f0 = st.forwards(fut, t0)
    sm0, r0, fx0 = tuple(smiles[t0][l] for l in st.legs), rho.loc[t0], fixing_state(st, fut, t0)
    k_put = zero_cost_put(st.k_call, f0, sm0, r0, fx0)
    call_mm = mm_price(st.k_call, True, f0, sm0, r0, fx0)
    put_mm = mm_price(k_put, False, f0, sm0, r0, fx0)
    call_mc = mc_price(st.k_call, True, f0, sm0, r0, fx0, p.mc_paths, p.seed)
    put_mc = mc_price(k_put, False, f0, sm0, r0, fx0, p.mc_paths, p.seed)
    print(f"\nEmission {t0:%Y-%m-%d}: K_put = {k_put:.4f}  (zero-cost collar: Call({st.k_call:g}) = Put({k_put:.2f}))\n")
    emission = pd.DataFrame({"Moment matching": [call_mm, put_mm], "Monte Carlo": [call_mc[0], put_mc[0]],
                             "MC std err": [call_mc[1], put_mc[1]]}, index=[f"Call {st.k_call:g}", f"Put {k_put:.2f}"])
    print(emission.to_string(formatters={"MC std err": "{:.6f}".format}, float_format=lambda v: f"{v:,.4f}"))

    # ---- validation
    F, sigma, K = 60.0, 0.35, 62.0
    val = validate_black(p.mc_paths, p.seed, F, sigma, K)
    val = val.assign(call=np.where(val.call, "Call", "Put")).rename(columns={
        "T": "Maturity", "call": "Type", "black": "Black-76", "mm": "MomMatch",
        "mc": "MonteCarlo", "se": "MC_SE", "z": "z"})
    print(f"\nValidation: option on ONE future (F={F:g}, K={K:g}, vol={sigma:.0%}) with a single fixing at maturity.")
    print("The basket collapses to a plain Black-76 option, so MomMatch and MonteCarlo must reproduce Black-76.")
    print("Maturity in years | Black-76 = exact price | MC_SE = MC standard error | z = (MonteCarlo - Black-76)/MC_SE\n")
    print(val.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\nOK if MomMatch = Black-76 and |z| < 2 (MC within its statistical error).")

    n, m, se = mc_convergence(st.k_call, True, f0, sm0, r0, fx0, p.mc_paths, p.seed)
    plots.plot_mc_convergence(n, m, se, call_mm, "Call MC vs moment matching", out / "mc_convergence.png")
    plots.plot_smiles(iv, smiles, st.legs, t0, out / "smiles.png")
    plots.plot_rho(rho.loc[st.start:st.end], out / "rho.png")

    # ---- hedging backtest
    res = backtest(st, p, k_put, fut, sofr, chains, smiles, rho)
    res.to_csv(out / "backtest.csv")
    plots.plot_portfolio(res, out / "portfolio.png")
    plots.plot_pnl2(res, out / "pnl2.png")
    plots.plot_hedge(res, out / "hedge.png")
    print(f"\nBacktest {res.index[0]:%Y-%m-%d} -> {res.index[-1]:%Y-%m-%d} ({len(res)} days)")
    print(f"  Mean squared daily P&L : {ranking_metric(res):,.4e} USD^2")
    print(f"  Final portfolio value  : {res.W.iloc[-1]:,.0f} USD")
    print(f"  Trading costs (bid/ask) : options {res.cost_opt.sum():,.0f} USD | futures {res.cost_fut.sum():,.0f} USD")
    print(f"  Outputs saved in       : {out.resolve()}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "test"], default="train")
    ap.add_argument("--options", nargs="+", default=[here / "unpivoted_bid_offer_train_clean.parquet", here / "full_df_train.parquet"])
    ap.add_argument("--xlsx", default=here / "COJ21_COZ22_historical_series.xlsx")
    ap.add_argument("--out", default=here / "out")
    main(ap.parse_args())
