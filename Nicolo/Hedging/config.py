from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class Structure:
    """Asian collar: fixing month, the two reference futures and the roll date (Annex A/B)."""
    legs: tuple
    front_expiry: pd.Timestamp          # futures expiry: 26 Feb 2021 (train), 31 Oct 2022 (test)
    year: int
    month: int
    start: pd.Timestamp
    k_call: float = 78.0
    # [DECIDE - ASK] the P&L^2 metric scales with notional^2: all groups must use the same convention (USD with 1e6 bbl as in the term sheet, or per barrel)
    notional: float = 1_000_000

    @property
    def fix_dates(self):
        first = pd.Timestamp(self.year, self.month, 1)
        return pd.bdate_range(first, first + pd.offsets.MonthEnd(0))

    @property
    def fix_leg(self):
        """0 = front contract on fixings strictly before its expiry, 1 = next contract from expiry on."""
        return (self.fix_dates >= self.front_expiry).astype(int)

    @property
    def end(self):
        return self.fix_dates[-1]

    def forwards(self, fut, t):
        return np.array([fut[l].settle.asof(t) for l in self.legs])


TRAIN = Structure(("COJ21", "COK21"), pd.Timestamp("2021-02-26"), 2021, 2, pd.Timestamp("2020-01-06"))
TEST = Structure(("COZ22", "COF23"), pd.Timestamp("2022-10-31"), 2022, 10, pd.Timestamp("2021-04-06"))


@dataclass(frozen=True)
class Params:
    # [DECIDE] calibrate smiles only on strikes that are multiples of strike_step (None = use all strikes).
    # full_df has quotes on every 1-USD strike only until 1 Jun 2020; using them makes the smile jump on 2 Jun 2020 (artificial P&L jump)
    strike_step: float = 5.0
    # [DECIDE] liquidity-filter thresholds are my choice; varying them moves K_put but not much
    min_bid: float = 0.05
    max_rel_spread: float = 0.75    # tuned (was 0.5)
    min_days: int = 7
    min_points: int = 4
    # smile / correlation
    # [DECIDE] pchip vs quad: K_put 44.17 vs 44.08, metric 1.6e9 vs 1.9e9
    smile: str = "pchip"            # "pchip" | "quad"
    rho_window: int = 30
    rho_min_obs: int = 10
    # [DECIDE] no history on 6 Jan 2020: rho is fixed until rho_min_obs returns are available (rolling rho in sample: 0.998-0.9999)
    rho_init: float = 0.999
    # pricing
    mc_paths: int = 200_000
    seed: int = 0
    # hedging
    # finite-difference steps for the Greeks
    bump_f: float = 1.0             # USD, finite-difference step on futures (was 0.05)
    bump_v: float = 0.005           # absolute step on vol (parallel smile shift)
    # [DECIDE] hedge_band and opt_tol were tuned on the training set: the metric varies a lot (opt_tol 0.1/0.25/0.5 -> 1.5e9/1.6e9/3.7e9).
    # Robust choice: do not over-optimise them in-sample, or agree on them with the group.
    hedge_band: float = 0.32        # keep held hedge option while |ln(K/F)| <= band (was 0.30)
    opt_tol: float = 0.45           # no-trade band: re-trade options only if target moves > tol (relative) (was 0.25)
