# Changes in v2 (27 Nov) – on top of Carlo's version of 26 Nov

All changes were kept only if they improve (or leave unchanged) the S4 metric on **all five** test collars:
training (Feb 2021), pseudo-OOS (Dec 2020), and the Dec 2020 collar re-emitted on 2 Mar, 1 Apr and 1 Jun 2020.

| Change | Where | Why |
|---|---|---|
| Smile calibrated on **settlement** prices instead of bid/ask mid | `config.py → smile_price`, `vol.py` | hedge options are marked every day at settlement: collar and hedges are now marked consistently |
| **Business-day time** for volatility (`/252`, weekends do not accrue variance) | `config.py → weekend_weight`, `fixing_calendar.year_frac`, `data.py`, `pricer.py`, `backtest.py` | removes the Monday theta jumps of the collar (largest near the averaging window); K_put unchanged |
| **Opportunistic build** of the static hedge: on days when an option's spread is tighter than on day 0, buy up to twice the daily slice (never behind schedule) | `backtest.py`, `strategy.py → static_opportunistic` | spreads of the same option vary a lot day to day (put 45 Apr21: 0.40 → 0.10) |
| Static wings **re-aligned to the collar vega during the whole life** (was: last 75 days only) | `strategy.py → static_trim_days` | same logic as Carlo's trim, applied throughout |

Metric of S4 (USD², ×1e8):

| Collar | Carlo 26/11 | v2 |
|---|---|---|
| Training (Feb 2021) | 2.18 | **1.96** |
| Pseudo-OOS (Dec 2020) | 2.51 | **2.01** |
| Re-emitted 2 Mar 2020 | 2.91 | **2.65** |
| Re-emitted 1 Apr 2020 | 4.23 | **3.71** |
| Re-emitted 1 Jun 2020 | 3.07 | **2.81** |

Further check on five more collars never used for any choice (S4, ×1e8):

| Collar | Carlo 26/11 | v2 |
|---|---|---|
| Dec 2020 average, emitted 3 Feb 2020 | 8.59 | 8.33 |
| Dec 2020 average, emitted 4 May 2020 | 4.86 | 5.39 |
| Dec 2020 average, emitted 1 Jul 2020 | 1.88 | 1.71 |
| Dec 2020 average, emitted 3 Aug 2020 | 1.15 | 1.23 |
| Jan 2021 average (Mar21/Apr21), emitted 6 Jan 2020 | 1.82 | 1.29 |

v2 is better on 8 of 10 collars (about −6% on average on the five extra ones). The least robust change is the
opportunistic build: it helps on the training (−6%), Dec 2020 (−12%) and Jan 2021 (−21%) collars but costs
+10% on the May and Aug 2020 emissions; it can be switched off with `static_opportunistic=False`.

Tried and discarded (worse on most collars): put/call twin with the lower spread for the static hedge;
static strikes chosen for the lowest cost per vega; dynamic option near the money with a slower build;
dynamic option on the front contract (better only on the training set); different bands and exit/close days.
A gamma layer with shorter-dated listed options was analysed but not implemented: in Jan-Mar 2020 the shortest
available options (Feb21, Mar21) have only ~1.2x the gamma per vega of the Apr21 options, so neutralising the
residual would require positions several times larger than the whole static hedge.

Tests: 43 passed (`python -m pytest -q tests`). The two volatility regression tests now pin the values
under the old convention (mid, calendar time); a new test checks the business-day time measure.

Note: README.md and STRATEGY.md still quote the previous numbers.
