# Hedging Competition – Python implementation (Federico)

Pricing and hedging of the zero-cost Asian collar on Brent (Computational Finance 2026/27, Group 16).
The market maker sells the collar (client: long Call 78, short Put K_put on the February 2021 average of
Apr21/May21 futures, 1M bbl) and hedges it daily. Ranking metric: (1/T) · Σ P&L_t² in USD² on 1M bbl.

## Results (training: collar on the Feb 2021 average, 6 Jan 2020 → 26 Feb 2021)

K_put = 44.22 USD/bbl (Call 78 = Put K_put = 1.09 USD/bbl). Version v2 (27 Nov): see `CHANGES_v2.md` for what changed and the robustness checks.

| Strategy | Metric (USD²) | Typical daily P&L (RMS) |
|---|---|---|
| S0 No hedge | 3.31e11 | 575k |
| S1 Delta with futures | 3.41e10 | 185k |
| S2 Delta + vega with one listed option | 2.00e9 | 45k |
| S3 Static "opposite collar" + delta | 2.19e8 | 15k |
| **S4 Static opposite collar + delta + vega (chosen)** | **1.96e8** | **14k** |

Second check on a collar the code was never tuned on (Dec 2020 average, Feb21/Mar21 futures): S1 4.42e10, S2 1.93e9, **S4 2.01e8**.

Conventions: day 0 (cost of the initial hedges) is included in the metric; futures and options are
traded in whole lots of 1,000 bbl; buy at ask, sell at bid; futures-style margining; cash earns SOFR.

## What the model does

1. **Data**: Apr21/May21 futures, listed options and SOFR from the professors' clean dataset.
2. **Fixing calendar**: 20 business days of Feb 2021 → 19 fixings on Apr21, the last one (26 Feb) on May21 (Q&A of 10 Nov).
3. **Implied vols**: Black-76 (r = 0) on liquid OTM options, from settlement prices (the same prices used to mark the hedges); time measured in business days (/252, no variance over weekends); every day a PCHIP smile in log-moneyness ln(K/F) for each leg, with smooth (C¹, tanh-damped) wings outside the quoted strikes.
4. **Correlation**: EWMA (λ = 0.94) of the two futures' log-returns.
5. **Pricing**: moment matching (lognormal approximation of the average), validated with Monte Carlo (antithetic + geometric control variate); K_put solved so that the collar costs zero.
6. **Greeks**: bump and reprice; the smile moves with the futures (sticky moneyness), also for the delta of the listed hedge options.
7. **Hedging (S4)**:
   - *static part*: sell Apr21 puts at strike 45 and buy Apr21 calls at 80/75 (60%/40%), sized so that their vega equals the vega of the collar's put and call legs. Built over k* = 3–12 days (adaptive: spread cost vs volatility risk), buying more on days when the spread is tight; realigned to the collar vega whenever it is over-hedged by more than 15%; closed gradually over 5 days before expiry (options whose time value is below the bid-ask are left to expire);
   - *dynamic part*: once the static hedge is built, one May21 option offsets the remaining vega (rebalanced only if the target moves by more than 5%);
   - *delta*: futures on both legs, in whole lots.

## Files

```
codice/
  config.py            all settings: dates, strikes, notional, filters, model choices; collars TRAIN, PSEUDO_OOS, TEST
  data.py              reads the parquet and the Excel file → futures, options, SOFR tables
  fixing_calendar.py   fixing dates and roll rule
  vol.py               Black-76, implied vols, liquidity filters, daily smiles
  corr.py              EWMA correlation
  pricer.py            collar price (moment matching + Monte Carlo), K_put
  greeks.py            delta and vega of the collar, smile-consistent delta of listed options
  strategy.py          strategies S0–S4 as parameter sets (BEST = S4) + variants used in the report
  backtest.py          prepare(): smiles, prices and Greeks for every day (computed once);
                       run_backtest(): daily simulation of a strategy (trades, costs, P&L and its decomposition)
  report_plots.py      all charts
  main.py              runs everything, prints a summary table, writes output/
  tests/               automated checks (see below)
  requirements.txt
dati/                  dataset of the professors (parquet + Excel futures series)
```

`output/` is not in the repository: it is created by `main.py` (charts, daily CSV of every strategy,
`risultati.json` with all numbers).

## How to run

```bash
cd codice
python -m venv ../.venv
source ../.venv/bin/activate        # Windows: ..\.venv\Scripts\activate
pip install -r requirements.txt
python main.py                      # training collar + second check (about 1-2 minutes)
python main.py --test               # Oct 2022 test collar, once the test data is in dati/ (file name in config.py → TEST.data_file)
```

## Tests: what they are for

`codice/tests/` contains 43 automated checks. They are **not** needed to run the project: they exist so
that anyone who edits the code can see in ~10 seconds whether something broke.

```bash
cd codice
python -m pytest -q tests           # expected: "43 passed"
```

They check, among other things:
- **data and calendar**: futures expiries; 19 fixings on Apr21 and 1 on May21 (20/1 for the test collar); every fixing day has a price;
- **volatility**: inverting Black-76 gives back the option price; put-call parity; vega matches a finite difference; prices outside no-arbitrage bounds give no vol; the liquidity filters are respected;
- **pricing**: with a single fixing the basket price equals Black-76; moment matching agrees with Monte Carlo; known fixings are handled correctly; Call 78 = Put K_put at emission;
- **backtest**: the five P&L components add up to the daily P&L; positions are whole multiples of 1,000 bbl; at the end the collar is worth its payoff; the static hedge is opened and closed; hedging reduces the metric (S4 < S2 < S1 < S0).

Run them after every change. If one fails, the message says which check and why.

## Where to change things

- a parameter (filters, smile method, bands, days to build the static hedge): `config.py` or `strategy.py`;
- a new strategy: add one line in `strategy.py`, the backtest does not need to change;
- the test collar: `config.py → TEST` (already set: emission 6 Apr 2021, Dec22/Jan23, Call 78).
