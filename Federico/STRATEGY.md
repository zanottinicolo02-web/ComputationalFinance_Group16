# Federico's Hedging Strategy (`S4`) — Detailed Technical Documentation

**Project:** Computational Finance 2025/2026 — Calibration and Hedging Lab (Group 16)  
**Product:** Zero-Cost Asian Collar on ICE Brent Crude Futures (1,000,000 bbl notional)  
**Primary Strategy:** `S4_statica_delta_vega` — *Phased Static Opposite Collar + Guarded Final-Month Trim + Dynamic Residual Vega + Smile-Consistent Daily Delta Hedge*  
**Competition Objective:** Minimize mean squared daily P&L,

$$\text{MSE} = \frac{1}{T}\sum_{t=1}^{T} \text{P\&L}_t^2 \quad [\text{USD}^2]$$

computed across the entire hedging horizon (including Day 0 initial hedging spread costs).

---

## 1. Executive Summary & Economic Intuition

The market maker sells an OTC zero-cost Asian collar to a client and is therefore:
- **Long** an Asian Put at $K_{\text{put}}$ (solved at emission so that $\text{Put}(K_{\text{put}}) = \text{Call}(78)$),
- **Short** an Asian Call at $K_{\text{call}} = 78.00\text{ USD/bbl}$,
- On $N_{\text{bbl}} = 1{,}000{,}000\text{ bbl}$ of the arithmetic average of first-nearby / next-nearby ICE Brent futures over the averaging month (`Feb 2021` in training, `Oct 2022` out-of-sample).

### Why `S4` Outperforms Pure Delta and Single-Option Vega Strategies
A collar has **opposite-sign wing exposures**:
- Around $K_{\text{put}}$, the market maker is **long vega and long gamma**;
- Around $K_{\text{call}} = 78$, the market maker is **short vega and short gamma**.

Near the middle of the collar, parallel vega partially cancels ($\mathcal{V}_{\text{put}} + \mathcal{V}_{\text{call}} \approx 0$), yet the book carries massive **skew, vanna, and convexity (gamma)** risk. Hedging with futures alone (`S1`) leaves all volatility and gamma risk open; hedging with a single ATM option (`S2`) only cancels net parallel vega and forces expensive dynamic re-hedging whenever Brent moves toward either wing.

**Federico's `S4` strategy** neutralizes the structure's full risk profile in four coordinated layers:
1. **Static Opposite Collar (Front Contract):** Sells listed puts bracketing $K_{\text{put}}$ and buys listed calls bracketing $K_{\text{call}}$ on the front contract (`COJ21` / `COZ22`), scaled to match each collar leg's vega. This cancels level vega, smile skew, vanna, and most gamma in a single one-time spread payment, built gradually over $k^*$ days to prevent a large squared Day-0 cost spike.
2. **Lifecycle & Final-Month Protection:** Inside the final 75 days, trims over-hedged static Leg-1 wings directly as Asian variance shrinks (avoiding opposite-sign Leg-2 calendar spreads), phases out any required early unwind over 5 days, lets both deep-OTM and deep-ITM low-time-value options expire at zero spread, and suppresses phantom vega from expiring penny quotes.
3. **Dynamic Residual Vega (Second Contract):** Uses one liquid OTM option on the next-nearby contract (`COK21` / `COF23`, which remains alive past the averaging month) with a 5% deadband to absorb the small residual Asian-vs-European vega mismatch.
4. **Daily Smile-Consistent Delta Hedge:** Hedges total residual delta (collar + static options + dynamic option) daily on both futures legs in integer ICE lots of 1,000 bbl, computing option deltas under the same **sticky-moneyness** smile dynamics as the collar.

On the training collar (`Feb 2021`), `S4` achieves an MSE of **$2.18 \times 10^8\ \text{USD}^2$** (RMS daily P&L of **\$14.8k**, or **1.48 ¢/bbl** — a **99.93%** variance reduction vs. unhedged and **99.36%** vs. delta-only). On the out-of-sample pseudo-test (`Dec 2020`), it achieves **$2.51 \times 10^8\ \text{USD}^2$** (RMS **\$15.9k**).

---

## 2. Codebase Architecture

All modules live in `Federico/codice/` and depend strictly on a single immutable `CollarConfig` object, allowing the exact same code to run on training (`TRAIN`), pseudo-OOS (`PSEUDO_OOS`), and the October 2022 test set (`TEST` via `python main.py --test`):

| Module | Responsibility |
|---|---|
| `config.py` | All contract definitions (`TRAIN`, `PSEUDO_OOS`, `TEST`), liquidity filters (`IVFilters`), and model choices (`ModelConfig`). |
| `data.py` | Loads parquet option chains, merges Excel futures history (covering post-option-expiry fixing days), builds aligned SOFR series, and runs data integrity checks (`_check_raw`). |
| `fixing_calendar.py` | ICE Brent contract code parsing, official futures expiry dates, business-day fixing schedule, and Annex B roll rule. |
| `vol.py` | Undiscounted Black-76 pricer/vega, vectorized bisection IV solver, liquidity filtering, and daily $C^1$-smooth PCHIP / quadratic smile calibration. |
| `corr.py` | Strictly causal RiskMetrics EWMA correlation estimator between the two futures legs. |
| `pricer.py` | Closed-form two-moment lognormal matching pricer (`price_mm`), exact bivariate Monte Carlo validator with antithetic + geometric control variates (`price_mc`), and zero-cost strike solver (`solve_k_put`). |
| `greeks.py` | Central finite-difference bump-and-reprice Greeks for the collar (`collar_greeks`) and sticky-moneyness delta/vega for listed hedge options (`option_delta_smile`, `option_vega_point`). |
| `strategy.py` | Declarative strategy definitions (`S0`–`S4`, `BEST = S4`, plus ablation `VARIANTS`). |
| `backtest.py` | Strategy-independent precomputation (`prepare`) and daily mark-to-market hedging engine (`run_backtest`) with exact 5-component P&L attribution. |
| `main.py` | End-to-end orchestration, MC validation, sensitivity sweeps, alternate emission-date stress tests, JSON/CSV exports, and report figures. |
| `report_plots.py` | Publication-ready matplotlib charts saved to `Federico/output/`. |
| `tests/` | 42 automated unit tests verifying data, calendar, volatility inversion, $C^1$ wing smoothness, pricing, accounting identity, lot sizing, and hedging efficacy. |

---

## 3. Data, Fixing Schedule & Roll Rule

### Data Alignment (`data.py`)
- **Option chains** come from `unpivoted_bid_offer_train_clean.parquet` (or the released test parquet). Mid prices are computed as `(bid + ask) / 2` and spreads directly as `ask - bid`. Time to expiry uses `ACT/365` (`ttm = ttm_days / 365.0`).
- **Futures settlements** are extracted per `(date, contract)` and supplemented with `COJ21_COZ22_historical_series.xlsx` for the final business days after the front contract's options expire (e.g., 24–26 Feb 2021 for `COJ21`). Overlapping dates are cross-checked for exact settlement equality (`np.allclose`).
- **SOFR overnight rates** are converted to decimals (`sofr_rate / 100.0`) and forward/back-filled on any missing trading day.

### Fixing Calendar & Annex B Roll Rule (`fixing_calendar.py`)
- **ICE Brent futures expiry**: Last business day of the second month preceding the delivery month (`COJ21` → 26 Feb 2021; `COZ22` → 31 Oct 2022).
- **Fixing dates**: All business days (Mon–Fri, excluding configured holidays) of the averaging month.
- **Roll rule (Annex B + 10 Nov Q&A)**: The front-month contract fixes up to and including the **second-to-last fixing day on or before expiry**; the final fixing day rolls to the next-nearby contract:
  - **Training (`Feb 2021`)**: $N = 20$ fixings — $n_1 = 19$ on `COJ21` (1–25 Feb, $w_1 = 0.95$) + $n_2 = 1$ on `COK21` (26 Feb, $w_2 = 0.05$).
  - **Test (`Oct 2022`)**: $N = 21$ fixings — $n_1 = 20$ on `COZ22` (3–28 Oct, $w_1 \approx 0.952$) + $n_2 = 1$ on `COF23` (31 Oct, $w_2 \approx 0.048$).

---

## 4. Volatility Surface & $C^1$-Smooth Smile Calibration (`vol.py`)

### Undiscounted Black-76 & Early-Exercise Premium
ICE Brent futures options are American-style but **futures-style margined**: no cash premium is exchanged upfront; gains and losses are settled daily via variation margin. Under risk-neutral martingale dynamics for futures and zero upfront financing ($r = 0$, Asay 1982; Lieu 1990), Jensen's inequality guarantees that the European value always dominates immediate intrinsic exercise, so the **early-exercise premium is identically zero** and American options price identically to undiscounted European Black-76:

$$C(F, K, T, \sigma) = F\Phi(d_1) - K\Phi(d_2), \qquad P(F, K, T, \sigma) = K\Phi(-d_2) - F\Phi(-d_1)$$

$$d_1 = \frac{\ln(F/K) + \frac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}.$$

Implied volatilities are inverted via vectorized 80-step bisection on $[10^{-4}, 5.0]$ (accuracy $< 10^{-10}$), returning `NaN` outside strict no-arbitrage bounds.

### Liquidity Filters (`IVFilters`)
Quotes are kept for smile calibration only if they satisfy all criteria (expressed in relative terms so they generalize cleanly to 2022 price levels):
1. Valid positive bid & ask (`bid > 0`);
2. **OTM only**: puts with $K < F$ and calls with $K \ge F$;
3. Minimum maturity: `ttm_days >= 5`;
4. Penny-option filter: `mid >= 0.10` USD/bbl;
5. Relative spread cap: `(ask - bid) / mid <= 0.60`;
6. Valid IV inside `[0.01, 3.00]`.

### Daily Smile Construction & $C^1$-Smooth Wing Extrapolation
For each trading day and each leg, implied volatility is modeled as a function of **log-moneyness** $x = \ln(K/F)$ using **PCHIP** (Piecewise Cubic Hermite Interpolating Polynomial) over all liquid quotes (`smile_min_points = 3`). PCHIP interpolates market quotes exactly without spline overshoot.

- **Fallback hierarchy** (e.g., after front-month options expire in late Feb): if a leg has $< 3$ liquid quotes on day $t$, it borrows the other leg's same-day smile in log-moneyness; if neither leg has enough quotes, it carries forward the previous day's smile.
- **$C^1$-smooth wing extrapolation (`wing_decay = 0.05`)**: Hard flat clamping outside $[x_{\min}, x_{\max}]$ creates a discontinuous first derivative $\partial\sigma/\partial x$ at the boundary knots. During extreme market moves (e.g., 9 Mar 2020 when no strikes below \$45 were quoted), a collar strike slightly outside $x_{\min}$ receives zero smile slope while an adjacent static hedge strike sitting on $x_{\min}$ receives a one-sided slope — creating tens of thousands of barrels of artificial delta mismatch.  
  Instead, outside $[x_{\min}, x_{\max}]$, the smile transitions smoothly via a bounded hyperbolic tangent:

  $$\sigma(x) = \begin{cases} \sigma(x_{\min}) + s_{\text{lo}}\,\delta\,\tanh\!\left(\dfrac{x - x_{\min}}{\delta}\right), & x < x_{\min} \\[6pt] \sigma(x_{\max}) + s_{\text{hi}}\,\delta\,\tanh\!\left(\dfrac{x - x_{\max}}{\delta}\right), & x > x_{\max} \end{cases}$$

  with $\delta = 0.05$ and edge slopes $s_{\text{lo}}, s_{\text{hi}}$ clamped to $[-0.5, 0.5]$. This guarantees $C^1$ continuity across the boundary while capping asymptotic wing drift to at most $\pm 2.5$ vol points from the edge quote (and flooring at `0.01`).

---

## 5. Correlation, Pricing & Greeks

### Strictly Causal EWMA Correlation (`corr.py`)
Under the baseline bivariate driftless GBM dynamics,

$$\frac{df(u, \tau_i)}{f(u, \tau_i)} = \sigma_i\,dW_i(u), \qquad d\langle W_1, W_2\rangle_u = \rho\,du,$$

$\rho(t)$ is updated daily from historical log-returns $r_i(t) = \ln(F_i(t)/F_i(t-1))$ via RiskMetrics EWMA ($\lambda = 0.94$):

$$\text{cov}_t = \lambda\,\text{cov}_{t-1} + (1-\lambda)\,r_{1,t}r_{2,t}, \qquad v_{i,t} = \lambda\,v_{i,t-1} + (1-\lambda)\,r_{i,t}^2, \qquad \rho_t = \frac{\text{cov}_t}{\sqrt{v_{1,t}v_{2,t}}}.$$

Until `corr_min_obs = 10` return pairs accumulate, a prior `rho_prior = 0.99` is used (no lookahead).

### Asian Basket Moment Matching & Discounting (`pricer.py`)
On valuation date $t$, let $S = \sum_{t_k \le t} f_{\text{leg}(k)}(t_k)$ be the sum of already-realized fixings and $U = \frac{1}{N}\sum_{t_k > t} f_{\text{leg}(k)}(t_k)$ be the remaining stochastic part. Then:

$$A = \frac{S}{N} + U \implies (A - K)^+ = (U - K')^+, \quad K' = K - \frac{S}{N}.$$

1. **Moments of $U$**:
   $$m = \mathbb{E}^{\mathbb{Q}}[U \mid \mathcal{F}_t] = \frac{1}{N}\sum_{t_k > t} f_{\text{leg}(k)}(t)$$
   $$\mathbb{E}^{\mathbb{Q}}[U^2 \mid \mathcal{F}_t] = \frac{1}{N^2}\sum_{t_k > t}\sum_{t_l > t} f_{\text{leg}(k)}(t)\,f_{\text{leg}(l)}(t)\exp\!\left(\rho_{kl}\,\sigma_{\text{leg}(k)}\,\sigma_{\text{leg}(l)}\,\min(\theta_k, \theta_l)\right)$$
   where $\rho_{kl} = 1$ on the same leg and $\rho(t)$ across legs, $\theta_k = (t_k - t)/365$, and volatilities are read from each leg's smile at the **effective basket log-moneyness** $x = \ln(K'/m)$ (generalizing the sticky-moneyness strike mapping $K_i = f_i K/F_{\text{avg}}$ into the averaging window).
2. **Lognormal Match & OTC Discounting**: Setting $v = \ln(\mathbb{E}[U^2]/m^2)$, the OTC collar legs (cash-settled at payment date $T$) are discounted using the overnight SOFR rate (`ACT/360`):
   $$D(t, T) = \exp\!\left(-r_{\text{SOFR}}(t)\,\frac{T - t}{360}\right), \qquad C(t; K) = D(t, T)\,[m\Phi(d_1) - K'\Phi(d_2)].$$
   Edge cases ($K' \le 0$ or no remaining fixings) evaluate analytically to the exact discounted intrinsic/payoff.
3. **Zero-Cost Put Strike**: Solved once at emission ($t = t_0$) via Brent's root-finder (`brentq`, tolerance $10^{-10}$) such that $\text{Put}(K_{\text{put}}) = \text{Call}(K_{\text{call}})$, and kept fixed throughout the backtest.
4. **Monte Carlo Validation**: `price_mc` simulates exact correlated Brownian increments over future fixing dates (200,000 antithetic paths + optimal-$\beta$ geometric-average control variate). Across emission, mid-life, and mid-averaging dates, max MM vs. MC error is **0.19 ¢/bbl**.

### Smile-Consistent Greeks (`greeks.py`)
- **Collar Greeks** (`collar_greeks`): Central finite differences on `price_mm` with relative futures bump $h = 10^{-3} F_i$ (yielding $\Delta_1, \Delta_2, \Gamma_1, \Gamma_2$) and parallel smile shift $\pm 0.01$ (yielding $\mathcal{V}_{\text{put}}, \mathcal{V}_{\text{call}}$, and $\mathcal{V} = \mathcal{V}_{\text{put}} + \mathcal{V}_{\text{call}}$). Because smiles are parameterized in $\ln(K/F)$, bumping $F_i$ shifts the smile automatically (**sticky moneyness**).
- **Listed Option Greeks** (`option_delta_smile`): Reprices each listed hedge option at $F \pm h$ while shifting its market IV by the local smile change $\sigma(K, F \pm h) - \sigma(K, F)$. Using pure Black-76 sticky-strike delta instead degrades `S4` MSE by nearly **20×** (from $2.18 \times 10^8$ to $4.27 \times 10^9$ on `TRAIN`).

---

## 6. Hedging Strategy Mechanics (`S4` — `strategy.py`, `backtest.py`)

Every position is traded in integer multiples of **1,000 bbl** (1 ICE Brent contract).

### Layer 1 — Phased Static Opposite Collar (Front Contract, Leg 0)
- **Strike Selection (`_static_portfolio`)**: On emission day $t_0$, selects liquid front-month puts bracketing $K_{\text{put}}$ (short) and liquid front-month calls bracketing $K_{\text{call}}$ (long). If $K$ lies between two quoted strikes $K_a < K < K_b$, linear weights $1 - w$ and $w = (K - K_a)/(K_b - K_a)$ are assigned (e.g., Call 78 → 40% Call 75 + 60% Call 80); if outside the quoted range, 100% goes to the nearest quoted strike.
- **Vega-Matched Sizing (`static_scale = "collar_vega"`)**: Because an arithmetic average has slightly lower variance than a single terminal fixing, 1M bbl of European options would over-hedge the collar's vega. Each side is therefore scaled so that its emission-day vega matches the corresponding collar leg's vega ($\mathcal{V}_{\text{put}}$ or $\mathcal{V}_{\text{call}}$), rounded to 1,000 bbl lots.
- **Volatility-Adaptive Entry Phasing (`static_phase_days = 0`)**: Entering ~1.8M–2.0M bbl of options on Day 0 creates a massive negative Day-0 spread P&L that is heavily penalized under $\sum \text{P\&L}_t^2$. Phasing entry linearly over $k$ days trades off cost-squared ($C_{\text{tot}}^2 / k$) against unhedged ramp-up risk ($\approx \frac{k}{3}\sigma_{\text{vega}}^2$), yielding the closed-form causal optimum (`_adaptive_phase_days`):

  $$k^* = \text{clip}\!\left(\text{round}\!\left(\sqrt{3}\,\frac{C_{\text{tot}}}{\sigma_{\text{vega}}}\right),\ 3,\ 12\right)$$

  where $C_{\text{tot}} = \sum |\text{target}_o|\cdot \text{spread}_o/2$ and $\sigma_{\text{vega}} = 0.5\,\max(|\mathcal{V}_{\text{put}}|, |\mathcal{V}_{\text{call}}|)\cdot 0.85\,\frac{100\,\sigma_{\text{ATM},0}}{\sqrt{252}}$ use **only Day-0 data**. This chooses 5 days in calm markets (`Jan 2020`, `Apr 2021`) and automatically tightens to 3–4 days in high-vol markets (`Mar/Apr 2020`).
- **Deferred Dynamic Hedge During Ramp-Up (`static_defer_dynamic = True`)**: Suppresses dynamic vega trades during days $t < k^*$ so the dynamic option never buys a large temporary hedge on Day 0 only to dump it two days later as the static legs finish phasing in.

### Layer 2 — Final-Month Lifecycle Management (2022-Ready)
1. **Guarded Over-Hedge Trimming (`static_trim_days = 75`)**: Within the final 75 days before option expiry, as the averaging window approaches and shrinks the Asian collar's variance faster than the European options' variance, each static wing is checked against its collar leg. If $|V_{\text{wing}}| - |V_{\text{collar}}| > \max(0.15\,|V_{\text{collar}}|,\ 0.002\cdot N_{\text{bbl}})$ and the option still has `time_val > spread`, the static wing is trimmed proportionally toward $|V_{\text{collar}}| / |V_{\text{wing}}|$ (capped at 25% of original target per day). This prevents opening large opposite-sign Leg-2 calendar spreads during the averaging month while protecting deep-ITM options from premature mid-life trimming.
2. **Phased Early Exit (`static_exit_days = 5`, `static_close_days = 10`)**: When an option approaches `static_close_days` with `time_val > spread`, its remaining size is ramped down linearly over 5 days rather than liquidated in a single spread spike.
3. **Time-Value Expiry Rule (`static_let_expire = True`)**: Near expiry, if `time_val <= spread` (catching **both** deep-OTM options and **deep-ITM options**, such as `COZ22` Calls if Brent rallies above \$78 in 2022), the option is left to expire at intrinsic value for **\$0 closing spread**.
4. **Zero Phantom Vega on Expiring Penny Options**: Once an option is flagged `letting_expire` (or has `time_val <= 0.02` with $< 25$ days left), its vega contribution is set to `0` and its delta to pure intrinsic (`0` OTM, `±1` ITM), preventing the dynamic hedge from chasing noisy IVs implied by \$0.01 tick floors.

### Layer 3 — Dynamic Residual Vega Hedge (Second Contract, Leg 1)
- Trades a single liquid OTM option on the second-nearby contract (`COK21` / `COF23`, filtered by `rel_spread <= 0.60` and `ttm_days >= 20`) targeting `-(g.vega + static_vega)`.
- Anchored to the collar strike (`K_put` or `K_call`) with the larger absolute vega (`vega_reference = "dominant"`).
- **Anti-churn rules**: Switches strike only if the current option becomes untradable or another strike is at least `option_switch_distance = $10` closer to the reference strike; adjusts quantity only when $|\psi^* - \psi| > 0.05\,|\psi^*| + 0.5\text{ lot}$ (`vega_band = 0.05`).
- If a held option temporarily disappears from daily exchange quotes, it is marked continuously using the calibrated smile (Black-76) rather than stale prices.

### Layer 4 — Daily Futures Delta Hedge
- Every day (with `delta_band_lots = 0`, because futures bid-ask spreads are tiny at \$0.01–\$0.02/bbl), sets futures positions on both legs to:

  $$q_i(t) = \text{round}_{1000}\!\big(-(\Delta_{\text{collar}, i}(t) + \Delta_{\text{options}, i}(t))\big), \qquad i \in \{1, 2\}.$$

- On the final fixing date, all remaining positions are closed and the collar settles at its exact realized arithmetic-average payoff.

---

## 7. Portfolio Accounting & Exact P&L Attribution

Because both futures and ICE Brent options are **futures-style margined**, their daily variation margin flows directly into the cash account $B_t$, leaving their post-settlement mark-to-market value at zero:

$$B_t = B_{t-1}\left(1 + r_{t-1}\frac{\Delta t}{360}\right) + \sum_{i=1}^2 q_i(t-1)\Delta F_i(t) + \sum_{j \in \text{opts}} \psi_j(t-1)\Delta P_j(t) - \text{Cost}_t$$

$$\Pi_t = V_{\text{collar}}(t) + B_t, \qquad \text{P\&L}_t = \Pi_t - \Pi_{t-1} \quad (\Pi_{-1} = 0, \text{ so } \text{P\&L}_0 = -\text{Cost}_0).$$

Every trade pays half the quoted bid-ask spread (`|Δqty| * spread / 2`, equivalent to buying at ask and selling at bid while marking at mid/settlement). Daily P&L decomposes **additively to machine precision** (verified by unit tests) into five components:
1. `pnl_prezzo`: collar price change from $\Delta F$ + futures variation margin + smile-shifted option price change from $\Delta F$;
2. `pnl_volatilita`: collar change from smile update + residual option P&L;
3. `pnl_tempo`: time decay (theta) and fixing lock-ins;
4. `pnl_interessi`: SOFR interest on cash $B_{t-1}$;
5. `pnl_costi`: negative bid-ask transaction costs.

---

## 8. Verified Performance & Robustness Results

### Training Collar (`Feb 2021`, `COJ21`/`COK21`, 297 Days)
Emission 6 Jan 2020: $K_{\text{put}} = 44.22\text{ USD/bbl}$, $\text{Call}(78) = \text{Put}(44.22) = 1.089\text{ USD/bbl}$ (\$1.09M), realized average = \$62.20/bbl.

| Strategy | Metric ($\text{USD}^2$) | Daily RMS P&L | Max $\|\text{P\&L}\|$ | Total Costs | Final $\Pi_T$ |
|---|---:|---:|---:|---:|---:|
| `S0` Unhedged | $3.31 \times 10^{11}$ | \$574.9k | \$3.97M | \$0 | \$0 |
| `S1` Delta (futures only) | $3.42 \times 10^{10}$ | \$184.9k | \$1.50M | \$105k | +\$482k |
| `S2` Delta + single dynamic option | $2.02 \times 10^{9}$ | \$45.0k | \$345k | \$734k | −\$992k |
| `S3` Static opposite collar (1M) + delta | $2.24 \times 10^{8}$ | \$15.0k | \$81k | \$349k | −\$406k |
| **`S4` Static collar + trim + dyn vega + delta (BEST)** | **$2.18 \times 10^{8}$** | **\$14.8k** | **\$88k** | **\$502k** | **−\$705k** |

### Pseudo-OOS Collar (`Dec 2020`, `COG21`/`COH21`, 257 Days)
Emission 6 Jan 2020: $K_{\text{call}} = 79.00$, $K_{\text{put}} = 44.52\text{ USD/bbl}$, premium = \$0.911/bbl, realized average = \$50.24/bbl.

| Strategy | Metric ($\text{USD}^2$) | Daily RMS P&L | Max $\|\text{P\&L}\|$ | Total Costs | Final $\Pi_T$ |
|---|---:|---:|---:|---:|---:|
| `S0` Unhedged | $4.71 \times 10^{11}$ | \$686.1k | \$4.40M | \$0 | \$0 |
| `S1` Delta (futures only) | $4.42 \times 10^{10}$ | \$210.3k | \$1.83M | \$92k | +\$1.33M |
| `S2` Delta + single dynamic option | $1.97 \times 10^{9}$ | \$44.4k | \$386k | \$578k | −\$678k |
| `S3` Static opposite collar (1M) + delta | $5.23 \times 10^{8}$ | \$22.9k | \$167k | \$366k | −\$315k |
| **`S4` Static collar + trim + dyn vega + delta (BEST)** | **$2.51 \times 10^{8}$** | **\$15.9k** | **\$81k** | **\$443k** | **−\$616k** |

### Cross-Regime Stress Tests (Alternate Emission Dates)
Testing `S4` across five distinct volatility and trend regimes confirms broad robustness without overfitting:

| Regime | Original Baseline `S4` | **Final Upgraded `S4`** | Reduction |
|---|---:|---:|---:|
| `TRAIN` (Jan 2020 → Feb 2021, crash & recovery) | $3.59 \times 10^8$ | **$2.18 \times 10^8$** | **−39.3%** |
| `PSEUDO_OOS` (Jan 2020 → Dec 2020) | $3.14 \times 10^8$ | **$2.51 \times 10^8$** | **−19.9%** |
| Emission 2 Mar 2020 (immediate pre-crash shock) | $1.04 \times 10^9$ | **$2.91 \times 10^8$** | **−72.0%** |
| Emission 1 Apr 2020 (peak crisis volatility) | $5.06 \times 10^8$ | **$4.23 \times 10^8$** | **−16.4%** |
| Emission 1 Jun 2020 (post-crash rally) | $3.56 \times 10^8$ | **$3.07 \times 10^8$** | **−13.7%** |
| **5-Collar Average MSE** | **$5.15 \times 10^8$** | **$2.98 \times 10^8$** | **−42.1%** |

### Key Ablations (`train_feb2021/risultati.json`)
- **Smile-consistent option delta is essential**: switching `option_delta_smile = False` raises `S4` MSE from $2.18 \times 10^8$ to $4.27 \times 10^9$ (+1,860%).
- **PCHIP beats quadratic smile**: switching to quadratic smile raises `S4` MSE from $2.18 \times 10^8$ to $5.76 \times 10^8$.
- **Front-month static beats second-month static**: placing the static collar on `May21` instead of `Apr21` raises `S4` MSE from $2.18 \times 10^8$ to $1.11 \times 10^9$.
- **Vega deadband stability**: `vega_band` between `0.00` and `0.10` gives virtually identical MSE ($2.16 \times 10^8$ to $2.18 \times 10^8$), confirming low parameter sensitivity.

---

## 9. Running the Code & Out-of-Sample October 2022 Test

```bash
cd Federico/codice
pip install -r requirements.txt

# 1. Run full training collar + pseudo-OOS validation + charts/CSVs (~50s)
python main.py

# 2. Run all 42 automated verification tests
python -m pytest -q tests

# 3. Run the out-of-sample October 2022 test collar (after placing test parquet in Federico/dati/
#    and updating TEST.data_file in config.py if the filename differs from full_df_test.parquet)
python main.py --test
```

On the `TEST` emission date (`2021-04-06`, already present in the training parquet):
- $F_1(\text{COZ22}) = 58.77$, $F_2(\text{COF23}) = 58.63$, ATM vol = 28.2%,
- Zero-cost strike $K_{\text{put}} = 38.43\text{ USD/bbl}$ ($\text{Call}(78) = \text{Put}(38.43) = 1.993\text{ USD/bbl}$),
- Both collar wings are cleanly bracketed by liquid `COZ22` strikes (`Put 35/40` and `Call 75/80`), with adaptive entry phasing $k^* = 5$ days.
