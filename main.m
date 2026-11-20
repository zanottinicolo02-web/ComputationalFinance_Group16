%% ========================================================================
%  HEDGING COMPETITION - MAIN BACKBONE SCRIPT
%  Computational Finance 2025/2026
%  Structure: Zero-Cost Asian Collar on ICE Brent Futures (Apr21 / May21)
%  ========================================================================
clear; clc; close all;

%% ========================================================================
%  STEP 1: LOAD DATASET & UNPACK COMFORTABLE DATA TYPES
%  ========================================================================
parquetFile = 'unpivoted_bid_offer_train_clean.parquet';
excelFile   = 'COJ21_COZ22_historical_series.xlsx';
leg1Code    = 'COJ21';  % First-nearby future (Apr21)
leg2Code    = 'COK21';  % Second-nearby future (May21)
avgYear     = 2021;     % Averaging window year
avgMonth    = 2;        % Averaging window month (February 2021)
n2RollDays  = 1;        % Number of final fixing days rolled to Leg 2 (Annex B)

% Load everything into a master struct
data = loadData(parquetFile, excelFile, leg1Code, leg2Code, avgYear, avgMonth, n2RollDays);

% --- 1.1 Contract & Term-Sheet Scalars (Annex A) ---
notional = data.notional;       % 1,000,000 bbl
K_call   = data.K_call;         % Fixed Call strike = 78.0 USD/bbl
t0       = data.t0;             % Emission date: 06-Jan-2020
T_settle = data.T_settle;       % Derivative settlement date: 26-Feb-2021

% --- 1.2 Backtest Calendar & Time Vectors (N_days x 1) ---
dates      = data.dates;        % (297 x 1) datetime vector of rebalancing dates
N_days     = data.N_days;       % Total rebalancing days in backtest (297)
dt_days    = data.dt_days;      % (297 x 1) calendar days between rebalancing dates
dt_yf      = data.dt_yf;        % (297 x 1) year fraction delta_t (ACT/365) for cash accrual
tau_settle = data.tau_settle;   % (297 x 1) year fraction (T - t) to final settlement

% --- 1.3 SOFR Interest Rate & Discount Factor Vectors (N_days x 1) ---
sofr      = data.sofr_rate;     % (297 x 1) decimal SOFR rate r(t)
df_settle = data.df_settle;     % (297 x 1) discount factor D(t, T) = exp(-r(t)*(T - t))

% --- 1.4 Futures Time Series Column Vectors (N_days x 1) & Matrices (N_days x 2) ---
% Leg 1 (COJ21 - Apr21)
F1        = data.F1_settle;     % (297 x 1) Leg 1 settlement price f(t, tau_1)
F1_bid    = data.F1_bid;        % (297 x 1) Leg 1 bid price
F1_ask    = data.F1_ask;        % (297 x 1) Leg 1 ask price
F1_spread = data.F1_spread;     % (297 x 1) Leg 1 bid-ask spread

% Leg 2 (COK21 - May21)
F2        = data.F2_settle;     % (297 x 1) Leg 2 settlement price f(t, tau_2)
F2_bid    = data.F2_bid;        % (297 x 1) Leg 2 bid price
F2_ask    = data.F2_ask;        % (297 x 1) Leg 2 ask price
F2_spread = data.F2_spread;     % (297 x 1) Leg 2 bid-ask spread

% Combined (N_days x 2) matrices [Leg1, Leg2] for vectorized hedging P&L
F_settle  = data.F_settle;      % (297 x 2) [F1, F2] settlement prices
F_bid     = data.F_bid;         % (297 x 2) [F1_bid, F2_bid]
F_ask     = data.F_ask;         % (297 x 2) [F1_ask, F2_ask]
F_spread  = data.F_spread;      % (297 x 2) [F1_spread, F2_spread]

% --- 1.5 Averaging Schedule & Basket Weights (Annex B) ---
fixing_dates = data.fixing_dates; % (20 x 1) datetime vector of Feb 2021 fixing days
fixing_idx   = data.fixing_idx;   % (20 x 1) indices of fixing_dates inside 'dates'
fixing_leg   = data.fixing_leg;   % (20 x 1) leg indicator (1 = COJ21, 2 = COK21)
K1_idx       = data.K1_idx;       % (n1 x 1) fixing indices belonging to Leg 1 (K_1)
K2_idx       = data.K2_idx;       % (n2 x 1) fixing indices belonging to Leg 2 (K_2)
N_fix        = data.N_fix;        % Total number of fixings N = 20
n1           = data.n1;           % Number of fixings on Leg 1
n2           = data.n2;           % Number of fixings on Leg 2
w1           = data.w1;           % Basket weight w1 = n1 / N
w2           = data.w2;           % Basket weight w2 = n2 / N
weights      = data.weights;      % [w1, w2]
F_avg        = data.F_avg;        % (297 x 1) basket forward F_avg(t) = w1*F1(t) + w2*F2(t)

% --- 1.6 Daily Option Chains (N_days x 1 Cell Arrays of Structs) ---
optChain1 = data.optChain1;     % (297 x 1) cell array for Leg 1 options (COJ21)
optChain2 = data.optChain2;     % (297 x 1) cell array for Leg 2 options (COK21)

fprintf('=== STEP 1: DATA LOADED SUCCESSFULLY ===\n');
fprintf('Backtest period : %s to %s (%d trading days)\n', ...
    datestr(t0, 'dd-mmm-yyyy'), datestr(T_settle, 'dd-mmm-yyyy'), N_days);
fprintf('Underlying legs : Leg 1 = %s (n1 = %d, w1 = %.2f), Leg 2 = %s (n2 = %d, w2 = %.2f)\n', ...
    leg1Code, n1, w1, leg2Code, n2, w2);
fprintf('Inception (t0)  : F1 = %.2f, F2 = %.2f, F_avg = %.4f, SOFR = %.2f%%, D(t0,T) = %.6f\n\n', ...
    F1(1), F2(1), F_avg(1), sofr(1)*100, df_settle(1));

%% ========================================================================
%  MODEL & STRATEGY CONFIGURATION (Modular Switches for Future Extensions)
%  ========================================================================
config = struct();

% 1. Volatility Model:
%    'flat_atm'         -> Constant ATM Black-76 vol per leg (simplest)
%    'smile_interp'     -> Interpolated Black-76 implied vol smile per leg
%    'svi'              -> [Extension Hook] SVI parametric smile
%    'stoch_vol_heston' -> [Extension Hook] Heston stochastic vol model
config.volModel      = 'smile_interp';
config.interpMethod  = 'pchip';       % 'linear' or 'pchip' for smile interpolation
config.quoteSource   = 'mid';         % 'mid' or 'settle' for option quotes
config.useOTMOnly    = true;          % Use OTM Puts (K <= F) and OTM Calls (K > F)
config.minOptionBid  = 0.01;          % Filter out quotes with bid < $0.01
config.maxRelSpread  = 0.80;          % Filter out quotes with (ask-bid)/mid > 80%

% 2. Forward Curve / Return Dynamics Model:
%    'black76_lognormal' -> Driftless bivariate GBM (baseline)
%    'ats'               -> [Extension Hook] Affine Term Structure (ATS)
%    'hjm_jump'          -> [Extension Hook] Jump-extended HJM dynamics
config.forwardModel  = 'black76_lognormal';

% 3. Correlation Estimator:
%    'expanding' | 'rolling' | 'ewma' | 'constant'
config.corrMethod    = 'expanding';
config.rhoPrior      = 0.99;          % Initial correlation prior before minCorrObs
config.minCorrObs    = 5;             % Minimum return observations before sample corr
config.corrWindow    = 60;            % Window length (days) if corrMethod = 'rolling'
config.ewmaLambda    = 0.94;          % Decay factor if corrMethod = 'ewma'

% 4. Pricing Engine:
%    'moment_matching' -> Closed-form lognormal moment matching (Section 6)
%    'monte_carlo'     -> Bivariate Lognormal MC + Geometric Control Variate (Section 5)
%    'custom_mc'       -> [Extension Hook] MC for ATS / Jump-HJM / Heston
config.pricerMethod      = 'moment_matching';
config.mcPaths           = 50000;     % Number of paths if pricerMethod = 'monte_carlo'
config.mcSeed            = 42;        % Common random number seed for smooth MC Greeks
config.useControlVariate = true;      % Kemna-Vorst exact geometric-average control variate

% 5. Hedging Strategy:
%    'delta_futures' -> Delta-neutral hedge on Leg 1 & Leg 2 futures (baseline)
%    'delta_vega'    -> Delta-Vega neutral hedge using ATM options + futures
%    'unhedged'      -> Unhedged short collar benchmark
%    'custom'        -> [Extension Hook] Custom multi-factor hedge optimizer
config.hedgeMethod      = 'delta_futures';
config.bumpRelF         = 1e-4;       % Relative bump dF/F for finite-difference Delta
config.bumpAbsVol       = 1e-3;       % Absolute vol bump (10 bps) for Vega
config.computeVega      = false;      % Set true to track daily structure Vega
config.unwindAtMaturity = false;      % Whether to charge spread on final settlement day

%% ========================================================================
%  STEP 2: INCEPTION CALIBRATION (t0 = 06-Jan-2020)
%  ========================================================================
calib0 = calibrateModel(1, data, config);

fprintf('=== STEP 2: INCEPTION CALIBRATION (%s) ===\n', datestr(t0, 'dd-mmm-yyyy'));
fprintf('Leg 1 (%s) ATM Vol : %.2f%% (%d liquid strikes calibrated)\n', ...
    leg1Code, 100 * calib0.atmVol1, numel(calib0.smile1.strikes));
fprintf('Leg 2 (%s) ATM Vol : %.2f%% (%d liquid strikes calibrated)\n', ...
    leg2Code, 100 * calib0.atmVol2, numel(calib0.smile2.strikes));
fprintf('Correlation rho    : %.4f\n\n', calib0.rho);

%% ========================================================================
%  STEP 3 & 4: INCEPTION PRICING & ZERO-COST PUT STRIKE (K_put)
%  ========================================================================
[K_put, C_star, P_star, callDet0, putDet0] = solveZeroCostStrike(1, K_call, data, calib0, config);

% Cross-check Moment Matching against Bivariate Lognormal Monte Carlo at t0
configMC = config;
configMC.pricerMethod = 'monte_carlo';
[C_mc, mcCallDet] = priceBasket(1, K_call, true,  F1(1), F2(1), data, calib0, configMC);
[P_mc, mcPutDet]  = priceBasket(1, K_put,  false, F1(1), F2(1), data, calib0, configMC);

fprintf('=== STEP 3 & 4: INCEPTION ZERO-COST COLLAR PRICING ===\n');
fprintf('Fixed Call Strike K_call : %.4f USD/bbl\n', K_call);
fprintf('Solved Put Strike K_put  : %.4f USD/bbl\n', K_put);
fprintf('Moment Matching          : Call(78) = %.6f | Put(K_put) = %.6f USD/bbl\n', C_star, P_star);
fprintf('Monte Carlo (CV, M=%dk)  : Call(78) = %.6f (SE=%.6f) | Put(K_put) = %.6f (SE=%.6f)\n\n', ...
    configMC.mcPaths/1000, C_mc, mcCallDet.se, P_mc, mcPutDet.se);

%% ========================================================================
%  STEP 5 & 6: FULL DAILY HEDGING BACKTEST (06-Jan-2020 to 26-Feb-2021)
%  ========================================================================
fprintf('Running daily hedging backtest across %d trading days...\n', N_days);
results = runBacktest(data, config);

% Unpack key backtest time-series vectors into main workspace
V_struct     = results.V_struct;        % (297 x 1) Structure value [USD/bbl]
cash         = results.cash;            % (297 x 1) Cash account [USD/bbl]
portVal      = results.portVal;         % (297 x 1) Net hedged portfolio value [USD/bbl]
dailyPnL     = results.dailyPnL;        % (297 x 1) Daily hedged P&L [USD/bbl]
sqDailyPnL   = results.sqDailyPnL;      % (297 x 1) Squared daily P&L [(USD/bbl)^2]
phi1         = results.phi1;            % (297 x 1) Leg 1 futures hedge ratio
phi2         = results.phi2;            % (297 x 1) Leg 2 futures hedge ratio
delta1       = results.delta1;          % (297 x 1) Structure Delta wrt F1
delta2       = results.delta2;          % (297 x 1) Structure Delta wrt F2
atmVol1      = results.atmVol1;         % (297 x 1) Daily Leg 1 ATM implied vol
atmVol2      = results.atmVol2;         % (297 x 1) Daily Leg 2 ATM implied vol
rho_ts       = results.rho;             % (297 x 1) Daily correlation rho(t)
mse_per_bbl  = results.mse_per_bbl;     % Scalar competition metric [(USD/bbl)^2]
mse_total    = results.mse_total;       % Scalar competition metric [USD^2 on 1M bbl]

fprintf('\n=== STEP 5 & 6: BACKTEST RESULTS SUMMARY ===\n');
fprintf('Final Structure Payoff V_struct(T) : %+.4f USD/bbl (%+.2f USD)\n', ...
    V_struct(end), results.V_struct_total(end));
fprintf('Final Cash Account Balance B(T)    : %+.4f USD/bbl (%+.2f USD)\n', ...
    cash(end), results.cash_total(end));
fprintf('Cumulative Bid-Ask Trans. Costs    :  %.4f USD/bbl ( %.2f USD)\n', ...
    results.cumTransCost(end), notional * results.cumTransCost(end));
fprintf('Final Net Hedged Portfolio Value   : %+.4f USD/bbl (%+.2f USD)\n', ...
    portVal(end), results.portVal_total(end));
fprintf('------------------------------------------------------------\n');
fprintf('Unhedged (1/T) sum(dV^2)           : %.6f (USD/bbl)^2  [%.4e USD^2]\n', ...
    results.mse_unhedged_bbl, results.mse_unhedged_total);
fprintf('Hedged   (1/T) sum(P&L_t^2)        : %.6f (USD/bbl)^2  [%.4e USD^2]\n', ...
    mse_per_bbl, mse_total);
fprintf('Daily Squared P&L Reduction        : %.2f%%\n', results.var_reduction_pct);

%% ========================================================================
%  PLOT DIAGNOSTICS & DELIVERABLE FIGURES
%  ========================================================================
plotResults(data, results, config);
