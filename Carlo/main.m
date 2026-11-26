%% ========================================================================
%  HEDGING COMPETITION - MAIN MULTI-MODEL SCRIPT (Step A + Step C)
%  Computational Finance 2025/2026
%  Structure: Zero-Cost Asian Collar on ICE Brent Futures (Apr21 / May21)
%
%  MODELS IMPLEMENTED (from Prof. Azzone's Lecture Notes, Note (3)):
%    1. 'black76' : Undiscounted Black-76 baseline (Moment Matching / MC)
%    2. 'merton'  : Merton Jump-Diffusion (Sec. 1.7)
%    3. 'kou'     : Kou Double-Exponential Jump-Diffusion (Sec. 1.7)
%    4. 'vg'      : Variance Gamma, alpha = 0 (Sec. 1.8)
%    5. 'nig'     : Normal Inverse Gaussian, alpha = 1/2 (Sec. 1.8)
%    6. 'cgmy'    : CGMY Tempered Stable (Sec. 1.9)
%    7. 'ats'     : Multi-Maturity Slice-by-Slice ATS (Sec. 4, Thm. 4.1, p. 40)
%    8. 'heston'  : Heston Stochastic Volatility (Sec. 5.1, pp. 45-46)
%    9. 'bates'   : Bates Stochastic Volatility + Jumps (Sec. 5.2, p. 47)
%   10. 'bns'     : BNS Non-Gaussian OU Stochastic Volatility (Sec. 5.4, p. 49)
%
%  HEDGING STRATEGIES TESTED (Step A):
%    - 'delta_futures'    : Baseline pure Delta hedging (COJ21 & COK21 futures)
%    - 'delta_vega'       : Sticky near-ATM COJ21 option (Vega) + futures (Delta)
%    - 'delta_vega_gamma' : Optional 2-option (Put+Call) Vega-Gamma + futures
%  ========================================================================
clear; clc; close all;

% Automatically add modular src/ folders to MATLAB path
rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir), rootDir = pwd; end
addpath(genpath(fullfile(rootDir, 'src')));

%% ========================================================================
%  STEP 1: LOAD DATASET & UNPACK COMFORTABLE DATA TYPES
%  (For Out-of-Sample Test Set per HedgingLab.pdf Sec. 2: switch to test
%   parquet/excel, leg1Code='COZ22', leg2Code='COF23', avgYear=2022, avgMonth=10)
%  ========================================================================
parquetFile = 'unpivoted_bid_offer_train_clean.parquet';
excelFile   = 'COJ21_COZ22_historical_series.xlsx';
leg1Code    = 'COJ21';  % First-nearby future (Apr21)
leg2Code    = 'COK21';  % Second-nearby future (May21)
avgYear     = 2021;     % Averaging window year
avgMonth    = 2;        % Averaging window month (February 2021)
n2RollDays  = 1;        % Number of final fixing days rolled to Leg 2 (Annex B)

data = loadData(parquetFile, excelFile, leg1Code, leg2Code, avgYear, avgMonth, n2RollDays);

% Unpack comfortable workspace vectors & scalars
notional     = data.notional;       K_call     = data.K_call;
t0           = data.t0;             T_settle   = data.T_settle;
dates        = data.dates;          N_days     = data.N_days;
dt_days      = data.dt_days;        dt_yf      = data.dt_yf;
tau_settle   = data.tau_settle;     sofr       = data.sofr_rate;
df_settle    = data.df_settle;

F1           = data.F1_settle;      F2         = data.F2_settle;
F1_bid       = data.F1_bid;         F2_bid     = data.F2_bid;
F1_ask       = data.F1_ask;         F2_ask     = data.F2_ask;
F1_spread    = data.F1_spread;      F2_spread  = data.F2_spread;
F_settle     = data.F_settle;       F_spread   = data.F_spread;

fixing_dates = data.fixing_dates;   fixing_idx = data.fixing_idx;
fixing_leg   = data.fixing_leg;     K1_idx     = data.K1_idx;
K2_idx       = data.K2_idx;         N_fix      = data.N_fix;
n1           = data.n1;             n2         = data.n2;
w1           = data.w1;             w2         = data.w2;
weights      = data.weights;        F_avg      = data.F_avg;

optChain1    = data.optChain1;      optChain2  = data.optChain2;
multiSurface = data.multiSurface;

fprintf('=== STEP 1: DATA LOADED SUCCESSFULLY ===\n');
fprintf('Backtest period : %s to %s (%d trading days)\n', ...
    string(t0, 'dd-MMM-yyyy'), string(T_settle, 'dd-MMM-yyyy'), N_days);
fprintf('Underlying legs : %s (n1=%d, w1=%.2f) | %s (n2=%d, w2=%.2f)\n', ...
    leg1Code, n1, w1, leg2Code, n2, w2);
fprintf('Multi-mat ATS   : %d liquid maturities available at t0\n\n', numel(multiSurface{1}));

%% ========================================================================
%  SHARED CONFIGURATION & FINALIST MODEL LIST (TOP 4 + MULTI-MATURITY ATS)
%  ========================================================================
%  All 10 models remain available in cfModels.m; here we run the Top 4
%  performers (black76, bns, merton, nig) plus Multi-Maturity ATS.
modelsToRun = {'black76', 'bns', 'merton', 'nig', 'ats'};

baseConfig = struct();
baseConfig.volModel          = 'smile_interp';
baseConfig.interpMethod      = 'pchip';
baseConfig.pricerMethod      = 'moment_matching';

% Liquidity filters (Note (3), Section 3.2)
baseConfig.quoteSource       = 'mid';
baseConfig.useOTMOnly        = true;
baseConfig.minOptionBid      = 0.01;
baseConfig.maxRelSpread      = 0.80;

% Correlation estimator
baseConfig.corrMethod        = 'expanding';
baseConfig.rhoPrior          = 0.99;
baseConfig.minCorrObs        = 5;
baseConfig.corrWindow        = 60;
baseConfig.ewmaLambda        = 0.94;

% Lewis FFT & CF-MC Numerical Parameters (Note (3), Sec. 2.3-2.4)
baseConfig.fftN              = 2048;
baseConfig.fftDu             = 0.15;
baseConfig.cdfN              = 2048;
baseConfig.calibMaxEval      = 60;
baseConfig.mcPathsInception  = 100000;  % High-precision 100k MC paths for inception K_put
baseConfig.mcPaths           = 100000;  % High-precision 100k MC paths for daily pricing & Greeks
baseConfig.mcSeed            = 42;
baseConfig.useControlVariate = true;    % Kemna-Vorst CV (Black-76 MC)
baseConfig.useCFControlVariate = true;  % European Lewis-FFT CV for CF/ATS/SV models
baseConfig.shapeRegWeight    = 0.03;    % Smooth shape-param regularization (vol level free)
baseConfig.useVegaWeightCalib = false;  % Optional IV-weighted calib (unweighted best overall)
baseConfig.useOTMParity      = false;   % Optional Put-Call parity side flip (native legs best)

% Step A Hedging Parameters (Futures + Listed Option Vega Hedging)
baseConfig.bumpRelF          = 1e-4;    % Relative bump dF/F for Delta/Gamma
baseConfig.bumpRelVol        = 0.02;    % 2% relative vol scale bump for Vega
baseConfig.computeVega       = false;
baseConfig.useMVDelta        = true;    % Upgrade #2: Minimum-Variance / Shadow Delta (spot-vol leverage)
baseConfig.optRollBand       = 7.5;     % Keep held option strike while |K - F1| <= $7.50
baseConfig.minOptTauDays     = 10;      % Unwind option hedge 10 days before expiry
baseConfig.maxOptPos         = 1.5;     % Position cap on listed option hedge ratio |psi|
baseConfig.vegaRebalBand     = 0.03;    % Turnover deadband on option hedge ratio (limits bid-ask churn)
baseConfig.futRebalBand      = 0.00;    % Futures deadband (keep 0.00: cheap futures spreads make daily Delta optimal)
baseConfig.unwindAtMaturity  = false;

% Safety Net Parameters (Calibration Guard + Black-76 Greeks Fallback)
baseConfig.useSafetyNet      = true;    % Enable safety net (set false for raw model behavior)
baseConfig.snMaxRMSE         = 0.060;   % Trigger if OTM smile RMSE > max($0.06, 3% of ATM price)
baseConfig.snMaxRelRMSE      = 0.030;   % Relative RMSE threshold vs ATM option price
baseConfig.snMinStrikes      = 4;       % Trigger if < 4 liquid OTM quotes
baseConfig.snMinTauDays      = 15;      % Revert Greeks to Black-76 inside final 15 days before expiry
baseConfig.snMaxParamJump    = 0.20;    % Trigger if parameter jumps > 20% of [lb, ub] without sharp fit
baseConfig.snMaxPsiJump      = 0.30;    % Trigger Black-76 Greeks if same-strike |dPsi| > 0.30

%% ========================================================================
%  RUN FULL 297-DAY BACKTEST AT 100,000 MC PATHS (TOP 4 + ATS)
%  Strategies: Collar1O (delta_vega_collar) & Strike-Bracket (delta_vega_bracket)
%  (plus instant MV-Delta reference via baseStructHistory reuse for Fig 4)
%  ========================================================================
nModels        = numel(modelsToRun);
deltaResults   = struct();
collarResults  = struct();
bracketResults = struct();

% Workspace comparison matrices for primary strategy (N_days x nModels)
portVal_mat    = zeros(N_days, nModels);
phi1_mat       = zeros(N_days, nModels);
psi1_mat       = zeros(N_days, nModels);
sqDailyPnL_mat = zeros(N_days, nModels);
runningMSE_mat = zeros(N_days, nModels);
calibRMSE1_mat = nan(N_days, nModels);

fprintf('=== FINALISTS (100k MC PATHS): COLLAR1O vs STRIKE-BRACKET ===\n');
fprintf('%-8s | %-6s | %-9s | %-9s | %-6s | %-6s | %-7s | %-6s\n', ...
    'Model', 'K_put', 'Collar1O', 'Bracket', 'ColTC', 'BrkTC', 'BrkRed', 'SN Day');
fprintf('%s\n', repmat('-', 1, 79));

for i = 1:nModels
    mKey = modelsToRun{i};

    % 1. Collar-Anchored 1-Option Hedge (calibrates 297d ONCE & computes base MC Greeks ONCE)
    cfgC = baseConfig;
    cfgC.modelName   = mKey;
    cfgC.hedgeMethod = 'delta_vega_collar';
    resC = runBacktest(data, cfgC);
    collarResults.(mKey) = resC;

    % Package precomputed calibrations, K_put, MC diagnostics & base structure Greeks
    precomp = struct('calibHistory',      {resC.calibHistory}, ...
                     'baseStructHistory', {resC.baseStructHistory}, ...
                     'K_put',             resC.K_put, ...
                     'C_inception',       resC.C_inception, ...
                     'P_inception',       resC.P_inception, ...
                     'mcDiag',            resC.mcDiag);

    % 2. Symmetric Strike-Bracketed Collar Hedge (reuses calib & base MC Greeks -> instant!)
    cfgB = baseConfig;
    cfgB.modelName   = mKey;
    cfgB.hedgeMethod = 'delta_vega_bracket';
    resB = runBacktest(data, cfgB, precomp);
    bracketResults.(mKey) = resB;

    % 3. Pure Futures MV-Delta reference (reuses calib & base MC Greeks -> < 0.05s!)
    cfgD = baseConfig;
    cfgD.modelName   = mKey;
    cfgD.hedgeMethod = 'delta_futures';
    resD = runBacktest(data, cfgD, precomp);
    deltaResults.(mKey) = resD;

    % Save Strike-Bracketed series into workspace matrices
    portVal_mat(:, i)    = resB.portVal;
    phi1_mat(:, i)       = resB.phi1;
    psi1_mat(:, i)       = resB.psi1;
    sqDailyPnL_mat(:, i) = resB.sqDailyPnL;
    runningMSE_mat(:, i) = resB.runningMSE;
    calibRMSE1_mat(:, i) = resB.calibRMSE1;

    fprintf('%-8s | %6.3f | %9.6f | %9.6f | %6.3f | %6.3f | %6.2f%% | %3d/297\n', ...
        upper(mKey), resC.K_put, resC.mse_per_bbl, resB.mse_per_bbl, ...
        resC.cumTransCost(end), resB.cumTransCost(end), resB.var_reduction_pct, resB.nSafetyDays);
end
fprintf('%s\n\n', repmat('-', 1, 79));

%% ========================================================================
%  REPORT: MC PRECISION, VANILLA CONVERGENCE & TIMING BREAKDOWN
%  ========================================================================
fprintf('=== MC PRECISION, VANILLA CONVERGENCE (t0, M=%d) & RUNTIME BREAKDOWN ===\n', baseConfig.mcPaths);
fprintf('%-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n', ...
    'Model', 'AvgRMSE1', 'VanMaxEr', 'VanMC_SE', 'AsianSE', 'MartErr', 'Calib(s)', 'Price(s)');
fprintf('%s\n', repmat('-', 1, 86));
for i = 1:nModels
    mKey = modelsToRun{i};
    r    = collarResults.(mKey);
    d    = r.mcDiag;
    fprintf('%-8s | %8.4f | %8.5f | %8.5f | %8.5f | %8.5f | %8.2f | %8.2f\n', ...
        upper(mKey), r.avgCalibRMSE1, d.vanillaMaxAbsErr, d.vanillaMeanSE, ...
        d.asianSE_put, d.martingaleErr, r.calibElapsedSec, r.priceHedgeElapsedSec);
end
fprintf('%s\n\n', repmat('-', 1, 86));

%% ========================================================================
%  PLOT HEDGE EVOLUTION OVER TIME (STRIKE-BRACKETED COLLAR + STEP A BARS)
%  ========================================================================
plotResults(data, bracketResults, modelsToRun, deltaResults, 'figures');
