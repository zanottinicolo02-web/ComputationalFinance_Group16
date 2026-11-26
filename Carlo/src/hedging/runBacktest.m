function results = runBacktest(data, config, precomp)
%RUNBACKTEST Execute full daily calibration, pricing, and hedging backtest.
%
%   results = runBacktest(data, config)
%   results = runBacktest(data, config, precomp)
%
%   Optional 'precomp' struct (with fields calibHistory, K_put, C_inception,
%   P_inception) reuses daily calibrations and inception strike across
%   hedging strategies on the same model, avoiding redundant calibrations.

    if nargin < 3
        precomp = [];
    end
    hasPreCalib = ~isempty(precomp) && isfield(precomp, 'calibHistory') && ...
                  numel(precomp.calibHistory) == data.N_days;
    hasPreKput  = ~isempty(precomp) && isfield(precomp, 'K_put') && isfinite(precomp.K_put);
    hasPreBase  = ~isempty(precomp) && isfield(precomp, 'baseStructHistory') && ...
                  numel(precomp.baseStructHistory) == data.N_days;

    tTimer   = tic;
    N_days   = data.N_days;
    notional = data.notional;
    K_call   = data.K_call;
    mName    = lower(strtrim(char(config.modelName)));

    if strcmpi(mName, 'black76')
        displayName = sprintf('Black-76 (%s)', strrep(config.volModel, '_', ' '));
        nParams     = 1;
        paramNames  = {'atmVol'};
    else
        spec        = cfModels(mName);
        displayName = spec.displayName;
        nParams     = numel(spec.paramNames);
        paramNames  = spec.paramNames;
    end

    % Preallocate time-series vectors (N_days x 1)
    V_struct     = zeros(N_days, 1);
    P_leg        = zeros(N_days, 1);
    C_leg        = zeros(N_days, 1);

    delta1       = zeros(N_days, 1);
    delta2       = zeros(N_days, 1);
    gamma1       = zeros(N_days, 1);
    gamma2       = zeros(N_days, 1);
    vegaScale    = zeros(N_days, 1);

    phi1         = zeros(N_days, 1);
    phi2         = zeros(N_days, 1);
    psi1         = zeros(N_days, 1);
    optStrike1   = nan(N_days, 1);

    cash         = zeros(N_days, 1);
    interestCF   = zeros(N_days, 1);
    futuresPnL   = zeros(N_days, 1);
    optionPnL    = zeros(N_days, 1);
    futTransCost = zeros(N_days, 1);
    optTransCost = zeros(N_days, 1);
    transCost    = zeros(N_days, 1);

    portVal      = zeros(N_days, 1);
    dailyPnL     = zeros(N_days, 1);
    unhedgedPnL  = zeros(N_days, 1);

    atmVol1      = zeros(N_days, 1);
    atmVol2      = zeros(N_days, 1);
    rho_ts       = zeros(N_days, 1);

    calibRMSE1   = nan(N_days, 1);
    calibRMSE2   = nan(N_days, 1);
    calibMAPE1   = nan(N_days, 1);
    calibMAPE2   = nan(N_days, 1);
    safetyFlag   = false(N_days, 1);
    params1_ts        = nan(N_days, nParams);
    params2_ts        = nan(N_days, nParams);
    calibHistory      = cell(N_days, 1);
    baseStructHistory = cell(N_days, 1);

    calib        = [];
    K_put        = NaN;
    C_inception  = NaN;
    P_inception  = NaN;
    openOpts     = struct('strike', {}, 'isCall', {}, 'psi', {}, 'lastMid', {});

    calibElapsedSec      = 0.0;
    priceHedgeElapsedSec = 0.0;
    mcDiag               = struct();

    for t = 1:N_days
        % 1. Daily Calibration (reuse precomp.calibHistory if provided)
        tCal = tic;
        if hasPreCalib
            calib = precomp.calibHistory{t};
            calib.mcCache = [];
        else
            calib = calibrateModel(t, data, config, calib);
        end
        calibElapsedSec = calibElapsedSec + toc(tCal);

        atmVol1(t)      = calib.atmVol1;
        atmVol2(t)      = calib.atmVol2;
        rho_ts(t)       = calib.rho;
        calibRMSE1(t)   = calib.rmse1;
        calibRMSE2(t)   = calib.rmse2;
        calibMAPE1(t)   = calib.mape1;
        calibMAPE2(t)   = calib.mape2;
        params1_ts(t,:) = calib.params1(:)';
        params2_ts(t,:) = calib.params2(:)';

        % 2. Inception (t = 1): Solve for Zero-Cost Put Strike K_put + MC Vanilla Check
        tPH = tic;
        if t == 1
            if hasPreKput
                K_put       = precomp.K_put;
                C_inception = precomp.C_inception;
                P_inception = precomp.P_inception;
            else
                [K_put, C_inception, P_inception, ~, ~, calib] = ...
                    solveZeroCostStrike(1, K_call, data, calib, config);
            end
        end

        % 3. Reprice Structure & Compute Target Futures + Option Hedges
        cachedBase_t = [];
        if hasPreBase
            cachedBase_t = precomp.baseStructHistory{t};
        end
        prevPhi = [];
        if t >= 2
            prevPhi = [phi1(t-1), phi2(t-1)];
        end
        hedge = computeHedge(t, data, calib, K_put, config, openOpts, cachedBase_t, prevPhi);
        baseStructHistory{t} = hedge.baseStruct;
        priceHedgeElapsedSec = priceHedgeElapsedSec + toc(tPH);

        if t == 1
            if ~isempty(precomp) && isfield(precomp, 'mcDiag') && ~isempty(precomp.mcDiag)
                mcDiag = precomp.mcDiag;
            else
                seP = 0.0; seC = 0.0;
                if isfield(hedge.putDet, 'se'),  seP = hedge.putDet.se;  end
                if isfield(hedge.callDet, 'se'), seC = hedge.callDet.se; end
                mcDiag = computeMCVanillaDiagnostics(data, calib, K_put, K_call, config, seP, seC);
            end
        end

        calibStore = calib;
        calibStore.mcCache = [];
        calibHistory{t} = calibStore;

        V_struct(t)   = hedge.V_struct;
        P_leg(t)      = hedge.P_val;
        C_leg(t)      = hedge.C_val;
        delta1(t)     = hedge.delta1;
        delta2(t)     = hedge.delta2;
        gamma1(t)     = hedge.gamma1;
        gamma2(t)     = hedge.gamma2;
        vegaScale(t)  = hedge.vegaScale;
        phi1(t)       = hedge.phi1;
        phi2(t)       = hedge.phi2;
        psi1(t)       = hedge.psi1;
        optStrike1(t) = hedge.optStrike1;
        safetyFlag(t) = hedge.usedBlackFallback;

        chain1_t = data.optChain1{t};

        % 4. Settle Overnight Interest + Futures MTM + Listed Option MTM
        if t == 1
            cashPre = 0.0;
        else
            r_prev     = data.sofr_rate(t-1);
            dt_t       = data.dt_yf(t);
            interest_t = cash(t-1) * (exp(r_prev * dt_t) - 1.0);

            fut_pnl_t  = phi1(t-1) * (data.F1_settle(t) - data.F1_settle(t-1)) + ...
                         phi2(t-1) * (data.F2_settle(t) - data.F2_settle(t-1));

            % Listed options on ICE Brent are futures-style margined (daily MTM)
            opt_pnl_t = 0.0;
            for j = 1:numel(openOpts)
                [midNow, ~, found] = lookupOptionQuote(chain1_t, openOpts(j).strike, openOpts(j).isCall);
                if ~found
                    if ~isempty(chain1_t) && chain1_t.tau > 5.0 / 365.0
                        % Live option temporarily outside dataset strike window: mark via Black-76 smile
                        midNow = black76Mark(data.F1_settle(t), openOpts(j).strike, ...
                            openOpts(j).isCall, chain1_t.tau, calib.sigma1_fn(openOpts(j).strike));
                    elseif openOpts(j).isCall
                        midNow = max(data.F1_settle(t) - openOpts(j).strike, 0.0);
                    else
                        midNow = max(openOpts(j).strike - data.F1_settle(t), 0.0);
                    end
                end
                opt_pnl_t = opt_pnl_t + openOpts(j).psi * (midNow - openOpts(j).lastMid);
            end

            interestCF(t) = interest_t;
            futuresPnL(t) = fut_pnl_t;
            optionPnL(t)  = opt_pnl_t;
            cashPre       = cash(t-1) + interest_t + fut_pnl_t + opt_pnl_t;
        end

        % 5. Rebalance Transaction Costs (Futures + Listed Options Half-Spread)
        if t == 1
            tc_f = 0.5 * data.F1_spread(1) * abs(phi1(1)) + ...
                   0.5 * data.F2_spread(1) * abs(phi2(1));
        elseif t == N_days && ~config.unwindAtMaturity
            tc_f = 0.0;
        else
            tc_f = 0.5 * data.F1_spread(t) * abs(phi1(t) - phi1(t-1)) + ...
                   0.5 * data.F2_spread(t) * abs(phi2(t) - phi2(t-1));
        end

        [tc_o, openOpts] = reconcileOptionPositions(openOpts, hedge.optTargets, chain1_t);

        futTransCost(t) = tc_f;
        optTransCost(t) = tc_o;
        transCost(t)    = tc_f + tc_o;

        cash(t)    = cashPre - transCost(t);
        portVal(t) = V_struct(t) + cash(t);
        if t == 1
            dailyPnL(1)    = 0.0;
            unhedgedPnL(1) = 0.0;
        else
            dailyPnL(t)    = portVal(t) - portVal(t-1);
            unhedgedPnL(t) = V_struct(t) - V_struct(t-1);
        end
    end

    % 6. Summary Metrics & Running Competition Score over Time
    pnl_steps          = dailyPnL(2:end);
    unhedged_steps     = unhedgedPnL(2:end);
    sqDailyPnL         = dailyPnL .^ 2;
    runningMSE         = [0; cumsum(pnl_steps .^ 2) ./ (1:numel(pnl_steps))'];

    mse_per_bbl        = mean(pnl_steps .^ 2);
    mse_unhedged_bbl   = mean(unhedged_steps .^ 2);
    mse_total          = mean((notional * pnl_steps) .^ 2);
    mse_unhedged_total = mean((notional * unhedged_steps) .^ 2);
    var_reduction_pct  = 100.0 * (1.0 - mse_per_bbl / mse_unhedged_bbl);

    results = struct();
    results.modelName          = mName;
    results.displayName        = displayName;
    results.hedgeMethod        = config.hedgeMethod;
    results.paramNames         = paramNames;
    results.dates              = data.dates;
    results.K_call             = K_call;
    results.K_put              = K_put;
    results.C_inception          = C_inception;
    results.P_inception          = P_inception;
    results.elapsedSec           = toc(tTimer);
    results.calibElapsedSec      = calibElapsedSec;
    results.priceHedgeElapsedSec = priceHedgeElapsedSec;
    results.calibMsPerDay        = 1000.0 * calibElapsedSec / N_days;
    results.priceHedgeMsPerDay   = 1000.0 * priceHedgeElapsedSec / N_days;
    results.mcDiag               = mcDiag;

    results.V_struct           = V_struct;
    results.P_leg              = P_leg;
    results.C_leg              = C_leg;
    results.delta1             = delta1;
    results.delta2             = delta2;
    results.gamma1             = gamma1;
    results.gamma2             = gamma2;
    results.vegaScale          = vegaScale;
    results.phi1               = phi1;
    results.phi2               = phi2;
    results.psi1               = psi1;
    results.optStrike1         = optStrike1;

    results.cash               = cash;
    results.interestCF         = interestCF;
    results.futuresPnL         = futuresPnL;
    results.cumFuturesPnL      = cumsum(futuresPnL);
    results.optionPnL          = optionPnL;
    results.cumOptionPnL       = cumsum(optionPnL);
    results.futTransCost       = futTransCost;
    results.optTransCost       = optTransCost;
    results.transCost          = transCost;
    results.cumTransCost       = cumsum(transCost);

    results.portVal            = portVal;
    results.dailyPnL           = dailyPnL;
    results.unhedgedPnL        = unhedgedPnL;
    results.sqDailyPnL         = sqDailyPnL;
    results.runningMSE         = runningMSE;

    results.V_struct_total     = notional * V_struct;
    results.cash_total         = notional * cash;
    results.portVal_total      = notional * portVal;
    results.dailyPnL_total     = notional * dailyPnL;
    results.unhedgedPnL_total  = notional * unhedgedPnL;
    results.sqDailyPnL_total   = (notional * dailyPnL) .^ 2;

    results.atmVol1            = atmVol1;
    results.atmVol2            = atmVol2;
    results.rho                = rho_ts;
    results.calibRMSE1         = calibRMSE1;
    results.calibRMSE2         = calibRMSE2;
    results.calibMAPE1         = calibMAPE1;
    results.calibMAPE2         = calibMAPE2;
    results.avgCalibRMSE1      = mean(calibRMSE1, 'omitnan');
    results.avgCalibRMSE2      = mean(calibRMSE2, 'omitnan');
    results.avgCalibMAPE1      = mean(calibMAPE1, 'omitnan');
    results.params1            = params1_ts;
    results.params2            = params2_ts;
    results.calibHistory       = calibHistory;
    results.baseStructHistory  = baseStructHistory;

    results.mse_per_bbl        = mse_per_bbl;
    results.mse_unhedged_bbl   = mse_unhedged_bbl;
    results.mse_total          = mse_total;
    results.mse_unhedged_total = mse_unhedged_total;
    results.var_reduction_pct  = var_reduction_pct;
    results.maxAbsDailyPnL     = max(abs(dailyPnL));
    results.finalPortVal       = portVal(end);
    results.safetyFlag         = safetyFlag;
    results.nSafetyDays        = sum(safetyFlag);
end

%% ========================================================================
%  HELPER 1: Lookup Listed Option Mid & Spread on Day t
%  ========================================================================
function [midVal, spVal, found] = lookupOptionQuote(chain, K, isCall)
    midVal = NaN; spVal = 0.0; found = false;
    if isempty(chain) || isempty(chain.strike)
        return;
    end
    idx = find(chain.strike == K & chain.isCall == isCall, 1);
    if ~isempty(idx) && ~isnan(chain.mid(idx)) && ~isnan(chain.bid(idx)) && ~isnan(chain.ask(idx))
        midVal = chain.mid(idx);
        spVal  = max(chain.ask(idx) - chain.bid(idx), 0.0);
        found  = true;
    end
end

%% ========================================================================
%  HELPER 2: Reconcile Open vs Target Listed Options & Charge Half-Spread
%  ========================================================================
function [tc_opt, nextOpen] = reconcileOptionPositions(prevOpen, targets, chain)
    tc_opt   = 0.0;
    nextOpen = struct('strike', {}, 'isCall', {}, 'psi', {}, 'lastMid', {});

    % Collect union of unique (strike, isCall) keys
    keysK = []; keysC = logical([]);
    for j = 1:numel(prevOpen)
        keysK(end+1) = prevOpen(j).strike; %#ok<AGROW>
        keysC(end+1) = prevOpen(j).isCall; %#ok<AGROW>
    end
    for j = 1:numel(targets)
        exists = any(keysK == targets(j).strike & keysC == targets(j).isCall);
        if ~exists
            keysK(end+1) = targets(j).strike; %#ok<AGROW>
            keysC(end+1) = targets(j).isCall; %#ok<AGROW>
        end
    end

    cnt = 0;
    for j = 1:numel(keysK)
        k_j = keysK(j);
        c_j = keysC(j);

        psiOld = 0.0;
        for a = 1:numel(prevOpen)
            if prevOpen(a).strike == k_j && prevOpen(a).isCall == c_j
                psiOld = prevOpen(a).psi;
                break;
            end
        end

        psiNew = 0.0;
        for b = 1:numel(targets)
            if targets(b).strike == k_j && targets(b).isCall == c_j
                psiNew = targets(b).psi;
                break;
            end
        end

        [midVal, spVal, found] = lookupOptionQuote(chain, k_j, c_j);
        dPsi = abs(psiNew - psiOld);
        if dPsi > 1e-10 && found
            tc_opt = tc_opt + 0.5 * spVal * dPsi;
        end

        if abs(psiNew) > 1e-10 && found
            cnt = cnt + 1;
            nextOpen(cnt) = struct('strike', k_j, 'isCall', c_j, 'psi', psiNew, 'lastMid', midVal);
        end
    end
end

function p = black76Mark(F, K, isCall, tau, sig)
    if tau <= 1e-6 || sig <= 1e-6
        if isCall, p = max(F - K, 0.0); else, p = max(K - F, 0.0); end
        return;
    end
    nc = @(x) 0.5 * erfc(-x / sqrt(2));
    s  = sig * sqrt(tau);
    d1 = (log(F / K) + 0.5 * s^2) / s;
    d2 = d1 - s;
    if isCall, p = F * nc(d1) - K * nc(d2);
    else,      p = K * nc(-d2) - F * nc(-d1); end
end

%% ========================================================================
%  HELPER: Inception MC Precision & European Vanilla Convergence Check
%  ========================================================================
function diag = computeMCVanillaDiagnostics(data, calib, K_put, K_call, config, asianSE_put, asianSE_call)
    % Benchmarks M-path MC against exact analytical / Lewis-FFT European COJ21
    % vanillas at [K_put (Put), F1_0 (ATM Call), K_call (Call)] at t0.
    F0     = data.F1_settle(1);
    tau    = calib.smile1.tau;
    Ks     = [K_put; F0; K_call];
    isCall = [false; true; true];
    M      = config.mcPaths;
    M_half = floor(M / 2);
    M_eff  = 2 * M_half;

    rng(config.mcSeed, 'twister');
    if strcmpi(calib.modelName, 'black76')
        % Exact closed-form Black-76 vs. M-path antithetic lognormal MC
        exactP = zeros(3, 1);
        mcP    = zeros(3, 1);
        mcSE   = zeros(3, 1);
        zh     = randn(1, M_half);
        z      = [zh, -zh];
        martErr = 0.0;
        for k = 1:3
            sig = calib.sigma1_fn(Ks(k));
            exactP(k) = black76Mark(F0, Ks(k), isCall(k), tau, sig);
            FT = F0 * exp(-0.5 * sig^2 * tau + sig * sqrt(tau) * z);
            if k == 2, martErr = abs(mean(FT) - F0); end
            if isCall(k), pay = max(FT - Ks(k), 0.0);
            else,         pay = max(Ks(k) - FT, 0.0); end
            mcP(k)  = mean(pay);
            mcSE(k) = std(pay) / sqrt(M_eff);
        end
    else
        spec = calib.spec;
        if spec.isMultiMat
            p1 = spec.evalSliceParams(tau, calib.atsCurve, 1.0);
        else
            p1 = calib.params1;
        end
        lcf = @(u) spec.logCF(u, tau, p1);
        C_ex = lewisFFT('call', lcf, F0, Ks, config.fftN, config.fftDu);
        exactP = C_ex;
        exactP(~isCall) = C_ex(~isCall) - (F0 - Ks(~isCall));

        if spec.isStochVol
            % Same 2D SDE Euler scheme used in priceBasket.m
            nSub = max(ceil(tau * 52.0), 1);
            dt   = tau / nSub;
            sqdt = sqrt(dt);
            X    = zeros(1, M_eff);
            V    = repmat(p1(1), 1, M_eff);
            if strcmpi(spec.name, 'bns')
                lam = p1(2); a = p1(3); b = p1(4); rho = p1(5);
                mu  = -a * rho / (b - rho);
                eL  = exp(-lam * dt); ehL = exp(-0.5 * lam * dt);
                for s = 1:nSub
                    zh = randn(1, M_half); w = [zh, -zh];
                    N  = samplePoisDiag(a * dt, M_eff);
                    dZ = sampleExpSumDiag(N, b);
                    Vn = eL * V + ehL * dZ;
                    IV = max((V - Vn + dZ) / lam, 0.0);
                    X  = X + mu * dt - 0.5 * IV + sqrt(IV) .* w + rho * dZ;
                    V  = Vn;
                end
            else
                kap = p1(2); eta = p1(3); th = p1(4); rho = p1(5);
                sr  = sqrt(max(1.0 - rho^2, 0.0));
                hasJ = strcmpi(spec.name, 'bates');
                if hasJ, lam = p1(6); muJ = p1(7); delJ = p1(8); kJ = exp(muJ + 0.5*delJ^2) - 1; end
                for s = 1:nSub
                    z1h = randn(1, M_half); zvh = randn(1, M_half);
                    w1  = [z1h, -z1h]; zv = rho * w1 + sr * [zvh, -zvh];
                    Vp  = max(V, 0.0); sqV = sqrt(Vp);
                    dX  = -0.5 * Vp * dt + sqV * sqdt .* w1;
                    if hasJ && lam > 0
                        Nj = samplePoisDiag(lam * dt, M_eff);
                        dX = dX - lam * kJ * dt + Nj * muJ + sqrt(Nj) * delJ .* randn(1, M_eff);
                    end
                    X = X + dX;
                    V = V + kap * (eta - Vp) * dt + th * sqV * sqdt .* zv;
                end
            end
            E_raw   = exp(X);
            martErr = abs(F0 * (mean(E_raw) - 1.0));
            FT      = F0 * (E_raw / mean(E_raw));
        else
            % Exact FFT-CDF inversion (same engine as priceBasket.m)
            [k_g, P_g] = lewisFFT('cdf', lcf, tau, config.cdfN);
            zh  = randn(1, M_half);
            U   = 0.5 * erfc(-[zh, -zh] / sqrt(2));
            Uc  = min(max(U, P_g(1)), P_g(end));
            X   = interp1(P_g, k_g, Uc, 'linear');
            E_raw   = exp(X);
            martErr = abs(F0 * (mean(E_raw) - 1.0));
            FT      = F0 * (E_raw / mean(E_raw));
        end

        mcP  = zeros(3, 1);
        mcSE = zeros(3, 1);
        for k = 1:3
            if isCall(k), pay = max(FT - Ks(k), 0.0);
            else,         pay = max(Ks(k) - FT, 0.0); end
            mcP(k)  = mean(pay);
            mcSE(k) = std(pay) / sqrt(M_eff);
        end
    end

    absErr = abs(mcP - exactP);
    diag = struct( ...
        'mcPaths',          M_eff, ...
        'strikes',          Ks, ...
        'exactVanilla',     exactP, ...
        'mcVanilla',        mcP, ...
        'vanillaAbsErr',    absErr, ...
        'vanillaMaxAbsErr', max(absErr), ...
        'vanillaMeanSE',    mean(mcSE), ...
        'martingaleErr',    martErr, ...
        'asianSE_put',      asianSE_put, ...
        'asianSE_call',     asianSE_call, ...
        'asianCI95_put',    1.96 * asianSE_put);
end

function N = samplePoisDiag(lam, M)
    N = zeros(1, M);
    if lam <= 0, return; end
    L = exp(-lam); p = ones(1, M); act = true(1, M);
    while any(act)
        p(act) = p(act) .* rand(1, sum(act));
        still = act & (p > L);
        N(still) = N(still) + 1;
        act = still;
    end
end

function S = sampleExpSumDiag(N, b)
    S = zeros(size(N));
    for r = 1:max(N)
        mask = (N >= r);
        S(mask) = S(mask) - log(max(rand(1, sum(mask)), 1e-15)) / b;
    end
end
