function calib = calibrateModel(t, data, config, prevCalib)
%CALIBRATEMODEL Daily calibration of Black-76, Levy, and Multi-Maturity ATS models.
%
%   calib = calibrateModel(t, data, config, prevCalib) calibrates the
%   selected model (config.modelName) on rebalancing day t using only
%   market data available up to day t:
%     - 'black76' : Undiscounted Black-76 (Asay, 1982) smile / flat ATM
%     - 'merton', 'kou', 'vg', 'nig', 'cgmy' : Per-leg Lewis FFT calibration
%     - 'ats'     : Multi-Maturity Joint Time-Changed Power-Law ATS
%                   (Note (3), Prop. 4.4, Eq. 16) calibrated across up to 5
%                   liquid Brent maturities (data.multiSurface{t}) simultaneously!

    if nargin < 4
        prevCalib = [];
    end

    F1_t = data.F1_settle(t);
    F2_t = data.F2_settle(t);

    calib = struct();
    calib.t         = t;
    calib.date      = data.dates(t);
    calib.F1        = F1_t;
    calib.F2        = F2_t;
    calib.modelName       = lower(strtrim(char(config.modelName)));
    calib.mcCache         = [];
    calib.atsCurve        = [];
    calib.safetyTriggered = false;
    calib.safetyReason    = '';

    %% 1. Extract Liquid OTM Option Quotes & Black-76 Implied Vols
    prevSmile1 = [];
    prevSmile2 = [];
    if ~isempty(prevCalib)
        prevSmile1 = prevCalib.smile1;
        prevSmile2 = prevCalib.smile2;
    end

    smile1 = extractLegSmile(data.optChain1{t}, F1_t, config, prevSmile1);
    smile2 = extractLegSmile(data.optChain2{t}, F2_t, config, prevSmile2);

    calib.smile1  = smile1;
    calib.smile2  = smile2;
    calib.atmVol1 = smile1.atmVol;
    calib.atmVol2 = smile2.atmVol;

    % Maintain strictly causal ATM vol history for Minimum-Variance / Shadow Delta
    if isempty(prevCalib) || ~isfield(prevCalib, 'atmVolHist')
        calib.atmVolHist = smile1.atmVol;
    else
        calib.atmVolHist = [prevCalib.atmVolHist; smile1.atmVol];
    end
    calib.betaVolF = estimateLeverageBeta(data.F1_settle(1:t), calib.atmVolHist);

    %% 2. Historical Correlation Estimation (Leg 1 vs Leg 2 Log-Returns)
    if t >= 2
        r1_hist = diff(log(data.F1_settle(1:t)));
        r2_hist = diff(log(data.F2_settle(1:t)));
    else
        r1_hist = [];
        r2_hist = [];
    end
    calib.rho = estimateCorrelation(r1_hist, r2_hist, config, prevCalib);

    %% 3. Model-Specific Calibration Dispatch
    if strcmpi(calib.modelName, 'black76')
        if strcmpi(config.volModel, 'flat_atm')
            calib.sigma1_fn = @(K) repmat(smile1.atmVol, size(K));
            calib.sigma2_fn = @(K) repmat(smile2.atmVol, size(K));
        else
            calib.sigma1_fn = buildSmileInterpolant(smile1.strikes, smile1.iv, config.interpMethod);
            calib.sigma2_fn = buildSmileInterpolant(smile2.strikes, smile2.iv, config.interpMethod);
        end
        calib.spec    = [];
        calib.params1 = smile1.atmVol;
        calib.params2 = smile2.atmVol;
        calib.rmse1   = 0.0; calib.rmse2 = 0.0;
        calib.mape1   = 0.0; calib.mape2 = 0.0;
        return;
    end

    spec = cfModels(calib.modelName);
    calib.spec = spec;
    calib.sigma1_fn = buildSmileInterpolant(smile1.strikes, smile1.iv, config.interpMethod);
    calib.sigma2_fn = buildSmileInterpolant(smile2.strikes, smile2.iv, config.interpMethod);

    if spec.isMultiMat
        % STEP C: Multi-Maturity Slice-by-Slice ATS Calibration (Theorem 4.1, p. 40)
        % Calibrates (sigma_m, k_m, eta_m) sequentially across all maturities in
        % data.multiSurface{t} enforcing non-decreasing g1, g2, g3 (Cont-Tankov).
        prevCurve = [];
        if ~isempty(prevCalib) && isfield(prevCalib, 'atsCurve') && ~isempty(prevCalib.atsCurve)
            prevCurve = prevCalib.atsCurve;
        end

        slices = data.multiSurface{t};
        [atsCurve, p1, p2, rmse1, rmse2, mape1, mape2] = ...
            calibrateMultiMaturityATS(spec, slices, data.leg1Code, data.leg2Code, prevCurve, config);

        calib.atsCurve = atsCurve;
        calib.params1  = p1;
        calib.params2  = p2;
        calib.rmse1    = rmse1;
        calib.rmse2    = rmse2;
        calib.mape1    = mape1;
        calib.mape2    = mape2;
    else
        % Single-leg Lewis FFT calibration for stationary Levy & SV models
        isCold = isempty(prevCalib) || ~isfield(prevCalib, 'params1') || isempty(prevCalib.params1);
        if ~isCold
            p1_init = prevCalib.params1;
            p2_init = prevCalib.params2;
        else
            p1_init = spec.p0;
            p2_init = spec.p0;
        end

        [p1, rmse1, mape1] = calibrateCFLeg(spec, smile1, F1_t, p1_init, config, isCold);
        [p2, rmse2, mape2] = calibrateCFLeg(spec, smile2, F2_t, p2_init, config, isCold);

        calib.params1 = p1;
        calib.params2 = p2;
        calib.rmse1   = rmse1;
        calib.rmse2   = rmse2;
        calib.mape1   = mape1;
        calib.mape2   = mape2;
    end

    %% 4. Safety Net Layer 1: Calibration Quality & Parameter-Jump Guard
    %  Triggers ONLY on genuine calibration breakdowns (spike in RMSE /
    %  high relative RMSE, sparse/carried-forward chain, near-expiry, or an
    %  unjustified parameter jump). Uses smooth damping (never hard-latches
    %  stale parameters) and flags computeHedge to revert Greeks to Black-76.
    if isfield(config, 'useSafetyNet') && config.useSafetyNet && ~isempty(prevCalib)
        maxRMSE   = getCfg(config, 'snMaxRMSE',      0.060);
        maxRelErr = getCfg(config, 'snMaxRelRMSE',   0.030);
        minStrk   = getCfg(config, 'snMinStrikes',   4);
        minTauD   = getCfg(config, 'snMinTauDays',   15);
        maxPJump  = getCfg(config, 'snMaxParamJump', 0.20);

        atmP      = max(smile1.mktPrice);
        if isempty(atmP) || ~isfinite(atmP), atmP = 1.0; end
        effMaxR   = max(maxRMSE, maxRelErr * atmP);
        prevR     = prevCalib.rmse1;
        rmseSpike = isfinite(prevR) && (prevR > 0) && (calib.rmse1 > max(0.045, 2.5 * prevR));

        badFit    = ~isfinite(calib.rmse1) || (calib.rmse1 > effMaxR) || rmseSpike;
        badChain  = smile1.carriedForward || (numel(smile1.strikes) < minStrk) || ...
                    (smile1.tau <= minTauD / 365.0);

        paramSpan = max(spec.ub(:)' - spec.lb(:)', 1e-6);
        relJump1  = max(abs(calib.params1(:)' - prevCalib.params1(:)') ./ paramSpan);
        % Only flag a parameter jump if fit is not already razor-sharp (prevents latching!)
        badJump   = ~isfinite(relJump1) || (relJump1 > maxPJump && calib.rmse1 > 0.025);

        if badFit || badChain || badJump
            calib.safetyTriggered = true;
            if badFit
                calib.safetyReason = 'high_rmse';
            elseif badJump
                calib.safetyReason = 'param_jump';
            else
                calib.safetyReason = 'sparse_or_expiry';
            end

            % On-manifold 1-parameter ATM volatility scaling of yesterday's clean shape
            % (Never linearly blend nonlinear parameter vectors!)
            vr1 = min(max(calib.atmVol1 / max(prevCalib.atmVol1, 1e-4), 0.80), 1.25);
            vr2 = min(max(calib.atmVol2 / max(prevCalib.atmVol2, 1e-4), 0.80), 1.25);

            if spec.isMultiMat && ~isempty(prevCalib.atsCurve)
                dt_y = 0.0;
                if t >= 2, dt_y = data.dt_yf(t); end
                cPrev = prevCalib.atsCurve;
                cSafe = cPrev;
                cSafe.taus = max(cPrev.taus - dt_y, 1e-4);
                cSafe.sigs = min(max(cPrev.sigs * vr1, spec.lb(1) + 1e-4), spec.ub(1) - 1e-4);
                allOk = true;
                for ms = 1:numel(cSafe.taus)
                    if ~spec.isValid([cSafe.sigs(ms), cSafe.kaps(ms), cSafe.etas(ms)], cSafe.taus(ms))
                        allOk = false; break;
                    end
                end
                if ~allOk, cSafe = cPrev; end
                p1_safe = spec.evalSliceParams(max(smile1.tau, 1e-4), cSafe, 1.0);
                rmseSafe = evalSmileRMSE(spec, p1_safe, smile1, F1_t, config);
                % Accept on-manifold frozen shape only if chain is bad or it fits well (prevents latching)
                if badChain || (isfinite(rmseSafe) && rmseSafe <= max(1.35 * calib.rmse1, effMaxR))
                    calib.atsCurve = cSafe;
                    calib.params1  = p1_safe;
                    calib.params2  = spec.evalSliceParams(max(smile2.tau, 1e-4), cSafe, 1.0);
                    if numel(smile1.strikes) < 3 || smile1.tau <= 5.0 / 365.0
                        calib.rmse1 = NaN;
                    elseif isfinite(rmseSafe)
                        calib.rmse1 = rmseSafe;
                    end
                end
            else
                p1s = spec.scaleVol(prevCalib.params1, vr1);
                p1s = min(max(p1s, spec.lb + 1e-4), spec.ub - 1e-4);
                if ~spec.isValid(p1s, max(smile1.tau, 1e-4))
                    p1s = prevCalib.params1;
                end
                rmseSafe = evalSmileRMSE(spec, p1s, smile1, F1_t, config);

                if badChain || (isfinite(rmseSafe) && rmseSafe <= max(1.35 * calib.rmse1, effMaxR))
                    calib.params1 = p1s;
                    if numel(smile1.strikes) < 3 || smile1.tau <= 5.0 / 365.0
                        calib.rmse1 = NaN;
                    elseif isfinite(rmseSafe)
                        calib.rmse1 = rmseSafe;
                    end

                    p2s = spec.scaleVol(prevCalib.params2, vr2);
                    p2s = min(max(p2s, spec.lb + 1e-4), spec.ub - 1e-4);
                    if spec.isValid(p2s, max(smile2.tau, 1e-4))
                        calib.params2 = p2s;
                    else
                        calib.params2 = prevCalib.params2;
                    end
                end
            end
        end
    end

    % Model-implied leverage slope for Heston/Bates (Upgrade #2: MV-Delta)
    if strcmpi(calib.modelName, 'heston') || strcmpi(calib.modelName, 'bates')
        v0_m = max(calib.params1(1), 1e-4);
        th_m = calib.params1(4);
        rh_m = calib.params1(5);
        betaMdl = (rh_m * th_m) / (2.0 * v0_m * max(F1_t, 1.0));
        if isfinite(betaMdl)
            calib.betaVolF = min(max(betaMdl, -0.06), 0.0);
        end
    end
end

function beta = estimateLeverageBeta(F_hist, vol_hist)
    % Strictly causal empirical spot-vol leverage beta = Cov(d log vol, dF) / Var(dF)
    beta = -0.02; % Conservative oil leverage prior when < 10 observations
    n = min(numel(F_hist), numel(vol_hist));
    if n < 10
        return;
    end
    i0 = max(1, n - 60); % 60-day rolling causal window
    dF = diff(F_hist(i0:n));
    dV = diff(log(max(vol_hist(i0:n), 1e-4)));
    ok = isfinite(dF) & isfinite(dV) & (abs(dV) < 0.35);
    if sum(ok) >= 8
        vF = var(dF(ok));
        if vF > 1e-8
            cMat = cov(dF(ok), dV(ok));
            beta = min(max(cMat(1, 2) / vF, -0.06), 0.0);
        end
    end
end

function rmse = evalSmileRMSE(spec, p, smile, F_t, config)
    if isempty(smile) || isempty(smile.strikes) || smile.tau <= 1e-4 || ~spec.isValid(p, smile.tau)
        rmse = Inf;
        return;
    end
    lcf   = @(u) spec.logCF(u, smile.tau, p);
    C_mdl = lewisFFT('call', lcf, F_t, smile.strikes, config.fftN, config.fftDu);
    P_mdl = C_mdl;
    P_mdl(~smile.isCall) = C_mdl(~smile.isCall) - (F_t - smile.strikes(~smile.isCall));
    err   = P_mdl - smile.mktPrice;
    if any(~isfinite(err))
        rmse = Inf;
    else
        rmse = sqrt(mean(err.^2));
    end
end

function v = getCfg(cfg, field, defaultVal)
    if isfield(cfg, field) && ~isempty(cfg.(field))
        v = cfg.(field);
    else
        v = defaultVal;
    end
end

%% ========================================================================
%  HELPER 1: Multi-Maturity Slice-by-Slice ATS Calibration (Theorem 4.1, p. 40)
%  ========================================================================
function [atsCurve, p1, p2, rmse1, rmse2, mape1, mape2] = ...
    calibrateMultiMaturityATS(spec, slices, leg1Code, leg2Code, prevCurve, config)

    nS = numel(slices);
    if nS == 0
        atsCurve = prevCurve;
        p1 = [prevCurve.sigs(1), prevCurve.kaps(1), prevCurve.etas(1)];
        p2 = [prevCurve.sigs(end), prevCurve.kaps(end), prevCurve.etas(end)];
        rmse1 = NaN; rmse2 = NaN; mape1 = NaN; mape2 = NaN;
        return;
    end

    taus  = zeros(nS, 1);
    sigs  = zeros(nS, 1);
    kaps  = zeros(nS, 1);
    etas  = zeros(nS, 1);
    codes = cell(nS, 1);

    lb     = spec.lb(:)';
    ub     = spec.ub(:)';
    N_fft  = config.fftN;
    du_fft = config.fftDu;

    g_prev = [];
    p_prev = [];
    rmse1  = NaN; mape1 = NaN;
    rmse2  = NaN; mape2 = NaN;
    p1     = spec.p0;
    p2     = spec.p0;

    isColdStart = isempty(prevCurve);
    maxIter     = config.calibMaxEval * (1 + isColdStart);
    shapeW      = getCfg(config, 'shapeRegWeight', 1e-3);
    useVW       = logical(getCfg(config, 'useVegaWeightCalib', false));
    if isColdStart
        regVec = 1e-3 * ones(3, 1);
    else
        regVec = [1e-3; shapeW; shapeW]; % Keep sigma_t free; regularize (kappa_t, eta_t)
    end

    for m = 1:nS
        sl    = slices(m);
        tau_m = sl.tau;
        Ks    = sl.strikes;
        isC   = sl.isCall;
        Pm    = sl.mktPrice;
        wSlice = computeVegaWeights(sl.F, Ks, tau_m, Pm, isC, useVW);

        % Warm-start this contract code from previous day if available,
        % falling back to preceding slice p_prev if Theorem 4.1 infeasible.
        p0 = spec.p0;
        if ~isempty(prevCurve)
            idxPrev = find(strcmp(prevCurve.codes, sl.code), 1);
            if ~isempty(idxPrev)
                p0 = [prevCurve.sigs(idxPrev), prevCurve.kaps(idxPrev), prevCurve.etas(idxPrev)];
            else
                p0 = spec.evalSliceParams(tau_m, prevCurve, 1.0);
            end
        elseif ~isempty(p_prev)
            p0 = p_prev;
        end
        p0 = min(max(p0, lb + 1e-4), ub - 1e-4);
        if ~spec.isValid(p0, tau_m)
            p0 = spec.p0;
        end
        if ~isempty(g_prev) && ~isempty(p_prev) && any(spec.gFuncs(tau_m, p0) < g_prev - 1e-6)
            % p_prev is always strictly feasible for tau_m > tau_{m-1}
            p0 = min(max(p_prev, lb + 1e-4), ub - 1e-4);
        end

        residFn = @(p) evalATSSliceResid(p, spec, sl, g_prev, p0, lb, ub, regVec, wSlice, N_fft, du_fft);

        if exist('lsqnonlin', 'file') == 2
            opts = optimoptions('lsqnonlin', ...
                'Display', 'off', ...
                'FunctionTolerance', 1e-6, ...
                'StepTolerance', 1e-6, ...
                'MaxIterations', maxIter, ...
                'MaxFunctionEvaluations', maxIter * (numel(p0) + 2));
            p_opt = lsqnonlin(residFn, p0, lb, ub, opts);
        else
            toUncon = @(p) log((p - lb) ./ (ub - p));
            toCon   = @(z) lb + (ub - lb) ./ (1.0 + exp(-min(max(z, -30), 30)));
            opts    = optimset('Display', 'off', 'TolFun', 1e-6, 'TolX', 1e-6, ...
                'MaxFunEvals', maxIter * 8);
            z_opt   = fminsearch(@(z) sum(residFn(toCon(z)).^2), toUncon(p0), opts);
            p_opt   = toCon(z_opt);
        end

        taus(m)  = tau_m;
        sigs(m)  = p_opt(1);
        kaps(m)  = p_opt(2);
        etas(m)  = p_opt(3);
        codes{m} = sl.code;
        p_prev   = p_opt;
        g_prev   = spec.gFuncs(tau_m, p_opt);

        if strcmp(sl.code, leg1Code) || strcmp(sl.code, leg2Code)
            lcf = @(u) spec.logCF(u, tau_m, p_opt);
            C_m = lewisFFT('call', lcf, sl.F, Ks, N_fft, du_fft);
            P_m = C_m;
            P_m(~isC) = C_m(~isC) - (sl.F - Ks(~isC));
            err = P_m - Pm;
            if strcmp(sl.code, leg1Code)
                p1    = p_opt;
                rmse1 = sqrt(mean(err.^2));
                mape1 = 100.0 * mean(abs(err) ./ max(Pm, 1e-4));
            else
                p2    = p_opt;
                rmse2 = sqrt(mean(err.^2));
                mape2 = 100.0 * mean(abs(err) ./ max(Pm, 1e-4));
            end
        end
    end

    atsCurve = struct('taus', taus, 'sigs', sigs, 'kaps', kaps, 'etas', etas, 'codes', {codes});
    if all(p1 == spec.p0), p1 = [sigs(1), kaps(1), etas(1)]; end
    if all(p2 == spec.p0), p2 = [sigs(end), kaps(end), etas(end)]; end
end

function r = evalATSSliceResid(p, spec, sl, g_prev, p0, lb, ub, regVec, wSlice, N_fft, du_fft)
    tau_m = sl.tau;
    Ks    = sl.strikes;
    isC   = sl.isCall;
    Pm    = sl.mktPrice;
    nTot  = numel(Pm) + 3 + numel(p0);

    if ~spec.isValid(p, tau_m)
        r = 1e3 * ones(nTot, 1);
        return;
    end
    lcf = @(u) spec.logCF(u, tau_m, p);
    C_m = lewisFFT('call', lcf, sl.F, Ks, N_fft, du_fft);
    P_m = C_m;
    P_m(~isC) = C_m(~isC) - (sl.F - Ks(~isC));
    err = wSlice .* (P_m - Pm);

    % Enforce Theorem 4.1 existence constraints: g1, g2, g3 non-decreasing
    pen = zeros(3, 1);
    if ~isempty(g_prev)
        g_cur  = spec.gFuncs(tau_m, p);
        pen(1) = 10.0 * max(g_prev(1) - g_cur(1), 0.0);
        pen(2) = 10.0 * max(g_prev(2) - g_cur(2), 0.0);
        pen(3) = 10.0 * max(g_prev(3) - g_cur(3), 0.0);
    end

    % Asymmetric Tikhonov damping: free vol level p(1), smooth shape params p(2:end)
    reg = regVec .* ((p(:) - p0(:)) ./ max(ub(:) - lb(:), 1e-6));
    r   = [err; pen; reg];
    if any(~isfinite(r))
        r = 1e3 * ones(nTot, 1);
    end
end

function w = computeVegaWeights(F, Ks, tau, Pm, isC, useVW)
    % Normalized clamped inverse-Vega weights (approx. Implied-Vol RMSE objective).
    % Caps wing amplification at 4x ATM and normalizes mean(w.^2) == 1.
    n = numel(Ks);
    w = ones(n, 1);
    if ~useVW || n == 0 || tau <= 1e-4 || F <= 1e-4
        return;
    end
    ivs = asayImpliedVol(Pm, F, Ks, tau, isC);
    medIV = median(ivs(isfinite(ivs) & ivs > 0.01));
    if isempty(medIV) || ~isfinite(medIV), medIV = 0.30; end
    ivs(~isfinite(ivs) | ivs <= 0.01) = medIV;

    s  = ivs .* sqrt(tau);
    d1 = (log(F ./ max(Ks, 1e-6)) + 0.5 * (s.^2)) ./ max(s, 1e-6);
    v  = F * sqrt(tau) * (exp(-0.5 * (d1.^2)) / sqrt(2.0 * pi));
    vMax = max(v);
    if ~isfinite(vMax) || vMax <= 1e-8
        return;
    end
    raw = vMax ./ max(v, 0.25 * vMax);
    w   = raw / sqrt(mean(raw.^2));
end

%% ========================================================================
%  HELPER 2: Calibrate Single-Leg Smile via Lewis FFT (Levy & SV Models)
%  ========================================================================
function [p_opt, rmse, mape] = calibrateCFLeg(spec, smile, F_t, p_init, config, isColdStart)
    if nargin < 6, isColdStart = false; end
    if isempty(smile) || numel(smile.strikes) < 3 || smile.tau <= 1e-4 || smile.carriedForward
        p_opt = p_init;
        rmse  = NaN;
        mape  = NaN;
        return;
    end

    tau    = smile.tau;
    K_mkt  = smile.strikes;
    isCall = smile.isCall;
    P_mkt  = smile.mktPrice;

    lb = spec.lb(:)';
    ub = spec.ub(:)';
    p0 = min(max(p_init(:)', lb + 1e-4), ub - 1e-4);
    if ~spec.isValid(p0, tau)
        p0 = spec.p0;
    end

    N_fft   = config.fftN;
    du_fft  = config.fftDu;
    nMkt    = numel(P_mkt);
    nPar    = numel(p0);
    scaleP  = max(ub(:) - lb(:), 1e-6);
    shapeW  = getCfg(config, 'shapeRegWeight', 1e-3);
    useVW   = logical(getCfg(config, 'useVegaWeightCalib', false));
    wMkt    = computeVegaWeights(F_t, K_mkt, tau, P_mkt, isCall, useVW);
    if isColdStart
        regVec = 1e-3 * ones(nPar, 1);
    else
        regVec = shapeW * ones(nPar, 1);
        regVec(1) = 1e-3; % Keep primary vol level p(1) free; smooth shape params
    end
    maxIter = config.calibMaxEval * (1 + isColdStart);

    function r = residFn(p)
        if ~spec.isValid(p, tau)
            r = 1e3 * ones(nMkt + nPar, 1);
            return;
        end
        logCF_fn = @(u) spec.logCF(u, tau, p);
        C_mdl = lewisFFT('call', logCF_fn, F_t, K_mkt, N_fft, du_fft);
        P_mdl = C_mdl;
        P_mdl(~isCall) = C_mdl(~isCall) - (F_t - K_mkt(~isCall));
        reg = regVec .* ((p(:) - p0(:)) ./ scaleP);
        r   = [wMkt .* (P_mdl - P_mkt); reg];
        if any(~isfinite(r))
            r = 1e3 * ones(nMkt + nPar, 1);
        end
    end

    if exist('lsqnonlin', 'file') == 2
        opts = optimoptions('lsqnonlin', ...
            'Display', 'off', ...
            'FunctionTolerance', 1e-6, ...
            'StepTolerance', 1e-6, ...
            'MaxIterations', maxIter, ...
            'MaxFunctionEvaluations', maxIter * (nPar + 2));
        p_opt = lsqnonlin(@residFn, p0, lb, ub, opts);
    else
        toUncon = @(p) log((p - lb) ./ (ub - p));
        toCon   = @(z) lb + (ub - lb) ./ (1.0 + exp(-min(max(z, -30), 30)));
        opts    = optimset('Display', 'off', 'TolFun', 1e-6, 'TolX', 1e-6, ...
            'MaxFunEvals', maxIter * (nPar + 4));
        z_opt   = fminsearch(@(z) sum(residFn(toCon(z)).^2), toUncon(p0), opts);
        p_opt   = toCon(z_opt);
    end

    r_fin = residFn(p_opt);
    % Report true unweighted dollar RMSE/MAPE for fair cross-run comparison
    r_mkt = r_fin(1:nMkt) ./ wMkt;
    rmse  = sqrt(mean(r_mkt.^2));
    mape  = 100.0 * mean(abs(r_mkt) ./ max(P_mkt, 1e-4));
end

%% ========================================================================
%  HELPER 3: Filter Liquid OTM Option Quotes & Extract Implied Vols
%  ========================================================================
function smile = extractLegSmile(chain, F_t, config, prevSmile)
    if isempty(chain) || isempty(chain.strike) || chain.tau <= 1e-4
        if ~isempty(prevSmile)
            smile = prevSmile;
            smile.carriedForward = true;
            return;
        else
            error('calibrateModel:EmptyChain', 'Option chain is empty on initial date.');
        end
    end

    K      = chain.strike;
    isCall = chain.isCall;
    bid    = chain.bid;
    ask    = chain.ask;
    mid    = chain.mid;
    settle = chain.settle;
    tau    = chain.tau;

    if strcmpi(config.quoteSource, 'settle')
        mktPrice = settle;
    else
        mktPrice = mid;
    end

    validQuote  = ~isnan(mktPrice) & (mktPrice > 0) & ~isnan(bid) & (bid >= config.minOptionBid);
    relSpread   = (ask - bid) ./ max(mid, 1e-6);
    validSpread = ~isnan(relSpread) & (relSpread >= 0) & (relSpread <= config.maxRelSpread);

    if config.useOTMOnly
        isOTM = (~isCall & (K <= F_t)) | (isCall & (K > F_t));
    else
        isOTM = true(size(K));
    end

    keep = validQuote & validSpread & isOTM;

    if sum(keep) >= 2
        K_filt     = K(keep);
        price_filt = mktPrice(keep);
        call_filt  = isCall(keep);
        bid_filt   = bid(keep);
        ask_filt   = ask(keep);
        mid_filt   = mid(keep);

        [iv_raw, vega_raw, ~] = asayImpliedVol(price_filt, F_t, K_filt, tau, call_filt);
        validIV = ~isnan(iv_raw) & (iv_raw > 0.01) & (iv_raw < 3.0);

        K_clean     = K_filt(validIV);
        iv_clean    = iv_raw(validIV);
        vega_clean  = vega_raw(validIV);
        price_clean = price_filt(validIV);
        bid_clean   = bid_filt(validIV);
        ask_clean   = ask_filt(validIV);
        mid_clean   = mid_filt(validIV);
        call_clean  = call_filt(validIV);

        [K_unique, ia] = unique(K_clean, 'stable');
        [K_unique, ord] = sort(K_unique);
        ia = ia(ord);

        iv_unique    = iv_clean(ia);
        vega_unique  = vega_clean(ia);
        price_unique = price_clean(ia);
        bid_unique   = bid_clean(ia);
        ask_unique   = ask_clean(ia);
        mid_unique   = mid_clean(ia);
        call_unique  = call_clean(ia);
    else
        K_unique = [];
    end

    if numel(K_unique) < 2
        if ~isempty(prevSmile)
            smile = prevSmile;
            smile.tau = tau;
            smile.carriedForward = true;
            return;
        else
            error('calibrateModel:InsufficientStrikes', ...
                'Fewer than 2 valid option strikes survived filtering.');
        end
    end

    atmFn  = buildSmileInterpolant(K_unique, iv_unique, config.interpMethod);
    atmVol = atmFn(F_t);

    smile = struct( ...
        'tau',            tau, ...
        'F',              F_t, ...
        'strikes',        K_unique, ...
        'iv',             iv_unique, ...
        'vega',           vega_unique, ...
        'mktPrice',       price_unique, ...
        'bid',            bid_unique, ...
        'ask',            ask_unique, ...
        'mid',            mid_unique, ...
        'isCall',         call_unique, ...
        'atmVol',         atmVol, ...
        'carriedForward', false);
end

function fn = buildSmileInterpolant(strikes, iv, method)
    K_min = min(strikes);
    K_max = max(strikes);
    fn = @(K_query) interp1(strikes, iv, min(max(K_query, K_min), K_max), method);
end

function rho = estimateCorrelation(r1, r2, config, prevCalib)
    nObs = numel(r1);
    if nObs < config.minCorrObs
        rho = config.rhoPrior;
        return;
    end

    switch lower(config.corrMethod)
        case 'constant'
            rho = config.rhoPrior;
        case 'expanding'
            C = corrcoef(r1, r2);
            rho = C(1, 2);
        case 'rolling'
            w = min(nObs, config.corrWindow);
            C = corrcoef(r1(end-w+1:end), r2(end-w+1:end));
            rho = C(1, 2);
        case 'ewma'
            lam = config.ewmaLambda;
            weights = (1 - lam) * (lam .^ ((nObs-1:-1:0)'));
            weights = weights / sum(weights);
            m1 = sum(weights .* r1);
            m2 = sum(weights .* r2);
            v1 = sum(weights .* (r1 - m1).^2);
            v2 = sum(weights .* (r2 - m2).^2);
            cov12 = sum(weights .* (r1 - m1) .* (r2 - m2));
            rho = cov12 / sqrt(max(v1 * v2, 1e-16));
        otherwise
            error('calibrateModel:UnknownCorrMethod', ...
                'Unknown corrMethod "%s".', config.corrMethod);
    end

    if isnan(rho)
        if ~isempty(prevCalib) && isfield(prevCalib, 'rho')
            rho = prevCalib.rho;
        else
            rho = config.rhoPrior;
        end
    end
    rho = min(max(rho, -0.9999), 0.9999);
end
