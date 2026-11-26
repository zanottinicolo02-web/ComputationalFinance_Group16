function [price, details] = priceBasket(t, K, isCall, f1_val, f2_val, data, calib, config, volScale, skewBump)
%PRICEBASKET Price an Asian basket Call or Put under Black-76 or any CF model.
%
%   [price, details] = priceBasket(t, K, isCall, f1_val, f2_val, data, calib, config)
%   [price, details] = priceBasket(..., volScale, skewBump)
%
%   Optional 'volScale' (default 1.0) applies a uniform relative volatility
%   scaling (parallel Vega). Optional 'skewBump' (default 0.0) applies a
%   model-consistent skew perturbation (Note (3), pp. 38-39).

    if nargin < 9 || isempty(volScale)
        volScale = 1.0;
    end
    if nargin < 10 || isempty(skewBump)
        skewBump = 0.0;
    end

    d_t       = data.dates(t);
    df_T      = data.df_settle(t);
    fix_dates = data.fixing_dates;
    fix_idx   = data.fixing_idx;
    N_fix     = data.N_fix;
    n1        = data.n1;
    n2        = data.n2;
    w1        = data.w1;
    w2        = data.w2;

    %% ====================================================================
    %  BRANCH A: CHARACTERISTIC-FUNCTION LEVY, ATS, & STOCHASTIC VOL MODELS
    %  ====================================================================
    if ~strcmpi(calib.modelName, 'black76')
        useCache = isfield(calib, 'mcCache') && ~isempty(calib.mcCache) && ...
                   isfield(calib.mcCache, 'volScale') && ...
                   (abs(calib.mcCache.volScale - volScale) < 1e-12) && ...
                   (abs(calib.mcCache.skewBump - skewBump) < 1e-12);
        if useCache
            mc = calib.mcCache;
        elseif calib.spec.isStochVol
            mc = simulateSVBasketFactors(t, data, calib, config, volScale, skewBump);
            mc.volScale = volScale;
            mc.skewBump = skewBump;
        else
            mc = simulateCFBasketFactors(t, data, calib, config, volScale, skewBump);
            mc.volScale = volScale;
            mc.skewBump = skewBump;
        end

        A_paths = f1_val * mc.B1 + f2_val * mc.B2 + mc.S_known;
        if isCall
            payoff = max(A_paths - K, 0.0);
        else
            payoff = max(K - A_paths, 0.0);
        end

        % Optional Upgrade #3: European Lewis-FFT Control Variate at midpoint fixing
        useCFCV = isfield(config, 'useCFControlVariate') && config.useCFControlVariate && ...
                  isfield(mc, 'hasCV') && mc.hasCV;
        if useCFCV
            F_eff = mc.w1_rem * f1_val + mc.w2_rem * f2_val;
            K_eff = K - mc.S_known;
            if F_eff > 1e-4 && K_eff > 1e-4 && mc.tau_cv > 1e-4
                if isCall
                    Y_cv = max(F_eff * mc.E_cv - K_eff, 0.0);
                else
                    Y_cv = max(K_eff - F_eff * mc.E_cv, 0.0);
                end
                spec = calib.spec;
                if spec.isMultiMat
                    p_cv = spec.evalSliceParams(mc.tau_cv, calib.atsCurve, volScale, skewBump);
                else
                    p_cv = spec.scaleVol(calib.params1, volScale);
                    if abs(skewBump) > 1e-12
                        p_cv = spec.bumpSkew(p_cv, skewBump);
                    end
                end
                if spec.isValid(p_cv, mc.tau_cv)
                    lcf_cv   = @(u) spec.logCF(u, mc.tau_cv, p_cv);
                    c_exact  = lewisFFT('call', lcf_cv, F_eff, K_eff, config.fftN, config.fftDu);
                    if isCall
                        exact_cv = max(c_exact(1), 0.0);
                    else
                        exact_cv = max(c_exact(1) - (F_eff - K_eff), 0.0);
                    end
                    if isfinite(exact_cv)
                        price = max(df_T * (mean(payoff) + (exact_cv - mean(Y_cv))), 0.0);
                        se    = df_T * std(payoff - Y_cv) / sqrt(numel(payoff));
                        details = struct('se', se, 'M1', mean(A_paths), 'rho', calib.rho, ...
                                         'mcCache', mc);
                        return;
                    end
                end
            end
        end

        price = df_T * mean(payoff);
        se    = df_T * std(payoff) / sqrt(numel(payoff));

        details = struct('se', se, 'M1', mean(A_paths), 'rho', calib.rho, ...
                         'mcCache', mc);
        return;
    end

    %% ====================================================================
    %  BRANCH B: BASELINE BLACK-76 (Moment Matching or Bivariate Lognormal MC)
    %  ====================================================================
    theta = max(days(fix_dates - d_t) / 365.0, 0.0);

    F_avg = w1 * f1_val + w2 * f2_val;
    if K > 0 && F_avg > 0
        K1 = f1_val * (K / F_avg);
        K2 = f2_val * (K / F_avg);
    else
        K1 = f1_val;
        K2 = f2_val;
    end

    rot1 = 1.0;
    rot2 = 1.0;
    if abs(skewBump) > 1e-12
        % Smooth smile rotation around ATM (Note (3), p. 38, item 3)
        rot1 = 1.0 + 0.05 * skewBump * tanh(log(f1_val / max(K1, 1e-4)) / 0.15);
        rot2 = 1.0 + 0.05 * skewBump * tanh(log(f2_val / max(K2, 1e-4)) / 0.15);
    end
    sig1 = calib.sigma1_fn(K1) * volScale * rot1;
    sig2 = calib.sigma2_fn(K2) * volScale * rot2;
    rho  = calib.rho;

    isFixed = (fix_dates <= d_t);
    M1 = 0.0;
    for k = 1:N_fix
        if isFixed(k)
            idx_k = fix_idx(k);
            if k <= n1, M1 = M1 + data.F1_settle(idx_k) / N_fix;
            else,       M1 = M1 + data.F2_settle(idx_k) / N_fix; end
        else
            if k <= n1, M1 = M1 + f1_val / N_fix;
            else,       M1 = M1 + f2_val / N_fix; end
        end
    end

    switch lower(config.pricerMethod)
        case 'moment_matching'
            [price, details] = priceMomentMatching( ...
                K, isCall, f1_val, f2_val, sig1, sig2, rho, ...
                theta, n1, n2, w1, w2, M1, df_T);

        case 'monte_carlo'
            [price, details] = priceBivariateMC( ...
                K, isCall, f1_val, f2_val, sig1, sig2, rho, ...
                theta, isFixed, fix_idx, n1, n2, N_fix, df_T, data, config);

        otherwise
            error('priceBasket:UnknownMethod', ...
                'Unknown pricerMethod "%s" for Black-76.', config.pricerMethod);
    end

    details.K1   = K1;
    details.K2   = K2;
    details.sig1 = sig1;
    details.sig2 = sig2;
    details.rho  = rho;
    details.M1   = M1;
end

%% ========================================================================
%  CF-MC ENGINE: Correlated Two-Leg Levy & Multi-Maturity ATS Simulation
%                (Note (3), Section 2.4, Corollary 2.12, Prop. 4.4)
%  ========================================================================
function mc = simulateCFBasketFactors(t, data, calib, config, volScale, skewBump)
    if nargin < 6 || isempty(skewBump), skewBump = 0.0; end
    d_t       = data.dates(t);
    fix_dates = data.fixing_dates;
    fix_idx   = data.fixing_idx;
    N_fix     = data.N_fix;
    n1        = data.n1;
    spec      = calib.spec;
    p1        = spec.scaleVol(calib.params1, volScale);
    p2        = spec.scaleVol(calib.params2, volScale);
    if abs(skewBump) > 1e-12
        p1 = spec.bumpSkew(p1, skewBump);
        p2 = spec.bumpSkew(p2, skewBump);
    end
    rho       = calib.rho;
    M         = config.mcPaths;
    N_cdf     = config.cdfN;

    theta   = max(days(fix_dates - d_t) / 365.0, 0.0);
    isFixed = (fix_dates <= d_t);
    S_known = 0.0;
    for k = 1:N_fix
        if isFixed(k)
            idx_k = fix_idx(k);
            if k <= n1, S_known = S_known + data.F1_settle(idx_k) / N_fix;
            else,       S_known = S_known + data.F2_settle(idx_k) / N_fix; end
        end
    end

    active = find(~isFixed & (theta > 0));
    nAct   = numel(active);
    if nAct == 0
        mc = struct('B1', zeros(1, M), 'B2', zeros(1, M), 'S_known', S_known, 'hasCV', false);
        return;
    end

    act1   = active(active <= n1);
    w1_rem = numel(act1) / N_fix;
    w2_rem = (nAct - numel(act1)) / N_fix;
    k_cv   = -1;
    if ~isempty(act1)
        k_cv = act1(ceil(numel(act1) / 2));
    end

    % In-place step-by-step accumulation (1 x M vectors instead of N_fix x M)
    rng(config.mcSeed, 'twister');
    M_half   = floor(M / 2);
    M_eff    = 2 * M_half;
    sr       = sqrt(max(1.0 - rho^2, 0.0));
    norm_cdf = @(x) 0.5 * erfc(-x / sqrt(2));

    X1 = zeros(1, M_eff);
    X2 = zeros(1, M_eff);
    B1 = zeros(1, M_eff);
    B2 = zeros(1, M_eff);
    E_cv = []; tau_cv = 0.0;

    prev_t     = 0.0;
    cacheKeys1 = []; cacheGrid1 = {};
    cacheKeys2 = []; cacheGrid2 = {};

    for j = 1:nAct
        k      = active(j);
        curr_t = theta(k);

        % Antithetic Common Random Numbers (CRN) generated one step at a time
        z1_h = randn(1, M_half);
        z2_h = randn(1, M_half);
        z1   = [z1_h, -z1_h];
        z2   = rho * z1 + sr * [z2_h, -z2_h];

        if spec.isMultiMat
            % Multi-Maturity Slice-by-Slice ATS (Theorem 4.1 & Corollary 2.12)
            dt_k = curr_t - prev_t;
            p_e  = spec.evalSliceParams(curr_t, calib.atsCurve, volScale, skewBump);
            if prev_t <= 1e-12
                lcf = @(u) spec.logCF(u, curr_t, p_e);
            else
                p_s = spec.evalSliceParams(prev_t, calib.atsCurve, volScale, skewBump);
                lcf = @(u) spec.logCF(u, curr_t, p_e) - spec.logCF(u, prev_t, p_s);
            end
            [k_g, P_g] = lewisFFT('cdf', lcf, dt_k, N_cdf);

            if k <= n1
                X1 = X1 + invertCDF(k_g, P_g, norm_cdf(z1));
                E1 = exp(X1);
                E1n = E1 / mean(E1);
                B1 = B1 + E1n / N_fix;
                if k == k_cv
                    E_cv   = E1n;
                    tau_cv = curr_t;
                end
            end
            X2 = X2 + invertCDF(k_g, P_g, norm_cdf(z2));
            if k > n1
                E2 = exp(X2);
                B2 = B2 + (E2 / mean(E2)) / N_fix;
            end
        else
            % Stationary Levy models: cache CDF grids by step size dt_k
            dt_k = curr_t - prev_t;
            if k <= n1
                [k1_g, P1_g, cacheKeys1, cacheGrid1] = getIncrementCDF( ...
                    spec, prev_t, curr_t, dt_k, p1, N_cdf, cacheKeys1, cacheGrid1);
                X1 = X1 + invertCDF(k1_g, P1_g, norm_cdf(z1));
                E1 = exp(X1);
                E1n = E1 / mean(E1);
                B1 = B1 + E1n / N_fix;
                if k == k_cv
                    E_cv   = E1n;
                    tau_cv = curr_t;
                end
            end

            [k2_g, P2_g, cacheKeys2, cacheGrid2] = getIncrementCDF( ...
                spec, prev_t, curr_t, dt_k, p2, N_cdf, cacheKeys2, cacheGrid2);
            X2 = X2 + invertCDF(k2_g, P2_g, norm_cdf(z2));
            if k > n1
                E2 = exp(X2);
                B2 = B2 + (E2 / mean(E2)) / N_fix;
            end
        end

        prev_t = curr_t;
    end

    mc = struct('B1', B1, 'B2', B2, 'S_known', S_known, ...
                'hasCV', ~isempty(E_cv), 'E_cv', E_cv, 'tau_cv', tau_cv, ...
                'w1_rem', w1_rem, 'w2_rem', w2_rem);
end

function [k_g, P_g, cacheKeys, cacheGrid] = getIncrementCDF(spec, t_s, t_e, dt_k, p, N_cdf, cacheKeys, cacheGrid)
    if ~spec.isAdditive
        key = round(dt_k * 1e6);
        hit = find(cacheKeys == key, 1);
        if ~isempty(hit)
            k_g = cacheGrid{hit}{1};
            P_g = cacheGrid{hit}{2};
            return;
        end
    end

    incLogCF_fn = @(u) spec.incLogCF(u, t_s, t_e, p);
    [k_g, P_g]  = lewisFFT('cdf', incLogCF_fn, dt_k, N_cdf);

    if ~spec.isAdditive
        cacheKeys(end + 1) = round(dt_k * 1e6);
        cacheGrid{end + 1} = {k_g, P_g};
    end
end

function X = invertCDF(k_g, P_g, U)
    if numel(k_g) < 3
        X = zeros(size(U));
    else
        U_c = min(max(U, P_g(1)), P_g(end));
        X   = interp1(P_g, k_g, U_c, 'linear');
    end
end

%% ========================================================================
%  SV-MC ENGINE: Correlated Two-Leg 2D Markov SDE Simulation
%                (Heston, Bates, BNS — Note (3), Section 5, pp. 45-50)
%  ========================================================================
function mc = simulateSVBasketFactors(t, data, calib, config, volScale, skewBump)
    if nargin < 6 || isempty(skewBump), skewBump = 0.0; end
    d_t       = data.dates(t);
    fix_dates = data.fixing_dates;
    fix_idx   = data.fixing_idx;
    N_fix     = data.N_fix;
    n1        = data.n1;
    spec      = calib.spec;
    q1        = spec.scaleVol(calib.params1, volScale);
    q2        = spec.scaleVol(calib.params2, volScale);
    if abs(skewBump) > 1e-12
        q1 = spec.bumpSkew(q1, skewBump);
        q2 = spec.bumpSkew(q2, skewBump);
    end
    rho_leg   = calib.rho;
    M         = config.mcPaths;

    theta   = max(days(fix_dates - d_t) / 365.0, 0.0);
    isFixed = (fix_dates <= d_t);
    S_known = 0.0;
    for k = 1:N_fix
        if isFixed(k)
            idx_k = fix_idx(k);
            if k <= n1, S_known = S_known + data.F1_settle(idx_k) / N_fix;
            else,       S_known = S_known + data.F2_settle(idx_k) / N_fix; end
        end
    end

    active = find(~isFixed & (theta > 0));
    nAct   = numel(active);
    if nAct == 0
        mc = struct('B1', zeros(1, M), 'B2', zeros(1, M), 'S_known', S_known, 'hasCV', false);
        return;
    end

    act1   = active(active <= n1);
    w1_rem = numel(act1) / N_fix;
    w2_rem = (nAct - numel(act1)) / N_fix;
    k_cv   = -1;
    if ~isempty(act1)
        k_cv = act1(ceil(numel(act1) / 2));
    end

    rng(config.mcSeed, 'twister');
    M_half = floor(M / 2);
    M_eff  = 2 * M_half;
    sr_leg = sqrt(max(1.0 - rho_leg^2, 0.0));

    X1 = zeros(1, M_eff); X2 = zeros(1, M_eff);
    B1 = zeros(1, M_eff); B2 = zeros(1, M_eff);
    V1 = repmat(q1(1), 1, M_eff);
    V2 = repmat(q2(1), 1, M_eff);
    E_cv = []; tau_cv = 0.0;

    prev_t = 0.0;
    isHB   = strcmpi(spec.name, 'heston') || strcmpi(spec.name, 'bates');
    hasJmp = strcmpi(spec.name, 'bates');

    for j = 1:nAct
        k      = active(j);
        curr_t = theta(k);
        span   = curr_t - prev_t;
        nSub   = max(ceil(span * 52.0), 1); % Weekly sub-stepping before Feb, exact daily inside Feb
        dt     = span / nSub;
        sqdt   = sqrt(dt);

        if isHB
            kap1 = q1(2); eta1 = q1(3); th1 = q1(4); rho1 = q1(5); sr1 = sqrt(max(1.0 - rho1^2, 0.0));
            kap2 = q2(2); eta2 = q2(3); th2 = q2(4); rho2 = q2(5); sr2 = sqrt(max(1.0 - rho2^2, 0.0));
            if hasJmp
                lam1 = q1(6); mu1 = q1(7); del1 = q1(8); kj1 = exp(mu1 + 0.5*del1^2) - 1.0;
                lam2 = q2(6); mu2 = q2(7); del2 = q2(8); kj2 = exp(mu2 + 0.5*del2^2) - 1.0;
            end

            for s = 1:nSub
                z1h = randn(1, M_half); z2h = randn(1, M_half); zvh = randn(1, M_half);
                w1  = [z1h, -z1h];
                w2  = rho_leg * w1 + sr_leg * [z2h, -z2h];
                zv  = [zvh, -zvh];
                zv1 = rho1 * w1 + sr1 * zv;
                zv2 = rho2 * w2 + sr2 * zv;

                V1p = max(V1, 0.0); sqV1 = sqrt(V1p);
                V2p = max(V2, 0.0); sqV2 = sqrt(V2p);

                dX1 = -0.5 * V1p * dt + sqV1 * sqdt .* w1;
                dX2 = -0.5 * V2p * dt + sqV2 * sqdt .* w2;

                if hasJmp && (lam1 > 0 || lam2 > 0)
                    Zj  = randn(1, M_eff);
                    Nj1 = samplePoissonVec(lam1 * dt, M_eff);
                    dX1 = dX1 - lam1 * kj1 * dt + Nj1 * mu1 + sqrt(Nj1) * del1 .* Zj;
                    if abs(lam2 - lam1) < 1e-3
                        Nj2 = Nj1;
                    else
                        Nj2 = samplePoissonVec(lam2 * dt, M_eff);
                    end
                    dX2 = dX2 - lam2 * kj2 * dt + Nj2 * mu2 + sqrt(Nj2) * del2 .* Zj;
                end

                if k <= n1, X1 = X1 + dX1; end
                X2 = X2 + dX2;
                V1 = V1 + kap1 * (eta1 - V1p) * dt + th1 * sqV1 * sqdt .* zv1;
                V2 = V2 + kap2 * (eta2 - V2p) * dt + th2 * sqV2 * sqdt .* zv2;
            end
        else
            % BNS Non-Gaussian OU: exact integrated variance IV = (V - V_new + dZ) / lam
            lam1 = q1(2); a1 = q1(3); b1 = q1(4); rho1 = q1(5); mu1 = -a1 * rho1 / (b1 - rho1);
            lam2 = q2(2); a2 = q2(3); b2 = q2(4); rho2 = q2(5); mu2 = -a2 * rho2 / (b2 - rho2);
            e1 = exp(-lam1 * dt); eh1 = exp(-0.5 * lam1 * dt);
            e2 = exp(-lam2 * dt); eh2 = exp(-0.5 * lam2 * dt);

            for s = 1:nSub
                z1h = randn(1, M_half); z2h = randn(1, M_half);
                w1  = [z1h, -z1h];
                w2  = rho_leg * w1 + sr_leg * [z2h, -z2h];

                Nj1 = samplePoissonVec(a1 * dt, M_eff);
                dZ1 = sampleExpSumVec(Nj1, b1);
                Nj2 = samplePoissonVec(a2 * dt, M_eff);
                dZ2 = sampleExpSumVec(Nj2, b2);

                V1n = e1 * V1 + eh1 * dZ1; IV1 = max((V1 - V1n + dZ1) / lam1, 0.0);
                V2n = e2 * V2 + eh2 * dZ2; IV2 = max((V2 - V2n + dZ2) / lam2, 0.0);

                if k <= n1
                    X1 = X1 + mu1 * dt - 0.5 * IV1 + sqrt(IV1) .* w1 + rho1 * dZ1;
                end
                X2 = X2 + mu2 * dt - 0.5 * IV2 + sqrt(IV2) .* w2 + rho2 * dZ2;
                V1 = V1n; V2 = V2n;
            end
        end

        if k <= n1
            E1  = exp(X1);
            E1n = E1 / mean(E1);
            B1  = B1 + E1n / N_fix;
            if k == k_cv
                E_cv   = E1n;
                tau_cv = curr_t;
            end
        else
            E2 = exp(X2); B2 = B2 + (E2 / mean(E2)) / N_fix;
        end
        prev_t = curr_t;
    end

    mc = struct('B1', B1, 'B2', B2, 'S_known', S_known, ...
                'hasCV', ~isempty(E_cv), 'E_cv', E_cv, 'tau_cv', tau_cv, ...
                'w1_rem', w1_rem, 'w2_rem', w2_rem);
end

% Toolbox-free vectorized exact Poisson(lam) sampler (fast 1st-step Knuth)
function N = samplePoissonVec(lam, M)
    if lam <= 0
        N = zeros(1, M);
        return;
    end
    L   = exp(-lam);
    p   = rand(1, M);
    act = (p > L);
    N   = double(act);
    while any(act)
        idx    = find(act);
        p(idx) = p(idx) .* rand(1, numel(idx));
        still  = p(idx) > L;
        idxS   = idx(still);
        N(idxS) = N(idxS) + 1;
        act(idx(~still)) = false;
    end
end

% Toolbox-free exact sum of N_i i.i.d. Exp(b) random variables (fast N>=1 path)
function S = sampleExpSumVec(N, b)
    S    = zeros(size(N));
    invB = 1.0 / b;
    idx1 = find(N >= 1);
    if isempty(idx1), return; end
    S(idx1) = -log(max(rand(1, numel(idx1)), 1e-15)) * invB;
    maxN = max(N(idx1));
    for r = 2:maxN
        idxR = idx1(N(idx1) >= r);
        S(idxR) = S(idxR) - log(max(rand(1, numel(idxR)), 1e-15)) * invB;
    end
end

%% ========================================================================
%  BASELINE ENGINE 1: Closed-Form Moment Matching (Note Hedging, Section 6)
%  ========================================================================
function [price, details] = priceMomentMatching( ...
    K, isCall, f1_val, f2_val, sig1, sig2, rho, theta, n1, n2, w1, w2, M1, df_T)

    norm_cdf = @(x) 0.5 * erfc(-x / sqrt(2));

    theta1 = theta(1:n1);
    theta2 = theta(n1+1:n1+n2);

    k_vec    = (1:n1)';
    var_A1   = (f1_val^2 / n1^2) * sum((2 * (n1 - k_vec) + 1) .* (exp(sig1^2 * theta1) - 1.0));
    j_vec    = (1:n2)';
    var_A2   = (f2_val^2 / n2^2) * sum((2 * (n2 - j_vec) + 1) .* (exp(sig2^2 * theta2) - 1.0));
    cov_A1A2 = (f1_val * f2_val / n1) * sum(exp(rho * sig1 * sig2 * theta1) - 1.0);

    var_A = max(w1^2 * var_A1 + w2^2 * var_A2 + 2 * w1 * w2 * cov_A1A2, 0.0);

    if var_A <= 1e-14 || K <= 0
        if isCall, price = df_T * max(M1 - K, 0.0);
        else,      price = df_T * max(K - M1, 0.0); end
        details = struct('var_A1', var_A1, 'var_A2', var_A2, ...
            'cov_A1A2', cov_A1A2, 'var_A', var_A, 'v', 0.0, 'se', 0.0);
        return;
    end

    v2 = log(1.0 + var_A / (M1^2));
    v  = sqrt(v2);
    d1 = (log(M1 / K) + 0.5 * v2) / v;
    d2 = d1 - v;

    if isCall
        price = df_T * (M1 * norm_cdf(d1) - K * norm_cdf(d2));
    else
        price = df_T * (K * norm_cdf(-d2) - M1 * norm_cdf(-d1));
    end

    details = struct('var_A1', var_A1, 'var_A2', var_A2, ...
        'cov_A1A2', cov_A1A2, 'var_A', var_A, 'v', v, 'd1', d1, 'd2', d2, 'se', 0.0);
end

%% ========================================================================
%  BASELINE ENGINE 2: Bivariate Lognormal MC + Kemna-Vorst Control Variate
%  ========================================================================
function [price, details] = priceBivariateMC( ...
    K, isCall, f1_val, f2_val, sig1, sig2, rho, theta, isFixed, fix_idx, n1, n2, N_fix, df_T, data, config)

    norm_cdf = @(x) 0.5 * erfc(-x / sqrt(2));
    remMask  = ~isFixed & (theta > 0);
    if ~any(remMask) || K <= 0
        f_realized = zeros(N_fix, 1);
        for k = 1:N_fix
            idx_k = fix_idx(k);
            f_realized(k) = (k <= n1) * data.F1_settle(idx_k) + (k > n1) * data.F2_settle(idx_k);
        end
        A_val = mean(f_realized);
        if isCall, price = df_T * max(A_val - K, 0.0);
        else,      price = df_T * max(K - A_val, 0.0); end
        details = struct('se', 0.0, 'price_raw', price);
        return;
    end

    leg_id  = [ones(n1, 1); 2 * ones(n2, 1)];
    sig_vec = (leg_id == 1) * sig1 + (leg_id == 2) * sig2;
    mu      = -0.5 * (sig_vec.^2) .* theta;

    Sigma = zeros(N_fix, N_fix);
    for a = 1:N_fix
        for b = 1:N_fix
            if a <= n1 && b <= n1
                Sigma(a, b) = sig1^2 * min(theta(a), theta(b));
            elseif a > n1 && b > n1
                Sigma(a, b) = sig2^2 * min(theta(a), theta(b));
            elseif a <= n1 && b > n1
                Sigma(a, b) = rho * sig1 * sig2 * theta(a);
            else
                Sigma(a, b) = rho * sig1 * sig2 * theta(b);
            end
        end
    end

    actIdx  = find(remMask);
    nAct    = numel(actIdx);
    L       = chol(Sigma(actIdx, actIdx) + 1e-12 * eye(nAct), 'lower');
    rng(config.mcSeed, 'twister');
    M       = config.mcPaths;
    Z       = randn(nAct, M);
    eta_act = mu(actIdx) + L * Z;

    f_paths = zeros(N_fix, M);
    for k = 1:N_fix
        if ~remMask(k)
            idx_k = fix_idx(k);
            if k <= n1, f_paths(k, :) = data.F1_settle(idx_k);
            else,       f_paths(k, :) = data.F2_settle(idx_k); end
        end
    end
    f0_act = (leg_id(actIdx) == 1) * f1_val + (leg_id(actIdx) == 2) * f2_val;
    f_paths(actIdx, :) = f0_act .* exp(eta_act);

    A_arith = mean(f_paths, 1);
    if isCall, pay_arith = max(A_arith - K, 0.0);
    else,      pay_arith = max(K - A_arith, 0.0); end
    price_raw = df_T * mean(pay_arith);
    se_raw    = df_T * std(pay_arith) / sqrt(M);

    if config.useControlVariate && all(remMask)
        G_geo = exp(mean(log(f_paths), 1));
        if isCall, pay_geo = max(G_geo - K, 0.0);
        else,      pay_geo = max(K - G_geo, 0.0); end

        a_vec = ones(N_fix, 1) / N_fix;
        w1 = n1 / N_fix; w2 = n2 / N_fix;
        m_G   = w1 * log(f1_val) + w2 * log(f2_val) + a_vec' * mu;
        v_G   = sqrt(a_vec' * Sigma * a_vec);
        F_geo = exp(m_G + 0.5 * v_G^2);
        d1_g  = (log(F_geo / K) + 0.5 * v_G^2) / v_G;
        d2_g  = d1_g - v_G;

        if isCall, exact_geo = df_T * (F_geo * norm_cdf(d1_g) - K * norm_cdf(d2_g));
        else,      exact_geo = df_T * (K * norm_cdf(-d2_g) - F_geo * norm_cdf(-d1_g)); end

        diff_cv = pay_arith - pay_geo;
        price   = df_T * mean(diff_cv) + exact_geo;
        se      = df_T * std(diff_cv) / sqrt(M);
    else
        price = price_raw;
        se    = se_raw;
    end

    details = struct('se', se, 'price_raw', price_raw, 'se_raw', se_raw);
end
