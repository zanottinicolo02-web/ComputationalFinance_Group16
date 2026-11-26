function [K_put, C_star, P_star, callDetails, putDetails, calib] = solveZeroCostStrike(t, K_call, data, calib, config)
%SOLVEZEROCOSTSTRIKE Solve for K_put such that Put(K_put) = Call(K_call).
%
%   Works identically for Black-76 (Moment Matching / MC) and all CF models
%   in cfModels.m (Merton, Kou, VG, NIG, CGMY, ATS). For CF models, the
%   simulated basket factors (B1, B2, S_known) are cached on the first call
%   and reused across all fzero iterations for fast, noise-free root-finding.

    f1_t  = data.F1_settle(t);
    f2_t  = data.F2_settle(t);
    F_avg = data.F_avg(t);

    % Use high-precision inception path count (e.g. 1,000,000) if configured
    cfgSolve = config;
    if isfield(config, 'mcPathsInception') && ~isempty(config.mcPathsInception)
        cfgSolve.mcPaths = config.mcPathsInception;
    end

    % 1. Price the fixed-strike Call leg C* = C(t; T, K_call)
    [C_star, callDetails] = priceBasket(t, K_call, true, f1_t, f2_t, data, calib, cfgSolve);

    % Cache simulated basket factors for CF models so fzero is instant & smooth
    if ~strcmpi(calib.modelName, 'black76') && isfield(callDetails, 'mcCache')
        calib.mcCache = callDetails.mcCache;
    end

    % 2. Root-finding objective h(K) = P(t; T, K) - C*
    objFn = @(K) priceBasket(t, K, false, f1_t, f2_t, data, calib, cfgSolve) - C_star;

    K_low  = 1.0;
    K_high = F_avg;
    while objFn(K_high) < 0 && K_high < 5 * F_avg
        K_high = 1.25 * K_high;
    end

    opts  = optimset('TolX', 1e-7, 'Display', 'off');
    K_put = fzero(objFn, [K_low, K_high], opts);

    % 3. Evaluate final Put price and details at K_put
    [P_star, putDetails] = priceBasket(t, K_put, false, f1_t, f2_t, data, calib, cfgSolve);

    % Clear inception 1M-path cache if daily backtest uses a different path count
    if cfgSolve.mcPaths ~= config.mcPaths
        calib.mcCache = [];
    end
end
