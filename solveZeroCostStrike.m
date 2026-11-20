function [K_put, C_star, P_star, callDetails, putDetails] = solveZeroCostStrike(t, K_call, data, calib, config)
%SOLVEZEROCOSTSTRIKE Solve for K_put such that Put(K_put) = Call(K_call).
%
%   [K_put, C_star, P_star, callDetails, putDetails] = solveZeroCostStrike(t, K_call, data, calib, config)
%   implements Section 7 of Note Hedging.pdf:
%     1. Prices the Call leg at K_call once: C_star = C(t; T, K_call).
%     2. Defines h(K) = P(t; T, K) - C_star, re-mapping K1(K), K2(K) and
%        re-reading sigma_1, sigma_2 off the smiles at each trial K.
%     3. Solves h(K_put) = 0 on [1e-4, F_avg(t)] using fzero.

    f1_t  = data.F1_settle(t);
    f2_t  = data.F2_settle(t);
    F_avg = data.F_avg(t);

    % 1. Price the fixed-strike Call leg C* = C(t; T, K_call)
    [C_star, callDetails] = priceBasket(t, K_call, true, f1_t, f2_t, data, calib, config);

    % 2. Root-finding objective h(K) = P(t; T, K) - C*
    objFn = @(K) priceBasket(t, K, false, f1_t, f2_t, data, calib, config) - C_star;

    % Bracket [K_low, K_high]: at K -> 0, P -> 0 < C*; increase K_high if needed
    K_low  = 1.0;
    K_high = F_avg;
    while objFn(K_high) < 0 && K_high < 5 * F_avg
        K_high = 1.25 * K_high;
    end

    opts  = optimset('TolX', 1e-8, 'Display', 'off');
    K_put = fzero(objFn, [K_low, K_high], opts);

    % 3. Evaluate final Put price and details at K_put
    [P_star, putDetails] = priceBasket(t, K_put, false, f1_t, f2_t, data, calib, config);
end

