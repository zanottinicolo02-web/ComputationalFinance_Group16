function hedge = computeHedge(t, data, calib, K_put, config)
%COMPUTEHEDGE Compute structure value, sensitivities (Greeks), and hedge ratios.
%
%   hedge = computeHedge(t, data, calib, K_put, config) evaluates the
%   market maker's short-collar position:
%       V_struct(t) = Put(t; T, K_put) - Call(t; T, K_call)
%   and computes the target hedge holdings according to config.hedgeMethod:
%     - 'unhedged'      : No hedge (benchmark)
%     - 'delta_futures' : Delta-neutral hedge using Leg 1 and Leg 2 futures
%     - 'delta_vega'    : Delta-Vega neutral hedge using ATM options + futures
%     - 'custom'        : [Extension Hook] Custom risk-factor hedging
%                         (e.g. ATS state factors, Heston vol-of-vol, carry)

    K_call = data.K_call;
    f1_t   = data.F1_settle(t);
    f2_t   = data.F2_settle(t);

    % Helper for market maker's structure value at arbitrary (f1, f2, calib_in)
    structValFn = @(f1_v, f2_v, cal) ...
        priceBasket(t, K_put,  false, f1_v, f2_v, data, cal, config) - ...
        priceBasket(t, K_call, true,  f1_v, f2_v, data, cal, config);

    % 1. Base Prices at Current Market State (f1_t, f2_t)
    [P_val, putDet]  = priceBasket(t, K_put,  false, f1_t, f2_t, data, calib, config);
    [C_val, callDet] = priceBasket(t, K_call, true,  f1_t, f2_t, data, calib, config);
    V_struct = P_val - C_val;

    % 2. Compute Futures Deltas (Central Finite Differences)
    eps1 = config.bumpRelF * f1_t;
    eps2 = config.bumpRelF * f2_t;

    V_up1 = structValFn(f1_t + eps1, f2_t, calib);
    V_dn1 = structValFn(f1_t - eps1, f2_t, calib);
    delta1 = (V_up1 - V_dn1) / (2.0 * eps1);
    gamma1 = (V_up1 - 2.0 * V_struct + V_dn1) / (eps1^2);

    V_up2 = structValFn(f1_t, f2_t + eps2, calib);
    V_dn2 = structValFn(f1_t, f2_t - eps2, calib);
    delta2 = (V_up2 - V_dn2) / (2.0 * eps2);
    gamma2 = (V_up2 - 2.0 * V_struct + V_dn2) / (eps2^2);

    % 3. Compute Structure Vegas (if requested or if using delta_vega hedge)
    vega1 = 0.0;
    vega2 = 0.0;
    if config.computeVega || strcmpi(config.hedgeMethod, 'delta_vega')
        dSig = config.bumpAbsVol;

        calib_v1_up = calib; calib_v1_up.sigma1_fn = @(K) calib.sigma1_fn(K) + dSig;
        calib_v1_dn = calib; calib_v1_dn.sigma1_fn = @(K) max(calib.sigma1_fn(K) - dSig, 1e-4);
        vega1 = (structValFn(f1_t, f2_t, calib_v1_up) - structValFn(f1_t, f2_t, calib_v1_dn)) / (2.0 * dSig);

        calib_v2_up = calib; calib_v2_up.sigma2_fn = @(K) calib.sigma2_fn(K) + dSig;
        calib_v2_dn = calib; calib_v2_dn.sigma2_fn = @(K) max(calib.sigma2_fn(K) - dSig, 1e-4);
        vega2 = (structValFn(f1_t, f2_t, calib_v2_up) - structValFn(f1_t, f2_t, calib_v2_dn)) / (2.0 * dSig);
    end

    % 4. Dispatch Hedging Strategy to Determine Target Positions
    phi1 = 0.0; % Target holding in Leg 1 future (per bbl notional)
    phi2 = 0.0; % Target holding in Leg 2 future (per bbl notional)
    psi1 = 0.0; % Target holding in Leg 1 ATM option (for Vega hedging)
    psi2 = 0.0; % Target holding in Leg 2 ATM option (for Vega hedging)

    switch lower(config.hedgeMethod)
        case 'unhedged'
            phi1 = 0.0;
            phi2 = 0.0;

        case 'delta_futures'
            % Baseline Delta-neutral hedge on both underlying futures legs
            phi1 = -delta1;
            phi2 = -delta2;

        case 'delta_vega'
            % Delta-Vega hedge using closest-to-ATM listed option on each leg
            % plus the two futures to neutralize residual Delta
            [psi1, optDelta1] = computeSingleLegVegaHedge(vega1, calib.smile1, f1_t);
            [psi2, optDelta2] = computeSingleLegVegaHedge(vega2, calib.smile2, f2_t);
            phi1 = -(delta1 + psi1 * optDelta1);
            phi2 = -(delta2 + psi2 * optDelta2);

        case 'custom'
            % [EXTENSION HOOK] Plug in custom multi-factor hedging (e.g.,
            % ATS slope/curvature factors, Heston vega/vanna/volga, or
            % spread-penalized L1 regularization on turnover).
            error('computeHedge:NotImplemented', ...
                'Custom hedging hook: plug your custom hedge optimizer here.');

        otherwise
            error('computeHedge:UnknownMethod', ...
                'Unknown hedgeMethod "%s".', config.hedgeMethod);
    end

    hedge = struct( ...
        'V_struct', V_struct, ...
        'P_val',    P_val, ...
        'C_val',    C_val, ...
        'delta1',   delta1, ...
        'delta2',   delta2, ...
        'gamma1',   gamma1, ...
        'gamma2',   gamma2, ...
        'vega1',    vega1, ...
        'vega2',    vega2, ...
        'phi1',     phi1, ...
        'phi2',     phi2, ...
        'psi1',     psi1, ...
        'psi2',     psi2, ...
        'putDet',   putDet, ...
        'callDet',  callDet);
end

%% Helper: Select Closest-to-ATM Listed Option for Vega Neutralization
function [psi, optDelta] = computeSingleLegVegaHedge(structVega, smile, F_t)
    if isempty(smile) || isempty(smile.strikes) || smile.tau <= 0.02 || abs(structVega) < 1e-8
        psi = 0.0;
        optDelta = 0.0;
        return;
    end
    [~, idxATM] = min(abs(smile.strikes - F_t));
    K_opt   = smile.strikes(idxATM);
    sig_opt = smile.iv(idxATM);
    vega_opt = smile.vega(idxATM);
    isCall  = smile.isCall(idxATM);
    tau_opt = smile.tau;

    if vega_opt < 1e-4
        psi = 0.0;
        optDelta = 0.0;
        return;
    end

    psi = -structVega / vega_opt;

    norm_cdf = @(x) 0.5 * erfc(-x / sqrt(2));
    d1 = (log(F_t / K_opt) + 0.5 * sig_opt^2 * tau_opt) / (sig_opt * sqrt(tau_opt));
    if isCall
        optDelta = norm_cdf(d1);
    else
        optDelta = norm_cdf(d1) - 1.0;
    end
end

