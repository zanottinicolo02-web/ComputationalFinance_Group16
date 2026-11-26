function hedge = computeHedge(t, data, calib, K_put, config, prevOpenOpts, cachedBase, prevPhi)
%COMPUTEHEDGE Compute structure value, model-consistent Greeks, and hedge targets.
%
%   hedge = computeHedge(t, data, calib, K_put, config, prevOpenOpts)
%   hedge = computeHedge(..., cachedBase, prevPhi) reuses base structure
%   valuation & Greeks across hedging strategies (exact, 3x faster) and
%   optionally applies a futures rebalancing deadband (config.futRebalBand).

    if nargin < 6
        prevOpenOpts = struct('strike', {}, 'isCall', {}, 'psi', {}, 'lastMid', {});
    end
    if nargin < 7
        cachedBase = [];
    end
    if nargin < 8
        prevPhi = [];
    end

    K_call = data.K_call;
    f1_t   = data.F1_settle(t);
    f2_t   = data.F2_settle(t);
    dv     = config.bumpRelVol;
    needsOpts = strcmpi(config.hedgeMethod, 'delta_vega') || ...
                strcmpi(config.hedgeMethod, 'delta_vega_collar') || ...
                strcmpi(config.hedgeMethod, 'delta_vega_collar_2leg') || ...
                strcmpi(config.hedgeMethod, 'delta_vega_bracket') || ...
                strcmpi(config.hedgeMethod, 'delta_vega_skew') || ...
                strcmpi(config.hedgeMethod, 'delta_vega_gamma');
    useMVD    = isfield(config, 'useMVDelta') && config.useMVDelta;
    chain1    = data.optChain1{t};
    canTradeOpts = ~isempty(chain1) && ~isempty(chain1.strike) && ...
                   (chain1.tau > config.minOptTauDays / 365.0) && ...
                   (t < data.N_days - data.N_fix - 2);

    useSN = isfield(config, 'useSafetyNet') && config.useSafetyNet && ...
            ~strcmpi(calib.modelName, 'black76');

    if ~isempty(cachedBase)
        V_struct  = cachedBase.V_struct;
        P_val     = cachedBase.P_val;
        C_val     = cachedBase.C_val;
        delta1    = cachedBase.delta1;
        delta2    = cachedBase.delta2;
        gamma1    = cachedBase.gamma1;
        gamma2    = cachedBase.gamma2;
        vegaScale = cachedBase.vegaScale;
        vegaPut   = cachedBase.vegaPut;
        vegaCall  = cachedBase.vegaCall;
        snActive  = cachedBase.snActive;
        putDet    = struct();
        callDet   = struct();
        if snActive
            calibG = buildBlack76FallbackCalib(calib);
            cfgG   = config;
            cfgG.modelName    = 'black76';
            cfgG.pricerMethod = 'moment_matching';
        else
            calibG = calib;
            cfgG   = config;
        end
    else
        % 1. Base Prices at Current Market State (populates mcCache for CF/ATS)
        [P_val, putDet] = priceBasket(t, K_put, false, f1_t, f2_t, data, calib, config, 1.0);
        if ~strcmpi(calib.modelName, 'black76') && isfield(putDet, 'mcCache')
            calib.mcCache = putDet.mcCache;
        end
        [C_val, callDet] = priceBasket(t, K_call, true, f1_t, f2_t, data, calib, config, 1.0);
        V_struct = P_val - C_val;

        structValFn = @(f1_v, f2_v, sv) ...
            priceBasket(t, K_put,  false, f1_v, f2_v, data, calib, config, sv) - ...
            priceBasket(t, K_call, true,  f1_v, f2_v, data, calib, config, sv);

        % 2. Futures Deltas & Gammas (Central Finite Differences, fixed params)
        snActive = useSN && isfield(calib, 'safetyTriggered') && calib.safetyTriggered;

        if snActive
            calibG = buildBlack76FallbackCalib(calib);
            cfgG   = config;
            cfgG.modelName    = 'black76';
            cfgG.pricerMethod = 'moment_matching';
            greekValFn = @(f1_v, f2_v, sv) ...
                priceBasket(t, K_put,  false, f1_v, f2_v, data, calibG, cfgG, sv) - ...
                priceBasket(t, K_call, true,  f1_v, f2_v, data, calibG, cfgG, sv);
            V_g0    = greekValFn(f1_t, f2_t, 1.0);
            relBump = config.bumpRelF;
        else
            calibG     = calib;
            cfgG       = config;
            greekValFn = structValFn;
            V_g0       = V_struct;
            if strcmpi(calib.modelName, 'black76') && strcmpi(config.pricerMethod, 'moment_matching')
                relBump = config.bumpRelF;
            else
                relBump = max(config.bumpRelF, 2e-3);
            end
        end
        eps1 = relBump * f1_t;
        eps2 = relBump * f2_t;

        V_up1  = greekValFn(f1_t + eps1, f2_t, 1.0);
        V_dn1  = greekValFn(f1_t - eps1, f2_t, 1.0);
        delta1 = (V_up1 - V_dn1) / (2.0 * eps1);
        gamma1 = (V_up1 - 2.0 * V_g0 + V_dn1) / (eps1^2);

        V_up2  = greekValFn(f1_t, f2_t + eps2, 1.0);
        V_dn2  = greekValFn(f1_t, f2_t - eps2, 1.0);
        delta2 = (V_up2 - V_dn2) / (2.0 * eps2);
        gamma2 = (V_up2 - 2.0 * V_g0 + V_dn2) / (eps2^2);

        % 3. Model-Consistent Leg-by-Leg & Net Scale Vega dV/d(volScale)
        vegaScale = 0.0; vegaPut = 0.0; vegaCall = 0.0;
        if (config.computeVega || needsOpts || useMVD) && (canTradeOpts || useMVD)
            [P_vu, det_vu] = priceBasket(t, K_put, false, f1_t, f2_t, data, calibG, cfgG, 1.0 + dv);
            calibTmp = calibG;
            if ~strcmpi(calibG.modelName, 'black76') && isfield(det_vu, 'mcCache')
                calibTmp.mcCache = det_vu.mcCache;
            end
            C_vu = priceBasket(t, K_call, true, f1_t, f2_t, data, calibTmp, cfgG, 1.0 + dv);

            [P_vd, det_vd] = priceBasket(t, K_put, false, f1_t, f2_t, data, calibG, cfgG, 1.0 - dv);
            if ~strcmpi(calibG.modelName, 'black76') && isfield(det_vd, 'mcCache')
                calibTmp.mcCache = det_vd.mcCache;
            end
            C_vd = priceBasket(t, K_call, true, f1_t, f2_t, data, calibTmp, cfgG, 1.0 - dv);
            vegaPut   = (P_vu - P_vd) / (2.0 * dv);
            vegaCall  = (C_vu - C_vd) / (2.0 * dv);
            vegaScale = vegaPut - vegaCall;
        end

        % Safety Net Layer 2B: Greek Sanity Check -> Revert to Black-76 if Greeks explode
        if useSN && ~snActive
            badGreeks = ~isfinite(delta1) || ~isfinite(delta2) || abs(delta1) > 1.25 || ...
                        (needsOpts && canTradeOpts && ~isfinite(vegaScale));
            if badGreeks
                snActive = true;
                calibG = buildBlack76FallbackCalib(calib);
                cfgG   = config;
                cfgG.modelName    = 'black76';
                cfgG.pricerMethod = 'moment_matching';
                greekValFn = @(f1_v, f2_v, sv) ...
                    priceBasket(t, K_put,  false, f1_v, f2_v, data, calibG, cfgG, sv) - ...
                    priceBasket(t, K_call, true,  f1_v, f2_v, data, calibG, cfgG, sv);
                V_g0   = greekValFn(f1_t, f2_t, 1.0);
                e1b    = config.bumpRelF * f1_t;
                e2b    = config.bumpRelF * f2_t;
                delta1 = (greekValFn(f1_t + e1b, f2_t, 1.0) - greekValFn(f1_t - e1b, f2_t, 1.0)) / (2.0 * e1b);
                gamma1 = (greekValFn(f1_t + e1b, f2_t, 1.0) - 2.0 * V_g0 + greekValFn(f1_t - e1b, f2_t, 1.0)) / (e1b^2);
                delta2 = (greekValFn(f1_t, f2_t + e2b, 1.0) - greekValFn(f1_t, f2_t - e2b, 1.0)) / (2.0 * e2b);
                gamma2 = (greekValFn(f1_t, f2_t + e2b, 1.0) - 2.0 * V_g0 + greekValFn(f1_t, f2_t - e2b, 1.0)) / (e2b^2);
                if (config.computeVega || needsOpts || useMVD) && (canTradeOpts || useMVD)
                    P_vu = priceBasket(t, K_put,  false, f1_t, f2_t, data, calibG, cfgG, 1.0 + dv);
                    P_vd = priceBasket(t, K_put,  false, f1_t, f2_t, data, calibG, cfgG, 1.0 - dv);
                    C_vu = priceBasket(t, K_call, true,  f1_t, f2_t, data, calibG, cfgG, 1.0 + dv);
                    C_vd = priceBasket(t, K_call, true,  f1_t, f2_t, data, calibG, cfgG, 1.0 - dv);
                    vegaPut   = (P_vu - P_vd) / (2.0 * dv);
                    vegaCall  = (C_vu - C_vd) / (2.0 * dv);
                    vegaScale = vegaPut - vegaCall;
                end
            end
        end
    end

    % Snapshot strategy-independent structure Greeks for exact reuse across strategies
    baseStruct = struct('V_struct', V_struct, 'P_val', P_val, 'C_val', C_val, ...
                        'delta1', delta1, 'delta2', delta2, 'gamma1', gamma1, ...
                        'gamma2', gamma2, 'vegaScale', vegaScale, ...
                        'vegaPut', vegaPut, 'vegaCall', vegaCall, 'snActive', snActive);

    % 4. Dispatch Hedging Strategy
    optTargets  = struct('strike', {}, 'isCall', {}, 'psi', {});
    optDeltaSum = 0.0;
    optVegaSum  = 0.0;

    switch lower(config.hedgeMethod)
        case 'unhedged'
            phi1 = 0.0;
            phi2 = 0.0;

        case 'delta_futures'
            mvAdj = 0.0;
            if useMVD && isfield(calib, 'betaVolF')
                mvAdj = vegaScale * calib.betaVolF;
            end
            phi1 = -(delta1 + mvAdj);
            phi2 = -delta2;

        case {'delta_vega', 'delta_vega_collar'}
            % Step A: Neutralize parallel Vega with 1 sticky COJ21 option
            %   - 'delta_vega'        : Sticky near-ATM option (rolls if |K - F1| > optRollBand)
            %   - 'delta_vega_collar' : Collar-anchored option fixed on the liquid strike
            %                           closest to K_put (K=45, zero strike rolls!)
            if canTradeOpts && abs(vegaScale) > 1e-6
                if strcmpi(config.hedgeMethod, 'delta_vega_collar')
                    [selK, selCall, ok] = selectCollarAnchoredOption(chain1, f1_t, K_put, K_call, prevOpenOpts, config);
                else
                    [selK, selCall, ok] = selectStickyATMOption(chain1, calib.smile1, f1_t, prevOpenOpts, config);
                end
                if ok
                    [d_o, ~, v_o] = computeListedOptionGreeks(f1_t, selK, selCall, chain1.tau, calibG, cfgG, dv, 0.0);
                    if abs(v_o) > 1e-4
                        psi_raw = -vegaScale / v_o;

                        % Safety Net Layer 2C: If same-strike psi jumps anomalously, revert to Black-76 Greeks
                        if useSN && ~snActive && isscalar(prevOpenOpts) && ...
                           prevOpenOpts(1).strike == selK && prevOpenOpts(1).isCall == selCall && ...
                           isfield(config, 'snMaxPsiJump') && abs(psi_raw - prevOpenOpts(1).psi) > config.snMaxPsiJump
                            snActive = true;
                            calibG = buildBlack76FallbackCalib(calib);
                            cfgG   = config;
                            cfgG.modelName    = 'black76';
                            cfgG.pricerMethod = 'moment_matching';
                            greekValFn = @(f1_v, f2_v, sv) ...
                                priceBasket(t, K_put,  false, f1_v, f2_v, data, calibG, cfgG, sv) - ...
                                priceBasket(t, K_call, true,  f1_v, f2_v, data, calibG, cfgG, sv);
                            e1b    = config.bumpRelF * f1_t;
                            e2b    = config.bumpRelF * f2_t;
                            delta1 = (greekValFn(f1_t + e1b, f2_t, 1.0) - greekValFn(f1_t - e1b, f2_t, 1.0)) / (2.0 * e1b);
                            delta2 = (greekValFn(f1_t, f2_t + e2b, 1.0) - greekValFn(f1_t, f2_t - e2b, 1.0)) / (2.0 * e2b);
                            vegaScale = (greekValFn(f1_t, f2_t, 1.0 + dv) - greekValFn(f1_t, f2_t, 1.0 - dv)) / (2.0 * dv);
                            [d_o, ~, v_o] = computeListedOptionGreeks(f1_t, selK, selCall, chain1.tau, calibG, cfgG, dv, 0.0);
                            if abs(v_o) > 1e-4
                                psi_raw = -vegaScale / v_o;
                            end
                        end

                        psi_tgt = min(max(psi_raw, -config.maxOptPos), config.maxOptPos);

                        % Optional turnover deadband to limit bid-ask churn
                        if isscalar(prevOpenOpts) && ...
                           prevOpenOpts(1).strike == selK && prevOpenOpts(1).isCall == selCall && ...
                           abs(psi_tgt - prevOpenOpts(1).psi) < config.vegaRebalBand
                            psi_tgt = prevOpenOpts(1).psi;
                        end

                        optTargets(1) = struct('strike', selK, 'isCall', selCall, 'psi', psi_tgt);
                        optDeltaSum   = psi_tgt * d_o;
                        optVegaSum    = psi_tgt * v_o;
                    end
                end
            end
            mvAdj = 0.0;
            if useMVD && isfield(calib, 'betaVolF')
                mvAdj = (vegaScale + optVegaSum) * calib.betaVolF;
            end
            phi1 = -(delta1 + optDeltaSum + mvAdj);
            phi2 = -delta2;

        case 'delta_vega_collar_2leg'
            % Idea #1: Leg-by-Leg Semi-Static Collar Replication (100% Symmetric!)
            % Hedges the Put leg (near K_put) and Call leg (near K_call) INDEPENDENTLY
            % with no 2x2 matrix inversion and a wide deadband so each wing is entered
            % once and cleanly unwound if it becomes illiquid/deep-OTM.
            if canTradeOpts
                wingBand = max(config.vegaRebalBand, 0.08);
                capCol   = min(config.maxOptPos, 1.15);

                % --- Leg 1: Sticky Listed Put near K_put ---
                [kP, okP] = selectStickyWingOption(chain1, K_put, false, prevOpenOpts);
                if okP && vegaPut > 0.05
                    [dP, ~, vP] = computeListedOptionGreeks(f1_t, kP, false, chain1.tau, calibG, cfgG, dv, 0.0);
                    if vP > 0.10
                        psiP  = min(max(-vegaPut / vP, -capCol), 0.0);
                        prevP = findPrevOptPsi(prevOpenOpts, kP, false);
                        if ~isnan(prevP) && abs(psiP - prevP) < wingBand
                            psiP = prevP;
                        end
                        if abs(psiP) > 0.05
                            optTargets(end+1) = struct('strike', kP, 'isCall', false, 'psi', psiP);
                            optDeltaSum = optDeltaSum + psiP * dP;
                            optVegaSum  = optVegaSum  + psiP * vP;
                        end
                    end
                end

                % --- Leg 2: Sticky Listed Call near K_call (only while liquid & active) ---
                [kC, okC] = selectStickyWingOption(chain1, K_call, true, prevOpenOpts);
                if okC && vegaCall > 0.05
                    [dC, ~, vC] = computeListedOptionGreeks(f1_t, kC, true, chain1.tau, calibG, cfgG, dv, 0.0);
                    idxC = find(chain1.strike == kC & chain1.isCall, 1);
                    midC = 0.0;
                    if ~isempty(idxC), midC = chain1.mid(idxC); end
                    % Keep Call leg only while meaningfully priced & active (avoids boundary dropoff)
                    if vC > 0.15 && midC >= 0.15
                        psiC  = min(max(+vegaCall / vC, 0.0), capCol);
                        prevC = findPrevOptPsi(prevOpenOpts, kC, true);
                        if ~isnan(prevC) && abs(psiC - prevC) < wingBand
                            psiC = prevC;
                        end
                        if abs(psiC) > 0.05
                            optTargets(end+1) = struct('strike', kC, 'isCall', true, 'psi', psiC);
                            optDeltaSum = optDeltaSum + psiC * dC;
                            optVegaSum  = optVegaSum  + psiC * vC;
                        end
                    end
                end
            end
            mvAdj = 0.0;
            if useMVD && isfield(calib, 'betaVolF')
                mvAdj = (vegaScale + optVegaSum) * calib.betaVolF;
            end
            phi1 = -(delta1 + optDeltaSum + mvAdj);
            phi2 = -delta2;

        case 'delta_vega_bracket'
            % Symmetric Strike-Bracketed Collar Replication:
            % Linearly brackets BOTH K_put (e.g. Put 40 & 45) and K_call (e.g. Call 75 & 80)
            % across adjacent listed strikes -> 100% symmetric for crashes and rallies!
            if canTradeOpts
                wingBand  = max(config.vegaRebalBand, 0.08);
                capCol    = min(config.maxOptPos, 1.15);
                useParity = isfield(config, 'useOTMParity') && config.useOTMParity;

                % Put wing (short listed puts matching Asian Put Vega)
                [optTargets, optDeltaSum, optVegaSum] = addBracketedWing( ...
                    optTargets, optDeltaSum, optVegaSum, chain1, f1_t, K_put, false, ...
                    -vegaPut, capCol, wingBand, 0.10, 0.10, calibG, cfgG, dv, prevOpenOpts, useParity);

                % Call wing (long listed calls matching Asian Call Vega, active only when liquid)
                [optTargets, optDeltaSum, optVegaSum] = addBracketedWing( ...
                    optTargets, optDeltaSum, optVegaSum, chain1, f1_t, K_call, true, ...
                    +vegaCall, capCol, wingBand, 0.15, 0.15, calibG, cfgG, dv, prevOpenOpts, useParity);
            end
            mvAdj = 0.0;
            if useMVD && isfield(calib, 'betaVolF')
                mvAdj = (vegaScale + optVegaSum) * calib.betaVolF;
            end
            phi1 = -(delta1 + optDeltaSum + mvAdj);
            phi2 = -delta2;

        case 'delta_vega_skew'
            % Upgrade #1 (Note (3), pp. 38-39): Sticky 2-Option Vega + Skew Hedge
            % Keeps both strikes inside the liquid core [F1-15, F1+15] so neither
            % option ever drops off the exchange quote window, neutralizing both
            % parallel volatility shifts and smile skew rotations.
            if canTradeOpts
                [kP, kC, hasP, hasC] = selectStickyCollarPair(chain1, f1_t, K_put, K_call, prevOpenOpts);
                ds = 0.25; % Skew perturbation step
                V_su = priceBasket(t, K_put,  false, f1_t, f2_t, data, calibG, cfgG, 1.0,  ds) - ...
                       priceBasket(t, K_call, true,  f1_t, f2_t, data, calibG, cfgG, 1.0,  ds);
                V_sd = priceBasket(t, K_put,  false, f1_t, f2_t, data, calibG, cfgG, 1.0, -ds) - ...
                       priceBasket(t, K_call, true,  f1_t, f2_t, data, calibG, cfgG, 1.0, -ds);
                skewSens = (V_su - V_sd) / (2.0 * ds);

                dP = 0; vP = 0; sP = 0;
                dC = 0; vC = 0; sC = 0;
                if hasP
                    [dP, ~, vP, sP] = computeListedOptionGreeks(f1_t, kP, false, chain1.tau, calibG, cfgG, dv, ds);
                end
                if hasC
                    [dC, ~, vC, sC] = computeListedOptionGreeks(f1_t, kC, true,  chain1.tau, calibG, cfgG, dv, ds);
                end

                psiP = 0.0; psiC = 0.0;
                A_mat = [vP, vC; sP, sC];
                activeP = hasP && (abs(vP) > 0.10);
                activeC = hasC && (abs(vC) > 0.10) && (~hasP || abs(kC - kP) >= 5.0);
                capCol  = min(config.maxOptPos, 1.25);

                if activeP && activeC && isfinite(rcond(A_mat)) && rcond(A_mat) > 1e-2
                    sol  = A_mat \ [-vegaScale; -skewSens];
                    psiP = min(max(sol(1), -capCol), capCol);
                    psiC = min(max(sol(2), -capCol), capCol);
                elseif activeP
                    psiP = min(max(-vegaScale / vP, -capCol), capCol);
                elseif activeC
                    psiC = min(max(-vegaScale / vC, -capCol), capCol);
                end

                % Per-leg sticky deadband (semi-static collar: minimizes wing bid-ask churn)
                wingBand = max(config.vegaRebalBand, 0.05);
                if hasP
                    prevP = findPrevOptPsi(prevOpenOpts, kP, false);
                    if ~isnan(prevP) && abs(psiP - prevP) < wingBand
                        psiP = prevP;
                    end
                    if abs(psiP) > 1e-4
                        optTargets(end+1) = struct('strike', kP, 'isCall', false, 'psi', psiP);
                        optDeltaSum = optDeltaSum + psiP * dP;
                        optVegaSum  = optVegaSum  + psiP * vP;
                    end
                end
                if hasC
                    prevC = findPrevOptPsi(prevOpenOpts, kC, true);
                    if ~isnan(prevC) && abs(psiC - prevC) < wingBand
                        psiC = prevC;
                    end
                    if abs(psiC) > 1e-4
                        optTargets(end+1) = struct('strike', kC, 'isCall', true, 'psi', psiC);
                        optDeltaSum = optDeltaSum + psiC * dC;
                        optVegaSum  = optVegaSum  + psiC * vC;
                    end
                end
            end
            mvAdj = 0.0;
            if useMVD && isfield(calib, 'betaVolF')
                mvAdj = (vegaScale + optVegaSum) * calib.betaVolF;
            end
            phi1 = -(delta1 + optDeltaSum + mvAdj);
            phi2 = -delta2;

        case 'delta_vega_gamma'
            % 2-option (Put near K_put + Call near K_call) Vega-Gamma hedge
            if canTradeOpts
                [kP, kC, ok] = selectWingOptionPair(chain1, K_put, K_call);
                if ok
                    [dP, gP, vP] = computeListedOptionGreeks(f1_t, kP, false, chain1.tau, calibG, cfgG, dv, 0.0);
                    [dC, gC, vC] = computeListedOptionGreeks(f1_t, kC, true,  chain1.tau, calibG, cfgG, dv, 0.0);
                    A_mat = [vP, vC; gP, gC];
                    if abs(det(A_mat)) > 1e-7
                        sol = A_mat \ [-vegaScale; -gamma1];
                        sol = min(max(sol, -config.maxOptPos), config.maxOptPos);
                        optTargets(1) = struct('strike', kP, 'isCall', false, 'psi', sol(1));
                        optTargets(2) = struct('strike', kC, 'isCall', true,  'psi', sol(2));
                        optDeltaSum   = sol(1) * dP + sol(2) * dC;
                    end
                end
            end
            phi1 = -(delta1 + optDeltaSum);
            phi2 = -delta2;

        otherwise
            error('computeHedge:UnknownMethod', ...
                'Unknown hedgeMethod "%s".', config.hedgeMethod);
    end

    % Optional Idea #3: Futures rebalancing deadband (skips tiny daily MC jitter)
    if ~isempty(prevPhi) && isfield(config, 'futRebalBand') && config.futRebalBand > 0
        if abs(phi1 - prevPhi(1)) < config.futRebalBand
            phi1 = prevPhi(1);
        end
        if abs(phi2 - prevPhi(2)) < config.futRebalBand
            phi2 = prevPhi(2);
        end
    end

    % Primary option holding summary for convenient time-series plotting
    psi1 = 0.0; optStrike1 = NaN;
    if ~isempty(optTargets)
        psi1       = optTargets(1).psi;
        optStrike1 = optTargets(1).strike;
    end

    hedge = struct( ...
        'V_struct',          V_struct, ...
        'P_val',             P_val, ...
        'C_val',             C_val, ...
        'delta1',            delta1, ...
        'delta2',            delta2, ...
        'gamma1',            gamma1, ...
        'gamma2',            gamma2, ...
        'vegaScale',         vegaScale, ...
        'phi1',              phi1, ...
        'phi2',              phi2, ...
        'psi1',              psi1, ...
        'optStrike1',        optStrike1, ...
        'optTargets',        optTargets, ...
        'usedBlackFallback', snActive, ...
        'baseStruct',        baseStruct, ...
        'putDet',            putDet, ...
        'callDet',           callDet);
end

function calibB76 = buildBlack76FallbackCalib(calib)
    calibB76           = calib;
    calibB76.modelName = 'black76';
    calibB76.spec      = [];
    calibB76.mcCache   = [];
    calibB76.params1   = calib.atmVol1;
    calibB76.params2   = calib.atmVol2;
end

%% ========================================================================
%  HELPER 1: Sticky Near-ATM Option Selector (Avoids Unnecessary Strike Rolls)
%  ========================================================================
function [selK, selCall, ok] = selectStickyATMOption(chain1, smile1, F1_t, prevOpenOpts, config)
    selK = NaN; selCall = false; ok = false;

    % Check if previously held option is still quoted and within roll band
    if isscalar(prevOpenOpts)
        kOld = prevOpenOpts(1).strike;
        cOld = prevOpenOpts(1).isCall;
        idxOld = find(chain1.strike == kOld & chain1.isCall == cOld, 1);
        if ~isempty(idxOld) && ~isnan(chain1.mid(idxOld)) && ~isnan(chain1.bid(idxOld)) && ...
           abs(kOld - F1_t) <= config.optRollBand
            selK    = kOld;
            selCall = cOld;
            ok      = true;
            return;
        end
    end

    % Otherwise pick closest-to-ATM liquid OTM option from calibrated smile1
    if isempty(smile1) || isempty(smile1.strikes)
        return;
    end
    [~, iATM] = min(abs(smile1.strikes - F1_t));
    kCand = smile1.strikes(iATM);
    cCand = smile1.isCall(iATM);
    idxC  = find(chain1.strike == kCand & chain1.isCall == cCand, 1);
    if ~isempty(idxC) && ~isnan(chain1.mid(idxC))
        selK    = kCand;
        selCall = cCand;
        ok      = true;
    end
end

%% ========================================================================
%  HELPER 1B: Collar-Anchored 1-Option Selector (Sticky K ≈ K_put, Zero Rolls)
%  ========================================================================
function [selK, selCall, ok] = selectCollarAnchoredOption(chain1, F1_t, K_put, K_call, prevOpenOpts, config)
    selK = NaN; selCall = false; ok = false;
    useParity = isfield(config, 'useOTMParity') && config.useOTMParity;

    % Stay glued to the previously held anchored option while quoted & within $30 of spot
    if isscalar(prevOpenOpts)
        kOld   = prevOpenOpts(1).strike;
        cOld   = prevOpenOpts(1).isCall;
        idxOld = find(chain1.strike == kOld & chain1.isCall == cOld, 1);
        if ~isempty(idxOld) && ~isnan(chain1.mid(idxOld)) && ~isnan(chain1.bid(idxOld)) && ...
           chain1.mid(idxOld) >= 0.05 && abs(kOld - F1_t) <= 30.0
            selK    = kOld;
            selCall = cOld;
            ok      = true;
            return;
        end
    end

    % Anchor to Put closest to K_put (or Call closest to K_call if F1 rallies near K_call)
    useCall = (F1_t > K_call - 10.0);
    valid   = ~isnan(chain1.mid) & ~isnan(chain1.bid) & (chain1.mid >= 0.05) & (chain1.isCall == useCall);
    candsK  = chain1.strike(valid);
    if isempty(candsK)
        valid  = ~isnan(chain1.mid) & ~isnan(chain1.bid) & (chain1.mid >= 0.05);
        candsK = chain1.strike(valid);
        if isempty(candsK), return; end
        [~, iK] = min(abs(candsK - K_put));
        selK    = candsK(iK);
        idx     = find(chain1.strike == selK & valid, 1);
        selCall = pickParitySide(chain1, selK, chain1.isCall(idx), F1_t, prevOpenOpts, useParity);
        ok      = true;
        return;
    end
    tgtK    = K_put;
    if useCall, tgtK = K_call; end
    [~, iK] = min(abs(candsK - tgtK));
    selK    = candsK(iK);
    selCall = pickParitySide(chain1, selK, useCall, F1_t, prevOpenOpts, useParity);
    ok      = true;
end

function isCallOut = pickParitySide(chain1, K, defaultIsCall, F1_t, prevOpenOpts, useParity)
    % Put-Call Parity Execution: keep already-held side sticky (zero flip spread);
    % on new entry, pick the tighter-spread / OTM side at strike K.
    isCallOut = defaultIsCall;
    if ~useParity
        return;
    end
    for j = 1:numel(prevOpenOpts)
        if prevOpenOpts(j).strike == K
            idx = find(chain1.strike == K & chain1.isCall == prevOpenOpts(j).isCall, 1);
            if ~isempty(idx) && ~isnan(chain1.mid(idx)) && ~isnan(chain1.bid(idx))
                isCallOut = prevOpenOpts(j).isCall;
                return;
            end
        end
    end
    iP = find(chain1.strike == K & ~chain1.isCall, 1);
    iC = find(chain1.strike == K &  chain1.isCall, 1);
    okP = ~isempty(iP) && ~isnan(chain1.mid(iP)) && ~isnan(chain1.bid(iP)) && chain1.mid(iP) >= 0.05;
    okC = ~isempty(iC) && ~isnan(chain1.mid(iC)) && ~isnan(chain1.bid(iC)) && chain1.mid(iC) >= 0.05;
    if okP && okC
        spP = chain1.spread(iP); spC = chain1.spread(iC);
        if isfinite(spC) && isfinite(spP) && spC < spP - 1e-4
            isCallOut = true;
        elseif isfinite(spC) && isfinite(spP) && spP < spC - 1e-4
            isCallOut = false;
        else
            isCallOut = (K > F1_t); % OTM tie-breaker
        end
    elseif okC
        isCallOut = true;
    elseif okP
        isCallOut = false;
    end
end

function [optTargets, optDeltaSum, optVegaSum] = addBracketedWing( ...
    optTargets, optDeltaSum, optVegaSum, chain1, f1_t, targetK, isCallWing, ...
    targetVega, capCol, wingBand, minMid, minVega, calibG, cfgG, dv, prevOpenOpts, useParity)
    % Symmetrically bracket a target collar strike K across adjacent listed strikes
    % K_lo <= targetK < K_hi (or fall back to nearest liquid strike within $7.5).
    if abs(targetVega) <= 0.05
        return;
    end
    valid = ~isnan(chain1.mid) & ~isnan(chain1.bid) & (chain1.mid >= minMid) & ...
            (chain1.isCall == isCallWing) & (abs(chain1.strike - targetK) <= 7.5);
    cands = sort(chain1.strike(valid));
    if isempty(cands)
        return;
    end

    loCands = cands(cands <= targetK);
    hiCands = cands(cands >= targetK);
    if ~isempty(loCands) && ~isempty(hiCands) && loCands(end) < hiCands(1)
        kLo = loCands(end); kHi = hiCands(1);
        wHi = (targetK - kLo) / (kHi - kLo);
        wLo = 1.0 - wHi;
        cLo = pickParitySide(chain1, kLo, isCallWing, f1_t, prevOpenOpts, useParity);
        cHi = pickParitySide(chain1, kHi, isCallWing, f1_t, prevOpenOpts, useParity);
        [dLo, ~, vLo] = computeListedOptionGreeks(f1_t, kLo, cLo, chain1.tau, calibG, cfgG, dv, 0.0);
        [dHi, ~, vHi] = computeListedOptionGreeks(f1_t, kHi, cHi, chain1.tau, calibG, cfgG, dv, 0.0);
        vMix = wLo * vLo + wHi * vHi;
        if vMix > minVega
            psiTot = min(max(targetVega / vMix, -capCol), capCol);
            psiLo  = wLo * psiTot;
            psiHi  = wHi * psiTot;
            pLoOld = findPrevOptPsi(prevOpenOpts, kLo, cLo);
            pHiOld = findPrevOptPsi(prevOpenOpts, kHi, cHi);
            if ~isnan(pLoOld) && abs(psiLo - pLoOld) < wingBand, psiLo = pLoOld; end
            if ~isnan(pHiOld) && abs(psiHi - pHiOld) < wingBand, psiHi = pHiOld; end
            if abs(psiLo) > 0.03
                optTargets(end+1) = struct('strike', kLo, 'isCall', cLo, 'psi', psiLo);
                optDeltaSum = optDeltaSum + psiLo * dLo;
                optVegaSum  = optVegaSum  + psiLo * vLo;
            end
            if abs(psiHi) > 0.03
                optTargets(end+1) = struct('strike', kHi, 'isCall', cHi, 'psi', psiHi);
                optDeltaSum = optDeltaSum + psiHi * dHi;
                optVegaSum  = optVegaSum  + psiHi * vHi;
            end
        end
    else
        [~, iK] = min(abs(cands - targetK));
        kS = cands(iK);
        cS = pickParitySide(chain1, kS, isCallWing, f1_t, prevOpenOpts, useParity);
        [dS, ~, vS] = computeListedOptionGreeks(f1_t, kS, cS, chain1.tau, calibG, cfgG, dv, 0.0);
        if vS > minVega
            psiS  = min(max(targetVega / vS, -capCol), capCol);
            prevS = findPrevOptPsi(prevOpenOpts, kS, cS);
            if ~isnan(prevS) && abs(psiS - prevS) < wingBand, psiS = prevS; end
            if abs(psiS) > 0.05
                optTargets(end+1) = struct('strike', kS, 'isCall', cS, 'psi', psiS);
                optDeltaSum = optDeltaSum + psiS * dS;
                optVegaSum  = optVegaSum  + psiS * vS;
            end
        end
    end
end

function [selK, ok] = selectStickyWingOption(chain1, targetK, isCallOpt, prevOpenOpts)
    selK = NaN; ok = false;
    % Keep previously held strike for this wing while quoted and within $7.5 of targetK
    for j = 1:numel(prevOpenOpts)
        if prevOpenOpts(j).isCall == isCallOpt
            kOld = prevOpenOpts(j).strike;
            idx  = find(chain1.strike == kOld & chain1.isCall == isCallOpt, 1);
            if ~isempty(idx) && ~isnan(chain1.mid(idx)) && ~isnan(chain1.bid(idx)) && ...
               chain1.mid(idx) >= 0.05 && abs(kOld - targetK) <= 7.5
                selK = kOld; ok = true; return;
            end
        end
    end
    valid = ~isnan(chain1.mid) & ~isnan(chain1.bid) & (chain1.mid >= 0.10) & ...
            (chain1.isCall == isCallOpt) & (abs(chain1.strike - targetK) <= 7.5);
    cands = chain1.strike(valid);
    if ~isempty(cands)
        [~, iK] = min(abs(cands - targetK));
        selK = cands(iK);
        ok   = true;
    end
end

%% ========================================================================
%  HELPER 2: Put + Call Wing Pair Selector (for 'delta_vega_gamma')
%  ========================================================================
function [kP, kC, ok] = selectWingOptionPair(chain1, K_put, K_call)
    kP = NaN; kC = NaN; ok = false;
    valid = ~isnan(chain1.mid) & ~isnan(chain1.bid) & (chain1.mid >= 0.05);
    putsK  = chain1.strike(valid & ~chain1.isCall);
    callsK = chain1.strike(valid &  chain1.isCall);
    if isempty(putsK) || isempty(callsK)
        return;
    end
    [~, iP] = min(abs(putsK  - K_put));
    [~, iC] = min(abs(callsK - K_call));
    kP = putsK(iP);
    kC = callsK(iC);
    if abs(kC - kP) >= 5.0
        ok = true;
    end
end

%% ========================================================================
%  HELPER 2B: Sticky Core Put & Call Selector (for 'delta_vega_skew')
%  ========================================================================
function [kP, kC, hasP, hasC] = selectStickyCollarPair(chain1, F1_t, K_put, K_call, prevOpenOpts)
    kP = NaN; kC = NaN; hasP = false; hasC = false;

    % Keep strikes safely inside the liquid core window so options never fall
    % off the exchange quote boundary during large Brent moves
    loBand = F1_t - 15.0;
    hiBand = F1_t + 15.0;

    for j = 1:numel(prevOpenOpts)
        kOld = prevOpenOpts(j).strike;
        cOld = prevOpenOpts(j).isCall;
        idx  = find(chain1.strike == kOld & chain1.isCall == cOld, 1);
        if ~isempty(idx) && ~isnan(chain1.mid(idx)) && ~isnan(chain1.bid(idx)) && ...
           chain1.mid(idx) >= 0.10 && kOld >= loBand && kOld <= hiBand
            if ~cOld && ~hasP
                kP = kOld; hasP = true;
            elseif cOld && ~hasC
                kC = kOld; hasC = true;
            end
        end
    end

    % Target strikes clamped inside [F1 - 12, F1 + 12] for generous liquidity buffer
    tgtP = min(max(K_put,  F1_t - 12.0), F1_t + 5.0);
    tgtC = min(max(K_call, F1_t + 5.0),  F1_t + 12.0);
    valid = ~isnan(chain1.mid) & ~isnan(chain1.bid) & (chain1.mid >= 0.15) & ...
            (chain1.strike >= F1_t - 13.0) & (chain1.strike <= F1_t + 13.0);

    if ~hasP
        putsK = chain1.strike(valid & ~chain1.isCall);
        if ~isempty(putsK)
            [~, iP] = min(abs(putsK - tgtP));
            kP = putsK(iP); hasP = true;
        end
    end
    if ~hasC
        callsK = chain1.strike(valid & chain1.isCall);
        if ~isempty(callsK)
            [~, iC] = min(abs(callsK - tgtC));
            kC = callsK(iC); hasC = true;
        end
    end
end

function prevPsi = findPrevOptPsi(prevOpenOpts, strike, isCall)
    prevPsi = NaN;
    for j = 1:numel(prevOpenOpts)
        if prevOpenOpts(j).strike == strike && prevOpenOpts(j).isCall == isCall
            prevPsi = prevOpenOpts(j).psi;
            return;
        end
    end
end

%% ========================================================================
%  HELPER 3: Model-Consistent Listed Option Greeks (Delta, Gamma, Vega, Skew)
%  ========================================================================
function [d_o, g_o, v_o, s_o] = computeListedOptionGreeks(F1_t, K_o, isCall_o, tau_o, calib, config, dv, ds)
    if nargin < 8 || isempty(ds), ds = 0.0; end
    e1 = 2e-3 * F1_t;
    p0 = evalListedOption(F1_t,      K_o, isCall_o, tau_o, calib, config, 1.0, 0.0);
    pu = evalListedOption(F1_t + e1, K_o, isCall_o, tau_o, calib, config, 1.0, 0.0);
    pd = evalListedOption(F1_t - e1, K_o, isCall_o, tau_o, calib, config, 1.0, 0.0);
    vu = evalListedOption(F1_t,      K_o, isCall_o, tau_o, calib, config, 1.0 + dv, 0.0);
    vd = evalListedOption(F1_t,      K_o, isCall_o, tau_o, calib, config, 1.0 - dv, 0.0);

    d_o = (pu - pd) / (2.0 * e1);
    g_o = (pu - 2.0 * p0 + pd) / (e1^2);
    v_o = (vu - vd) / (2.0 * dv);
    s_o = 0.0;
    if ds > 1e-12
        su  = evalListedOption(F1_t, K_o, isCall_o, tau_o, calib, config, 1.0,  ds);
        sd  = evalListedOption(F1_t, K_o, isCall_o, tau_o, calib, config, 1.0, -ds);
        s_o = (su - sd) / (2.0 * ds);
    end
end

function price = evalListedOption(F1_v, K_o, isCall_o, tau_o, calib, config, volScale, skewBump)
    if nargin < 8 || isempty(skewBump), skewBump = 0.0; end
    if strcmpi(calib.modelName, 'black76')
        rot = 1.0;
        if abs(skewBump) > 1e-12
            rot = 1.0 + 0.05 * skewBump * tanh(log(F1_v / max(K_o, 1e-4)) / 0.15);
        end
        sig = calib.sigma1_fn(K_o) * volScale * rot;
        if tau_o <= 1e-6 || sig <= 1e-6
            if isCall_o, price = max(F1_v - K_o, 0.0);
            else,        price = max(K_o - F1_v, 0.0); end
            return;
        end
        norm_cdf = @(x) 0.5 * erfc(-x / sqrt(2));
        s  = sig * sqrt(tau_o);
        d1 = (log(F1_v / K_o) + 0.5 * s^2) / s;
        d2 = d1 - s;
        if isCall_o, price = F1_v * norm_cdf(d1) - K_o * norm_cdf(d2);
        else,        price = K_o * norm_cdf(-d2) - F1_v * norm_cdf(-d1); end
        return;
    end

    spec = calib.spec;
    if spec.isMultiMat
        p_o = spec.evalSliceParams(tau_o, calib.atsCurve, volScale, skewBump);
        lcf = @(u) spec.logCF(u, tau_o, p_o);
    else
        p_s = spec.scaleVol(calib.params1, volScale);
        if abs(skewBump) > 1e-12
            p_s = spec.bumpSkew(p_s, skewBump);
        end
        lcf = @(u) spec.logCF(u, tau_o, p_s);
    end
    c_val = lewisFFT('call', lcf, F1_v, K_o, config.fftN, config.fftDu);
    if isCall_o
        price = c_val(1);
    else
        price = c_val(1) - (F1_v - K_o);
    end
end
