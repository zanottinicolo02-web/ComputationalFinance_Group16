function results = runBacktest(data, config)
%RUNBACKTEST Execute the full daily calibration, pricing, and hedging backtest.
%
%   results = runBacktest(data, config) runs from emission date t0
%   (06-Jan-2020) to final settlement T (26-Feb-2021). At each rebalancing
%   date t:
%     1. Recalibrates volatility smiles, forward parameters, and correlation rho.
%     2. At inception (t = 1), solves for the zero-cost Put strike K_put.
%     3. Reprices the collar structure and computes hedge ratios.
%     4. Updates the cash account (SOFR interest + futures MTM - bid/ask TC)
%        and records daily P&L and the competition metric (1/T) * sum(P&L_t^2).

    N_days   = data.N_days;
    notional = data.notional;
    K_call   = data.K_call;

    % Preallocate time-series vectors (N_days x 1)
    V_struct     = zeros(N_days, 1); % Market maker structure value (Put - Call) [USD/bbl]
    P_leg        = zeros(N_days, 1); % Long Put(K_put) value [USD/bbl]
    C_leg        = zeros(N_days, 1); % Short Call(K_call) value [USD/bbl]

    delta1       = zeros(N_days, 1); % dV_struct / dF1
    delta2       = zeros(N_days, 1); % dV_struct / dF2
    gamma1       = zeros(N_days, 1); % d^2V_struct / dF1^2
    gamma2       = zeros(N_days, 1); % d^2V_struct / dF2^2
    vega1        = zeros(N_days, 1); % dV_struct / dSigma1
    vega2        = zeros(N_days, 1); % dV_struct / dSigma2

    phi1         = zeros(N_days, 1); % Target Leg 1 futures position [bbl / bbl]
    phi2         = zeros(N_days, 1); % Target Leg 2 futures position [bbl / bbl]
    psi1         = zeros(N_days, 1); % Target Leg 1 option position
    psi2         = zeros(N_days, 1); % Target Leg 2 option position

    cash         = zeros(N_days, 1); % Cash account balance [USD/bbl]
    interestCF   = zeros(N_days, 1); % Daily overnight SOFR interest [USD/bbl]
    futuresPnL   = zeros(N_days, 1); % Daily futures settlement P&L [USD/bbl]
    transCost    = zeros(N_days, 1); % Daily bid-ask transaction cost [USD/bbl]

    portVal      = zeros(N_days, 1); % Total hedged portfolio value V_struct + cash [USD/bbl]
    dailyPnL     = zeros(N_days, 1); % Daily hedged portfolio P&L [USD/bbl]
    unhedgedPnL  = zeros(N_days, 1); % Daily unhedged structure P&L [USD/bbl]

    atmVol1      = zeros(N_days, 1); % Calibrated Leg 1 ATM implied vol
    atmVol2      = zeros(N_days, 1); % Calibrated Leg 2 ATM implied vol
    rho_ts       = zeros(N_days, 1); % Calibrated correlation rho(t)
    calibHistory = cell(N_days, 1);  % Full calibration structs per day

    calib = [];
    K_put = NaN;
    C_inception = NaN;
    P_inception = NaN;

    for t = 1:N_days
        % 1. Daily Calibration (Volatility Smiles + Forward Model + Correlation)
        calib = calibrateModel(t, data, config, calib);
        calibHistory{t} = calib;
        atmVol1(t)      = calib.atmVol1;
        atmVol2(t)      = calib.atmVol2;
        rho_ts(t)       = calib.rho;

        % 2. Inception (t = 1): Solve for Zero-Cost Put Strike K_put
        if t == 1
            [K_put, C_inception, P_inception] = solveZeroCostStrike(1, K_call, data, calib, config);
        end

        % 3. Reprice Structure & Compute Target Hedge Positions
        hedge = computeHedge(t, data, calib, K_put, config);

        V_struct(t) = hedge.V_struct;
        P_leg(t)    = hedge.P_val;
        C_leg(t)    = hedge.C_val;
        delta1(t)   = hedge.delta1;
        delta2(t)   = hedge.delta2;
        gamma1(t)   = hedge.gamma1;
        gamma2(t)   = hedge.gamma2;
        vega1(t)    = hedge.vega1;
        vega2(t)    = hedge.vega2;
        phi1(t)     = hedge.phi1;
        phi2(t)     = hedge.phi2;
        psi1(t)     = hedge.psi1;
        psi2(t)     = hedge.psi2;

        % 4. Portfolio Cash Account, Transaction Costs, and Daily MTM P&L
        if t == 1
            % At inception, V_struct(1) = 0; pay half-spread to establish initial hedge
            tc_1 = 0.5 * data.F1_spread(1) * abs(phi1(1)) + ...
                   0.5 * data.F2_spread(1) * abs(phi2(1));
            transCost(1)   = tc_1;
            cash(1)        = -tc_1;
            portVal(1)     = V_struct(1) + cash(1);
            dailyPnL(1)    = 0.0;
            unhedgedPnL(1) = 0.0;
        else
            % Overnight SOFR interest accrued on previous day's cash balance
            r_prev = data.sofr_rate(t-1);
            dt_t   = data.dt_yf(t);
            interest_t = cash(t-1) * (exp(r_prev * dt_t) - 1.0);

            % Daily mark-to-market settlement P&L from futures held over [t-1, t]
            fut_pnl_t = phi1(t-1) * (data.F1_settle(t) - data.F1_settle(t-1)) + ...
                        phi2(t-1) * (data.F2_settle(t) - data.F2_settle(t-1));

            % Bid-ask spread transaction cost on hedge rebalancing
            if t == N_days && ~config.unwindAtMaturity
                tc_t = 0.0;
            else
                tc_t = 0.5 * data.F1_spread(t) * abs(phi1(t) - phi1(t-1)) + ...
                       0.5 * data.F2_spread(t) * abs(phi2(t) - phi2(t-1));
            end

            interestCF(t)  = interest_t;
            futuresPnL(t)  = fut_pnl_t;
            transCost(t)   = tc_t;
            cash(t)        = cash(t-1) + interest_t + fut_pnl_t - tc_t;
            portVal(t)     = V_struct(t) + cash(t);
            dailyPnL(t)    = portVal(t) - portVal(t-1);
            unhedgedPnL(t) = V_struct(t) - V_struct(t-1);
        end
    end

    % 5. Summary Metrics & Competition Ranking Score: (1/T) * sum(P&L_t^2)
    % Computed over the T = N_days - 1 daily return intervals (t = 2..N_days)
    pnl_steps          = dailyPnL(2:end);
    unhedged_steps     = unhedgedPnL(2:end);

    mse_per_bbl        = mean(pnl_steps .^ 2);
    mse_unhedged_bbl   = mean(unhedged_steps .^ 2);

    dailyPnL_total     = notional * dailyPnL;
    unhedgedPnL_total  = notional * unhedgedPnL;
    portVal_total      = notional * portVal;
    V_struct_total     = notional * V_struct;
    cash_total         = notional * cash;

    mse_total          = mean((notional * pnl_steps) .^ 2);
    mse_unhedged_total = mean((notional * unhedged_steps) .^ 2);
    var_reduction_pct  = 100.0 * (1.0 - mse_per_bbl / mse_unhedged_bbl);

    % Pack into output struct
    results = struct();
    results.dates              = data.dates;
    results.K_call             = K_call;
    results.K_put              = K_put;
    results.C_inception        = C_inception;
    results.P_inception        = P_inception;

    % Per-barrel series [USD/bbl]
    results.V_struct           = V_struct;
    results.P_leg              = P_leg;
    results.C_leg              = C_leg;
    results.delta1             = delta1;
    results.delta2             = delta2;
    results.gamma1             = gamma1;
    results.gamma2             = gamma2;
    results.vega1              = vega1;
    results.vega2              = vega2;
    results.phi1               = phi1;
    results.phi2               = phi2;
    results.psi1               = psi1;
    results.psi2               = psi2;
    results.cash               = cash;
    results.interestCF         = interestCF;
    results.futuresPnL         = futuresPnL;
    results.transCost          = transCost;
    results.cumTransCost       = cumsum(transCost);
    results.portVal            = portVal;
    results.dailyPnL           = dailyPnL;
    results.unhedgedPnL        = unhedgedPnL;
    results.sqDailyPnL         = dailyPnL .^ 2;

    % Total Notional series (1,000,000 bbl) [USD]
    results.V_struct_total     = V_struct_total;
    results.cash_total         = cash_total;
    results.portVal_total      = portVal_total;
    results.dailyPnL_total     = dailyPnL_total;
    results.unhedgedPnL_total  = unhedgedPnL_total;
    results.sqDailyPnL_total   = dailyPnL_total .^ 2;

    % Calibration trajectories
    results.atmVol1            = atmVol1;
    results.atmVol2            = atmVol2;
    results.rho                = rho_ts;
    results.calibHistory       = calibHistory;

    % Competition performance metrics
    results.mse_per_bbl        = mse_per_bbl;
    results.mse_unhedged_bbl   = mse_unhedged_bbl;
    results.mse_total          = mse_total;
    results.mse_unhedged_total = mse_unhedged_total;
    results.var_reduction_pct  = var_reduction_pct;
end

