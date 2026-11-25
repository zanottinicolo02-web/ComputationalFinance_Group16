function out = runPipeline(data, cfg)
% RUNPIPELINE  Full pipeline on one dataset: calendar check, implied vols, smiles, correlation,
% zero-cost K_put at the first date, and the hedging backtest. The same code is used for the
% training collar (Apr21/May21, Feb 2021) and the test collar (Dec22/Jan23, Oct 2022).
%
%   data  struct returned by loadData (fields as used in the backbone script)
%   cfg   leg1Year, leg1Month   contract month of the first-nearby leg (e.g. 2021, 4 for Apr21)
%         avgYear, avgMonth     averaging window
%         hedge                 options struct for runHedgeBacktestOpt
%         optsIV, optsRho       (optional) structs for extractImpliedVols / estimateCorrelation
%         compare               (optional, default false) also run futures-only and unhedged
    if ~isfield(cfg, 'optsIV'),   cfg.optsIV   = struct(); end
    if ~isfield(cfg, 'optsRho'),  cfg.optsRho  = struct(); end
    if ~isfield(cfg, 'compare'),  cfg.compare  = false; end

    dates = data.dates;  N_days = data.N_days;
    F1 = data.F1_settle;  F2 = data.F2_settle;
    notional = data.notional;  K_call = data.K_call;

    %% Calendar: our rule must reproduce the fixing schedule of the data (Annex B + Q&A)
    expiry1 = brentFuturesExpiry(cfg.leg1Year, cfg.leg1Month);
    sched   = buildFixingSchedule(dates, cfg.avgYear, cfg.avgMonth, expiry1);
    assert(sched.n2 == data.n2 && isequal(sched.fixing_idx(:), data.fixing_idx(:)), ...
        'Calendar mismatch: rule gives n1=%d, n2=%d; data has n2=%d. Check expiry and n2RollDays.', ...
        sched.n1, sched.n2, data.n2);
    fprintf('Calendar OK: leg-1 expiry %s, last leg-1 fixing %s, n1 = %d, n2 = %d\n', ...
        datestr(expiry1, 'dd-mmm-yyyy'), datestr(sched.lastLeg1Day, 'dd-mmm-yyyy'), sched.n1, sched.n2);

    %% Implied vols and daily smiles (stale smile carried forward if < 3 usable quotes)
    iv1 = cell(N_days,1);  iv2 = cell(N_days,1);  st1 = cell(N_days,1);  st2 = cell(N_days,1);
    smiles1 = cell(N_days,1);  smiles2 = cell(N_days,1);
    stale1 = false(N_days,1);  stale2 = false(N_days,1);  prev1 = [];  prev2 = [];
    for t = 1:N_days
        [iv1{t}, st1{t}] = extractImpliedVols(data.optChain1{t}, F1(t), dates(t), cfg.optsIV);
        [iv2{t}, st2{t}] = extractImpliedVols(data.optChain2{t}, F2(t), dates(t), cfg.optsIV);
        if height(iv1{t}) >= 3, smiles1{t} = fitSmileQuad(iv1{t}.LogMoneyness, iv1{t}.IV); prev1 = smiles1{t};
        else, smiles1{t} = prev1; stale1(t) = true; end
        if height(iv2{t}) >= 3, smiles2{t} = fitSmileQuad(iv2{t}.LogMoneyness, iv2{t}.IV); prev2 = smiles2{t};
        else, smiles2{t} = prev2; stale2(t) = true; end
    end
    if isempty(smiles1{1}) || isempty(smiles2{1})
        error('runPipeline:smile', 'Fewer than 3 usable quotes on the first date: relax cfg.optsIV filters.');
    end
    fprintf('Stale smile days: leg 1 = %d, leg 2 = %d\n', nnz(stale1), nnz(stale2));

    %% Correlation (rolling window + shrinkage, no look-ahead)
    rho = zeros(N_days,1);
    for t = 1:N_days, rho(t) = estimateCorrelation(F1, F2, t, cfg.optsRho); end

    %% Pricing at the first date: zero-cost strike (solved once)
    st = buildPricingState(1, dates, data.fixing_dates, data.fixing_idx, data.fixing_leg, F1, F2, data.N_fix);
    st.rho = rho(1);  st.smile1 = smiles1{1};  st.smile2 = smiles2{1};
    st.kRange1 = [min(iv1{1}.LogMoneyness), max(iv1{1}.LogMoneyness)];
    st.kRange2 = [min(iv2{1}.LogMoneyness), max(iv2{1}.LogMoneyness)];
    [K_put, zc] = solveZeroCostStrike(K_call, st);
    fprintf('K_put = %.4f | Call(%g) = Put(K_put) = %.4f USD/bbl | premium %.0f USD | vols %.3f / %.3f\n', ...
        K_put, K_call, zc.target, zc.target * notional, zc.vols(1), zc.vols(2));
    if zc.inRange == 0
        warning('K_put lies outside the strike range used for the smile fit: its vol is extrapolated.');
    end

    %% Backtest
    bt = struct();
    bt.dates = dates;  bt.F1 = F1;  bt.F2 = F2;
    bt.F1_spread = data.F1_spread;  bt.F2_spread = data.F2_spread;  bt.sofr = data.sofr_rate;
    bt.rho = rho;  bt.smiles1 = smiles1;  bt.smiles2 = smiles2;
    bt.fixing_dates = data.fixing_dates;  bt.fixing_idx = data.fixing_idx;  bt.fixing_leg = data.fixing_leg;
    bt.N_fix = data.N_fix;  bt.K_call = K_call;  bt.K_put = K_put;  bt.notional = notional;
    bt.iv1 = iv1;  bt.chain1 = data.optChain1;

    R = runHedgeBacktestOpt(bt, cfg.hedge);
    fprintf('Hedge: mean P&L^2 = %.4e | costs fut %.0f opt %.0f | final Pi = %.0f | max|P&L| = %.0f\n', ...
        R.meanSqPnL, sum(R.costFut), sum(R.costOpt), R.Pi(end), max(abs(R.pnl)));

    out = struct('K_put', K_put, 'zeroCost', zc, 'R', R, 'bt', bt, 'rho', rho, 'iv1', {iv1}, 'iv2', {iv2}, ...
                 'filterStats1', {st1}, 'filterStats2', {st2}, 'stale1', stale1, 'stale2', stale2, 'sched', sched);
    if cfg.compare
        h0 = cfg.hedge;  h0.useOptions = false;
        out.R_futOnly = runHedgeBacktestOpt(bt, h0);
        h0.hedgeScale = 0;
        out.R_unhedged = runHedgeBacktestOpt(bt, h0);
        fprintf('Futures-only: %.4e | unhedged: %.4e\n', out.R_futOnly.meanSqPnL, out.R_unhedged.meanSqPnL);
    end
end
