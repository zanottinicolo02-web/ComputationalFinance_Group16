%% ========================================================================
%  TRAINING-SET ANALYSIS FOR THE REPORT  (not needed for the test run: use main.m)
%  Runs the pipeline once, then produces: filter statistics, smiles, correlation, pricer checks,
%  K_put sensitivities, Greeks at inception, strategy comparison, parameter sweeps,
%  sub-period robustness and the worst-day P&L decomposition.
%  ========================================================================
clear; clc; close all;

%% 0. Pipeline
cfg  = getConfig('train');
data = loadData(cfg.parquetFile, cfg.excelFile, cfg.leg1Code, cfg.leg2Code, ...
                cfg.avgYear, cfg.avgMonth, cfg.n2RollDays);
out  = runPipeline(data, cfg);
bt = out.bt;  dates = bt.dates;  K_call = bt.K_call;  K_put = out.K_put;  notional = bt.notional;
iv = {out.iv1, out.iv2};  stats = {out.filterStats1, out.filterStats2};
chains = {data.optChain1, data.optChain2};

%% 1. Quote filtering statistics (per leg, summed over the sample)
for leg = 1:2
    S = stats{leg};
    nTot  = sum(cellfun(@nQuotes, chains{leg}));
    first = find(~cellfun(@isempty, S), 1);
    names = {S{first}.name};
    removed = zeros(1, numel(names));
    for t = 1:numel(S), if ~isempty(S{t}), removed = removed + [S{t}.removed]; end, end
    fprintf('\nLeg %d: %d quotes in total, %d kept\n', leg, nTot, nTot - sum(removed));
    disp(table(names', removed', 'VariableNames', {'Filter', 'Removed'}));
end

%% 2. Smiles on day 1 and correlation
kk = linspace(-0.45, 0.35, 100);
figure;
for leg = 1:2
    subplot(1,2,leg);  Q = iv{leg}{1};  sm = bt.(sprintf('smiles%d', leg)){1};
    plot(Q.LogMoneyness, Q.IV, 'o', kk, sm(kk), '-'); grid on;
    xlabel('ln(K/F)'); ylabel('implied vol'); title(sprintf('Leg %d smile, %s', leg, datestr(dates(1))));
    legend('quotes', 'quadratic fit');
end
figure; plot(dates, out.rho); grid on; title('Correlation of the two futures (rolling + shrinkage)'); ylabel('\rho');

%% 3. State at inception, pricer checks
st = buildPricingState(1, dates, bt.fixing_dates, bt.fixing_idx, bt.fixing_leg, bt.F1, bt.F2, bt.N_fix);
st.rho = out.rho(1);  st.smile1 = bt.smiles1{1};  st.smile2 = bt.smiles2{1};
kmin1 = min(iv{1}{1}.LogMoneyness);  kmin2 = min(iv{2}{1}.LogMoneyness);

pMM = basketMomentMatching(true, 62, bt.F1(1), bt.F2(1), 0.30, 0.30, 0.98, 0.5, 1, 1, 0);
fprintf('\nVanilla check: MM = %.6f, Black-76 = %.6f\n', pMM, black76(bt.F1(1), 62, 0.30, 0.5, true));

[callPx, a1, a2] = basketOptionPrice(K_call, true, st);
nB = 10;  nPer = 2e5;  mcM = zeros(nB,1);  mcS = zeros(nB,1);
for b = 1:nB, [mcM(b), mcS(b)] = basketMCcheck(true, K_call, st, a1, a2, nPer, 100 + b); end
mcSE = sqrt(sum(mcS.^2)) / nB;
fprintf('Call(%g): MM = %.4f | MC (%.0e paths) = %.4f +- %.4f | gap = %.2f SE\n', ...
    K_call, callPx, nB*nPer, mean(mcM), mcSE, (mean(mcM) - callPx) / mcSE);

%% 4. K_put and its sensitivity to rho and to the wing vol
fprintf('\nK_put = %.4f | premium %.0f USD | vols at K_put %.3f / %.3f | in smile range: %d\n', ...
    K_put, out.zeroCost.target * notional, out.zeroCost.vols, out.zeroCost.inRange);
for r = [0.90 0.95 0.98 0.999]
    s2 = st;  s2.rho = r;
    fprintf('  rho = %.3f -> K_put = %.4f\n', r, solveZeroCostStrike(K_call, s2));
end
for d = [-0.04 -0.02 0 0.02 0.04]          % shift of the extrapolated wing vol only
    s2 = st;
    s2.smile1 = @(k) st.smile1(k) + d * (k < kmin1);
    s2.smile2 = @(k) st.smile2(k) + d * (k < kmin2);
    fprintf('  wing vol shift %+.2f -> K_put = %.4f\n', d, solveZeroCostStrike(K_call, s2));
end

%% 5. Greeks at inception under both smile dynamics
for dyn = {'sticky_strike', 'sticky_moneyness'}
    G = structureGreeks(K_put, K_call, st, notional, struct('dynamics', dyn{1}));
    fprintf('\n%s: structure delta = [%.0f %.0f] bbl | gamma = %.0f | vega = [%.0f %.0f] USD/volpt | value %.2f\n', ...
        dyn{1}, G.delta, G.gamma, G.vega, G.value);
end

%% 6. Strategy comparison
base  = struct('qTol', 0.6, 'band1', 0.35, 'keepStrikeAcrossType', true);
cases = { ...
    'unhedged',                  struct('useOptions', false, 'hedgeScale', 0);
    'futures only (strike)',     struct('useOptions', false, 'dynamics', 'sticky_strike');
    'futures only (moneyness)',  struct('useOptions', false, 'dynamics', 'sticky_moneyness');
    'vega-only (strike)',        struct('useGamma', false, 'dynamics', 'sticky_strike');
    'vega-only (moneyness)',     struct('useGamma', false, 'dynamics', 'sticky_moneyness');
    'vega+gamma (strike)',       struct('dynamics', 'sticky_strike');
    'vega+gamma (moneyness)',    struct('dynamics', 'sticky_moneyness')};
res = cell(size(cases, 1), 1);
fprintf('\n%-26s %12s %10s %10s %12s %11s\n', 'Strategy', 'mean P&L^2', 'cost fut', 'cost opt', 'final Pi', 'max|P&L|');
for j = 1:size(cases, 1)
    res{j} = runHedgeBacktestOpt(bt, mergeOpts(base, cases{j,2}));  r = res{j};
    fprintf('%-26s %12.4e %10.0f %10.0f %12.0f %11.0f\n', cases{j,1}, r.meanSqPnL, ...
        sum(r.costFut), sum(r.costOpt), r.Pi(end), max(abs(r.pnl)));
end

%% 7. Parameter sweeps for the chosen strategy (cfg.hedge)
fprintf('\nband1 sweep (chosen strategy)\n%7s %12s %10s %12s\n', 'band1', 'mean P&L^2', 'cost opt', 'final Pi');
for b1 = [0.08 0.15 0.25 0.35 0.45]
    o = cfg.hedge;  o.band1 = b1;  r = runHedgeBacktestOpt(bt, o);
    fprintf('%7.2f %12.4e %10.0f %12.0f\n', b1, r.meanSqPnL, sum(r.costOpt), r.Pi(end));
end
fprintf('\nqTol sweep (chosen strategy)\n%7s %12s %10s %12s\n', 'qTol', 'mean P&L^2', 'cost opt', 'final Pi');
for qt = [0.1 0.3 0.6 1.0]
    o = cfg.hedge;  o.qTol = qt;  r = runHedgeBacktestOpt(bt, o);
    fprintf('%7.2f %12.4e %10.0f %12.0f\n', qt, r.meanSqPnL, sum(r.costOpt), r.Pi(end));
end

%% 8. Sub-period robustness of the smile dynamics (before / after end of April 2020)
early = dates <= datetime(2020, 4, 30);
fprintf('\nSub-periods (chosen strategy)\n');
for dyn = {'sticky_strike', 'sticky_moneyness'}
    o = cfg.hedge;  o.dynamics = dyn{1};  r = runHedgeBacktestOpt(bt, o);
    p = r.pnl.^2;  [~, i] = max(abs(r.pnl));
    fprintf('%-18s | to Apr-2020: %.3e | after: %.3e | worst day %s (%.0f)\n', ...
        dyn{1}, mean(p(early)), mean(p(~early)), datestr(dates(i)), r.pnl(i));
end

%% 9. Worst day of the futures-only hedge: price, hedge and vol effects
rF = runHedgeBacktestOpt(bt, mergeOpts(base, struct('useOptions', false, 'dynamics', cfg.hedge.dynamics)));
[~, i] = max(abs(rF.pnl));
V0 = structureGreeks(K_put, K_call, mkState(bt, i-1, i-1), notional).value;
Vp = structureGreeks(K_put, K_call, mkState(bt, i,   i-1), notional).value;   % new prices, old smile
V1 = structureGreeks(K_put, K_call, mkState(bt, i,   i),   notional).value;   % new prices, new smile
fprintf('\nWorst day %s: price effect %.0f | futures hedge %.0f | residual gamma %.0f | vol effect %.0f | P&L %.0f\n', ...
    datestr(dates(i)), Vp - V0, rF.futuresPnL(i), Vp - V0 + rF.futuresPnL(i), V1 - Vp, rF.pnl(i));

%% 10. Final figures: chosen strategy vs benchmarks
rC = runHedgeBacktestOpt(bt, cfg.hedge);
rU = res{1};  rB = rF;
fig1 = figure; plot(dates, rC.Pi / 1e6, dates, rB.Pi / 1e6, dates, rU.Pi / 1e6); grid on; ylabel('USD million');
legend({'chosen (vega + futures)', 'futures only', 'unhedged'}, 'Location', 'best'); title('Portfolio value \Pi_t');
fig2 = figure; semilogy(dates, rC.pnl.^2 + eps, dates, rB.pnl.^2 + eps); grid on; ylabel('USD^2');
legend({'chosen', 'futures only'}); title('Squared daily P&L');
fig3 = figure; plot(dates, rC.q(:,1)); grid on; ylabel('bbl'); title('Hedge option quantity (chosen strategy)');
saveas(fig1, 'train_portfolio_value.png');  saveas(fig2, 'train_squared_pnl.png');  saveas(fig3, 'train_option_qty.png');

%% ------------------------------------------------------------------------
function n = nQuotes(c)
    n = 0;
    if isstruct(c) && isfield(c, 'strike'), n = numel(c.strike); end
end

function o = mergeOpts(a, b)
    o = a;  f = fieldnames(b);
    for k = 1:numel(f), o.(f{k}) = b.(f{k}); end
end

function s = mkState(bt, t, tSmile)
% Pricing state at day t (prices/fixings of day t) with the smile and rho of day tSmile.
    s = buildPricingState(t, bt.dates, bt.fixing_dates, bt.fixing_idx, bt.fixing_leg, bt.F1, bt.F2, bt.N_fix);
    s.rho = bt.rho(tSmile);  s.smile1 = bt.smiles1{tSmile};  s.smile2 = bt.smiles2{tSmile};
end
