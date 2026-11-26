%% ========================================================================
%  HEDGING COMPETITION - MAIN ENTRY SCRIPT (training and test run)
%  Computational Finance 2025/2026
%  Zero-cost Asian collar on ICE Brent futures, hedged by a market maker.
%  Set 'mode' and run: the same code is used for the training collar
%  (Apr21/May21, Feb 2021) and the test collar (Dec22/Jan23, Oct 2022).
%  All settings live in getConfig.m; the report analysis is in analysis_training.m.
%  ========================================================================
clear; clc; close all;
mode = 'train';                      % 'train' or 'test'

cfg  = getConfig(mode);
data = loadData(cfg.parquetFile, cfg.excelFile, cfg.leg1Code, cfg.leg2Code, ...
                cfg.avgYear, cfg.avgMonth, cfg.n2RollDays);
fprintf('Data loaded: %s to %s (%d days) | legs %s / %s | n1 = %d, n2 = %d\n\n', ...
    datestr(data.dates(1), 'dd-mmm-yyyy'), datestr(data.dates(end), 'dd-mmm-yyyy'), data.N_days, ...
    cfg.leg1Code, cfg.leg2Code, data.n1, data.n2);

out = runPipeline(data, cfg);

%% Deliverable: portfolio value, squared daily P&L, ranking metric
R = out.R;
fig = figure('Position', [100 100 900 700]);
subplot(2,1,1); plot(R.dates, R.Pi / 1e6); grid on; ylabel('USD million');
title(sprintf('Portfolio value \\Pi_t  (%s run)', mode));
subplot(2,1,2); semilogy(R.dates, R.pnl.^2 + eps); grid on; ylabel('USD^2');
title(sprintf('Squared daily P&L   |   (1/T) \\Sigma P&L^2 = %.4e', mean(R.pnl.^2)));
saveas(fig, sprintf('hedging_%s.png', mode));
fprintf('\nRanking metric (1/T) sum P&L^2 = %.6e over T = %d days\n', mean(R.pnl.^2), numel(R.pnl));
