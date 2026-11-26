function plotResults(data, resultsInput, arg3, refDeltaResults, saveDir)
%PLOTRESULTS Plot hedge evolution over time and cleanly overwrite PNG figures.
%
%   USAGE 1 (Multi-Model Comparison — All Models):
%       plotResults(data, vegaResults, modelList, deltaResults, 'figures')
%
%   USAGE 2 (Single-Model Diagnostic):
%       plotResults(data, results, config, [], 'figures')

    if nargin < 4
        refDeltaResults = [];
    end
    if nargin < 5
        saveDir = '';
    end

    % Resolve saveDir and clear previous run's PNG figures so every run
    % cleanly substitutes the figures with those of the new run.
    if ~isempty(saveDir)
        if ~startsWith(saveDir, filesep) && ~(numel(saveDir) >= 2 && saveDir(2) == ':')
            saveDir = fullfile(pwd, saveDir);
        end
        if ~isfolder(saveDir)
            mkdir(saveDir);
        else
            oldPngs = dir(fullfile(saveDir, 'fig*.png'));
            for k = 1:numel(oldPngs)
                delete(fullfile(saveDir, oldPngs(k).name));
            end
        end
    end

    if iscell(arg3) || isstring(arg3)
        plotMultiModelComparison(data, resultsInput, cellstr(arg3), refDeltaResults, saveDir);
    else
        plotSingleModel(data, resultsInput, arg3, saveDir);
    end
end

%% ========================================================================
%  MODE 1: MULTI-MODEL HEDGE EVOLUTION & PERFORMANCE COMPARISON (ALL MODELS)
%  ========================================================================
function plotMultiModelComparison(data, allResults, modelList, refDeltaResults, saveDir)
    dates   = data.dates;
    nModels = numel(modelList);
    colors  = buildDistinctPalette(nModels);
    styles  = {'-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--'};
    refRes  = allResults.(modelList{1});
    hTag    = refRes.hedgeMethod;
    figVis  = 'on';
    if ~usejava('desktop'), figVis = 'off'; end

    %% FIGURE 1: Evolution of Hedged Portfolio Value & Running MSE (All Models)
    f1 = figure('Name', ['Hedge Evolution Over Time (' hTag ')'], 'Color', 'w', ...
                'Position', [50, 70, 1280, 780], 'Visible', figVis);

    % 1A: Hedged Portfolio Value Pi(t) — Unhedged removed so all models are clearly visible
    subplot(2, 1, 1); hold on;
    for i = 1:nModels
        r  = allResults.(modelList{i});
        lw = 1.5 + 0.7 * strcmpi(modelList{i}, 'black76');
        plot(dates, r.portVal, 'Color', colors(i,:), 'LineStyle', styles{mod(i-1,numel(styles))+1}, ...
            'LineWidth', lw, 'DisplayName', sprintf('%s (End: %+.2f)', upper(modelList{i}), r.finalPortVal));
    end
    yline(0, 'k:', 'HandleVisibility', 'off');
    grid on; xlabel('Date'); ylabel('Portfolio Value \Pi(t) [USD/bbl]');
    title(sprintf('Evolution of Net Hedged Portfolio Value Over Time (%s — All %d Models)', hTag, nModels), ...
        'Interpreter', 'none');
    legend('Location', 'southwest', 'NumColumns', 2, 'FontSize', 8);

    % 1B: Running Competition Metric (1/t) * sum_{s<=t} P&L_s^2
    subplot(2, 1, 2); hold on;
    for i = 1:nModels
        r  = allResults.(modelList{i});
        lw = 1.5 + 0.7 * strcmpi(modelList{i}, 'black76');
        plot(dates(2:end), r.runningMSE(2:end), 'Color', colors(i,:), ...
            'LineStyle', styles{mod(i-1,numel(styles))+1}, 'LineWidth', lw, ...
            'DisplayName', sprintf('%s (MSE: %.5f)', upper(modelList{i}), r.mse_per_bbl));
    end
    grid on; xlabel('Date'); ylabel('Running (1/t) \Sigma P&L_s^2 [(USD/bbl)^2]');
    title(sprintf('Evolution of Cumulative Squared Daily P&L Over Time (%s)', hTag), 'Interpreter', 'none');
    legend('Location', 'northeast', 'NumColumns', 2, 'FontSize', 8);
    saveFigIfRequested(f1, saveDir, 'fig1_hedge_evolution.png');

    %% FIGURE 2: Dynamic Futures & Option Hedge Positions + Transaction Costs
    f2 = figure('Name', ['Dynamic Hedge Ratios (' hTag ')'], 'Color', 'w', ...
                'Position', [90, 90, 1280, 860], 'Visible', figVis);

    subplot(3, 1, 1); hold on;
    for i = 1:nModels
        r = allResults.(modelList{i});
        plot(dates, r.phi1, 'Color', colors(i,:), 'LineStyle', styles{mod(i-1,numel(styles))+1}, ...
            'LineWidth', 1.4, 'DisplayName', upper(modelList{i}));
    end
    yline(0, 'k:', 'HandleVisibility', 'off'); grid on;
    ylabel('\phi_1(t) [bbl/bbl]');
    title(sprintf('Evolution of Leg 1 (%s) Futures Hedge Position \\phi_1(t)', data.leg1Code));
    legend('Location', 'northwest', 'NumColumns', 5, 'FontSize', 8);

    subplot(3, 1, 2); hold on;
    hasOptHedge = any(abs(refRes.psi1) > 1e-8);
    if hasOptHedge
        for i = 1:nModels
            r = allResults.(modelList{i});
            plot(dates, r.psi1, 'Color', colors(i,:), 'LineStyle', styles{mod(i-1,numel(styles))+1}, ...
                'LineWidth', 1.4, 'DisplayName', upper(modelList{i}));
        end
        yline(0, 'k:', 'HandleVisibility', 'off'); grid on;
        ylabel('\psi_1(t) [opt/bbl]');
        title(sprintf('Evolution of Listed %s Option Vega-Hedge Position \\psi_1(t)', data.leg1Code));
    else
        basePhi1 = refRes.phi1;
        for i = 1:nModels
            r = allResults.(modelList{i});
            plot(dates, r.phi1 - basePhi1, 'Color', colors(i,:), ...
                'LineStyle', styles{mod(i-1,numel(styles))+1}, 'LineWidth', 1.4, ...
                'DisplayName', sprintf('%s - %s', upper(modelList{i}), upper(modelList{1})));
        end
        yline(0, 'k:', 'HandleVisibility', 'off'); grid on;
        ylabel('\Delta \phi_1(t) [bbl/bbl]');
        title(sprintf('Hedge Ratio Divergence Relative to %s', upper(modelList{1})));
    end
    legend('Location', 'northeast', 'NumColumns', 5, 'FontSize', 8);

    subplot(3, 1, 3); hold on;
    for i = 1:nModels
        r = allResults.(modelList{i});
        plot(dates, r.cumTransCost, 'Color', colors(i,:), ...
            'LineStyle', styles{mod(i-1,numel(styles))+1}, 'LineWidth', 1.4, ...
            'DisplayName', sprintf('%s (%.2f)', upper(modelList{i}), r.cumTransCost(end)));
    end
    grid on; xlabel('Date'); ylabel('Cumulative TC [USD/bbl]');
    title('Cumulative Bid-Ask Transaction Costs Over Time (Futures + Options)');
    legend('Location', 'northwest', 'NumColumns', 2, 'FontSize', 8);
    saveFigIfRequested(f2, saveDir, 'fig2_hedge_ratios_tc.png');

    %% FIGURE 3: Calibration Smile Fit (Inception) & Daily Calibration RMSE
    cfIdx = find(~strcmpi(modelList, 'black76'));
    if ~isempty(cfIdx)
        f3 = figure('Name', 'Calibration Quality (All Parametric Models)', 'Color', 'w', ...
                    'Position', [130, 110, 1280, 780], 'Visible', figVis);

        subplot(2, 1, 1); hold on;
        c0 = allResults.(modelList{cfIdx(1)}).calibHistory{1};
        s1 = c0.smile1;
        plot(s1.strikes, 100 * s1.iv, 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'k', ...
            'DisplayName', sprintf('Market OTM (%s)', data.leg1Code));

        K_dense = linspace(min(s1.strikes), max(s1.strikes), 80)';
        isCallD = K_dense > c0.F1;
        for j = 1:numel(cfIdx)
            idx  = cfIdx(j);
            mKey = modelList{idx};
            c_m  = allResults.(mKey).calibHistory{1};
            spec = cfModels(mKey);
            if spec.isMultiMat
                p_o = spec.evalSliceParams(s1.tau, c_m.atsCurve, 1.0);
                lcf = @(u) spec.logCF(u, s1.tau, p_o);
            else
                lcf = @(u) spec.logCF(u, s1.tau, c_m.params1);
            end
            C_fit = lewisFFT('call', lcf, c0.F1, K_dense, 2048, 0.15);
            P_fit = C_fit;
            P_fit(~isCallD) = C_fit(~isCallD) - (c0.F1 - K_dense(~isCallD));
            iv_fit = asayImpliedVol(P_fit, c0.F1, K_dense, s1.tau, isCallD);
            plot(K_dense, 100 * iv_fit, 'Color', colors(idx,:), 'LineWidth', 1.6, ...
                'LineStyle', styles{mod(idx-1,numel(styles))+1}, ...
                'DisplayName', sprintf('%s (%.4f)', upper(mKey), c_m.rmse1));
        end
        xline(c0.F1, 'k:', 'F_1(t_0)', 'HandleVisibility', 'off');
        grid on; xlabel('Strike K [USD/bbl]'); ylabel('Implied Volatility [%]');
        title(sprintf('Inception Smile Fit (%s, %s, \\tau = %.2f yr — All Parametric Models)', ...
            string(dates(1), 'dd-MMM-yyyy'), data.leg1Code, s1.tau));
        legend('Location', 'northeast', 'NumColumns', 2, 'FontSize', 8);

        subplot(2, 1, 2); hold on;
        for j = 1:numel(cfIdx)
            idx  = cfIdx(j);
            mKey = modelList{idx};
            r    = allResults.(mKey);
            plot(dates, r.calibRMSE1, 'Color', colors(idx,:), ...
                'LineStyle', styles{mod(idx-1,numel(styles))+1}, 'LineWidth', 1.3, ...
                'DisplayName', sprintf('%s (Avg: %.4f)', upper(mKey), r.avgCalibRMSE1));
        end
        grid on; xlabel('Date'); ylabel('Daily Smile RMSE [USD/bbl]');
        title(sprintf('Daily OTM Option Price Calibration RMSE Over Time (%s)', data.leg1Code));
        legend('Location', 'northeast', 'NumColumns', 3, 'FontSize', 8);
        saveFigIfRequested(f3, saveDir, 'fig3_calibration_quality.png');
    end

    %% FIGURE 4: Daily Squared P&L Spikes & All-Model Ranking Bar Charts
    f4 = figure('Name', 'Step A Impact: Delta vs Delta-Vega & All-Model Ranking', 'Color', 'w', ...
                'Position', [170, 130, 1280, 780], 'Visible', figVis);

    subplot(2, 2, [1, 2]); hold on;
    for i = 1:nModels
        r = allResults.(modelList{i});
        plot(dates(2:end), r.sqDailyPnL(2:end), 'Color', colors(i,:), ...
            'LineWidth', 1.2, 'DisplayName', upper(modelList{i}));
    end
    grid on; xlabel('Date'); ylabel('Squared Daily P&L [(USD/bbl)^2]');
    title(sprintf('Daily Squared P&L Timeline (%s — All %d Models)', hTag, nModels), 'Interpreter', 'none');
    legend('Location', 'northeast', 'NumColumns', 5, 'FontSize', 8);

    labels   = upper(modelList);
    mseCurr  = zeros(nModels, 1);
    kputVals = zeros(nModels, 1);
    for i = 1:nModels
        mseCurr(i)  = allResults.(modelList{i}).mse_per_bbl;
        kputVals(i) = allResults.(modelList{i}).K_put;
    end

    subplot(2, 2, 3);
    if ~isempty(refDeltaResults)
        mseDelta = zeros(nModels, 1);
        for i = 1:nModels
            mseDelta(i) = refDeltaResults.(modelList{i}).mse_per_bbl;
        end
        bar([mseDelta, mseCurr]);
        set(gca, 'XTick', 1:nModels, 'XTickLabel', labels);
        xtickangle(25);
        ylabel('(1/T) \Sigma P&L_t^2 [(USD/bbl)^2]');
        title('Step A Impact: Pure Delta vs Delta+Vega MSE (Lower Wins)');
        legend({'Pure Delta (Futures)', 'Delta + Vega (Futures + Opt)'}, 'Location', 'northwest');
    else
        bar(mseCurr, 'FaceColor', [0.20 0.45 0.75]);
        set(gca, 'XTick', 1:nModels, 'XTickLabel', labels);
        xtickangle(25);
        ylabel('(1/T) \Sigma P&L_t^2 [(USD/bbl)^2]');
        title('Competition Metric by Model (Lower Wins)');
    end
    grid on;

    subplot(2, 2, 4);
    bar(kputVals, 'FaceColor', [0.80 0.35 0.20]);
    set(gca, 'XTick', 1:nModels, 'XTickLabel', labels);
    xtickangle(25);
    ylim([min(kputVals) - 1.0, max(kputVals) + 0.8]);
    ylabel('Zero-Cost Strike K_{put} [USD/bbl]');
    title('Inception Zero-Cost K_{put} by Model (K_{call} = 78)');
    grid on;
    saveFigIfRequested(f4, saveDir, 'fig4_stepA_ranking.png');
end

%% ========================================================================
%  MODE 2: SINGLE-MODEL DIAGNOSTIC PLOTS
%  ========================================================================
function plotSingleModel(data, results, config, saveDir)
    dates  = data.dates;
    figVis = 'on';
    if ~usejava('desktop'), figVis = 'off'; end
    f1 = figure('Name', ['Hedging Evolution - ' results.displayName], 'Color', 'w', ...
                'Position', [100, 100, 1150, 760], 'Visible', figVis);

    subplot(3, 1, 1);
    plot(dates, results.portVal, 'b-', 'LineWidth', 2.0);
    yline(0, 'k:'); grid on; ylabel('Portfolio Value \Pi(t) [USD/bbl]');
    title(sprintf('%s | Hedge: %s | MSE=%.5f (%.1f%% Red.)', ...
        results.displayName, config.hedgeMethod, results.mse_per_bbl, results.var_reduction_pct), ...
        'Interpreter', 'none');

    subplot(3, 1, 2);
    plot(dates, results.phi1, 'b-', 'LineWidth', 1.6); hold on;
    plot(dates, results.phi2, 'r--', 'LineWidth', 1.4);
    if any(abs(results.psi1) > 1e-8)
        plot(dates, results.psi1, 'm-.', 'LineWidth', 1.5);
        legend({['\phi_1 (' data.leg1Code ')'], ['\phi_2 (' data.leg2Code ')'], '\psi_1 (Option Vega Hedge)'}, 'Location', 'best');
    else
        legend({['\phi_1 (' data.leg1Code ')'], ['\phi_2 (' data.leg2Code ')']}, 'Location', 'best');
    end
    yline(0, 'k:'); grid on; ylabel('Position');
    title('Dynamic Hedge Holdings (\phi_1, \phi_2, \psi_1)');

    subplot(3, 1, 3);
    plot(dates(2:end), results.sqDailyPnL(2:end), 'b-', 'LineWidth', 1.4); hold on;
    plot(dates(2:end), results.runningMSE(2:end), 'r-', 'LineWidth', 1.8);
    grid on; xlabel('Date'); ylabel('[(USD/bbl)^2]');
    title('Daily Squared P&L_t^2 and Running Mean Squared P&L');
    legend({'Daily P&L_t^2', 'Running Mean (1/t) \Sigma P&L_s^2'}, 'Location', 'northeast');
    saveFigIfRequested(f1, saveDir, sprintf('fig_single_%s.png', results.modelName));
end

function colors = buildDistinctPalette(n)
    base = [ ...
        0.0000, 0.4470, 0.7410;  % Blue (Black-76)
        0.8500, 0.3250, 0.0980;  % Orange-Red (Merton)
        0.9290, 0.6940, 0.1250;  % Gold (Kou)
        0.4940, 0.1840, 0.5560;  % Purple (VG)
        0.4660, 0.6740, 0.1880;  % Green (NIG)
        0.3010, 0.7450, 0.9330;  % Cyan (CGMY)
        0.6350, 0.0780, 0.1840;  % Crimson (ATS)
        0.1000, 0.1000, 0.1000;  % Charcoal (Heston)
        0.9000, 0.2500, 0.6500;  % Magenta (Bates)
        0.0000, 0.5000, 0.5000]; % Teal (BNS)
    if n <= size(base, 1)
        colors = base(1:n, :);
    else
        colors = [base; lines(n - size(base, 1))];
    end
end

function saveFigIfRequested(figHandle, saveDir, fileName)
    if ~isempty(saveDir)
        try theme(figHandle, 'light'); catch, end
        set(figHandle, 'Color', 'w');
        outPath = fullfile(saveDir, fileName);
        exportgraphics(figHandle, outPath, 'Resolution', 150, 'BackgroundColor', 'white');
    end
end
