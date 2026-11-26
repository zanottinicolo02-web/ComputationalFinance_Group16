function plotResults(data, results, config)
%PLOTRESULTS Plot calibration diagnostics, hedging trajectory, and P&L metrics.
%
%   plotResults(data, results, config) generates three figures covering:
%     1. Calibration Diagnostics (Inception Volatility Smiles, ATM Vol, Rho)
%     2. Hedging Performance (Portfolio Value, Structure vs Cash, Delta Positions)
%     3. Competition Deliverables (Daily P&L and Squared Daily P&L_t^2)

    dates = data.dates;
    calib0 = results.calibHistory{1};

    %% Figure 1: Calibration Diagnostics
    figure('Name', 'Calibration Diagnostics', 'Color', 'w', 'Position', [100, 100, 1100, 720]);

    % 1A: Inception Implied Volatility Smiles
    subplot(2, 2, 1);
    K_grid = linspace(40, 90, 200)';
    plot(K_grid, 100 * calib0.sigma1_fn(K_grid), 'b-', 'LineWidth', 1.8); hold on;
    plot(calib0.smile1.strikes, 100 * calib0.smile1.iv, 'bo', 'MarkerFaceColor', 'b');
    plot(K_grid, 100 * calib0.sigma2_fn(K_grid), 'r--', 'LineWidth', 1.8);
    plot(calib0.smile2.strikes, 100 * calib0.smile2.iv, 'rs', 'MarkerFaceColor', 'r');
    xline(results.K_put, 'k:', sprintf('K_{put}=%.2f', results.K_put), 'LineWidth', 1.2);
    xline(results.K_call, 'k--', sprintf('K_{call}=%.1f', results.K_call), 'LineWidth', 1.2);
    grid on;
    xlabel('Strike K [USD/bbl]');
    ylabel('Implied Volatility [%]');
    title(sprintf('Inception Smiles (%s) - Vol Model: %s', ...
        datestr(dates(1), 'dd-mmm-yyyy'), config.volModel), 'Interpreter', 'none');
    legend({[data.leg1Code ' Fit'], [data.leg1Code ' Mkt'], ...
            [data.leg2Code ' Fit'], [data.leg2Code ' Mkt']}, 'Location', 'northeast');

    % 1B: Futures Price Trajectories
    subplot(2, 2, 2);
    plot(dates, data.F1_settle, 'b-', 'LineWidth', 1.5); hold on;
    plot(dates, data.F2_settle, 'r--', 'LineWidth', 1.5);
    yline(results.K_call, 'k--', 'K_{call} = 78');
    yline(results.K_put, 'k:', sprintf('K_{put} = %.2f', results.K_put));
    grid on;
    xlabel('Date');
    ylabel('Futures Settlement Price [USD/bbl]');
    title('Underlying Futures Trajectories & Collar Strikes');
    legend({data.leg1Code, data.leg2Code, 'K_{call}', 'K_{put}'}, 'Location', 'best');

    % 1C: Daily Calibrated ATM Implied Volatilities
    subplot(2, 2, 3);
    plot(dates, 100 * results.atmVol1, 'b-', 'LineWidth', 1.4); hold on;
    plot(dates, 100 * results.atmVol2, 'r--', 'LineWidth', 1.4);
    grid on;
    xlabel('Date');
    ylabel('ATM Implied Vol [%]');
    title('Calibrated ATM Implied Volatility Evolution');
    legend({data.leg1Code, data.leg2Code}, 'Location', 'best');

    % 1D: Historical Correlation Trajectory
    subplot(2, 2, 4);
    plot(dates, results.rho, 'k-', 'LineWidth', 1.5);
    ylim([min(0.95, min(results.rho) - 0.01), 1.001]);
    grid on;
    xlabel('Date');
    ylabel('Correlation \rho');
    title(sprintf('Estimated Correlation \\rho (%s)', config.corrMethod), 'Interpreter', 'tex');

    %% Figure 2: Portfolio Value & Hedge Positions
    figure('Name', 'Hedging Performance & Positions', 'Color', 'w', 'Position', [140, 140, 1100, 720]);

    % 2A: Portfolio Value vs Unhedged Structure Value (USD/bbl)
    subplot(2, 1, 1);
    plot(dates, results.V_struct, 'r--', 'LineWidth', 1.4); hold on;
    plot(dates, -results.cash, 'g:', 'LineWidth', 1.4);
    plot(dates, results.portVal, 'b-', 'LineWidth', 2.0);
    yline(0, 'k-');
    grid on;
    xlabel('Date');
    ylabel('Value [USD/bbl]');
    title(sprintf('Portfolio Value Trajectory (Hedge: %s | MSE Reduction: %.1f%%)', ...
        config.hedgeMethod, results.var_reduction_pct), 'Interpreter', 'none');
    legend({'Structure V_{struct}(t) (Unhedged)', 'Replicating Hedge (-Cash)', ...
            'Net Hedged Portfolio \Pi(t)'}, 'Location', 'northwest');

    % 2B: Futures Hedge Ratios phi_1(t) and phi_2(t)
    subplot(2, 1, 2);
    plot(dates, results.phi1, 'b-', 'LineWidth', 1.6); hold on;
    plot(dates, results.phi2, 'r--', 'LineWidth', 1.6);
    yline(0, 'k:');
    grid on;
    xlabel('Date');
    ylabel('Hedge Position \phi_i [bbl / bbl]');
    title('Dynamic Futures Hedge Holdings (\phi_1 and \phi_2)');
    legend({['\phi_1 (' data.leg1Code ')'], ['\phi_2 (' data.leg2Code ')']}, 'Location', 'best');

    %% Figure 3: Competition Deliverable Plots (Portfolio Value & Squared Daily P&L)
    figure('Name', 'Competition Deliverable Metrics', 'Color', 'w', 'Position', [180, 180, 1050, 680]);

    subplot(2, 1, 1);
    plot(dates, results.portVal_total / 1e6, 'b-', 'LineWidth', 1.8); hold on;
    plot(dates, results.V_struct_total / 1e6, 'r--', 'LineWidth', 1.2);
    yline(0, 'k:');
    grid on;
    xlabel('Date');
    ylabel('Portfolio Value [Million USD]');
    title('Total Portfolio Value (Notional = 1,000,000 bbl)');
    legend({'Hedged Portfolio', 'Unhedged Short Collar'}, 'Location', 'northwest');

    subplot(2, 1, 2);
    plot(dates(2:end), results.sqDailyPnL(2:end), 'b-', 'LineWidth', 1.5);
    grid on;
    xlabel('Date');
    ylabel('Squared Daily P&L [(USD/bbl)^2]');
    title(sprintf('Squared Daily P&L  |  (1/T) \\Sigma P&L_t^2 = %.6f (USD/bbl)^2  [%.3e USD^2]', ...
        results.mse_per_bbl, results.mse_total));
end

