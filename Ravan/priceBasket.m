function [price, details] = priceBasket(t, K, isCall, f1_val, f2_val, data, calib, config)
%PRICEBASKET Price an Asian basket Call or Put on two futures legs on day t.
%
%   [price, details] = priceBasket(t, K, isCall, f1_val, f2_val, data, calib, config)
%   implements both baseline pricing engines from Note Hedging.pdf:
%     - 'moment_matching' : Closed-form Turnbull-Wakeman / Levy lognormal
%                           moment matching (Section 6, Eqs. 15-24).
%     - 'monte_carlo'     : Exact vectorized Bivariate Lognormal Monte Carlo
%                           with Kemna-Vorst geometric-average control variate
%                           (Section 5, Eqs. 7-11).
%     - 'custom_mc'       : [Extension Hook] Simulation pricer for ATS,
%                           Jump-HJM, or Heston stochastic vol models.
%
%   INPUTS:
%       t      : integer day index in data.dates (1 <= t <= N_days)
%       K      : scalar strike price of the basket option
%       isCall : logical (true = Call, false = Put)
%       f1_val : current/bumped Leg 1 futures price f(t, tau_1)
%       f2_val : current/bumped Leg 2 futures price f(t, tau_2)
%       data   : dataset struct from loadData()
%       calib  : calibrated model struct on day t from calibrateModel()
%       config : configuration struct

    d_t       = data.dates(t);
    df_T      = data.df_settle(t);
    fix_dates = data.fixing_dates;
    fix_idx   = data.fixing_idx;
    N_fix     = data.N_fix;
    n1        = data.n1;
    n2        = data.n2;
    w1        = data.w1;
    w2        = data.w2;

    % Year fraction theta_k = max(t_k - t, 0) / 365 for each fixing date k
    theta = max(days(fix_dates - d_t) / 365.0, 0.0);

    % Equivalent single-name strikes via sticky log-moneyness (Section 4.2, Eq. 5)
    F_avg = w1 * f1_val + w2 * f2_val;
    if K > 0 && F_avg > 0
        K1 = f1_val * (K / F_avg);
        K2 = f2_val * (K / F_avg);
    else
        K1 = f1_val;
        K2 = f2_val;
    end

    % Read volatilities off each leg's calibrated smile (Eq. 6)
    sig1 = calib.sigma1_fn(K1);
    sig2 = calib.sigma2_fn(K2);
    rho  = calib.rho;

    % Conditional expectation M1 = E^Q[A | F_t] accounting for realized fixings (t_k <= d_t)
    isFixed = (fix_dates <= d_t);
    M1 = 0.0;
    for k = 1:N_fix
        if isFixed(k)
            idx_k = fix_idx(k);
            if k <= n1
                M1 = M1 + data.F1_settle(idx_k) / N_fix;
            else
                M1 = M1 + data.F2_settle(idx_k) / N_fix;
            end
        else
            if k <= n1
                M1 = M1 + f1_val / N_fix;
            else
                M1 = M1 + f2_val / N_fix;
            end
        end
    end

    switch lower(config.pricerMethod)
        case 'moment_matching'
            [price, details] = priceMomentMatching( ...
                K, isCall, f1_val, f2_val, sig1, sig2, rho, ...
                theta, n1, n2, w1, w2, M1, df_T);

        case 'monte_carlo'
            [price, details] = priceBivariateMC( ...
                K, isCall, f1_val, f2_val, sig1, sig2, rho, ...
                theta, isFixed, fix_idx, n1, n2, N_fix, df_T, data, config);

        case 'custom_mc'
            % [EXTENSION HOOK] Simulate paths under ATS / Jump-HJM / Heston
            % using calib.forwardParams and calib.volParams.
            error('priceBasket:NotImplemented', ...
                'Custom MC pricer hook: plug your ATS / Jump-HJM / Heston simulator here.');

        otherwise
            error('priceBasket:UnknownMethod', ...
                'Unknown pricerMethod "%s".', config.pricerMethod);
    end

    details.K1   = K1;
    details.K2   = K2;
    details.sig1 = sig1;
    details.sig2 = sig2;
    details.rho  = rho;
    details.M1   = M1;
end

%% ========================================================================
%  ENGINE 1: Closed-Form Moment Matching (Note Hedging.pdf Section 6)
%  ========================================================================
function [price, details] = priceMomentMatching( ...
    K, isCall, f1_val, f2_val, sig1, sig2, rho, theta, n1, n2, w1, w2, M1, df_T)

    norm_cdf = @(x) 0.5 * erfc(-x / sqrt(2));

    theta1 = theta(1:n1);
    theta2 = theta(n1+1:n1+n2);

    % Eq. (15): Var(A_1) single-sum closed form
    k_vec  = (1:n1)';
    var_A1 = (f1_val^2 / n1^2) * sum((2 * (n1 - k_vec) + 1) .* (exp(sig1^2 * theta1) - 1.0));

    % Eq. (16): Var(A_2) single-sum closed form
    j_vec  = (1:n2)';
    var_A2 = (f2_val^2 / n2^2) * sum((2 * (n2 - j_vec) + 1) .* (exp(sig2^2 * theta2) - 1.0));

    % Eq. (18): Cov(A_1, A_2) collapsed single sum over K_1 fixings
    cov_A1A2 = (f1_val * f2_val / n1) * sum(exp(rho * sig1 * sig2 * theta1) - 1.0);

    % Eq. (20): Total basket variance Var(A)
    var_A = w1^2 * var_A1 + w2^2 * var_A2 + 2 * w1 * w2 * cov_A1A2;
    var_A = max(var_A, 0.0);

    % Degenerate / matured case (Var(A) == 0 or K <= 0)
    if var_A <= 1e-14 || K <= 0
        if isCall
            price = df_T * max(M1 - K, 0.0);
        else
            price = df_T * max(K - M1, 0.0);
        end
        details = struct('var_A1', var_A1, 'var_A2', var_A2, ...
            'cov_A1A2', cov_A1A2, 'var_A', var_A, 'v', 0.0, 'se', 0.0);
        return;
    end

    % Eq. (21): Lognormal variance parameter v^2
    v2 = log(1.0 + var_A / (M1^2));
    v  = sqrt(v2);

    % Eqs. (22)-(24): Closed-form Black-76 basket price
    d1 = (log(M1 / K) + 0.5 * v2) / v;
    d2 = d1 - v;

    if isCall
        price = df_T * (M1 * norm_cdf(d1) - K * norm_cdf(d2));
    else
        price = df_T * (K * norm_cdf(-d2) - M1 * norm_cdf(-d1));
    end

    details = struct('var_A1', var_A1, 'var_A2', var_A2, ...
        'cov_A1A2', cov_A1A2, 'var_A', var_A, 'v', v, 'd1', d1, 'd2', d2, 'se', 0.0);
end

%% ========================================================================
%  ENGINE 2: Vectorized Bivariate Lognormal Monte Carlo + Control Variate
%            (Note Hedging.pdf Section 5, Eqs. 7-11)
%  ========================================================================
function [price, details] = priceBivariateMC( ...
    K, isCall, f1_val, f2_val, sig1, sig2, rho, theta, isFixed, fix_idx, n1, n2, N_fix, df_T, data, config)

    norm_cdf = @(x) 0.5 * erfc(-x / sqrt(2));

    % Separate already-fixed dates (theta == 0) from future dates (theta > 0)
    remMask = ~isFixed & (theta > 0);
    if ~any(remMask) || K <= 0
        % All fixings are already known
        f_realized = zeros(N_fix, 1);
        for k = 1:N_fix
            idx_k = fix_idx(k);
            f_realized(k) = (k <= n1) * data.F1_settle(idx_k) + (k > n1) * data.F2_settle(idx_k);
        end
        A_val = mean(f_realized);
        if isCall
            price = df_T * max(A_val - K, 0.0);
        else
            price = df_T * max(K - A_val, 0.0);
        end
        details = struct('se', 0.0, 'price_raw', price);
        return;
    end

    % Build drift vector mu (Eq. 7) and covariance matrix Sigma (Eq. 8)
    leg_id = [ones(n1, 1); 2 * ones(n2, 1)];
    sig_vec = (leg_id == 1) * sig1 + (leg_id == 2) * sig2;
    mu = -0.5 * (sig_vec.^2) .* theta;

    Sigma = zeros(N_fix, N_fix);
    for a = 1:N_fix
        for b = 1:N_fix
            if a <= n1 && b <= n1
                Sigma(a, b) = sig1^2 * min(theta(a), theta(b));
            elseif a > n1 && b > n1
                Sigma(a, b) = sig2^2 * min(theta(a), theta(b));
            elseif a <= n1 && b > n1
                Sigma(a, b) = rho * sig1 * sig2 * theta(a);
            else
                Sigma(a, b) = rho * sig1 * sig2 * theta(b);
            end
        end
    end

    % Simulate active (unfixed) indices via Cholesky factor L
    actIdx = find(remMask);
    nAct   = numel(actIdx);
    Sigma_act = Sigma(actIdx, actIdx) + 1e-12 * eye(nAct);
    L = chol(Sigma_act, 'lower');

    % Use common random numbers (CRN) for smooth finite-difference Greeks
    rng(config.mcSeed, 'twister');
    M = config.mcPaths;
    Z = randn(nAct, M);
    eta_act = mu(actIdx) + L * Z;

    % Reconstruct futures prices across all N_fix dates and M paths (Step 4)
    f_paths = zeros(N_fix, M);
    for k = 1:N_fix
        if ~remMask(k)
            idx_k = fix_idx(k);
            if k <= n1
                f_paths(k, :) = data.F1_settle(idx_k);
            else
                f_paths(k, :) = data.F2_settle(idx_k);
            end
        end
    end
    f0_act = (leg_id(actIdx) == 1) * f1_val + (leg_id(actIdx) == 2) * f2_val;
    f_paths(actIdx, :) = f0_act .* exp(eta_act);

    % Arithmetic average A^(j) (Step 5)
    A_arith = mean(f_paths, 1);
    if isCall
        pay_arith = max(A_arith - K, 0.0);
    else
        pay_arith = max(K - A_arith, 0.0);
    end
    price_raw = df_T * mean(pay_arith);
    se_raw    = df_T * std(pay_arith) / sqrt(M);

    % Exact Geometric-Average Control Variate when all fixings are active (Section 5.3)
    if config.useControlVariate && all(remMask)
        G_geo = exp(mean(log(f_paths), 1));
        if isCall
            pay_geo = max(G_geo - K, 0.0);
        else
            pay_geo = max(K - G_geo, 0.0);
        end

        % Exact Black-76 geometric average price C_geo / P_geo (Eq. 10-11)
        a_vec = ones(N_fix, 1) / N_fix;
        w1 = n1 / N_fix;
        w2 = n2 / N_fix;
        m_G = w1 * log(f1_val) + w2 * log(f2_val) + a_vec' * mu;
        v_G = sqrt(a_vec' * Sigma * a_vec);
        F_geo = exp(m_G + 0.5 * v_G^2);
        d1_g = (log(F_geo / K) + 0.5 * v_G^2) / v_G;
        d2_g = d1_g - v_G;

        if isCall
            exact_geo = df_T * (F_geo * norm_cdf(d1_g) - K * norm_cdf(d2_g));
        else
            exact_geo = df_T * (K * norm_cdf(-d2_g) - F_geo * norm_cdf(-d1_g));
        end

        diff_cv = pay_arith - pay_geo;
        price   = df_T * mean(diff_cv) + exact_geo;
        se      = df_T * std(diff_cv) / sqrt(M);
    else
        price = price_raw;
        se    = se_raw;
    end

    details = struct('se', se, 'price_raw', price_raw, 'se_raw', se_raw);
end

