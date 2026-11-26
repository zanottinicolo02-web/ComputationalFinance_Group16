function spec = cfModels(modelName)
%CFMODELS Unified registry of Characteristic-Function Levy, Additive & SV models.
%
%   spec = cfModels(modelName) returns a standardized model specification
%   struct for any model studied in Prof. Azzone's Lecture Notes (Note (3)):
%     - 'merton' : Merton Jump-Diffusion (Sec. 1.7, p. 14) [4 params]
%     - 'kou'    : Kou Double-Exponential Jump-Diffusion (Sec. 1.7, p. 14-15) [5 params]
%     - 'vg'     : Variance Gamma / LTS alpha = 0 (Sec. 1.8, p. 20) [3 params]
%     - 'nig'    : Normal Inverse Gaussian / LTS alpha = 1/2 (Sec. 1.8, p. 20) [3 params]
%     - 'cgmy'   : CGMY Tempered Stable (Sec. 1.9, p. 20) [4 params]
%     - 'ats'    : Multi-Maturity Slice-by-Slice ATS, alpha = 1/2
%                  (Sec. 4, Theorem 4.1, p. 40) [3 params/slice]
%     - 'heston' : Heston Stochastic Volatility (Sec. 5.1, pp. 45-46) [5 params]
%     - 'bates'  : Bates Stochastic Volatility + Jumps (Sec. 5.2, p. 47) [8 params]
%     - 'bns'    : BNS Non-Gaussian OU Stochastic Vol (Sec. 5.4, p. 49) [5 params]
%
%   HOW TO ADD NEW MODELS LATER:
%     Simply add ONE 'case' block below defining paramNames, p0, lb, ub,
%     isValid, logCF, scaleVol, and strip. No other files need to be created!

    key = lower(strtrim(char(modelName)));
    spec = struct();
    spec.name       = key;
    spec.isAdditive = false;
    spec.isMultiMat = false;
    spec.isStochVol = false;

    switch key
        %% 1. MERTON JUMP-DIFFUSION (Note (3), Section 1.7, p. 14)
        %  params = [sigma, lambda, mu_J, delta_J]
        case 'merton'
            spec.displayName = 'Merton Jump-Diffusion';
            spec.paramNames  = {'sigma', 'lambda', 'mu_J', 'delta_J'};
            spec.p0          = [0.20, 0.80, -0.10, 0.15];
            spec.lb          = [0.01, 0.00, -1.00, 0.01];
            spec.ub          = [2.00, 10.0,  0.50, 1.50];
            spec.isValid     = @(p, ~) all(isfinite(p)) && p(1) > 0 && p(2) >= 0 && p(4) > 0;
            spec.logCF       = @logCF_merton;
            spec.scaleVol    = @(p, s) [p(1)*s, p(2)*(s^2), p(3), p(4)];
            spec.bumpSkew    = @(p, d) [p(1), p(2), min(max(p(3) - 0.15*d, -0.95), 0.45), p(4)];
            spec.strip       = @(~, ~) [Inf, Inf];

        %% 2. KOU DOUBLE-EXPONENTIAL JUMP-DIFFUSION (Note (3), Section 1.7, p. 14-15)
        %  params = [sigma, lambda, p_up, lambda_plus, lambda_minus]
        case 'kou'
            spec.displayName = 'Kou Double-Exponential';
            spec.paramNames  = {'sigma', 'lambda', 'p_up', 'lambda_plus', 'lambda_minus'};
            spec.p0          = [0.18, 1.00, 0.30, 10.0, 6.0];
            spec.lb          = [0.01, 0.00, 0.01,  1.2, 0.5];
            spec.ub          = [2.00, 10.0, 0.99, 50.0, 50.0];
            spec.isValid     = @(p, ~) all(isfinite(p)) && p(1) > 0 && p(2) >= 0 && ...
                                       p(3) > 0 && p(3) < 1 && p(4) > 1.05 && p(5) > 0.1;
            spec.logCF       = @logCF_kou;
            spec.scaleVol    = @(p, s) [p(1)*s, p(2)*(s^2), p(3), p(4), p(5)];
            spec.bumpSkew    = @(p, d) [p(1), p(2), min(max(p(3) - 0.10*d, 0.02), 0.98), p(4), p(5)];
            spec.strip       = @(~, p) [p(5), p(4) - 1.0];

        %% 3. VARIANCE GAMMA (VG) - LTS alpha = 0 (Note (3), Section 1.8, p. 19-20)
        %  params = [sigma, kappa, eta] with mu = -sigma^2 * (eta + 1/2)
        case 'vg'
            spec.displayName = 'Variance Gamma (VG)';
            spec.paramNames  = {'sigma', 'kappa', 'eta'};
            spec.p0          = [0.25, 0.50, 2.00];
            spec.lb          = [0.05, 0.01, -4.0];
            spec.ub          = [2.00, 5.00, 25.0];
            spec.isValid     = @(p, ~) all(isfinite(p)) && p(1) > 0 && p(2) > 0 && ...
                                       (1.0 + p(2) * p(1)^2 * p(3)) > 0.05;
            spec.logCF       = @logCF_vg;
            spec.scaleVol    = @(p, s) [p(1)*s, p(2), p(3)];
            spec.bumpSkew    = @(p, d) [p(1), p(2), min(max(p(3) + 1.5*d, -3.5), 24.5)];
            spec.strip       = @(~, p) strip_lts(p, 0.0);

        %% 4. NORMAL INVERSE GAUSSIAN (NIG) - LTS alpha = 1/2 (Note (3), Section 1.8, p. 20)
        %  params = [sigma, kappa, eta] with mu = -sigma^2 * (eta + 1/2)
        case 'nig'
            spec.displayName = 'Normal Inverse Gaussian (NIG)';
            spec.paramNames  = {'sigma', 'kappa', 'eta'};
            spec.p0          = [0.25, 0.50, 2.00];
            spec.lb          = [0.05, 0.01, -4.0];
            spec.ub          = [2.00, 5.00, 25.0];
            spec.isValid     = @(p, ~) all(isfinite(p)) && p(1) > 0 && p(2) > 0 && ...
                                       (1.0 + 2.0 * p(2) * p(1)^2 * p(3)) > 0.05;
            spec.logCF       = @logCF_nig;
            spec.scaleVol    = @(p, s) [p(1)*s, p(2), p(3)];
            spec.bumpSkew    = @(p, d) [p(1), p(2), min(max(p(3) + 1.5*d, -3.5), 24.5)];
            spec.strip       = @(~, p) strip_lts(p, 0.5);

        %% 5. CGMY TEMPERED STABLE (Note (3), Section 1.9, p. 20)
        %  params = [C, G (lambda_-), M (lambda_+), Y (alpha)]
        case 'cgmy'
            spec.displayName = 'CGMY Tempered Stable';
            spec.paramNames  = {'C', 'G_lamMinus', 'M_lamPlus', 'Y_alpha'};
            spec.p0          = [1.00, 5.00, 12.0, 0.30];
            spec.lb          = [0.01, 1.10,  1.5, 0.05];
            spec.ub          = [30.0, 50.0, 50.0, 0.95];
            spec.isValid     = @(p, ~) all(isfinite(p)) && p(1) > 0 && p(2) > 0.5 && ...
                                       p(3) > 1.1 && p(4) > 0.01 && p(4) < 1.95 && abs(p(4) - 1.0) > 0.02;
            spec.logCF       = @logCF_cgmy;
            spec.scaleVol    = @(p, s) [p(1)*(s^2), p(2), p(3), p(4)];
            spec.bumpSkew    = @(p, d) [p(1), max(p(2)*(1.0 - 0.15*d), 1.15), min(p(3)*(1.0 + 0.15*d), 49.0), p(4)];
            spec.strip       = @(~, p) [p(2), p(3) - 1.0];

        %% 6. MULTI-MATURITY SLICE-BY-SLICE ATS (Note (3), Sec. 4, Thm. 4.1, p. 40)
        %  Cont & Tankov (2003) slice-by-slice calibration across all liquid
        %  maturities T_1 < ... < T_M in data.multiSurface{t} with alpha = 1/2
        %  enforcing Theorem 4.1 monotonicity: g1(T_m) >= g1(T_{m-1}),
        %  g2(T_m) >= g2(T_{m-1}), g3(T_m) >= g3(T_{m-1}).
        case 'ats'
            spec.displayName     = 'Multi-Maturity ATS (Thm. 4.1)';
            spec.isAdditive      = true;
            spec.isMultiMat      = true;
            spec.paramNames      = {'sigma_t', 'kappa_t', 'eta_t'};
            spec.p0              = [0.28, 0.40, 2.50];
            spec.lb              = [0.05, 0.01, -3.0];
            spec.ub              = [2.00, 15.0, 25.0];
            spec.isValid         = @(p, ~) all(isfinite(p)) && p(1) > 0 && p(2) > 0 && ...
                                           (1.0 + 2.0 * p(2) * p(1)^2 * p(3)) > 0.05;
            spec.logCF           = @logCF_nig;
            spec.gFuncs          = @gFuncs_ats;
            spec.evalSliceParams = @evalATSSliceParams;
            spec.scaleVol        = @(p, s) [p(1)*s, p(2), p(3)];
            spec.bumpSkew        = @(p, d) [p(1), p(2), min(max(p(3) + 1.5*d, -2.5), 24.5)];
            spec.strip           = @(~, p) strip_lts(p, 0.5);

        %% 7. HESTON STOCHASTIC VOLATILITY (Note (3), Sec. 5.1, pp. 45-46)
        %  params = [v0, kappa, eta, theta, rho]
        case 'heston'
            spec.displayName = 'Heston Stochastic Vol';
            spec.isStochVol  = true;
            spec.paramNames  = {'v0', 'kappa', 'eta', 'theta', 'rho'};
            spec.p0          = [0.06, 2.00, 0.06, 0.45, -0.55];
            spec.lb          = [0.002, 0.10, 0.005, 0.05, -0.98];
            spec.ub          = [2.50, 15.0,  2.00,  3.00,  0.95];
            spec.isValid     = @(p, ~) all(isfinite(p)) && p(1) > 0 && p(2) > 0 && ...
                                       p(3) > 0 && p(4) > 0 && abs(p(5)) < 0.995;
            spec.logCF       = @logCF_heston;
            spec.scaleVol    = @(p, s) [p(1)*(s^2), p(2), p(3)*(s^2), p(4)*s, p(5)];
            spec.bumpSkew    = @(p, d) [p(1), p(2), p(3), p(4), min(max(p(5) - 0.12*d, -0.97), 0.94)];
            spec.strip       = @(~, ~) [1.0, 1.0];

        %% 8. BATES STOCHASTIC VOL + JUMPS (Note (3), Sec. 5.2, p. 47)
        %  params = [v0, kappa, eta, theta, rho, lambda, mu_J, delta_J]
        case 'bates'
            spec.displayName = 'Bates (Heston + Jumps)';
            spec.isStochVol  = true;
            spec.paramNames  = {'v0', 'kappa', 'eta', 'theta', 'rho', 'lambda', 'mu_J', 'delta_J'};
            spec.p0          = [0.05, 2.00, 0.05, 0.40, -0.50, 0.50, -0.10, 0.15];
            spec.lb          = [0.002, 0.10, 0.005, 0.05, -0.98, 0.00, -0.80, 0.01];
            spec.ub          = [2.50, 15.0,  2.00,  3.00,  0.95, 6.00,  0.40, 1.00];
            spec.isValid     = @(p, ~) all(isfinite(p)) && p(1) > 0 && p(2) > 0 && ...
                                       p(3) > 0 && p(4) > 0 && abs(p(5)) < 0.995 && ...
                                       p(6) >= 0 && p(8) > 0;
            spec.logCF       = @logCF_bates;
            % Keep jump arrival rate lambda fixed under scaleVol and scale jump
            % amplitudes (mu_J, delta_J) linearly by s to preserve exact CRN in MC Vega.
            spec.scaleVol    = @(p, s) [p(1)*(s^2), p(2), p(3)*(s^2), p(4)*s, p(5), p(6), p(7)*s, p(8)*s];
            spec.bumpSkew    = @(p, d) [p(1), p(2), p(3), p(4), min(max(p(5) - 0.12*d, -0.97), 0.94), p(6), p(7), p(8)];
            spec.strip       = @(~, ~) [1.0, 1.0];

        %% 9. BNS NON-GAUSSIAN OU STOCHASTIC VOL (Note (3), Sec. 5.4, pp. 49-50)
        %  Exp-Compound-Poisson subordinator Z_t (rate a, jump ~ Exp(b), rho <= 0)
        %  params = [v0, lambda, a, b, rho]
        case 'bns'
            spec.displayName = 'BNS Non-Gaussian OU SV';
            spec.isStochVol  = true;
            spec.paramNames  = {'v0', 'lambda', 'a_rate', 'b_exp', 'rho'};
            spec.p0          = [0.06, 2.00, 2.00, 15.0, -1.50];
            spec.lb          = [0.002, 0.10, 0.05,  1.0, -8.00];
            spec.ub          = [2.50, 15.0, 20.0, 80.0, -0.01];
            spec.isValid     = @(p, ~) all(isfinite(p)) && p(1) > 0 && p(2) > 0 && ...
                                       p(3) > 0 && p(4) > 0 && p(5) <= 0 && (p(4) - p(5)) > 0.1;
            spec.logCF       = @logCF_bns;
            spec.scaleVol    = @(p, s) [p(1)*(s^2), p(2), p(3), p(4)/(s^2), p(5)/s];
            spec.bumpSkew    = @(p, d) [p(1), p(2), p(3), p(4), min(max(p(5) - 0.35*d, -7.8), -0.02)];
            spec.strip       = @(~, p) [p(4), p(4)];

        otherwise
            error('cfModels:UnknownModel', ...
                'Unknown CF model "%s". Supported: merton, kou, vg, nig, cgmy, ats, heston, bates, bns.', modelName);
    end

    % Unified increment log-CF for stationary Levy models: Psi_{s, t}(u)
    spec.incLogCF = @(u, t_s, t_e, p) spec.logCF(u, t_e - t_s, p);
end

%% ========================================================================
%  CUMULATIVE LOG-CHARACTERISTIC FUNCTIONS (Martingale-Normalized: logCF(-i) = 0)
%  ========================================================================
function val = logCF_merton(u, t, p)
    sig   = p(1); lam = p(2); mu_j = p(3); del_j = p(4);
    b     = -0.5 * sig^2 - lam * (exp(mu_j + 0.5 * del_j^2) - 1.0);
    psi   = 1i * u * b - 0.5 * sig^2 * (u.^2) + lam * (exp(1i * u * mu_j - 0.5 * del_j^2 * (u.^2)) - 1.0);
    val   = t * psi;
end

function val = logCF_kou(u, t, p)
    sig = p(1); lam = p(2); prob = p(3); lp = p(4); lm = p(5);
    b   = -0.5 * sig^2 - lam * (prob / (lp - 1.0) - (1.0 - prob) / (lm + 1.0));
    psi = 1i * u * b - 0.5 * sig^2 * (u.^2) + ...
          lam * (prob * 1i * u ./ (lp - 1i * u) - (1.0 - prob) * 1i * u ./ (lm + 1i * u));
    val = t * psi;
end

function val = logCF_vg(u, t, p)
    sig = p(1); kap = p(2); eta = p(3);
    phi = (1.0 / kap) * log(1.0 + kap * sig^2 * eta);
    arg = 1.0 + kap * (1i * u * (0.5 + eta) * sig^2 + 0.5 * (u.^2) * sig^2);
    val = -(t / kap) * log(arg) + 1i * u * phi * t;
end

function val = logCF_nig(u, t, p)
    sig = p(1); kap = p(2); eta = p(3);
    phi = -(1.0 / kap) * (1.0 - sqrt(1.0 + 2.0 * kap * sig^2 * eta));
    arg = 1.0 + 2.0 * kap * (1i * u * (0.5 + eta) * sig^2 + 0.5 * (u.^2) * sig^2);
    val = (t / kap) * (1.0 - sqrt(arg)) + 1i * u * phi * t;
end

function val = logCF_cgmy(u, t, p)
    C = p(1); G = p(2); M = p(3); Y = p(4);
    g_mY = gamma(-Y);
    psi0 = C * g_mY * ((M - 1i * u).^Y - M^Y + (G + 1i * u).^Y - G^Y);
    phi  = -C * g_mY * ((M - 1.0)^Y - M^Y + (G + 1.0)^Y - G^Y);
    val  = t * (psi0 + 1i * u * phi);
end

function val = logCF_heston(u, s, p)
    % Heston (1993) Little-Heston-Trap formulation (branch-cut-free, Note (3), p. 46)
    v0 = p(1); kap = p(2); eta = p(3); th = p(4); rho = p(5);
    xi     = kap - 1i * rho * th * u;
    d      = sqrt(xi.^2 + (th^2) * (u.^2 + 1i * u));
    g      = (xi - d) ./ (xi + d);
    exp_ds = exp(-d * s);
    D      = ((xi - d) / (th^2)) .* ((1.0 - exp_ds) ./ (1.0 - g .* exp_ds));
    C      = (kap * eta / (th^2)) * ((xi - d) * s - 2.0 * log((1.0 - g .* exp_ds) ./ (1.0 - g)));
    val    = C + v0 * D;
end

function val = logCF_bates(u, s, p)
    % Bates (1996): Heston + independent Merton compound Poisson jumps (Note (3), p. 47)
    lcf_h = logCF_heston(u, s, p(1:5));
    lam = p(6); mu_j = p(7); del_j = p(8);
    k_j   = exp(mu_j + 0.5 * del_j^2) - 1.0;
    lcf_j = s * (-1i * u * lam * k_j + lam * (exp(1i * u * mu_j - 0.5 * del_j^2 * (u.^2)) - 1.0));
    val   = lcf_h + lcf_j;
end

function val = logCF_bns(u, s, p)
    % BNS (2001) Non-Gaussian OU with Exp-CP subordinator (Note (3), Sec. 5.4, p. 50)
    v0 = p(1); lam = p(2); a = p(3); b = p(4); rho = p(5);
    mu_drift = -a * rho / (b - rho);
    e_ls     = exp(-lam * s);
    f1       = 1i * u * rho;
    f2       = -0.5 * (1i * u + u.^2) / lam;
    B        = b - f1 - f2;
    int_psi  = (a ./ B) .* ((f1 + f2) * s + (b / lam) * log((b - f1 - f2 * (1.0 - e_ls)) ./ (b - f1)));
    val      = 1i * u * mu_drift * s - 0.5 * (1i * u + u.^2) * v0 * ((1.0 - e_ls) / lam) + int_psi;
end

%% ========================================================================
%  THEOREM 4.1 EXISTENCE FUNCTIONS & SLICE INTERPOLATION (Note (3), p. 40)
%  ========================================================================
function g = gFuncs_ats(t, p)
    % Theorem 4.1 (p. 40) for alpha = 1/2: g1, g2, g3 must be non-decreasing in t
    sig = p(1); kap = p(2); eta = p(3);
    rad = sqrt((0.5 + eta)^2 + 1.0 / (sig^2 * kap));
    g1  =  (0.5 + eta) - rad;
    g2  = -(0.5 + eta) - rad;
    g3  = (t^2 * sig^2 / kap) * rad;
    g   = [g1, g2, g3];
end

function p_u = evalATSSliceParams(t_query, atsCurve, volScale, skewBump)
    if nargin < 3 || isempty(volScale), volScale = 1.0; end
    if nargin < 4 || isempty(skewBump), skewBump = 0.0; end
    taus = atsCurve.taus(:);
    t_c  = min(max(t_query, taus(1)), taus(end));
    if isscalar(taus)
        sig_u = atsCurve.sigs(1);
        kap_u = atsCurve.kaps(1);
        eta_u = atsCurve.etas(1);
    else
        sig_u = interp1(taus, atsCurve.sigs(:), t_c, 'linear');
        kap_u = interp1(taus, atsCurve.kaps(:), t_c, 'linear');
        eta_u = interp1(taus, atsCurve.etas(:), t_c, 'linear');
    end
    eta_b = min(max(eta_u + 1.5 * skewBump, -2.5), 24.5);
    p_u   = [sig_u * volScale, kap_u, eta_b];
end

function st = strip_lts(p, alpha)
    sig = p(1); kap = p(2); eta = p(3);
    rad = sqrt((0.5 + eta)^2 + 2.0 * (1.0 - alpha) / (sig^2 * kap));
    g1  = (0.5 + eta) - rad;
    g2  = -(0.5 + eta) - rad;
    st  = [-g1, -g2 - 1.0];
end
