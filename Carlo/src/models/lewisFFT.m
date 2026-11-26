function varargout = lewisFFT(mode, varargin)
%LEWISFFT Unified Fourier pricing & FFT-CDF simulation engine (Note (3), Sec. 2.3-2.4).
%
%   Implements the Lewis (2001) FFT option pricer and FFT-CDF inversion
%   simulator from Prof. Michele Azzone's Computational Finance notes:
%
%   MODES:
%     1. Call Option Pricing (Section 2.3, Theorem 2.6 with alpha = -1/2):
%        C = lewisFFT('call', logCF_fn, F, K_vec)
%        C = lewisFFT('call', logCF_fn, F, K_vec, N, du)
%        Computes undiscounted Call prices F * E[(exp(X_T) - K/F)^+] for a
%        vector of strikes K_vec given the log-characteristic function
%        logCF_fn(u) = log(E[exp(i*u*X_T)]) with martingale condition logCF(-i) = 0.
%
%     2. CDF Grid Construction (Section 2.4, Gil-Pelaez / Lewis alpha = 0):
%        [k_mono, P_mono] = lewisFFT('cdf', logCF_fn, dt)
%        [k_mono, P_mono] = lewisFFT('cdf', logCF_fn, dt, N)
%        Computes the strictly monotonic CDF P(X <= k) on an adaptive grid
%        for any Levy or Additive increment with log-CF logCF_fn(u).
%
%     3. Inverse-CDF Sampling (Section 2.4, p. 34-35):
%        X = lewisFFT('sample', logCF_fn, dt, U)
%        X = lewisFFT('sample', logCF_fn, dt, U, N)
%        Maps uniform draws U in (0, 1) to samples X = P^{-1}(U).
%
%   Because every model is passed only via its log-characteristic function
%   logCF_fn(u), this single engine works identically for Merton, Kou, VG,
%   NIG, CGMY, ATS, and any future CF model (Heston, Bates, BNS, ...).

    switch lower(mode)
        case 'call'
            logCF_fn = varargin{1};
            F        = varargin{2};
            K_vec    = double(varargin{3}(:));
            if numel(varargin) >= 4 && ~isempty(varargin{4}), N  = varargin{4}; else, N  = 2048; end
            if numel(varargin) >= 5 && ~isempty(varargin{5}), du = varargin{5}; else, du = 0.15; end
            varargout{1} = lewisCallFFT(logCF_fn, F, K_vec, N, du);

        case 'cdf'
            logCF_fn = varargin{1};
            dt       = varargin{2};
            if numel(varargin) >= 3 && ~isempty(varargin{3}), N = varargin{3}; else, N = 2048; end
            [k_mono, P_mono] = lewisCDFFFT(logCF_fn, dt, N);
            varargout{1} = k_mono;
            varargout{2} = P_mono;

        case 'sample'
            logCF_fn = varargin{1};
            dt       = varargin{2};
            U        = varargin{3};
            if numel(varargin) >= 4 && ~isempty(varargin{4}), N = varargin{4}; else, N = 2048; end
            [k_mono, P_mono] = lewisCDFFFT(logCF_fn, dt, N);
            if numel(k_mono) < 3
                X = zeros(size(U));
            else
                U_clamped = min(max(U, P_mono(1)), P_mono(end));
                X = interp1(P_mono, k_mono, U_clamped, 'linear');
            end
            varargout{1} = X;

        otherwise
            error('lewisFFT:UnknownMode', ...
                'Unknown mode "%s". Use ''call'', ''cdf'', or ''sample''.', mode);
    end
end

%% ========================================================================
%  1. Lewis FFT Call Pricer (alpha = -1/2, R_{-1/2, C} = 1)
%  ========================================================================
function C = lewisCallFFT(logCF_fn, F, K_vec, N, du)
    alpha  = -0.5;
    dk     = 2.0 * pi / (N * du);
    n      = (0:N-1)';
    u      = (n + 0.5) * du;
    k1     = -0.5 * N * dk;
    k_grid = k1 + n * dk;

    % Damped Fourier transform C_hat_alpha(u) = phi(u - (alpha+1)i) / ((alpha+iu)*(1+alpha+iu))
    % For alpha = -1/2: u - (alpha+1)i = u - 0.5i, denom = -(u^2 + 0.25)
    log_phi = logCF_fn(u - (alpha + 1.0) * 1i);
    c_hat   = exp(log_phi) ./ ((alpha + 1i * u) .* (1.0 + alpha + 1i * u));

    % Midpoint-rule FFT (Note (3), Section 2.3, Step 4)
    x      = exp(-1i * n * du * k1) .* c_hat;
    fft_x  = fft(x);
    c_grid = 1.0 + (exp(-alpha * k_grid) * du / pi) .* real(exp(-0.5i * du * k_grid) .* fft_x);

    % Interpolate onto requested log-moneyness k = log(K / F)
    k_q = log(K_vec / F);
    c_q = interp1(k_grid, c_grid, k_q, 'pchip');

    % Enforce no-arbitrage lower bound max(F - K, 0)
    C = F * max(c_q, max(1.0 - K_vec / F, 0.0));
end

%% ========================================================================
%  2. Lewis FFT CDF Inversion Grid (alpha = 0, R_{0, D} = 1/2)
%  ========================================================================
function [k_mono, P_mono] = lewisCDFFFT(logCF_fn, dt, N)
    % Scale Fourier step du inversely with sqrt(dt) so both 1-year and 1-day
    % increments have tight k-grid resolution and negligible truncation error.
    du     = min(max(0.15 / sqrt(max(dt, 1e-4)), 0.15), 2.5);
    dk     = 2.0 * pi / (N * du);
    n      = (0:N-1)';
    u      = (n + 0.5) * du;
    k1     = -0.5 * N * dk;
    k_grid = k1 + n * dk;

    % Digital call D(k) = 0.5 + (1/pi) int_0^inf Re[e^{-iuk} phi(u)/(i*u)] du
    % CDF P(X <= k) = 1 - D(k) = 0.5 - (1/pi) int_0^inf Re[e^{-iuk} phi(u)/(i*u)] du
    phi_u  = exp(logCF_fn(u));
    d_hat  = phi_u ./ (1i * u);
    x      = exp(-1i * n * du * k1) .* d_hat;
    fft_x  = fft(x);
    P_grid = 0.5 - (du / pi) .* real(exp(-0.5i * du * k_grid) .* fft_x);

    % Extract maximal strictly increasing interval around median (Section 2.4, p. 34)
    [~, midIdx] = min(abs(P_grid - 0.5));
    lo = midIdx;
    while lo > 1 && P_grid(lo - 1) < P_grid(lo) && P_grid(lo - 1) > 1e-7
        lo = lo - 1;
    end
    hi = midIdx;
    while hi < N && P_grid(hi + 1) > P_grid(hi) && P_grid(hi + 1) < (1.0 - 1e-7)
        hi = hi + 1;
    end

    k_mono = k_grid(lo:hi);
    P_mono = P_grid(lo:hi);
end
