function [iv, vega, modelPrice] = asayImpliedVol(mktPrice, F, K, tau, isCall)
%ASAYIMPLIEDVOL Invert undiscounted Black-76 formula (Asay, 1982; r = 0).
%
%   [iv, vega, modelPrice] = asayImpliedVol(mktPrice, F, K, tau, isCall)
%   extracts implied volatilities for futures-style margined options where
%   D(t, T) = 1 (r = 0). Under futures-style margining and r = 0, early
%   exercise is never optimal (Asay, 1982; Lieu, 1990), so the American
%   early-exercise premium is zero.
%
%   INPUTS:
%       mktPrice : (M x 1) vector of option market prices (mid or settle)
%       F        : scalar or (M x 1) underlying futures price f(t, tau)
%       K        : (M x 1) vector of option strikes
%       tau      : scalar or (M x 1) time to option expiry in years
%       isCall   : (M x 1) logical vector (true = Call, false = Put)
%
%   OUTPUTS:
%       iv         : (M x 1) implied volatilities (NaN if no-arbitrage violated)
%       vega       : (M x 1) undiscounted Black-76 Vega dPrice/dSigma
%       modelPrice : (M x 1) fitted option price at 'iv'

    mktPrice = double(mktPrice(:));
    K        = double(K(:));
    isCall   = logical(isCall(:));
    M        = numel(K);

    if isscalar(F),   F   = repmat(double(F), M, 1);   else, F   = double(F(:));   end
    if isscalar(tau), tau = repmat(double(tau), M, 1); else, tau = double(tau(:)); end

    iv         = nan(M, 1);
    vega       = nan(M, 1);
    modelPrice = nan(M, 1);

    % Standard normal CDF and PDF (toolbox-free)
    norm_cdf = @(x) 0.5 * erfc(-x / sqrt(2));
    norm_pdf = @(x) exp(-0.5 * x.^2) / sqrt(2 * pi);

    for i = 1:M
        p_i   = mktPrice(i);
        f_i   = F(i);
        k_i   = K(i);
        tau_i = tau(i);
        c_i   = isCall(i);

        if isnan(p_i) || p_i <= 0 || tau_i <= 1e-6 || f_i <= 0 || k_i <= 0
            continue;
        end

        % Intrinsic value and upper no-arbitrage bound (undiscounted, D = 1)
        if c_i
            intrinsic = max(f_i - k_i, 0);
            upperBnd  = f_i;
        else
            intrinsic = max(k_i - f_i, 0);
            upperBnd  = k_i;
        end

        if p_i <= intrinsic + 1e-7 || p_i >= upperBnd - 1e-7
            continue;
        end

        % Initial guess (Brenner-Subrahmanyam style approximation)
        sigma = max(min(sqrt(2 * pi / tau_i) * (p_i / f_i), 1.5), 0.15);
        sigLow  = 1e-4;
        sigHigh = 5.0;

        % Safeguard Newton-Raphson + Bisection hybrid
        for iter = 1:60
            s_sqrt = sigma * sqrt(tau_i);
            d1 = (log(f_i / k_i) + 0.5 * sigma^2 * tau_i) / s_sqrt;
            d2 = d1 - s_sqrt;

            if c_i
                price_est = f_i * norm_cdf(d1) - k_i * norm_cdf(d2);
            else
                price_est = k_i * norm_cdf(-d2) - f_i * norm_cdf(-d1);
            end

            diff_p = price_est - p_i;
            if abs(diff_p) < 1e-9
                break;
            end

            if diff_p > 0
                sigHigh = sigma;
            else
                sigLow = sigma;
            end

            vega_i = f_i * sqrt(tau_i) * norm_pdf(d1);
            if vega_i > 1e-8
                sigma_new = sigma - diff_p / vega_i;
                if sigma_new > sigLow && sigma_new < sigHigh
                    sigma = sigma_new;
                else
                    sigma = 0.5 * (sigLow + sigHigh);
                end
            else
                sigma = 0.5 * (sigLow + sigHigh);
            end
        end

        % Final evaluation at converged sigma
        s_sqrt = sigma * sqrt(tau_i);
        d1 = (log(f_i / k_i) + 0.5 * sigma^2 * tau_i) / s_sqrt;
        d2 = d1 - s_sqrt;

        iv(i)   = sigma;
        vega(i) = f_i * sqrt(tau_i) * norm_pdf(d1);
        if c_i
            modelPrice(i) = f_i * norm_cdf(d1) - k_i * norm_cdf(d2);
        else
            modelPrice(i) = k_i * norm_cdf(-d2) - f_i * norm_cdf(-d1);
        end
    end
end

