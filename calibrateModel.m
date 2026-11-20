function calib = calibrateModel(t, data, config, prevCalib)
%CALIBRATEMODEL Daily calibration of volatility smiles, correlation, and dynamics.
%
%   calib = calibrateModel(t, data, config, prevCalib) calibrates the
%   chosen volatility model (config.volModel), correlation estimator
%   (config.corrMethod), and forward dynamics (config.forwardModel) using
%   only market information available up to rebalancing day t.
%
%   ARCHITECTURE FOR EXTENSIONS:
%     - Volatility models (switch config.volModel):
%         'flat_atm'          : Constant ATM implied vol per leg (simplest)
%         'smile_interp'      : Interpolated Black-76 implied vol smile per leg
%         'svi'               : [Hook] SVI parametric smile parameterization
%         'stoch_vol_heston'  : [Hook] Spot-driven Heston stochastic volatility
%     - Forward dynamics (switch config.forwardModel):
%         'black76_lognormal' : Driftless bivariate GBM (baseline)
%         'ats'               : [Hook] Affine Term Structure forward model
%         'hjm_jump'          : [Hook] Jump-extended HJM forward curve model
%     - Correlation methods (switch config.corrMethod):
%         'expanding'         : Expanding historical window of log-returns
%         'rolling'           : Rolling window of length config.corrWindow
%         'ewma'              : Exponentially Weighted Moving Average
%         'constant'          : Fixed correlation config.rhoPrior

    if nargin < 4
        prevCalib = [];
    end

    F1_t = data.F1_settle(t);
    F2_t = data.F2_settle(t);

    calib = struct();
    calib.t    = t;
    calib.date = data.dates(t);
    calib.F1   = F1_t;
    calib.F2   = F2_t;

    %% ====================================================================
    %  1. EXTRACT LIQUID OPTION QUOTES & UNDISCOUNTED BLACK-76 IMPLIED VOLS
    %  ====================================================================
    prevSmile1 = [];
    prevSmile2 = [];
    if ~isempty(prevCalib)
        prevSmile1 = prevCalib.smile1;
        prevSmile2 = prevCalib.smile2;
    end

    smile1 = extractLegSmile(data.optChain1{t}, F1_t, config, prevSmile1);
    smile2 = extractLegSmile(data.optChain2{t}, F2_t, config, prevSmile2);

    calib.smile1  = smile1;
    calib.smile2  = smile2;
    calib.atmVol1 = smile1.atmVol;
    calib.atmVol2 = smile2.atmVol;

    %% ====================================================================
    %  2. VOLATILITY SMILE / SURFACE MODELIZATION
    %  ====================================================================
    switch lower(config.volModel)
        case 'flat_atm'
            % Simplest model: constant ATM volatility across all strikes
            calib.sigma1_fn = @(K) repmat(smile1.atmVol, size(K));
            calib.sigma2_fn = @(K) repmat(smile2.atmVol, size(K));
            calib.volParams = struct('type', 'flat_atm');

        case 'smile_interp'
            % Baseline smile model: shape-preserving interpolation on liquid
            % strikes with flat clamping on the wings
            calib.sigma1_fn = buildSmileInterpolant(smile1.strikes, smile1.iv, config.interpMethod);
            calib.sigma2_fn = buildSmileInterpolant(smile2.strikes, smile2.iv, config.interpMethod);
            calib.volParams = struct('type', 'smile_interp', 'method', config.interpMethod);

        case 'svi'
            % [EXTENSION HOOK] Fit SVI parameters (a, b, rho_svi, m, s) to
            % total implied variance w(k) = iv.^2 * tau on log-moneyness k = log(K/F)
            error('calibrateModel:NotImplemented', ...
                'SVI smile parameterization hook: plug your SVI fitter here.');

        case 'stoch_vol_heston'
            % [EXTENSION HOOK] Jointly calibrate Heston parameters
            % (v0, kappa, theta_v, xi, rho_sv) + cost-of-carry Lambda(t, tau_i)
            % to smile1 and smile2. Store in calib.volParams.
            error('calibrateModel:NotImplemented', ...
                'Heston stochastic vol hook: plug your Heston calibration here.');

        otherwise
            error('calibrateModel:UnknownVolModel', ...
                'Unknown volModel "%s".', config.volModel);
    end

    %% ====================================================================
    %  3. FORWARD RETURNS MODEL & HISTORICAL CORRELATION ESTIMATION
    %  ====================================================================
    switch lower(config.forwardModel)
        case 'black76_lognormal'
            % Baseline driftless lognormal futures (Eq. 1-2 in Note Hedging.pdf):
            % Innovations are daily log-returns d ln f(u, tau_i)
            if t >= 2
                r1_hist = diff(log(data.F1_settle(1:t)));
                r2_hist = diff(log(data.F2_settle(1:t)));
            else
                r1_hist = [];
                r2_hist = [];
            end
            calib.forwardParams = struct('type', 'black76_lognormal');

        case 'ats'
            % [EXTENSION HOOK] Affine Term Structure (ATS) model for futures:
            % Filter latent state factors X_t and extract model innovations
            % r1_hist, r2_hist and factor loadings for pricing/hedging.
            error('calibrateModel:NotImplemented', ...
                'ATS forward model hook: plug your ATS state filter & parameter estimator here.');

        case 'hjm_jump'
            % [EXTENSION HOOK] Jump-extended HJM forward model (Benth et al., 2008):
            % Calibrate diffusion + jump intensity/distribution parameters.
            error('calibrateModel:NotImplemented', ...
                'Jump-HJM forward model hook: plug your HJM-Jump calibration here.');

        otherwise
            error('calibrateModel:UnknownForwardModel', ...
                'Unknown forwardModel "%s".', config.forwardModel);
    end

    % Estimate correlation rho between Leg 1 and Leg 2 innovations up to day t
    calib.rho = estimateCorrelation(r1_hist, r2_hist, config, prevCalib);
end

%% ========================================================================
%  HELPER 1: Filter Option Chain & Extract Undiscounted Black-76 Implied Vols
%  ========================================================================
function smile = extractLegSmile(chain, F_t, config, prevSmile)
    if isempty(chain) || isempty(chain.strike) || chain.tau <= 1e-4
        if ~isempty(prevSmile)
            smile = prevSmile;
            return;
        else
            error('calibrateModel:EmptyChain', 'Option chain is empty on initial date.');
        end
    end

    K      = chain.strike;
    isCall = chain.isCall;
    bid    = chain.bid;
    ask    = chain.ask;
    mid    = chain.mid;
    settle = chain.settle;
    tau    = chain.tau;

    % Choose price source ('mid' or 'settle')
    if strcmpi(config.quoteSource, 'settle')
        mktPrice = settle;
    else
        mktPrice = mid;
    end

    % Liquidity filters:
    % 1. Valid positive price and bid >= minOptionBid
    validQuote = ~isnan(mktPrice) & (mktPrice > 0) & ~isnan(bid) & (bid >= config.minOptionBid);

    % 2. Relative bid-ask spread filter: (ask - bid) / mid <= maxRelSpread
    relSpread = (ask - bid) ./ max(mid, 1e-6);
    validSpread = ~isnan(relSpread) & (relSpread >= 0) & (relSpread <= config.maxRelSpread);

    % 3. OTM selection (Puts for K <= F, Calls for K > F)
    if config.useOTMOnly
        isOTM = (~isCall & (K <= F_t)) | (isCall & (K > F_t));
    else
        isOTM = true(size(K));
    end

    keep = validQuote & validSpread & isOTM;

    if sum(keep) >= 2
        K_filt     = K(keep);
        price_filt = mktPrice(keep);
        call_filt  = isCall(keep);
        bid_filt   = bid(keep);
        ask_filt   = ask(keep);
        mid_filt   = mid(keep);

        [iv_raw, vega_raw, ~] = asayImpliedVol(price_filt, F_t, K_filt, tau, call_filt);
        validIV = ~isnan(iv_raw) & (iv_raw > 0.01) & (iv_raw < 3.0);

        K_clean    = K_filt(validIV);
        iv_clean   = iv_raw(validIV);
        vega_clean = vega_raw(validIV);
        bid_clean  = bid_filt(validIV);
        ask_clean  = ask_filt(validIV);
        mid_clean  = mid_filt(validIV);
        call_clean = call_filt(validIV);

        % Deduplicate strikes if both Call and Put were kept
        [K_unique, ia] = unique(K_clean, 'stable');
        iv_unique   = iv_clean(ia);
        vega_unique = vega_clean(ia);
        bid_unique  = bid_clean(ia);
        ask_unique  = ask_clean(ia);
        mid_unique  = mid_clean(ia);
        call_unique = call_clean(ia);
    else
        K_unique = [];
    end

    % Fallback to previous day's smile if fewer than 2 strikes survived near expiry
    if numel(K_unique) < 2
        if ~isempty(prevSmile)
            smile = prevSmile;
            smile.tau = tau;
            return;
        else
            error('calibrateModel:InsufficientStrikes', ...
                'Fewer than 2 valid option strikes survived filtering.');
        end
    end

    % Evaluate ATM implied volatility at K = F_t
    atmFn  = buildSmileInterpolant(K_unique, iv_unique, config.interpMethod);
    atmVol = atmFn(F_t);

    smile = struct( ...
        'tau',     tau, ...
        'F',       F_t, ...
        'strikes', K_unique, ...
        'iv',      iv_unique, ...
        'vega',    vega_unique, ...
        'bid',     bid_unique, ...
        'ask',     ask_unique, ...
        'mid',     mid_unique, ...
        'isCall',  call_unique, ...
        'atmVol',  atmVol);
end

%% ========================================================================
%  HELPER 2: Build Strike Interpolant with Flat Clamped Extrapolation
%  ========================================================================
function fn = buildSmileInterpolant(strikes, iv, method)
    K_min = min(strikes);
    K_max = max(strikes);
    fn = @(K_query) interp1(strikes, iv, min(max(K_query, K_min), K_max), method);
end

%% ========================================================================
%  HELPER 3: Historical Correlation Estimator
%  ========================================================================
function rho = estimateCorrelation(r1, r2, config, prevCalib)
    nObs = numel(r1);
    if nObs < config.minCorrObs
        rho = config.rhoPrior;
        return;
    end

    switch lower(config.corrMethod)
        case 'constant'
            rho = config.rhoPrior;

        case 'expanding'
            C = corrcoef(r1, r2);
            rho = C(1, 2);

        case 'rolling'
            w = min(nObs, config.corrWindow);
            C = corrcoef(r1(end-w+1:end), r2(end-w+1:end));
            rho = C(1, 2);

        case 'ewma'
            lam = config.ewmaLambda;
            weights = (1 - lam) * (lam .^ ((nObs-1:-1:0)'));
            weights = weights / sum(weights);
            m1 = sum(weights .* r1);
            m2 = sum(weights .* r2);
            v1 = sum(weights .* (r1 - m1).^2);
            v2 = sum(weights .* (r2 - m2).^2);
            cov12 = sum(weights .* (r1 - m1) .* (r2 - m2));
            rho = cov12 / sqrt(max(v1 * v2, 1e-16));

        otherwise
            error('calibrateModel:UnknownCorrMethod', ...
                'Unknown corrMethod "%s".', config.corrMethod);
    end

    if isnan(rho)
        if ~isempty(prevCalib) && isfield(prevCalib, 'rho')
            rho = prevCalib.rho;
        else
            rho = config.rhoPrior;
        end
    end

    % Clamp to open interval (-1, 1) for positive-definite covariance
    rho = min(max(rho, -0.9999), 0.9999);
end

