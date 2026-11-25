function [rho, nObs] = estimateCorrelation(F1, F2, tIdx, opts)
% ESTIMATECORRELATION  Correlation of daily log-returns of two futures at date tIdx.
%
%   F1, F2  price series of the two contracts (column vectors, same dates). Include
%           any pre-inception history (e.g. 2-3 Jan 2020) at the start of the vectors.
%   tIdx    index of the current date in F1/F2 (only data up to and including tIdx is used)
%   opts    optional struct:
%             window  (default 60)   max number of most recent returns used
%             minObs  (default 10)   below this, return the prior
%             m       (default 20)   shrinkage strength (pseudo-observations at the prior)
%             prior   (default 0.98) prior correlation of adjacent Brent contracts (ASSUMPTION)
%             cap     (default 0.999) upper bound, keeps the covariance matrix non-singular
%
%   rho = (n*r + m*prior) / (n + m), r = rolling sample correlation over the last n returns.
%   No look-ahead: nothing after tIdx enters the estimate.
    if nargin < 4, opts = struct(); end
    d = struct('window', 60, 'minObs', 10, 'm', 20, 'prior', 0.98, 'cap', 0.999);
    f = fieldnames(d);
    for i = 1:numel(f)
        if ~isfield(opts, f{i}), opts.(f{i}) = d.(f{i}); end
    end

    r1 = diff(log(F1(1:tIdx)));
    r2 = diff(log(F2(1:tIdx)));
    ok = isfinite(r1) & isfinite(r2);
    r1 = r1(ok);  r2 = r2(ok);
    n  = min(numel(r1), opts.window);
    nObs = n;

    if n < opts.minObs
        rho = opts.prior;
    else
        r1 = r1(end-n+1:end);  r2 = r2(end-n+1:end);
        C  = corrcoef(r1, r2);
        r  = C(1,2);
        rho = (n*r + opts.m*opts.prior) / (n + opts.m);
    end
    rho = min(max(rho, -opts.cap), opts.cap);
end
