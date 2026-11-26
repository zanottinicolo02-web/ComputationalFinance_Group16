function [out, stats] = extractImpliedVols(chain, F, ~, opts)
% EXTRACTIMPLIEDVOLS  Filter one day's option chain of one contract and invert Black-76.
%
%   chain     struct as in optChain1{t}: scalar fields expiration, tau (ACT/365 years
%             to option expiry) and vector fields strike, isCall, bid, ask
%   F         futures settlement price of the underlying contract on that day
%   opts      struct, see defaults below (third input kept for call compatibility)
%
%   Early-exercise premium (EEP): quotes are American, but with futures-style
%   margining and r = 0, F is a martingale and the European value
%   E[(F_T-K)^+] >= (F_t-K)^+ (Jensen), so early exercise has no value and
%   American = European. Black-76 can be inverted directly. To limit residual
%   quote noise we use OTM options only (calls K>=F, puts K<F) and drop quotes
%   at/below intrinsic value.
    if nargin < 4, opts = struct(); end
    opts = setDefaults(opts);
    stats = struct('name', {}, 'removed', {});
    if isempty(chain) || ~isfield(chain, 'strike') || isempty(chain.strike)
        out = table(); return;
    end

    K = chain.strike(:);  bid = chain.bid(:);  ask = chain.ask(:);
    isCall = logical(chain.isCall(:));
    n   = numel(K);
    T   = chain.tau * ones(n,1);                 % years to option expiry (ACT/365)
    mid = 0.5*(bid + ask);
    k   = log(K ./ F);
    intrinsic = max(isCall.*(F-K) + (~isCall).*(K-F), 0);

    keep = true(n, 1);
    [keep, stats] = applyFilter(keep, stats, 'bid<=0',            bid > opts.minBid);
    [keep, stats] = applyFilter(keep, stats, 'crossed',           ask >= bid);
    [keep, stats] = applyFilter(keep, stats, 'T<=0',              T > 0);
    [keep, stats] = applyFilter(keep, stats, 'far from forward',  abs(k) <= opts.maxAbsLogMoneyness);
    [keep, stats] = applyFilter(keep, stats, 'wide spread',       (ask-bid)./mid <= opts.maxRelSpread);
    [keep, stats] = applyFilter(keep, stats, 'low price',         mid >= opts.minMid);
    [keep, stats] = applyFilter(keep, stats, 'at/below intrinsic', mid > intrinsic + opts.minTimeValue);
    if opts.otmOnly
        [keep, stats] = applyFilter(keep, stats, 'ITM', (isCall & K >= F) | (~isCall & K < F));
    end

    iv = nan(n, 1);
    iv(keep) = impliedVolBlack76(mid(keep), F, K(keep), T(keep), isCall(keep));
    [keep, stats] = applyFilter(keep, stats, 'no implied vol', ~isnan(iv));

    out = table(K(keep), isCall(keep), bid(keep), ask(keep), mid(keep), T(keep), ...
                repmat(F, nnz(keep), 1), k(keep), iv(keep), ...
        'VariableNames', {'Strike','IsCall','Bid','Ask','Mid','T','F','LogMoneyness','IV'});
    out = sortrows(out, 'Strike');
end

function [keep, stats] = applyFilter(keep, stats, name, passMask)
    before = nnz(keep);
    keep   = keep & passMask;
    stats(end+1) = struct('name', name, 'removed', before - nnz(keep));
end

function opts = setDefaults(opts)
    d.minBid = 0;  d.maxRelSpread = 0.5;  d.minMid = 0.02;  d.minTimeValue = 0;
    d.maxAbsLogMoneyness = 0.35;  d.otmOnly = true;
    f = fieldnames(d);
    for i = 1:numel(f)
        if ~isfield(opts, f{i}), opts.(f{i}) = d.(f{i}); end
    end
end
