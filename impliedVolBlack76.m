function iv = impliedVolBlack76(price, F, K, T, isCall)
% IMPLIEDVOLBLACK76  Invert undiscounted Black-76 by vectorised bisection.
%   Price is strictly increasing in sigma, so the root is unique if it exists.
%   Returns NaN where no valid vol exists (price <= intrinsic, above the
%   no-arbitrage upper bound, T <= 0, or root outside [sLo, sHi]).
    [price, F, K, T, isCall] = deal(price(:), F(:), K(:), T(:), logical(isCall(:)));
    n = numel(price);
    [F, K, T, isCall] = deal(F.*ones(n,1), K.*ones(n,1), T.*ones(n,1), isCall.*true(n,1));

    sLo = 1e-4;  sHi = 5;
    lo  = sLo * ones(n,1);   hi = sHi * ones(n,1);

    intrinsic = max(isCall.*(F-K) + (~isCall).*(K-F), 0);
    upper     = isCall.*F + (~isCall).*K;           % undiscounted bounds: call<=F, put<=K
    valid     = T > 0 & price > intrinsic & price < upper;
    valid     = valid & black76(F, K, sHi, T, isCall) > price;   % root bracketed

    for it = 1:100
        mid = 0.5*(lo + hi);
        tooLow = black76(F, K, mid, T, isCall) < price;
        lo(tooLow)  = mid(tooLow);
        hi(~tooLow) = mid(~tooLow);
    end
    iv = 0.5*(lo + hi);
    iv(~valid) = NaN;
end
