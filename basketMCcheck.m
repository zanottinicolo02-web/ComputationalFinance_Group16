function [px, se] = basketMCcheck(isCall, K, st, s1, s2, nPaths, seed)
% BASKETMCCHECK  Monte Carlo benchmark for basketMomentMatching (same inputs, GBM exact steps).
    if nargin < 7, seed = 1; end
    rng(seed);
    tau = st.tau(:);  m = numel(tau);
    dt  = diff([0; tau]);
    Z1  = randn(nPaths, m);
    Z2  = st.rho * Z1 + sqrt(1 - st.rho^2) * randn(nPaths, m);
    W1  = cumsum(Z1 .* sqrt(dt'), 2);
    W2  = cumsum(Z2 .* sqrt(dt'), 2);
    X1  = st.F1t * exp(-0.5*s1^2*tau' + s1*W1);
    X2  = st.F2t * exp(-0.5*s2^2*tau' + s2*W2);
    X   = X1 .* (st.leg' == 1) + X2 .* (st.leg' == 2);
    A   = st.knownSum + sum(X, 2) / st.N;
    if isCall, pay = max(A - K, 0); else, pay = max(K - A, 0); end
    px = mean(pay);  se = std(pay) / sqrt(nPaths);
end
