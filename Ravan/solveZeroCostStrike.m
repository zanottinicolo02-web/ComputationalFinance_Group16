function [Kput, info] = solveZeroCostStrike(Kcall, st)
% SOLVEZEROCOSTSTRIKE  Find Kput such that Put(Kput) = Call(Kcall) on the basket pricer.
%   Put price is increasing in strike, so the root is unique. Bracketed root-finding (fzero).
%   Optional st.kRange1 / st.kRange2 = [min max] of the log-moneyness of the quotes used in
%   each leg's smile; if given, info.inRange tells whether Kput lies inside both ranges.
%   NOTE: solve ONCE at inception and keep Kput fixed for the whole backtest (it is a
%   contract term, not something to re-solve at each rebalancing date).
    target = basketOptionPrice(Kcall, true, st);
    f      = @(K) basketOptionPrice(K, false, st) - target;

    fwdA = st.knownSum + sum((st.leg == 1) * st.F1t + (st.leg == 2) * st.F2t) / st.N;

    Khigh = fwdA;                                   % ATM put should exceed an OTM call premium
    for i = 1:10
        if f(Khigh) > 0, break; end
        Khigh = 1.25 * Khigh;
    end
    if f(Khigh) <= 0, error('solveZeroCostStrike:bracket', 'No upper bracket found.'); end

    Klow = 0.5 * fwdA;
    for i = 1:40
        if f(Klow) < 0, break; end
        Klow = 0.5 * Klow;
    end
    if f(Klow) >= 0, error('solveZeroCostStrike:bracket', 'No lower bracket found.'); end

    Kput = fzero(f, [Klow, Khigh], optimset('TolX', 1e-10));

    [putPx, s1, s2] = basketOptionPrice(Kput, false, st);
    info.target   = target;                         % call premium, USD/bbl
    info.putPx    = putPx;
    info.Kput     = Kput;
    info.bracket  = [Klow, Khigh];
    info.vols     = [s1, s2];
    info.logMoney = [log(Kput / st.F1t), log(Kput / st.F2t)];
    info.inRange  = NaN;
    if isfield(st, 'kRange1') && isfield(st, 'kRange2')
        info.inRange = info.logMoney(1) >= st.kRange1(1) && info.logMoney(1) <= st.kRange1(2) && ...
                       info.logMoney(2) >= st.kRange2(1) && info.logMoney(2) <= st.kRange2(2);
    end
end
