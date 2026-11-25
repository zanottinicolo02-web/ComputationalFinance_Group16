function st = buildPricingState(t, dates, fixing_dates, fixing_idx, fixing_leg, F1, F2, N_fix)
% BUILDPRICINGSTATE  Split the fixings at backtest index t into known and future.
%   A fixing on date t itself is already known (its settle is in F1/F2).
%   Returns the quantities basketMomentMatching needs. Add st.rho, st.smile1, st.smile2
%   afterwards (see basketOptionPrice).
    past   = fixing_idx(:) <= t;
    leg    = fixing_leg(:);
    knownVals = zeros(N_fix, 1);
    for i = find(past)'
        if leg(i) == 1, knownVals(i) = F1(fixing_idx(i)); else, knownVals(i) = F2(fixing_idx(i)); end
    end
    st.F1t       = F1(t);
    st.F2t       = F2(t);
    st.N         = N_fix;
    st.knownSum  = sum(knownVals(past)) / N_fix;
    st.leg       = leg(~past);
    st.tau       = days(fixing_dates(~past) - dates(t)) / 365;   % ACT/365
    st.nFuture   = nnz(~past);
end
