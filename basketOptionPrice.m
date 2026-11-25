function [price, s1, s2] = basketOptionPrice(K, isCall, st)
% BASKETOPTIONPRICE  Price (USD/bbl) of a call/put on the average at strike K.
%   Each leg's flat vol is read off its own smile at the log-moneyness of K against
%   that leg's futures price: s_k = smile_k( ln(K / F_k) ).
%   st: struct from buildPricingState plus st.rho, st.smile1, st.smile2.
    s1 = st.smile1(log(K / st.F1t));
    s2 = st.smile2(log(K / st.F2t));
    price = basketMomentMatching(isCall, K, st.F1t, st.F2t, s1, s2, st.rho, ...
                                 st.tau, st.leg, st.N, st.knownSum);
end
