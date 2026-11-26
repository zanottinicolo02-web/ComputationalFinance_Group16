function p = black76(F, K, sigma, T, isCall)
% BLACK76  Undiscounted Black-76 price (r = 0, futures-style margining).
%   Inputs broadcast against each other (R2016b+). isCall is logical.
    sd = sigma .* sqrt(T);
    d1 = (log(F ./ K) + 0.5 * sd.^2) ./ sd;
    d2 = d1 - sd;
    Phi  = @(x) 0.5 * erfc(-x / sqrt(2));
    call = F .* Phi(d1)  - K .* Phi(d2);
    put  = K .* Phi(-d2) - F .* Phi(-d1);
    p = isCall .* call + (~isCall) .* put;
end
