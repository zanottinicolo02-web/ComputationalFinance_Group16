function price = basketMomentMatching(isCall, K, F1t, F2t, s1, s2, rho, tau, leg, N, knownSum)
% BASKETMOMENTMATCHING  Undiscounted price of an Asian option on the average of two
% correlated driftless-GBM futures, by two-moment lognormal matching.
%
%   A = knownSum + S,   S = (1/N) * sum_i F_{leg(i)}(tau_i)   (future fixings only)
%   isCall   true = call on A, false = put on A
%   K        strike on the average A
%   F1t,F2t  current futures prices (Leg 1 = Apr21, Leg 2 = May21)
%   s1,s2    flat volatilities of the two futures (annualised)
%   rho      correlation between the two futures
%   tau      (m x 1) year fractions from today to each FUTURE fixing date
%   leg      (m x 1) leg (1 or 2) referenced by each future fixing
%   N        total number of fixings of the structure
%   knownSum sum of the already-fixed values, divided by N
%
%   Exact moments (martingale GBMs, s <= u):  E[F_a(s) F_b(u)] = F_a F_b exp(rho_ab s_a s_b s)
%   M1 = E[S], M2 = E[S^2]; S ~ lognormal with mean M1 and log-variance v2 = ln(M2/M1^2).
%   Matching the exact moments of the weighted sum is what the "match each average, then
%   match the weighted sum" recipe reduces to when the cross moment E[A1*A2] is exact.
    tau = tau(:);  leg = leg(:);
    Kp  = K - knownSum;                         % strike on the remaining random part
    if isempty(tau)                             % everything already fixed
        payoff = max(knownSum - K, 0);  if ~isCall, payoff = max(K - knownSum, 0); end
        price = payoff; return;
    end

    Fv = (leg == 1) * F1t + (leg == 2) * F2t;   % forward of each fixing
    sv = (leg == 1) * s1  + (leg == 2) * s2;    % vol of each fixing's contract
    c  = 1 / N;

    M1 = c * sum(Fv);
    R  = ones(numel(leg));  R(leg ~= leg') = rho;            % correlation matrix of fixings
    E  = (Fv * Fv') .* exp(R .* (sv * sv') .* min(tau, tau'));
    M2 = c^2 * sum(E(:));
    v2 = log(M2 / M1^2);
    v  = sqrt(max(v2, 0));

    if Kp <= 0                                  % S > 0, so the call is always exercised
        call = M1 - Kp;
        put  = 0;
    elseif v < 1e-10                            % deterministic limit
        call = max(M1 - Kp, 0);
        put  = max(Kp - M1, 0);
    else
        Phi  = @(x) 0.5 * erfc(-x / sqrt(2));
        d1   = (log(M1 / Kp) + 0.5 * v2) / v;
        d2   = d1 - v;
        call = M1 * Phi(d1) - Kp * Phi(d2);
        put  = Kp * Phi(-d2) - M1 * Phi(-d1);
    end
    if isCall, price = call; else, price = put; end
end
