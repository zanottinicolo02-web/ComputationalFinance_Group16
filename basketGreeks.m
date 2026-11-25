function g = basketGreeks(K, isCall, st, opts)
% BASKETGREEKS  Deltas (w.r.t. each futures), gamma w.r.t. F1 and vegas (per leg) of one basket option,
% by bump-and-revalue on the closed-form moment-matching pricer.
%
%   opts.dynamics  'sticky_moneyness' (default) | 'sticky_strike'
%       sticky_strike    : vol at the fixed strike K stays at its base value when F moves
%       sticky_moneyness : vol is re-read from the smile at ln(K/F_bumped)
%   opts.relBump   relative futures bump for delta (default 1e-4)
%   opts.gammaBump relative futures bump for gamma (default 1e-3)
%   opts.volBump   absolute parallel vol shift for vega (default 0.005)
%
%   Output (per 1 bbl of notional):
%     g.price   USD/bbl
%     g.delta   [d/dF1, d/dF2]
%     g.gamma   d2/dF1^2 (Leg 1 only; Leg 2 carries ~5% of the weight)
%     g.vega    [leg1, leg2]  USD/bbl per +1 vol point (0.01) parallel shift of that leg
%     g.vols    [s1, s2] base vols at strike K
    if nargin < 4, opts = struct(); end
    if ~isfield(opts, 'dynamics'),  opts.dynamics  = 'sticky_moneyness'; end
    if ~isfield(opts, 'relBump'),   opts.relBump   = 1e-4; end
    if ~isfield(opts, 'gammaBump'), opts.gammaBump = 1e-3; end
    if ~isfield(opts, 'volBump'),   opts.volBump   = 0.005; end

    [p0, s1, s2] = basketOptionPrice(K, isCall, st);
    F1 = st.F1t;  F2 = st.F2t;
    pr = @(f1, f2, a, b) basketMomentMatching(isCall, K, f1, f2, a, b, st.rho, ...
                                              st.tau, st.leg, st.N, st.knownSum);
    h1 = opts.relBump * F1;  h2 = opts.relBump * F2;  hg = opts.gammaBump * F1;

    switch opts.dynamics
        case 'sticky_strike'
            d1 = (pr(F1+h1, F2, s1, s2) - pr(F1-h1, F2, s1, s2)) / (2*h1);
            d2 = (pr(F1, F2+h2, s1, s2) - pr(F1, F2-h2, s1, s2)) / (2*h2);
            gm = (pr(F1+hg, F2, s1, s2) - 2*p0 + pr(F1-hg, F2, s1, s2)) / hg^2;
        case 'sticky_moneyness'
            vol1 = @(f) st.smile1(log(K / f));
            vol2 = @(f) st.smile2(log(K / f));
            d1 = (pr(F1+h1, F2, vol1(F1+h1), s2) - pr(F1-h1, F2, vol1(F1-h1), s2)) / (2*h1);
            d2 = (pr(F1, F2+h2, s1, vol2(F2+h2)) - pr(F1, F2-h2, s1, vol2(F2-h2))) / (2*h2);
            gm = (pr(F1+hg, F2, vol1(F1+hg), s2) - 2*p0 + pr(F1-hg, F2, vol1(F1-hg), s2)) / hg^2;
        otherwise
            error('basketGreeks:dynamics', 'Unknown dynamics "%s".', opts.dynamics);
    end

    dv = opts.volBump;
    v1 = (pr(F1, F2, s1+dv, s2) - pr(F1, F2, s1-dv, s2)) / (2*dv) * 0.01;
    v2 = (pr(F1, F2, s1, s2+dv) - pr(F1, F2, s1, s2-dv)) / (2*dv) * 0.01;

    g.price = p0;
    g.delta = [d1, d2];
    g.gamma = gm;
    g.vega  = [v1, v2];
    g.vols  = [s1, s2];
end
