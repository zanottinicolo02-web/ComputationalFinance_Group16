function g = optionGreeks(K, isCall, F, T, sigma0, smile, opts)
% OPTIONGREEKS  Delta, gamma, vega of a listed (vanilla) futures option, undiscounted Black-76.
%   sigma0  the option's own implied vol (from its quote)
%   smile   handle of the contract's smile (used only for 'sticky_moneyness')
%   Dynamics mirror basketGreeks: sticky_strike keeps sigma0 when F moves;
%   sticky_moneyness shifts sigma0 by the smile's change along ln(K/F).
%   vega is per +1 vol point (0.01), delta/gamma per bbl.
    if ~isfield(opts, 'dynamics'),  opts.dynamics  = 'sticky_moneyness'; end
    if ~isfield(opts, 'relBump'),   opts.relBump   = 1e-4; end
    if ~isfield(opts, 'gammaBump'), opts.gammaBump = 1e-3; end
    if ~isfield(opts, 'volBump'),   opts.volBump   = 0.005; end

    switch opts.dynamics
        case 'sticky_strike'
            vol = @(f) sigma0;
        case 'sticky_moneyness'
            s0  = smile(log(K / F));
            vol = @(f) sigma0 + smile(log(K / f)) - s0;
        otherwise
            error('optionGreeks:dynamics', 'Unknown dynamics "%s".', opts.dynamics);
    end
    pr = @(f) black76(f, K, vol(f), T, isCall);
    h  = opts.relBump * F;  hg = opts.gammaBump * F;  dv = opts.volBump;

    g.price = pr(F);
    g.delta = (pr(F+h) - pr(F-h)) / (2*h);
    g.gamma = (pr(F+hg) - 2*pr(F) + pr(F-hg)) / hg^2;
    g.vega  = (black76(F, K, sigma0+dv, T, isCall) - black76(F, K, sigma0-dv, T, isCall)) / (2*dv) * 0.01;
end
