function G = structureGreeks(Kput, Kcall, st, notional, opts)
% STRUCTUREGREEKS  Value and Greeks of the market maker's position: long Put(Kput), short Call(Kcall).
%   G.value  USD            (zero at inception by construction)
%   G.delta  [bbl, bbl]     exposure to Apr21 / May21 futures; hedge by trading -G.delta
%   G.gamma  bbl per USD/bbl  (second derivative w.r.t. F1)
%   G.vega   [USD, USD]     per +1 vol point on each leg's smile
    if nargin < 5, opts = struct(); end
    gp = basketGreeks(Kput,  false, st, opts);
    gc = basketGreeks(Kcall, true,  st, opts);
    G.value = (gp.price - gc.price) * notional;
    G.delta = (gp.delta - gc.delta) * notional;
    G.gamma = (gp.gamma - gc.gamma) * notional;
    G.vega  = (gp.vega  - gc.vega)  * notional;
    G.put   = gp;
    G.call  = gc;
end
