function idx = selectHedgeOptions(Q, prev, opts)
% SELECTHEDGEOPTIONS  Pick two liquid hedge options from the day's kept quotes Q (table from
% extractImpliedVols: Strike, IsCall, LogMoneyness, ...).
%   Slot 1: nearest to ATM (target ln(K/F) = 0).  Slot 2: nearest to opts.opt2Moneyness.
%   Hysteresis: a previously held instrument (prev(j,:) = [strike isCall], NaN if none) is kept
%   while it is still quoted and within opts.band1 / opts.band2 of the slot's target,
%   to avoid paying two spreads for every small move of the underlying.
%   Returns row indices into Q ([NaN NaN] if two distinct strikes are not available).
    targets = [0, opts.opt2Moneyness];
    bands   = [opts.band1, opts.band2];
    idx = [NaN NaN];
    for j = 1:2
        other = idx(setdiff(1:2, j));  other = other(~isnan(other));
        otherStrike = Q.Strike(other);
        if ~isnan(prev(j,1))
            if isfield(opts, 'keepStrikeAcrossType') && opts.keepStrikeAcrossType
                m = find(Q.Strike == prev(j,1), 1);     % put/call at one strike: same vega and gamma
            else
                m = find(Q.Strike == prev(j,1) & Q.IsCall == logical(prev(j,2)), 1);
            end
            if ~isempty(m) && abs(Q.LogMoneyness(m) - targets(j)) <= bands(j) && ~any(Q.Strike(m) == otherStrike)
                idx(j) = m;  continue;
            end
        end
        [~, order] = sort(abs(Q.LogMoneyness - targets(j)));
        for m = order(:)'
            if ~any(Q.Strike(m) == otherStrike), idx(j) = m; break; end
        end
    end
end
