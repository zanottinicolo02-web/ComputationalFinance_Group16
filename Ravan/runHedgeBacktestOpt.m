function R = runHedgeBacktestOpt(bt, opts)
% RUNHEDGEBACKTESTOPT  Delta-gamma-vega hedge: two listed Apr21 options + Apr21/May21 futures.
%
%   Each day (all data up to t only):
%     1. value the structure and compute its Greeks (delta, gamma on F1, vega on Leg 1);
%     2. select two liquid hedge options (ATM and a put-side wing, with hysteresis);
%     3. solve for quantities q1,q2 (bbl) with zero net vega and zero net gamma
%        (fallback to vega-only if the 2x2 system is ill-conditioned);
%     4. set the futures so that the total delta is zero.
%   Options are futures-style margined like the futures: daily P&L = q*(mark_t - mark_{t-1}),
%   no premium cash flow, margin costs neglected; every trade pays half the quoted bid-ask spread.
%   Marks use the quote mid when valid, otherwise Black-76 at the smile vol.
%   Options are closed when time to option expiry falls below opts.minTau.
%
%   bt: as for runHedgeBacktest plus bt.iv1 (cell of kept-quote tables) and bt.chain1 (optChain1).
%   opts (optional): dynamics, relBump, gammaBump, volBump, hedgeScale (1), useOptions (true),
%     rebalEvery is NOT used here (futures rebalanced daily), band1 (0.08), band2 (0.10),
%     opt2Moneyness (-0.15), qTol (0.10), maxCond (1e3), qMax (10*notional), minTau (7/365).
%   useGamma (default true): false = vega-only hedge with one option (second option unused).
%   With useOptions = false this reproduces runHedgeBacktest (daily futures delta hedge).
    if nargin < 2, opts = struct(); end
    d = struct('dynamics', 'sticky_moneyness', 'relBump', 1e-4, 'gammaBump', 1e-3, 'volBump', 0.005, ...
               'hedgeScale', 1, 'useOptions', true, 'band1', 0.08, 'band2', 0.10, ...
               'opt2Moneyness', -0.15, 'qTol', 0.10, 'maxCond', 1e3, 'qMax', 10*bt.notional, ...
               'minTau', 7/365, 'keepStrikeAcrossType', true, 'useGamma', true);
    f = fieldnames(d);
    for i = 1:numel(f), if ~isfield(opts, f{i}), opts.(f{i}) = d.(f{i}); end, end

    N   = numel(bt.dates);
    Fm  = [bt.F1(:), bt.F2(:)];
    spr = [bt.F1_spread(:), bt.F2_spread(:)];
    dtc = [0; days(diff(bt.dates(:))) / 365];

    [S, B, costF, costO, fpnl, opnl, intr, Pi, pnl] = deal(zeros(N, 1));
    [H, D, V, Qopt, Kopt] = deal(zeros(N, 2));
    Hs = nan(2, 2);  qh = [0 0];  mkh = [0 0];  sprh = [0 0];     % held options: [strike isCall], qty, mark, spread

    for t = 1:N
        st = buildPricingState(t, bt.dates, bt.fixing_dates, bt.fixing_idx, bt.fixing_leg, ...
                               bt.F1, bt.F2, bt.N_fix);
        st.rho = bt.rho(t);  st.smile1 = bt.smiles1{t};  st.smile2 = bt.smiles2{t};
        G = structureGreeks(bt.K_put, bt.K_call, st, bt.notional, opts);
        S(t) = G.value;  D(t,:) = G.delta;  V(t,:) = G.vega;
        chain = bt.chain1{t};  Qt = bt.iv1{t};

        % --- P&L of options held over the night
        if t > 1 && any(qh ~= 0)
            for j = 1:2
                if qh(j) ~= 0
                    mk = getMark(chain, Hs(j,1), logical(Hs(j,2)), bt.F1(t), bt.smiles1{t});
                    opnl(t) = opnl(t) + qh(j) * (mk - mkh(j));
                end
            end
        end

        % --- target option positions
        tauOpt = NaN;
        if isstruct(chain) && isfield(chain, 'tau') && ~isempty(chain.tau), tauOpt = chain.tau; end
        ok = opts.useOptions && t < N && istable(Qt) && height(Qt) >= 2 && ~isnan(tauOpt) && tauOpt >= opts.minTau;
        idx = [NaN NaN];  HsNew = nan(2,2);  qNew = [0 0];  og = cell(1,2);  sprNew = [0 0];
        if ok
            prevSel = Hs;  prevSel(qh == 0, :) = NaN;
            idx = selectHedgeOptions(Qt, prevSel, opts);
            ok = ~any(isnan(idx));
        end
        if ok
            for j = 1:2
                r = Qt(idx(j), :);
                typeJ = r.IsCall;  sprJ = r.Ask - r.Bid;
                if opts.keepStrikeAcrossType && qh(j) ~= 0 && Hs(j,1) == r.Strike && Hs(j,2) ~= r.IsCall
                    % Held option went ITM and left the OTM-only quote set: keep it. At one strike put and
                    % call have the same vega/gamma (deltas differ by exactly 1), so flipping only costs spread.
                    typeJ = Hs(j,2);
                    sprJ  = getSpread(chain, r.Strike, logical(typeJ), sprh(j));
                end
                og{j} = optionGreeks(r.Strike, logical(typeJ), bt.F1(t), r.T, r.IV, bt.smiles1{t}, opts);
                HsNew(j,:) = [r.Strike, typeJ];
                sprNew(j)  = sprJ;
            end
            v  = [og{1}.vega,  og{2}.vega];
            gm = [og{1}.gamma, og{2}.gamma];
            A  = [v; gm];   b = -[G.vega(1); G.gamma];
            An = A ./ sqrt(sum(A.^2, 2));                       % row-normalised for the conditioning test
            if opts.useGamma && all(isfinite(A(:))) && rcond(An) > 1 / opts.maxCond
                qf = A \ b;
            else
                qf = [b(1) / v(1); 0];                          % vega-only fallback
            end
            qf   = max(min(qf(:)', opts.qMax), -opts.qMax);
            qTgt = opts.hedgeScale * qf;
            for j = 1:2
                same = qh(j) ~= 0 && isequal(Hs(j,:), HsNew(j,:));
                if same && abs(qTgt(j) - qh(j)) <= opts.qTol * abs(qh(j)), qNew(j) = qh(j); else, qNew(j) = qTgt(j); end
            end
        end

        % --- option trading costs (exits of dropped instruments, entries / adjustments)
        for j = 1:2
            if qh(j) ~= 0 && ~(ok && isequal(Hs(j,:), HsNew(j,:)))
                costO(t) = costO(t) + 0.5 * abs(qh(j)) * getSpread(chain, Hs(j,1), logical(Hs(j,2)), sprh(j));
            end
        end
        if ok
            for j = 1:2
                if qh(j) ~= 0 && isequal(Hs(j,:), HsNew(j,:)), dq = qNew(j) - qh(j); else, dq = qNew(j); end
                costO(t) = costO(t) + 0.5 * abs(dq) * sprNew(j);
            end
        end

        % --- update option state and marks
        if ok
            Hs = HsNew;  qh = qNew;  sprh = sprNew;
            for j = 1:2, mkh(j) = getMark(chain, Hs(j,1), logical(Hs(j,2)), bt.F1(t), bt.smiles1{t}); end
            Qopt(t,:) = qh;  Kopt(t,:) = Hs(:,1)';
            optDelta = sum(qh .* [og{1}.delta, og{2}.delta]);
        else
            Hs = nan(2,2);  qh = [0 0];  mkh = [0 0];  sprh = [0 0];  optDelta = 0;
        end

        % --- futures: total delta zero
        if t == 1, Hprev = [0 0]; else, Hprev = H(t-1,:); end
        H(t,:) = -opts.hedgeScale * G.delta - [optDelta, 0];
        costF(t) = 0.5 * sum(abs(H(t,:) - Hprev) .* spr(t,:));

        % --- cash account and portfolio value
        if t == 1
            B(t) = -(costF(t) + costO(t));
            pnl(t) = S(t) + B(t);
        else
            fpnl(t) = Hprev * (Fm(t,:) - Fm(t-1,:))';
            intr(t) = B(t-1) * bt.sofr(t-1) * dtc(t);
            B(t)    = B(t-1) + intr(t) + fpnl(t) + opnl(t) - costF(t) - costO(t);
            pnl(t)  = (S(t) + B(t)) - Pi(t-1);
        end
        Pi(t) = S(t) + B(t);
    end

    R = struct('dates', bt.dates, 'S', S, 'B', B, 'H', H, 'q', Qopt, 'optStrikes', Kopt, 'delta', D, 'vega', V, ...
               'costFut', costF, 'costOpt', costO, 'futuresPnL', fpnl, 'optionPnL', opnl, 'interest', intr, ...
               'Pi', Pi, 'pnl', pnl, 'meanSqPnL', mean(pnl.^2), 'totalCost', sum(costF) + sum(costO), ...
               'optDays', nnz(any(Qopt ~= 0, 2)), 'opts', opts);
end

function m = getMark(chain, K, isCall, F, smile)
% Quote mid if the option has a valid quote today, else Black-76 at the smile vol (intrinsic if expired).
    tau = 0;
    if isstruct(chain) && isfield(chain, 'strike') && ~isempty(chain.strike)
        i = find(chain.strike == K & logical(chain.isCall) == isCall, 1);
        if ~isempty(i) && chain.bid(i) > 0 && chain.ask(i) >= chain.bid(i)
            m = 0.5 * (chain.bid(i) + chain.ask(i));  return;
        end
        tau = chain.tau;
    end
    if tau > 0
        m = black76(F, K, smile(log(K / F)), tau, isCall);
    else
        m = max(isCall * (F - K) + (~isCall) * (K - F), 0);
    end
end

function s = getSpread(chain, K, isCall, sDefault)
% Today's quoted spread of the option if valid, else the spread it was entered with.
    s = sDefault;
    if isstruct(chain) && isfield(chain, 'strike') && ~isempty(chain.strike)
        i = find(chain.strike == K & logical(chain.isCall) == isCall, 1);
        if ~isempty(i) && chain.bid(i) > 0 && chain.ask(i) >= chain.bid(i)
            s = chain.ask(i) - chain.bid(i);
        end
    end
end
