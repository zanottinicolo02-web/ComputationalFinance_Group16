function data = loadData(parquetFile, excelFile, leg1Code, leg2Code, avgYear, avgMonth, n2RollDays)
%LOADDATA Load and structure futures, options, and SOFR data for the Hedging Lab.
%
%   data = loadData() loads the default in-sample training dataset
%   (Apr21 'COJ21' and May21 'COK21', averaging over February 2021).
%
%   data = loadData(parquetFile, excelFile, leg1Code, leg2Code, avgYear, avgMonth, n2RollDays)
%   allows customizing file paths, leg tickers, and the averaging month
%   (e.g., for the out-of-sample test set on 'COZ22' and 'COF23' in Oct 2022).
%
%   OUTPUT:
%       data : struct containing aligned daily vectors, matrices, fixing
%              schedule, contract parameters, and daily option chains.

    %% 1. Default Arguments (Training Set: Feb 2021 Collar on COJ21 / COK21)
    if nargin < 1 || isempty(parquetFile)
        parquetFile = 'unpivoted_bid_offer_train_clean.parquet';
    end
    if nargin < 2 || isempty(excelFile)
        excelFile = 'COJ21_COZ22_historical_series.xlsx';
    end
    if nargin < 3 || isempty(leg1Code)
        leg1Code = 'COJ21'; % First-nearby leg (Apr21)
    end
    if nargin < 4 || isempty(leg2Code)
        leg2Code = 'COK21'; % Second-nearby leg (May21)
    end
    if nargin < 5 || isempty(avgYear)
        avgYear = 2021;
    end
    if nargin < 6 || isempty(avgMonth)
        avgMonth = 2;       % February 2021
    end
    if nargin < 7 || isempty(n2RollDays)
        % Annex B: Leg 1 applies up to and including the second-to-last
        % fixing day before expiry; afterwards it rolls to Leg 2.
        % Set n2RollDays = 1 if penultimate day of contract is included in K1
        % (n1 = 19, n2 = 1), or n2RollDays = 2 if "before expiry" excludes
        % the expiry date itself (n1 = 18, n2 = 2).
        n2RollDays = 1;
    end

    %% 2. Read Raw Files
    rawParquet = parquetread(parquetFile);
    rawParquet.date = datetime(rawParquet.date);
    rawParquet.expiration = datetime(rawParquet.expiration);
    rawParquet.future_contract_code = string(rawParquet.future_contract_code);
    rawParquet.option_type = string(rawParquet.option_type);

    % Read Leg 1 complete futures history from Excel (covers up to futures expiry)
    histLeg1 = readtable(excelFile, 'Sheet', char(leg1Code));
    histLeg1.date = datetime(histLeg1.date);
    histLeg1 = sortrows(histLeg1, 'date');

    %% 3. Filter Option Tables for Leg 1 and Leg 2
    optTable1 = rawParquet(rawParquet.future_contract_code == string(leg1Code), :);
    optTable2 = rawParquet(rawParquet.future_contract_code == string(leg2Code), :);

    optTable1 = sortrows(optTable1, {'date', 'strike', 'option_type'});
    optTable2 = sortrows(optTable2, {'date', 'strike', 'option_type'});

    %% 4. Define Backtest Calendar (Emission Date t0 to Settlement Date T)
    % Emission date is the first date available in the training parquet (06-Jan-2020)
    % Settlement date T is the last fixing day of the averaging month (expiry of Leg 1)
    t0 = min(rawParquet.date);
    isAvgMonth = (year(histLeg1.date) == avgYear) & (month(histLeg1.date) == avgMonth);
    T_settle = max(histLeg1.date(isAvgMonth));

    % Extract daily Leg 2 futures & SOFR series from optTable2 over [t0, T_settle]
    [~, uniqueIdx2] = unique(optTable2.date, 'first');
    dailyLeg2 = optTable2(uniqueIdx2, {'date', 'underlying_settlement', ...
        'underlying_bid', 'underlying_ask', 'underlying_spread', 'sofr_rate'});
    dailyLeg2 = sortrows(dailyLeg2, 'date');
    dailyLeg2 = dailyLeg2(dailyLeg2.date >= t0 & dailyLeg2.date <= T_settle, :);

    % Align Leg 1 (from Excel) and Leg 2 (from Parquet) on common trading dates
    [dates, idx1, idx2] = intersect(histLeg1.date, dailyLeg2.date, 'stable');
    N_days = numel(dates);

    %% 5. Extract Aligned Futures Time Series Vectors (N_days x 1)
    % Leg 1 (COJ21) futures vectors
    F1_settle = double(histLeg1.settlement(idx1));
    F1_bid    = double(histLeg1.bid(idx1));
    F1_ask    = double(histLeg1.ask(idx1));
    F1_spread = F1_ask - F1_bid;

    % Leg 2 (COK21) futures vectors
    F2_settle = double(dailyLeg2.underlying_settlement(idx2));
    F2_bid    = double(dailyLeg2.underlying_bid(idx2));
    F2_ask    = double(dailyLeg2.underlying_ask(idx2));
    F2_spread = F2_ask - F2_bid;

    % Combined (N_days x 2) matrices for convenient vectorized portfolio math
    F_settle  = [F1_settle, F2_settle];
    F_bid     = [F1_bid,    F2_bid];
    F_ask     = [F1_ask,    F2_ask];
    F_spread  = [F1_spread, F2_spread];

    %% 6. Extract SOFR Rate & Discounting Vectors (N_days x 1)
    % Convert SOFR from percentage (e.g. 1.55) to decimal (0.0155)
    sofr_rate = double(dailyLeg2.sofr_rate(idx2)) / 100.0;

    % Year fraction between consecutive rebalancing dates (for cash accrual)
    dt_days = [0; days(diff(dates))];
    dt_yf   = dt_days / 365.0;

    % Time to derivative settlement T (ACT/365) and discount factor D(t, T)
    tau_settle = days(T_settle - dates) / 365.0;
    df_settle  = exp(-sofr_rate .* tau_settle);

    %% 7. Build Fixing Schedule & Weights (Annex A & Annex B)
    fixing_idx   = find((year(dates) == avgYear) & (month(dates) == avgMonth));
    fixing_dates = dates(fixing_idx);
    N_fix        = numel(fixing_dates);

    n1 = N_fix - n2RollDays;
    n2 = n2RollDays;
    K1_idx = (1:n1)';
    K2_idx = (n1+1:N_fix)';

    % Indicator vector: 1 if fixing k uses Leg 1, 2 if fixing k uses Leg 2
    fixing_leg = [ones(n1, 1); 2 * ones(n2, 1)];

    % Basket weights w1 = n1/N, w2 = n2/N
    w1 = n1 / N_fix;
    w2 = n2 / N_fix;
    weights = [w1, w2];

    % Model-free basket forward F_avg(t) = w1*F1(t) + w2*F2(t) (N_days x 1)
    F_avg = w1 * F1_settle + w2 * F2_settle;

    %% 8. Pre-Group Daily Option Chains & Multi-Maturity Surface (N_days x 1)
    % Pre-grouping avoids filtering the 138k-row table inside the daily loop.
    optChain1    = buildDailyOptionChain(optTable1, dates);
    optChain2    = buildDailyOptionChain(optTable2, dates);
    multiSurface = buildMultiMaturitySurface(rawParquet, dates, leg1Code, leg2Code);

    %% 9. Pack Everything into Output Struct
    data = struct();

    % Metadata & Contract Term Sheet (Annex A)
    data.leg1Code     = char(leg1Code);
    data.leg2Code     = char(leg2Code);
    data.notional     = 1e6;          % 1,000,000 bbl
    data.K_call       = 78.0;         % Fixed Call strike (USD/bbl)
    data.t0           = t0;           % Emission date (06-Jan-2020)
    data.T_settle     = T_settle;     % Settlement date (26-Feb-2021)

    % Calendar & Time Vectors
    data.dates        = dates;        % (N_days x 1) datetime
    data.N_days       = N_days;       % scalar (297 days)
    data.dt_days      = dt_days;      % (N_days x 1) calendar days between steps
    data.dt_yf        = dt_yf;        % (N_days x 1) year fraction between steps
    data.tau_settle   = tau_settle;   % (N_days x 1) year fraction T - t

    % Rates & Discount Factors
    data.sofr_rate    = sofr_rate;    % (N_days x 1) decimal SOFR rate r(t)
    data.df_settle    = df_settle;    % (N_days x 1) discount factor D(t, T)

    % Futures Time Series Vectors (N_days x 1) & Matrices (N_days x 2)
    data.F1_settle    = F1_settle;
    data.F1_bid       = F1_bid;
    data.F1_ask       = F1_ask;
    data.F1_spread    = F1_spread;

    data.F2_settle    = F2_settle;
    data.F2_bid       = F2_bid;
    data.F2_ask       = F2_ask;
    data.F2_spread    = F2_spread;

    data.F_settle     = F_settle;
    data.F_bid        = F_bid;
    data.F_ask        = F_ask;
    data.F_spread     = F_spread;
    data.F_avg        = F_avg;

    % Fixing Schedule & Basket Weights (Annex B)
    data.fixing_dates = fixing_dates; % (N_fix x 1) datetime
    data.fixing_idx   = fixing_idx;   % (N_fix x 1) index in data.dates
    data.fixing_leg   = fixing_leg;   % (N_fix x 1) 1 for Leg 1, 2 for Leg 2
    data.K1_idx       = K1_idx;       % (n1 x 1) indices of Leg 1 fixings
    data.K2_idx       = K2_idx;       % (n2 x 1) indices of Leg 2 fixings
    data.N_fix        = N_fix;        % total fixings (20)
    data.n1           = n1;           % Leg 1 fixings
    data.n2           = n2;           % Leg 2 fixings
    data.w1           = w1;           % weight w1 = n1 / N_fix
    data.w2           = w2;           % weight w2 = n2 / N_fix
    data.weights      = weights;      % [w1, w2]

    % Daily Option Chains & Multi-Maturity Surface (for Multi-Maturity ATS)
    data.optChain1    = optChain1;
    data.optChain2    = optChain2;
    data.multiSurface = multiSurface;
    data.optTable1    = optTable1;
    data.optTable2    = optTable2;
    data.histLeg1     = histLeg1;
end

%% Helper Function: Build Daily Option Chain Structs
function chains = buildDailyOptionChain(optTable, dates)
    N_days = numel(dates);
    chains = cell(N_days, 1);

    % Map each row's date to the index in 'dates'
    [isMember, dayIdx] = ismember(optTable.date, dates);
    validRows = find(isMember);
    dayIdx = dayIdx(isMember);

    for t = 1:N_days
        rows_t = validRows(dayIdx == t);
        if isempty(rows_t)
            chains{t} = struct( ...
                'date',       dates(t), ...
                'expiration', NaT, ...
                'tau',        0, ...
                'strike',     [], ...
                'isCall',     logical([]), ...
                'settle',     [], ...
                'bid',        [], ...
                'ask',        [], ...
                'mid',        [], ...
                'spread',     [], ...
                'atm_strike', NaN, ...
                'moneyness',  []);
            continue;
        end

        sub = optTable(rows_t, :);
        bid_t = double(sub.option_bid);
        ask_t = double(sub.option_ask);

        chains{t} = struct( ...
            'date',       dates(t), ...
            'expiration', sub.expiration(1), ...
            'tau',        double(sub.time_to_expiry_days(1)) / 365.0, ...
            'strike',     double(sub.strike), ...
            'isCall',     sub.option_type == "Call", ...
            'settle',     double(sub.settlement_price), ...
            'bid',        bid_t, ...
            'ask',        ask_t, ...
            'mid',        0.5 * (bid_t + ask_t), ...
            'spread',     ask_t - bid_t, ...
            'atm_strike', double(sub.atm_strike(1)), ...
            'moneyness',  double(sub.moneyness_usd));
    end
end

%% Helper Function: Build Daily Multi-Maturity Option Surface (for ATS)
function surface = buildMultiMaturitySurface(rawTable, dates, leg1Code, leg2Code)
    N_days  = numel(dates);
    surface = cell(N_days, 1);

    % Pre-filter to valid backtest dates and clean medium/long maturities
    % (exclude ultra-short front-month < 45d spikes except target legs >= 11d)
    t_days = double(rawTable.time_to_expiry_days);
    isTgt  = (rawTable.future_contract_code == string(leg1Code)) | ...
             (rawTable.future_contract_code == string(leg2Code));
    inWin  = (rawTable.date >= dates(1)) & (rawTable.date <= dates(end)) & ...
             (t_days <= 660) & ((t_days >= 45) | (isTgt & t_days >= 11));
    subAll = rawTable(inWin, {'date', 'future_contract_code', 'strike', ...
        'option_type', 'option_bid', 'option_ask', 'underlying_settlement', 'time_to_expiry_days'});

    [isMem, dayIdx] = ismember(subAll.date, dates);
    subAll = subAll(isMem, :);
    dayIdx = dayIdx(isMem);

    % Pre-compute numeric arrays for fast inner slicing
    codesAll = subAll.future_contract_code;
    K_all    = double(subAll.strike);
    isC_all  = subAll.option_type == "Call";
    bid_all  = double(subAll.option_bid);
    ask_all  = double(subAll.option_ask);
    mid_all  = 0.5 * (bid_all + ask_all);
    F_all    = double(subAll.underlying_settlement);
    tau_all  = double(subAll.time_to_expiry_days) / 365.0;

    leg1Str  = string(leg1Code);
    leg2Str  = string(leg2Code);

    for t = 1:N_days
        rows_t = find(dayIdx == t);
        if isempty(rows_t)
            surface{t} = struct([]);
            continue;
        end

        c_t = codesAll(rows_t);
        uCodes = unique(c_t, 'stable');
        nC = numel(uCodes);
        cand = struct('code', {}, 'tau', {}, 'F', {}, 'strikes', {}, 'isCall', {}, 'mktPrice', {});
        cnt = 0;

        for j = 1:nC
            rc = rows_t(c_t == uCodes(j));
            F_c   = F_all(rc(1));
            tau_c = tau_all(rc(1));
            K_c   = K_all(rc);
            isC_c = isC_all(rc);
            bid_c = bid_all(rc);
            ask_c = ask_all(rc);
            mid_c = mid_all(rc);

            % Liquid OTM quotes only
            isOTM = (~isC_c & (K_c <= F_c)) | (isC_c & (K_c > F_c));
            relSp = (ask_c - bid_c) ./ max(mid_c, 1e-6);
            keep  = isOTM & ~isnan(mid_c) & ~isnan(bid_c) & (bid_c >= 0.01) & (relSp <= 0.80);

            if sum(keep) >= 5
                [Ks, ord] = sort(K_c(keep));
                isCs = isC_c(keep); isCs = isCs(ord);
                mids = mid_c(keep); mids = mids(ord);
                cnt = cnt + 1;
                cand(cnt) = struct('code', char(uCodes(j)), 'tau', tau_c, ...
                    'F', F_c, 'strikes', Ks, 'isCall', isCs, 'mktPrice', mids);
            end
        end

        if cnt == 0
            surface{t} = struct([]);
            continue;
        end

        % Sort candidate slices by time to maturity
        [~, sOrd] = sort([cand.tau]);
        cand = cand(sOrd);

        % Always keep Leg 1 and Leg 2 if available + up to 3 spanning maturities
        isTarget = strcmp({cand.code}, char(leg1Str)) | strcmp({cand.code}, char(leg2Str));
        sel = cand(isTarget);
        others = cand(~isTarget);
        if ~isempty(others)
            nAdd = min(3, numel(others));
            pickIdx = unique(round(linspace(1, numel(others), nAdd)));
            sel = [sel, others(pickIdx)]; %#ok<AGROW>
        end
        [~, fOrd] = sort([sel.tau]);
        surface{t} = sel(fOrd);
    end
end

