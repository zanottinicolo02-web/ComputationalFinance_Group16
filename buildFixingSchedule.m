function sched = buildFixingSchedule(dates, avgYear, avgMonth, expiry1, holidays, includeExpiryDay)
% BUILDFIXINGSCHEDULE  Fixing dates and Leg1/Leg2 assignment (Annex B).
%
%   dates            datetime vector of backtest dates (rebalancing calendar)
%   avgYear/avgMonth averaging window (e.g. 2021, 2)
%   expiry1          expiry of the first-nearby contract (leg 1), see brentFuturesExpiry
%   holidays         (optional) datetime vector of non-fixing business days
%   includeExpiryDay (optional, default true) whether the expiry day itself is a
%                    fixing day that counts when looking for the "second-to-last
%                    fixing day before expiry". Default true gives n2 = 1 for
%                    Feb 2021 and Oct 2022. Check the reading of Annex B.
%
%   Rule: leg 1 up to and including the second-to-last fixing day on/before expiry,
%         leg 2 from the next fixing day onward.
    if nargin < 5 || isempty(holidays), holidays = datetime.empty(0,1); end
    if nargin < 6, includeExpiryDay = true; end

    first   = datetime(avgYear, avgMonth, 1);
    allDays = (first : caldays(1) : dateshift(first, 'end', 'month'))';
    isBiz   = ~ismember(weekday(allDays), [1 7]) & ~ismember(allDays, holidays);
    fixing_dates = allDays(isBiz);
    N = numel(fixing_dates);

    if includeExpiryDay
        cand = fixing_dates(fixing_dates <= expiry1);
    else
        cand = fixing_dates(fixing_dates <  expiry1);
    end
    if numel(cand) < 2
        error('buildFixingSchedule:roll', 'Fewer than two fixing days before expiry.');
    end
    lastLeg1    = cand(end-1);                  % second-to-last fixing day
    fixing_leg  = 1 + double(fixing_dates > lastLeg1);

    % map fixing dates onto the backtest calendar
    [tf, loc] = ismember(dateshift(fixing_dates,'start','day'), dateshift(dates,'start','day'));
    if ~all(tf)
        error('buildFixingSchedule:missing', ...
            'Fixing dates not in the data calendar: %s. Pass them via "holidays" or check the dataset.', ...
            strjoin(string(fixing_dates(~tf), 'dd-MMM-yyyy'), ', '));
    end

    sched.fixing_dates = fixing_dates;
    sched.fixing_idx   = loc;                   % indices into 'dates'
    sched.fixing_leg   = fixing_leg;
    sched.K1_idx       = find(fixing_leg == 1); % indices into the fixing vector
    sched.K2_idx       = find(fixing_leg == 2);
    sched.N_fix        = N;
    sched.n1           = numel(sched.K1_idx);
    sched.n2           = numel(sched.K2_idx);
    sched.w1           = sched.n1 / N;
    sched.w2           = sched.n2 / N;
    sched.weights      = [sched.w1, sched.w2];
    sched.lastLeg1Day  = lastLeg1;
end
