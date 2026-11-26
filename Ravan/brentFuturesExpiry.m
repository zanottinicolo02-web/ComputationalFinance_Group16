function d = brentFuturesExpiry(contractYear, contractMonth, holidays)
% BRENTFUTURESEXPIRY  Expiry of an ICE Brent futures contract.
%   Rule: last business day of the second month preceding the contract month.
%   Example: Apr21 -> last business day of Feb 2021 ; Dec22 -> last business day of Oct 2022.
%   NOTE: ICE has exceptions (e.g. around Christmas). Verify against the official
%   ICE expiry calendar and pass UK/ICE holidays via 'holidays' if needed.
    if nargin < 3, holidays = datetime.empty(0,1); end
    d = dateshift(datetime(contractYear, contractMonth, 1) - calmonths(2), 'end', 'month');
    while any(weekday(d) == [1 7]) || any(d == holidays)
        d = d - caldays(1);
    end
end
