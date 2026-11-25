function [smileFun, coef] = fitSmileQuad(k, iv, w)
% FITSMILEQUAD  Quadratic smile in log-moneyness k = ln(K/F), flat outside the data range.
%   iv(k) = a + b*k + c*k^2 fitted by (weighted) least squares.
    k = k(:); iv = iv(:);
    if nargin < 3, w = ones(size(k)); end
    w = w(:);
    A = [ones(size(k)), k, k.^2];
    coef = (w .* A) \ (w .* iv);
    kmin = min(k); kmax = max(k);
    smileFun = @(kk) [ones(numel(kk),1), min(max(kk(:),kmin),kmax), min(max(kk(:),kmin),kmax).^2] * coef;
end
