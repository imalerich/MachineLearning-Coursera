function X_rec = recoverData(Z, U, K)

% RECOVERDATA Recovers an approximation of the original data when using the 
% projected data
% X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
% original data that has been reduced to K dimensions. It returns the
% approximate reconstruction in X_rec.

% This will be the projection matrix to apply to Z.
U_reduce = U(:, 1:K);

% Approximate our original X.
X_rec = Z * U_reduce';

end
