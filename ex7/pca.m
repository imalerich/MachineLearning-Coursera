function [U, S] = pca(X)

% PCA Run principal component analysis on the dataset X
% [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
% Returns the eigenvectors U, the eigenvalues (on diagonal) in S

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).

% Useful values
[m, n] = size(X);

% First calculate the covariance matrix for this data set.
SIGMA = (1/m) * X' * X;

% U & S are returned by this function.
[U, S, V] = svd(SIGMA);

end
