function Z = projectData(X, U, K)

% PROJECTDATA Computes the reduced data representation when projecting only 
% on to the top k eigenvectors
% Z = projectData(X, U, K) computes the projection of 
% the normalized inputs X into the reduced dimensional space spanned by
% the first K columns of U. It returns the projected examples in Z.

% This will be the projection matrix to apply to X.
U_reduce = U(:, 1:K);

% Apply the projection to our data.
Z = X * U_reduce;

end
