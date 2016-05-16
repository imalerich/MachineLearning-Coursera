function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

% LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
% regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y);

% first calculate the cost
t = (lambda/(2*m)) * theta .^ 2;
t(1) = 0;
c = (X * theta - y) .^ 2;
J = (1/(2*m)) * sum(c);
J += sum(t);

% and next the gradients
t = (lambda/m) * theta;
t(1) = 0;

grad = sum((X * theta - y) .* X)' ./ m;
grad += t;

% Unwrap the gradient matrix.
grad = grad(:);

end
