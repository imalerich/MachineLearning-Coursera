function [J, grad] = lrCostFunction(theta, X, y, lambda)

% LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
% regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples
s = sigmoid(X * theta);

% first calculate the cost
t = (lambda/(2*m)) * theta .^ 2;
t(1) = 0;
c = (-y' * log(s)) - ((1-y') * log(1 - s));
J = (1/m) * sum(c);
J += sum(t);

% and next the gradients
t = (lambda/m) * theta;
t(1) = 0;

grad = (X' * (s - y)) ./ m;
grad += t;
grad = grad(:);

end
