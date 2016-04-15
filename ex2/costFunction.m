function [J, grad] = costFunction(theta, X, y)

% COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y);
s = sigmoid(X * theta);

c = (-y' * log(s)) - ((1-y') * log(1 - s));

J = (1/m) * sum(c);
grad = (X' * (s - y)) ./ m;

end
