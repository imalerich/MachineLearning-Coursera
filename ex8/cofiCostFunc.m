function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)

% COFICOSTFUNC Collaborative filtering cost function
%    [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%    num_features, lambda) returns the cost and gradient for the
%    collaborative filtering problem.

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

% This will be used a few times, so only compute it once.
tmp = X * Theta' - Y;

% Compute the cost for the input data.
J = 0.5 * sum(sum(R .* tmp .^ 2));

% Add the regularization parametr to the cost function.
J += (lambda/2) * sum(sum(X .^ 2));
J += (lambda/2) * sum(sum(Theta .^ 2));

% Compute the needed gradients.
X_grad = (tmp .* R) * Theta;
Theta_grad = (tmp .* R)' * X;

% Add the regularization parameter to the gradients.
X_grad += lambda * X;
Theta_grad += lambda * Theta;

grad = [X_grad(:); Theta_grad(:)];

end
