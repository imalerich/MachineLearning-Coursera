function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)

% LEARNINGCURVE Generates the train and cross validation set errors needed 
% to plot a learning curve
% [error_train, error_val] = ...
%	  LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%	  cross validation set errors for a learning curve. In particular, 
%	  it returns two vectors of the same length - error_train and 
%	  error_val. Then, error_train(i) contains the training error for
%	  i examples (and similarly for error_val(i)).
%
%	  In this function, you will compute the train and test errors for
%	  dataset sizes from 1 up to m. In practice, when working with larger
%	  datasets, you might want to do this in larger intervals.

% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% Number of training examples
m = size(X, 1);

% These will be column vectors.
error_train = [];
error_val   = [];

% Compute errors needed to plot a learning curve.
% We will compute errors starting with 1 training example,
% then incrementally add an additionaly training example.
for i = 1:m
	% Train the training set.
	theta = trainLinearReg(X(1:i,:), y(1:i), lambda);

	% Compute training errors using training examples.
	c = linearRegCostFunction(X(1:i,:), y(1:i), theta, 0);
	error_train = [error_train; c ];

	% Compute train errors using training examples.
	c = linearRegCostFunction(Xval, yval, theta, 0);
	error_val = [error_val; c ];
end

end
