function [all_theta] = oneVsAll(X, y, num_labels, lambda)

% ONEVSALL trains multiple logistic regression classifiers and returns all
% the classifiers in a matrix all_theta, where the i-th row of all_theta 
% corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% m will be the number of training examples.
m = size(X, 1);

% n will be the number of parameters used for training.
n = size(X, 2);

% Add one's for the bias parametor of the input matrix.
X = [ones(m, 1) X];

% This will be our output data to be built in the 'for' loop below.
all_theta = [];

% Loop through each class and minimize 'theta' for that class.
for c = 1:num_labels

	% Set the options we will use for the minimization function.
	options = optimset('GradObj', 'on', 'MaxIter', 50);

	% Initial theta value for this iteration.
	init_theta = zeros(n + 1, 1);
	
	% Minimize theta values for this data set.
	theta = fmincg(@(t)(lrCostFunction(t, X, (y == c), lambda)), init_theta, options);

	% Append the results to our output array.
	all_theta = [all_theta ; theta'];

end

end
