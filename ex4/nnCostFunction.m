function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% NNCOSTFUNCTION Implements the neural network cost function for a two layer
% neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% The number of examples to include in our error calculation.
m = size(X, 1);

% Number of possible lables is the number of rows in Theta2.
num_labels = size(Theta2, 1);

% Add ones to the X matrix.
X = [ones(m, 1) X];

% Pass the input X through the first layer.
HIDDEN = sigmoid(X * Theta1');

% Add ones to the HIDDEN matrix.
HIDDEN = [ones(m, 1) HIDDEN];

% Pass the output of the first layer into the outpu layer.
s = sigmoid(HIDDEN * Theta2');

% Convert y from values 1..K to a matrix of 1's and 0's.
Y = [];

for i = 1:m
	% Create the vector which will be appended to Y.
	tmp = zeros(num_labels, 1);
	tmp(y(i)) = 1;

	% And append the vector to Y.
	Y = [Y tmp];
end

% We have everything now to calculate our cost.
c = (-Y' .* log(s)) - ((1-Y)' .* log(1 - s));
J = (1/m) * sum(sum(c));

% Add the regularization parameters.
reg = 0;

theta = Theta1;
theta(:, 1) = 0;
theta = theta .^ 2;
reg += sum(sum(theta));
         
theta = Theta2;
theta(:, 1) = 0;
theta = theta .^ 2;
reg += sum(sum(theta));

% Add the regularization parameter to our total cost.
J += (lambda * reg) / (2 * m);

% Perform the Backpropogation algorithm for gradient calculation.
Theta2_grad = zeros(size(Theta2));
Theta1_grad = zeros(size(Theta1));

for t = 1:m
	% Forward propogate this example through the neural network.
	a_1 = X(t, :)';
	z_2 = Theta1 * a_1;
	a_2 = [1; sigmoid(z_2)];
	z_3 = Theta2 * a_2;
	a_3 = sigmoid(z_3);

	delta_3 = a_3 - Y(:, t);
	delta_2 = (Theta2' * delta_3);
	delta_2 = delta_2 .* [1; sigmoidGradient(z_2)];
	delta_2 = delta_2(2:end);

	Theta2_grad += delta_3 * a_2';
	Theta1_grad += delta_2 * a_1';
end

% Divide the gradients by m.
Theta2_grad *= (1 / m);
Theta1_grad *= (1 / m);

% Zero out column 0 of our Theta matrixes, these will not be penalized by regularization.
Theta2(:, 1) = 0;
Theta1(:, 1) = 0;

% Add the regularization term for the gradients.
Theta2_grad += (lambda / m) * Theta2;
Theta1_grad += (lambda / m) * Theta1;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
