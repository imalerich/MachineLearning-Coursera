function p = predict(Theta1, Theta2, X)

% PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.

% number of inputs to test
m = size(X, 1); 

% number of outputs to the neural network
% one for each class in the problem
num_labels = size(Theta2, 1); 

% Add ones to the X matrix
X = [ones(m, 1) X];

% pass the input through the the first layer of the neural network
Y = sigmoid(X * Theta1');

% Add ones to the Y matrix
Y = [ones(m, 1) Y];

% pass the output of the first layer into the second (output) layer
s = sigmoid(Y * Theta2');
[s, p] = max(s, [], 2);

end
