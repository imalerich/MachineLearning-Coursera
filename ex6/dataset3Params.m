function [C, sigma] = dataset3Params(X, y, Xval, yval)

% EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
% where you select the optimal (C, sigma) learning parameters to use for SVM
% with RBF kernel
%	  [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%	  sigma. You should complete this function to return the optimal C and 
%	  sigma based on a cross-validation set.

% % We are looking for the smallest error
% % over our cross validation set.
% minErr = -1;
% C = 1;
% sigma = 1;
% 
% % Loop through each possible combination of C and sigma.
% for c=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
% 	for s=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
% 
% 		% Train our model with the given 'C' value and 'sigma' value.
% 		model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s)); 
% 
% 		% Compute the error for this example.
% 		predictions = svmPredict(model, Xval);
% 		err = mean(double(predictions ~= yval));
% 
% 		% Did we find a new optimum value?
% 		if (err < minErr || minErr == -1)
% 			minErr = err
% 			C = c
% 			sigma = s
% 		end
% 
% 	end % end for sigma
% end % end for C

% Results found from the training above.
C = 1;
sigma = 0.1;

end
