function centroids = computeCentroids(X, idx, K)

% COMPUTECENTROIDS returs the new centroids by computing the means of the 
% data points assigned to each centroid.
% centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
% computing the means of the data points assigned to each centroid. It is
% given a dataset X where each row is a single data point, a vector
% idx of centroid assignments (i.e. each entry in range [1..K]) for each
% example, and K, the number of centroids. You should return a matrix
% centroids, where each row of centroids is the mean of the data points
% assigned to it.

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

% For each centroid.
for k = 1:K
	count = 0;
	total = zeros(1, n);

	% Add each example to its corresponding centroid.
	for i = 1:m
		% The centroid this example belongs to.
		c = idx(i);

		if c == k
			% Add this value to that centroid.
			total += X(i,:);
			count++;
		end
	end

	centroids(k,:) = total / count;
end

end
