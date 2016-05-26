function idx = findClosestCentroids(X, centroids)

% FINDCLOSESTCENTROIDS computes the centroid memberships for every example
% idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
% in idx for a dataset X where each row is a single example. idx = m x 1 
% vector of centroid assignments (i.e. each entry in range [1..K])

% Set K, the number of groups we are finding.
K = size(centroids, 1);

% Returned data, start with an empty column vector.
idx = [];

% Loop through each example in X.
for i = 1:size(X,1)
	x = X(i,:);

	% Keep track of the minimum distance and its index.
	min_val = norm(x - centroids(1,:)) .^ 2;
	min_idx = 1;

	% Loop through each centroid.
	for k = 2:K
		val = norm(x - centroids(k,:)) .^ 2;

		% Check if we have found a new minimum value.
		if (val < min_val)
			min_idx = k;
			min_val = val;
		end
	end

	% Add that index to our column vector.
	idx = [idx; min_idx];
end

end
