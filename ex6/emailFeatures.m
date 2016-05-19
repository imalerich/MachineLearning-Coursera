function x = emailFeatures(word_indices)

% EMAILFEATURES takes in a word_indices vector and produces a feature vector
% from the word indices
%	  x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%	  produces a feature vector from the word indices. 

% Total number of words in the dictionary
n = 1899;

% Output 1 if the word is in the dictionary
% 0 otherwise, this vector will serve as the
% features for our training.
x = [];

% Loop through each word in the dictionary.
for i=1:n
	if (ismember(i, word_indices))
		x = [x ; 1];
	else
		x = [x ; 0];
	end
end

end
