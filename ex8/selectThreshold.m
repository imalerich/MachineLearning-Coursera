function [bestEpsilon bestF1] = selectThreshold(yval, pval)

% SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
% outliers
%    [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%    threshold to use for selecting outliers based on the results from a
%    validation set (pval) and the ground truth (yval).

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)

	% Count our true positives and false positives/negatives.
	tp = sum(yval .* (pval < epsilon));
	fp = sum(yval .* (pval >= epsilon));
	fn = sum(abs(yval - 1) .* (pval < epsilon));

	% Evaluate the precision, recall, and F1 score from tp, fp, and fn.
	prec = tp / (tp + fp);
	rec = tp / (tp + fn);
	F1 = (2 * prec * rec) / (prec + rec);

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
