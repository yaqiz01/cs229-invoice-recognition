function error=test(X_test, y_test, k, phi_k, phi_y)
	% [spmatrix, tokenlist, category] = readMatrix('MATRIX.TEST');
	
	%testMatrix = full(spmatrix);
	%X_test = [0 1 0; 1 0 1; 0 0 1; 1 0 1; 1 0 0; 0 1 1];
	%numTestDocs = size(testMatrix, 1);
	% #training docs
	%m = size(X_test, 1);
	% numTokens = size(testMatrix, 2);
	% #tokens
	%n = size(X_test, 2);
	[m,n] = size(X_test);
	% threshold
	threshold = log(0.6);
	
	% Assume nb_train.m has just been executed, and all the parameters computed/needed
	% by your classifier are in memory through that execution. You can also assume 
	% that the columns in the test set are arranged in exactly the same way as for the
	% training set (i.e., the j-th column represents the same token in the test data 
	% matrix as in the original training data matrix).
	
	% Write code below to classify each document in the test set (ie, each row
	% in the current document word matrix) as 1 for SPAM and 0 for NON-SPAM.
	
	% Construct the (numTestDocs x 1) vector 'output' such that the i-th entry 
	% of this vector is the predicted class (1/0) for the i-th  email (i-th row 
	% in testMatrix) in the test set.
	log_y_k = zeros(m, k);
	
	log_phi_k_one = log(phi_k);
	log_phi_k_zero = log(1-phi_k);
	
	for i = 1 : k
	    log_y_k(:, i) = sum(X_test .* log_phi_k_one(i, :) + (1-X_test) .* log_phi_k_zero(i, :), 2) + log(phi_y(i, 1));
	end
	
	[y_value, y_index] = max(log_y_k');
	
	
	% Compute the error on the test set
	% y = full(category);
	% y = y(:);
	error = sum(y_test ~= y_index') / m;
end
