function [X_training, y_training, X_test, y_test] = generate_data(MODE, TEST_INDEX, X, y)
	%[m, n] = size(data);
	[m, n] = size(X); % changed m,n to size of X
	
	switch MODE
		case 'TRAINING_ERR'
			X_training = X;
			X_test = X;
			y_training = y;
			y_test = y;
		case 'TESTING_ERR'
			split_point = m - floor(m/3);
			X_training = X((1: split_point),:);
			X_test = X((split_point +1 : m),:);
			
			y_training = y((1: split_point),:);
			y_test = y((split_point +1 : m),:);
	  case 'LEAVE-ONE-OUT'
	    if TEST_INDEX == 1
	       X_training = X((TEST_INDEX+1:m),:);
	       X_test = X(TEST_INDEX,:);
	
	       y_training = y((TEST_INDEX+1:m),:);
	       y_test = y(TEST_INDEX,:); 
	    elseif TEST_INDEX == m
	       X_training = X((1:TEST_INDEX-1),:);
	       X_test = X(TEST_INDEX,:);
	
	       y_training = y((1:TEST_INDEX-1),:);
	       y_test = y(TEST_INDEX,:); 
	    else
	       X_training = X([1:TEST_INDEX-1, TEST_INDEX+1:m],:);
	       X_test = X(TEST_INDEX,:);
	
	       y_training = y([1:TEST_INDEX-1, TEST_INDEX+1:m],:);
	       y_test = y(TEST_INDEX,:); 
	    end
		otherwise
	end
end
