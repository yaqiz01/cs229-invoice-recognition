function [phi_k, phi_y, k] = training(X_training, y_training)
	% [spmatrix, tokenlist, trainCategory] = readMatrix('MATRIX.TRAIN');
	
	%X_training = [0 0 1; 0 0 1; 0 0 1; 1 0 1; 1 0 0; 0 1 1];
	% invoice number: 1; invoice date: 2; total amount: 3
	%y = [1 1 2 2 2 1]';
	[m, n] = size(X_training);
	% #classification classes
	k = max(y_training); %TODO: should this be k+1? handle training labels happen to not contain
											 %one type of label
	
	phi_k = zeros(k, n);
	phi_y = zeros(k, 1);
	
	for i = 1 : k
	    phi_k(i, :) = (sum((y_training == i) .* X_training) + 1) / (sum(y_training==i) + k);
	    phi_y(i, 1) = (sum(y_training == i) + 1) / (size(y_training, 1) + k);
	end
end
