
close all, clearvars -except 'INPUT_PATH' 'MODE'

% Allow passing in INPUT_PATH and MODE from command line
if (~exist('INPUT_PATH', 'var'))
	INPUT_PATH = '../experiments/Training_Input/training_input.csv';
end
if (~exist('MODE', 'var'))
	MODE = 'LEAVE-ONE-OUT';
end

[X,y] = read_data(INPUT_PATH);
[m,n] = size(X);

switch MODE
	case 'TRAINING_ERR'
		[X_training, y_training, X_test, y_test] = generate_data(MODE, 0, X, y);
		[phi_k, phi_y, k] = training(X_training, y_training);
		testerror = test(X_test, y_test, k, phi_k, phi_y);
	case 'TESTING_ERR'
		[X_training, y_training, X_test, y_test] = generate_data(MODE, 0, X, y);
		[phi_k, phi_y, k] = training(X_training, y_training);
		testerror = test(X_test, y_test, k, phi_k, phi_y);
	case 'LEAVE-ONE-OUT'
		testerror = 0;
		for i = 1:m
		    TEST_INDEX = i;
		    [X_training, y_training, X_test, y_test] = generate_data(MODE, TEST_INDEX, X, y);
		    [phi_k, phi_y, k] = training(X_training, y_training);
		    error = test(X_test, y_test, k, phi_k, phi_y);
		    testerror = testerror + error;
		end
	otherwise
end

%Print out the classification error on the test set
fprintf(1, '%s ,m=%d, n=%d, Test error: %1.4f\n', INPUT_PATH, m, n, testerror);


