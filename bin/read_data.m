function [X,y] = read_data(INPUT_PATH)
	% if (~exist('INPUT_PATH'))
	% 	INPUT_PATH = '../data/Training_Input/training_input.csv';
	% end
	% MODE = 'TESTINT_ERR';
	
	data = csvread(INPUT_PATH, 1, 1);
	[m, n] = size(data); %TODO: shouldn't this be size(X)?? Doesn't matters here
	fprintf('Reading file %s\n', INPUT_PATH)
	
	X = data(:, (1: n -1));
	y = data(:, n);
	%text = csvread('../training_input.csv', 1, 0, [1,0,m,0]);
end
