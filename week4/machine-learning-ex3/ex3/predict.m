function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add x0 = 1
X = [ones(size(X,1), 1) X];

% Layer 2: a2 is 5000x25
a2 = sigmoid(X * Theta1');

% add a(2)(0) = 1
a2 = [ones(size(a2,1), 1) a2];

% Layer 3: 5000x10
a3 = sigmoid(a2 * Theta2');

% get the max probability prosition => class
[max_value, p] = max(a3, [], 2);

% p(find(p==10),:) = 0;

% =========================================================================


end
