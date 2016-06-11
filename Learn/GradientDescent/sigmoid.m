function y = sigmoid( x )

% sigmoid computes the Sigmoid Function for logistic regression

y = 1 ./ (1 + exp(-x));

end