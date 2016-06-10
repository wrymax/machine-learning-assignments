function [jVal, gradient] = costFunction(theta)

% costFunction for optimized gradient descent

jVal = sum((theta - 5) .^ 2); % (theta(1) - 5)^2 + (theta(2) - 5)^2

gradient = 2 * (theta - 5);

end