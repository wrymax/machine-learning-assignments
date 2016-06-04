function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% Pre-process the theta vector
theta_new = [];
sizeTheta = size(theta);

if sizeTheta(1) == 1
    theta_new = theta';
elseif sizeTheta(2) == 1
    theta_new = theta;
else
    disp('Error: theta is not properly initialized.');
end

% prediction vector
prediction = X * theta_new;
    
J = sum((prediction - y) .^ 2) / (2 * m);

% =========================================================================

end
