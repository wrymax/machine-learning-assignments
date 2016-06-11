function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


prediction = predict(theta, X);

cost = -y.*log(prediction) - (1-y).*log(1-prediction);

% dont regularize theta(0)
J = sum(cost)/m + lambda/(2*m)*sum(theta(2:size(theta,1),:).^2);

% grad = X' * (prediction - y) / m;

theta_new = theta;
% for j = 0: no regularization
theta_new(1) = 0;
grad = X'*(prediction - y)/m + lambda/m*theta_new;

% =============================================================

end
