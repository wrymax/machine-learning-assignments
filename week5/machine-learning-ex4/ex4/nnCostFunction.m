function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% =========================================================================
% Assignment Implementation:
% Author: Max Wang
% 1.1 Cost Funtion without regularization

% loop for all classes, from k = 1 to k = K ( K = 10 here)
[a_last, prediction] = predict(Theta1, Theta2, X);
% prediction = log(prediction);
K = size(Theta2,1);
y_eye = eye(K);

% recode y into a logistic vector
y_recode = zeros(size(y,1), K);
for index_1 = 1:size(y,1)
    y_recode(index_1,:) = y_eye(:, rem(y(index_1), 11))';
end

for index_2 = 1:m
    yk = y_recode(index_2,:)';
    hx = a_last(index_2,:);
    
    cost = - (...
        log(hx) * yk + ...
        log(1 - hx) * (1 - yk) ...
    );
    J = J + cost;    
end
    
J = J / m;

% 1.2 Regularization part
J = J + (sum(sum(Theta1(:, 2:end) .^ 2)) +...
    sum(sum(Theta2(:, 2:end) .^ 2))) * lambda / (2 * m);


% 2. Compute Gradient via ?Back Propagation?algorithm

% set training examples X as a_1, a_n is activation matrices

% a_1: 5000x401
a_1 = [ones(m, 1) X];
% z_2: 5000x25
z_2 = a_1 * Theta1';
% a_2: 5000x26
a_2 = [ones(m, 1) sigmoid(z_2)];
% z_3: 5000x10
z_3 = a_2 * Theta2';
% a_3: 50000x10
a_3 = sigmoid(z_3);

% delta3: 5000x10
delta3 = a_3 - y_recode;
% delta2: 5000x25
delta2 = delta3 * Theta2(:,2:end) .* sigmoidGradient(z_2);

% Theta2_grad: 10x26
Theta2_grad = Theta2_grad + delta3' * a_2 / m;

% Theta1_grad: 25x401
Theta1_grad = Theta1_grad + delta2' * a_1 / m;


% 3. Regularized Neural Networks
t2_grad_r = [zeros(size(Theta2, 1), 1) lambda / m * Theta2(:, 2:end)];
Theta2_grad = Theta2_grad + t2_grad_r;
           
t1_grad_r = [zeros(size(Theta1, 1), 1) lambda / m * Theta1(:, 2:end)];
Theta1_grad = Theta1_grad + t1_grad_r;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
