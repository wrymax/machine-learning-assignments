function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Process to get C and sigma
% c_canidates = [0.01 0.03 0.1 0.3 1 3 10 30];
% sigma_canidates = [0.01 0.03 0.1 0.3 1 3 10 30];
% s = size(c_canidates, 2);
% errors = zeros(1, s^2);
% 
% % loop all C and sigma
% for i = 1:s
%     for j = 1:s
%         % train the SVM model of current C and aigma
%         model= svmTrain( ...
%             X, y, c_canidates(i), ...
%             @(x1, x2) gaussianKernel(x1, x2, sigma_canidates(j))...
%             ); 
%         % predict by Cross-Validation set Xval
%         predictions = svmPredict(model, Xval);
%         % record the errors caused by C and sigma
%         errors((i-1)*s + j) = mean(double(predictions ~= yval));
%     end
% end
% 
% % find the minimum error
% [v, index] = min(errors);
% 
% % get the index
% if rem(index, s) == 0
%     s_index = s;
%     c_index = floor(index / s) ;
% else
%     s_index = rem(index, s);
%     c_index = floor(index / s) + 1;
% end
% 
% % get C and sigma
% C = c_canidates(c_index);
% sigma = sigma_canidates(s_index);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

C = 1;
sigma = 0.1;



% =========================================================================

end
