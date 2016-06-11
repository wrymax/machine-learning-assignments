function y = logisticPrediction( theta, X )
%LOGISTICPREDICTION Summary of this function goes here
%   Detailed explanation goes here

y = 1 / (1 + exp(-theta' * X));

end

