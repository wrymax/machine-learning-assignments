% Compute the result of prediction equation

function output = prediction(theta, X)
    theta = columnize(theta);
    X = columnize(X);

    output = theta' * X;
end

% make input vectors as column vectors
function result = columnize(vector)
    if size(vector, 2) ~= 1 && size(vector, 1) == 1
        result = vector';
    else
        result = vector;
    end  
end