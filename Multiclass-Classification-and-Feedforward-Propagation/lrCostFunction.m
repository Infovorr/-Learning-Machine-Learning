function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h = sigmoid(X * theta);

redactedTheta = theta;
redactedTheta(1) = [];

unregJ = (1 / m) * sum(((-1 * y) .* log(h)) - ((1 - y) .* log(1 - h)));

regJ = lambda / (2 * m) * sum(theta(2:end) .^ 2);

J = unregJ + regJ;

unregGrad = X' * (h - y);

regGrad = lambda / m * [0; theta(2:end)];

grad = (1 / m) * unregGrad + regGrad;

grad = grad(:);

grad = grad(:);

end
