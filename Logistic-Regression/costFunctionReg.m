function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h = sigmoid(X * theta);

redactedTheta = theta;
redactedTheta(1) = [];

J = (1 / m) * sum(((-1 * y) .* log(h)) - ((1 - y) .* log(1 - h))) + (lambda / (2 * m)) * sum(redactedTheta .^ 2);

for i = 1:m,
	grad = grad + ((h(i) - y(i)) * X(i, :)');
end

grad = (1 / m) * grad;

for i = 1:size(redactedTheta),
	grad(i+1) = grad(i+1) + (lambda / m) * redactedTheta(i);
end

end
