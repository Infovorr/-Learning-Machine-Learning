function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h = sigmoid(X * theta);

J = (1 / m) * sum(((-1 * y) .* log(h)) - ((1 - y) .* log(1 - h)));

disp(size(J));

for i = 1:m,
	grad = grad + ((h(i) - y(i)) * X(i, :)');
end

grad = (1 / m) * grad;

end
