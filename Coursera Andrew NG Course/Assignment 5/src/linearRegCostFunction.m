function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%sumErr = 0;
h = (X*theta);
error = h-y;
v_sqr = error.^2;
sumError = sum(v_sqr);
n = length(theta); 
regularizationCostFactor = zeros(size(theta));
regularizationCostFactor = (lambda/(2*m))*sum(theta(2:n).^2);
J = (1/(2*m))*sumError + regularizationCostFactor;

%As we already have the values of h, compute the error and sumup everything
% Liner Combination was applied
error = ((h-y)'*X)';
 
%Calculates the Regularization Factor for Gradient
regularizationGradFactor = zeros(size(error));
regularizationGradFactor =(lambda/m)*theta;
%The first parameter of theta is not regularized (its value is 1), so 
%we restore the original value, and set the regularization factor for its
%as 0
theta(1)=1;
regularizationGradFactor(1) = 0;
%compute the gradient
grad = ((1/m)* error) + regularizationGradFactor ;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
