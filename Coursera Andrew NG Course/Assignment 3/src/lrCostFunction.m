function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


%Calculate the linear combination X and Theta
g = X * theta;
%element wise operation to compute 1/(1+e^z) Z = X * theta
h =  1./(1+exp(-g));
%calculates the cost function (without the sum)
cost = -y.*log(h) - (1-y).*log(1-h);

%sum up all values of the cost vector
sumCost = sum(cost);

%Calculates the Regularization factor for Cost Function (to penalizes huge values of theta)
n = length(theta); 
regularizationCostFactor = zeros(size(theta));
regularizationCostFactor = (lambda/(2*m))*sum(theta(2:n).^2);

%Calculates J
J = (1/m) * sumCost + regularizationCostFactor;

% =============================================================

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
 grad = ((1/m)* error) + regularizationGradFactor;
end
