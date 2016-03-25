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

%Theta Transverse multiplied by X
g = X * theta;

%Returns the sigmoid value of each element of g. h = g(theta'*X), where g is the 
%sigmoid function 1/(1+e^-z)
h = sigmoid(g);

%Auxiliary Matriz to store the values of each logistic error
auxJMatrix = zeros(size(y));
%Compute the logistic error for all training examples
for i=1:m
  auxJMatrix(i) = -y(i)*log(h(i)) - (1-y(i))*log(1-h(i));
end
 
 %sum up the error values
 sumCost = sum(auxJMatrix);
 
 
%Calculates the Regularization factor for Cost Function (to penalizes huge values of theta)
n = length(theta); 
regularizationCostFactor = zeros(size(theta));
regularizationCostFactor = (lambda/(2*m))*sum(theta(2:n).^2);
 
%Returns the cost function
 J = ((1/m)*sumCost)+regularizationCostFactor;
 
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




% =============================================================

end
