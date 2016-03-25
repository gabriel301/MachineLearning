function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%sumErr = 0;
h = (X*theta);
error = h-y;
v_sqr = error.^2;
sumError = sum(v_sqr);
J = (1/(2*m))*sumError;
%for ind=1:m
%  ind
%  hyp = ((theta(1)*X(ind,1))+(theta(2)*X(ind,2)))
%  err = (hyp - y(ind))^2
%  sumErr = sumErr + err
%  pause
%end
%J = (1/(2*m))*sumErr;


% =========================================================================

end
