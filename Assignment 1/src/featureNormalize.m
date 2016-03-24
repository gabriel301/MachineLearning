function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
mu = mean(X); %calculate the mean of each column of X  - Returns a vector 1xn, if X is mxn
sigma = std(X); %calculate the standard deviation of each column of X  - Return a vector 1xn, if X is mxn
m = size(X, 1); % extract the number of rows of X; if X is mxn, returns m
mu_matrix = ones(m, 1) * mu  ; %creates an vector m x 1 filled all with 1's, an multiply with the mean vector (1xn)
                                %returning a mxn matrix.
sigma_matrix = ones(m, 1) * sigma; %creates an vector m x 1 filled all with 1's, an multiply with the standard deviation vector (1xn)
                                %returning a mxn matrix.
X_norm = (X - mu_matrix);
for i=1:size(X, 1)
  for j=1:size(X, 2)
    X_norm(i,j) = X_norm(i,j)/sigma_matrix(i,j);
  end
end





% ============================================================

end
