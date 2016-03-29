function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%Adds the bias unit in the input layer
a1= [ones(m, 1) X];

%Calculate the linear combination of the input layer and the weitghs for the first layer
z2 = a1*Theta1';

%Computes the input values for the hiiden layer
a2 = sigmoid(z2);

%Adds the bias unit in the hidden layer
a2 = [ones(size(a2,1), 1) a2];

%Computes the values for the output layer
a3 = a2*Theta2';

%extracts the indexes where the max element is found (per column)
[values,p] = (max(a3,[],2));







% =========================================================================


end
