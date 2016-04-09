function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
       
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


%%FOWARD PROGAPAGION
%Adds the bias unit in the input layer
a1= [ones(m, 1) X];
%Calculate the linear combination of the input layer and the weitghs for the first layer
z2 = a1*Theta1';

%Computes the input values for the hiiden layer
a2 = sigmoid(z2);

%Adds the bias unit in the hidden layer
a2 = [ones(size(a2,1), 1) a2];

%Computes the values for the output layer
z3 = a2*Theta2';
 
a3 = sigmoid(z3);

%%COST FUNCTION PROPAGATION
%vector to indicate the current class tested
current_class = eye(num_labels)(y,:);

sumcost = 0; %sum of the cost of each example
cost = 0; %total cost

%Perform the calculation of the cost function 
sumation = (-(current_class).*log(a3)) - ((1-current_class).*log(1-a3)); 

%sum the cost for each example
for i=1:m     
  sumcost(i) = sum(sumation(i,:));
end

%sum the cost for all examples
cost = sum(sumcost);

%%REGULARIZATION


%gets dimensions of Theta
T1size = size(Theta1); 
T2size = size(Theta2);

regularizationFactor=0;

%Calculates the Regularization factor for Cost Function (to penalizes huge values of theta)
if lambda > 0
    
      %Square all of elements of Theta
      regularizationCostFactorT1 = Theta1.^2;
      regularizationCostFactorT2 = Theta2.^2;
      
      %Removes the first column (bias unit) - We do no regularize them
      regularizationCostFactorT1(:,1) = [];
      regularizationCostFactorT2(:,1) = [];
      
      sumRegularizationT2 = zeros(length(T2size(1)));
      sumRegularizationT1 = zeros(length(T1size(1)));
      
      %calculates the sum for each feature
      for i= 1: T1size(1)    
        sumRegularizationT1(i) = sum(regularizationCostFactorT1(i,:));
      end

      for i= 1: T2size(1)    
        sumRegularizationT2(i) = sum(regularizationCostFactorT2(i,:));
      end

      %Calculate them sum for each input/hidden unit
      regularizationT1 = sum(sumRegularizationT1);
      regularizationT2 = sum(sumRegularizationT2);
      
      %Compute the regularization factor
      regularizationFactor = ((lambda/(2*m))*(regularizationT1+regularizationT2));

end

%compute the cost function
J = ((1/m) * cost) + regularizationFactor;
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

%from the foward propagation em J function implementation

%compute delta lower case 3
d3 = a3 - current_class;
%compute delta lower case 2. As Theta2 has the bias unit at the first
%column, we remove this column from the computation
d2 = (d3*Theta2(:,2:T2size(2))).*sigmoidGradient(z2);

%computes capital deltas
delta1 = ((a1)'*d2)';
delta2 = ((a2)'*d3)';

regularizationGradient1 = (lambda/m).*Theta1;
regularizationGradient2 = (lambda/m).*Theta2;

%First columns should not me regularized because they are the bias units
regularizationGradient1(:,1) = 0;
regularizationGradient2(:,1) = 0;

%compute the gradients w/ regularization 
Theta1_grad = ((1/m).*delta1) + regularizationGradient1;
Theta2_grad = (1/m).*delta2 + regularizationGradient2;




















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
