function [dW1, db1, dW2, db2] = backward_prop(X, batch_preds, y, batch_size, h_output, W1, b1, W2, b2, lambda)
%% backward propagation for our 1 layer network
%% input parameters
% X is our m x n dataset, where m = number of samples, n = number of
% features
% y is the length m label vector for each sample
% W1 is our h1 x n weight matrix, where h1 = number of hidden units in
% layer 1
% b1 is the length h1 column vector of bias terms associated with layer 1
% W2 is the c x h1 weight matrix, where c = number of classes
% b2 is the length h2 column vector of bias terms associated with the output
%% output parameters
% returns the gradient of W1, b1, W2, b2 as dW1, db1, dW2, db2
%% Your code here
h_output = h_output';

delta_2 = (batch_preds - y)';
delta_1 = (W2'*delta_2).*(h_output.*(1-h_output));

dW2 = zeros(1, 50);
dW1 = zeros(50, 11);
for i=1:batch_size
    dW2 = dW2 + delta_2(:, i)*h_output(:, i)';
    dW1 = dW1 + delta_1(:, i)*X(i, :);
end
dW2 = dW2/batch_size + lambda*W2;
dW1 = dW1/batch_size + lambda*W1;
db2 = sum(delta_2, 2)/batch_size;
db1 = sum(delta_1, 2)/batch_size;
% Your code end here
end