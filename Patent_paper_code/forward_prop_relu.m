function [h_output, prob, loss] = forward_prop(X, y, W1, b1, W2, b2, lambda)
%% forward propagation for our 1 layer network
%% input parameters
% X is our m x n dataset, where m = number of samples, n = number of
% features
% W1 is our h1 x n weight matrix, where h1 = number of hidden units in
% layer 1
% b1 is the length h1 column vector of bias terms associated with layer 1
% W2 is the c x h1 weight matrix, where c = number of classes
% b2 is the length h2 column vector of bias terms associated with the output
%% output parameters
% returns a probability matrix of dimension m x c, where the element in
% position (i, j) corresponds to the probability that sample i is in class
% j
%% Your code here
[m, ~] = size(X);
z1 = W1*X' + repmat(b1, 1, m);
a1 = relu_func(z1);

h_output = a1';

z1 = W2*a1 + repmat(b2, 1, m);
prob = sigmoid_func(z1');

loss = log_loss(prob, y);
% Your code end here
end