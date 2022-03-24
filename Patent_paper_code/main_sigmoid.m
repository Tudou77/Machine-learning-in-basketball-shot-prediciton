%% importing, normalizing and splitting data
rng(20171115, 'twister');
if (exist('train_data', 'var') ~= 1)
    train_data = load('train_data_big.csv');
    test_data = load('test_data.csv');
end

y_train = train_data(:, 1);
y_test = test_data(:, 1);

train_data = train_data(:, 2:end);
test_data = test_data(:, 2:end);

n = size(train_data, 2);
c = 1;
train_size = 90000;
p = randperm(size(train_data, 1));

X_train_all = train_data(p, :);
y_train_all = y_train(p, :);
X_train = X_train_all(1:train_size, :);
X_dev = X_train_all(train_size+1:end, :);
y_train = y_train_all(1:train_size, :);
y_dev = y_train_all(train_size+1:end, :);
X_test = test_data;


%normalize the data
avg = mean(mean(X_train));
s = std(reshape(X_train, [], 1));

X_train = (X_train - avg) / s;
X_dev = (X_dev - avg) / s;
X_test = (X_test - avg) / s;

%% training the network (without regularization)
m = train_size;
h1 = 50; % 50 units in hidden layer

% initialize the parameters
W1 = randn(h1, n);
b1 = zeros(h1, 1);
W2 = randn(c, h1);
b2 = zeros(c, 1);

num_epoch = 25;
batch_size = 1000;
num_batch = m / batch_size;
learning_rate = 5;
lambda = 0.0001;
train_loss = zeros(num_epoch, 1);
dev_loss = zeros(num_epoch, 1);
train_accuracy = zeros(num_epoch, 1);
dev_accuracy = zeros(num_epoch, 1);

%% Your code here
for i=1:num_epoch
    for j=1:num_batch
        start_ind = 1 + (j-1)*batch_size;
        X_batch = X_train(start_ind:(start_ind + batch_size - 1), :);
        y_batch = y_train(start_ind:(start_ind + batch_size - 1), :);
        [hidden_output, batch_preds, batch_loss] = forward_prop_sigmoid(X_batch, y_batch, W1, b1, W2, b2, lambda);
        [dW1, db1, dW2, db2] = backward_prop_sigmoid(X_batch, batch_preds, y_batch, batch_size, hidden_output...
            ,W1, b1, W2, b2, lambda);
        W1 = W1 - learning_rate*dW1;
        b1 = b1 - learning_rate*db1;
        W2 = W2 - learning_rate*dW2;
        b2 = b2 - learning_rate*db2;

        train_loss(i) = train_loss(i) + batch_loss*batch_size;
        
        y_c_train_pred = batch_preds > 0.5;
        train_accuracy(i) = train_accuracy(i) + sum(y_c_train_pred == y_batch);
    end
    train_loss(i) = train_loss(i)/m;
    train_accuracy(i) = train_accuracy(i)/m;
    
    % perform forward prop on dev set
    [~, dev_preds, dev_set_loss] = forward_prop_sigmoid(X_dev, y_dev, W1, b1, W2, b2, lambda);
    dev_loss(i) = dev_set_loss;
    
    y_c_dev_pred = dev_preds > 0.5;
    dev_accuracy(i) = sum(y_c_dev_pred == y_dev) / size(y_dev, 1);
end
% Your code end here
%% plotting and displaying results
figure(1);
plot(1:num_epoch, train_loss, 'r', ...
    1:num_epoch, dev_loss, 'b');
legend('training set', 'dev set');
xlabel('epochs');
ylabel('loss');
figure(2);
plot(1:num_epoch, train_accuracy, 'r', ...
    1:num_epoch, dev_accuracy, 'b');
legend('training set', 'dev set');
xlabel('epochs');
ylabel('accuracy');


lambda = 0.0001;

[~, test_pred, ~] = forward_prop_sigmoid(X_test, y_test, W1, b1, W2, b2, lambda);
y_c_test_pred = test_pred > 0.5;
test_accuracy = sum(y_c_test_pred == y_test) / size(y_test, 1);
fprintf('test set accuracy: %f \n', test_accuracy);
