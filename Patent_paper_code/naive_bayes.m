clear;
rng(1)%for reproducibility


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (exist('train_data', 'var') ~= 1)
    train_data = load('train_data.csv');
    test_data = load('test_data.csv');
end
%train_data = train_data(1:200,:);


data = [train_data; test_data];
%bin continuous data
% continuous_features = [5 6 8 9 11 12 15];
% %continuous_features = [9 11 12 15];
% for i=1:length(continuous_features)
%     avg = mean(data(:, continuous_features(i)));
%     std_dev = std(data(:, continuous_features(i)));
%     normalized = (data(:, continuous_features(i)) - avg)/std_dev;
%     categorical_vector = double(normalized <= (avg - std_dev));
%     categorical_vector = categorical_vector + ...
%         2*double(normalized > (avg - std_dev) & normalized <= avg);
%     categorical_vector = categorical_vector + ...
%         3*double(normalized > avg & normalized <= (avg + std_dev));
%     categorical_vector = categorical_vector + ...
%         4*double(normalized > (avg + std_dev));
%     data(:, continuous_features(i)) = categorical_vector;
% end

num_features = size(train_data, 2) - 1;
k_values = zeros(num_features, 1);
for i=1:num_features%replace all values with categorical ones
    [~, ~, ic] = unique(data(:, i + 1));%get all unique values
    k_values(i) = max(ic);
    data(:, i + 1) = ic;
end

train_data = data(1:size(train_data, 1), :);
test_data = data((size(train_data, 1) + 1):end, :);
%%%%%%%%%%%%%

y_train = train_data(:, 1);
y_test = test_data(:, 1);

train_data = train_data(:, 2:end);
test_data = test_data(:, 2:end);

c = 1;
num_train_examples = size(train_data, 1);
p = randperm(size(train_data, 1));

X_train = train_data(p, :);
y_train = y_train(p, :);
X_test = test_data;
num_test_examples = size(test_data, 1);

phis = cell(num_features, 1);
for i=1:num_features
    num_unique_values = k_values(i);
    phi_multinom = zeros(2, num_unique_values);
    for j=1:num_unique_values
        phi_multinom(1, j) = (sum(y_train == 0 & X_train(:, i) == j) + 1)/...
            (num_train_examples + num_unique_values);
        phi_multinom(2, j) = (sum(y_train == 1 & X_train(:, i) == j) + 1)/...
            (num_train_examples + num_unique_values);
    end
    phis{i} = phi_multinom;
end
phi_y = sum(y_train == 1)/num_train_examples;

% num_test_examples = 10;
% y_test = y_test(1:num_test_examples);
output = zeros(num_test_examples, 1);
for i=1:num_test_examples
    post_pos = 0;
    post_neg = 0;
    for j=1:num_features
        phi_multinom = phis{j};
        post_pos = post_pos + log(phi_multinom(2, X_test(i, j)));
        post_neg = post_neg + log(phi_multinom(1, X_test(i, j)));
    end
    post_pos = post_pos + log(phi_y);
    post_neg = post_neg + log(1 - phi_y);
    
    prob_pos = 1/(1 + exp(post_neg - post_pos));
    if prob_pos >= 0.5
        output(i) = 1;
    end
end
disp(['Test accuracy: ', num2str(sum(output == y_test)/num_test_examples)]);

