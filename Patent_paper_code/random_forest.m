%TAKES RANDOM SAMPLES OF SAMPLE OBSERVATIONS FROM _all_ COLUMNS-TREEBAGGER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

rng(1)%for reproducibility

numTimesToSample = 1;%10

numTrees = 80;
numMinLeaf = 50;%usually 50

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (exist('train_data', 'var') ~= 1)
    train_data = load('train_data.csv');
    test_data = load('test_data.csv');
end

%train_data = [train_data(:, 1) train_data(:, 8) train_data(:, 10:11) train_data(:, 14)];
%test_data = [test_data(:, 1) test_data(:, 8) test_data(:, 10:11) test_data(:, 14)];
%%%%%%%%%%%%%

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
% avg = mean(mean(X_train));
% s = std(reshape(X_train, [], 1));
% 
% X_train = (X_train - avg) / s;
% X_dev = (X_dev - avg) / s;
% X_test = (X_test - avg) / s;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%store predictions here
prediction_vector = zeros(size(y_test));




%place test prediction variables in an array

%testPredictors = [];




tic;



%SAMPLE FROM NEW POPULATION EVERY TIME, OR USE SAME POP MULTIPLE TIMES????

%     
%     
%     trainPredictors = sample(:, 1:numCols);
%     y =  sample(:, numCols + 1);
%     clear sample;


disp(['training ensemble...']);
cat_preds = [true, true, true, false, false, true, false, false, false, false, false, true, true, false, true];
BaggedEnsemble = TreeBagger(numTrees, X_train, y_train, 'Method', 'classification', ...
'Minleaf', numMinLeaf, 'FBoot', 0.5, 'SampleWithReplacement', 'Off');%, 'CategoricalPredictors', cat_preds);

%     clear trainPredictors;
%     clear Clicked;

%Predictions for new data
disp('predicting...');
[~, scores] = predict(BaggedEnsemble, X_train);
pred_train = double(scores(:, 2) > 0.5);
train_accuracy = sum(pred_train == y_train)/length(y_train)

[~, scores] = predict(BaggedEnsemble, X_dev);
%prediction_vector = prediction_vector + scores(:,2);
pred_dev = double(scores(:, 2) > 0.5);
dev_accuracy = sum(pred_dev == y_dev)/length(y_dev)

[~, scores] = predict(BaggedEnsemble, X_test);
pred_test = double(scores(:, 2) > 0.5);
test_accuracy = sum(pred_test == y_test)/length(y_test)


toc; 






prediction_vector = prediction_vector ./(numTimesToSample);

%worse score
%     |         0.6135 40 trees - some categorical
%     V         0.6192 40 trees - none categorical

%better score
