function loss = log_loss(pred, y)
loss = mean(y.*log(pred) + (1-y).*log(1-pred));
end