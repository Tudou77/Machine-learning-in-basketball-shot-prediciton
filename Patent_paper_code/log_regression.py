from __future__ import division
import numpy as np

lambda_reg = 0.0001

try:
    xrange
except NameError:
    xrange = range

def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X

def load_data(filename):
    D = np.loadtxt(filename, delimiter=',')
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y

def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y)) + lambda_reg*theta

    return grad

def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    for i in xrange(int(2*1e6)):
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta  - learning_rate * (grad)
        if (i + 1) % 100000 == 0:
            print('Finished %d iterations' % (i+1))
	    print('theta{}: {}'.format(i, theta))
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return theta

def evaluate(output, label):
    error = (output != label).sum()*1. /len(output)
    print 'Error: %1.4f' %error

def main():
    print('==== Training model on data set A ====')
    Xtrain, Ytrain = load_data('logistic_training_small.csv')
    theta = logistic_regression(Xtrain, Ytrain)

    Xtest, Ytest = load_data('logistic_test_small.csv')
    probs = 1/(1+np.exp(-Xtest.dot(theta)))
    labels = np.rint(probs).astype(int)
    labels[np.where(labels == 0)] = -1
    evaluate(labels, Ytest)

    probs = 1/(1+np.exp(-Xtrain.dot(theta)))
    labels = np.rint(probs).astype(int)
    labels[np.where(labels == 0)] = -1
    evaluate(labels, Ytrain)
    return

if __name__ == '__main__':
    main()
