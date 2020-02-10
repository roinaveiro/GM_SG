# Some useful functions
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad

# Auxiliar function to standarize dataset
def stand(x, mean, std):
    x = x - mean
    x = x/std
    return(x)

# Root Mean Squared Error
def rmse(y, pred):
    return torch.sqrt( torch.mean( (pred - y )**2 ) )

## Linear Model
def model(x, w, b):
    return( x @ w.t() + b )

## Cost for raw ridge regression
def mse(t1, t2, w, lmb):
    diff = t1 - t2
    return( torch.sum(diff * diff) / diff.numel() + lmb*w @ w.t() )

## Attack dataset
def attack(w, b, x, c_d, z):
    c_d = ( 1/c_d + w @ w.t() )**(-1)
    p1 = torch.diag( c_d[0] )
    p2 = ( x @ w.t() + b - z)@w
    out = x - p1 @ p2
    return(out)

## Cost in adversary aware case
def learner_cost(w, b, x, y, lmb, c_d, z):
    out = attack(w, b, x, c_d, z)
    return torch.sum( (out @ w.t() + b - y)**2 ) +  lmb * w @ w.t()

## Train raw ridge regression model
def train_model(model, mse, x_train, y_train, lr, epochs, lmb):
    ##
    w = torch.randn(1, x_train.shape[1], requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    ##
    for epoch in range(epochs):
        epoch += 1
        preds = model(x_train, w, b)
        loss = mse(preds, y_train, w, lmb)
        loss.backward()
        with torch.no_grad():
            w -= w.grad * lr
            b -= b.grad * lr
            w.grad.zero_()
            b.grad.zero_()
    return(w, b)

## Train Nash defense
def train_nash_model(learner_cost, x_train, y_train, mean, z, lr, epochs, lmb):
    ##
    w_nash = torch.randn(1, x_train.shape[1], requires_grad=True)
    b_nash = torch.randn(1, requires_grad=True)
    ##
    c_d = torch.ones(len(y_train))*mean
    for epoch in range(epochs):
        epoch += 1
        loss = learner_cost(w_nash, b_nash, x_train, y_train, lmb, c_d, z)
        loss.backward()
        with torch.no_grad():
            w_nash -= w_nash.grad * lr
            b_nash -= b_nash.grad * lr
            w_nash.grad.zero_()
            b_nash.grad.zero_()
    return(w_nash, b_nash)

## Train Bayes-Nash defense
def train_bayes_model(learner_cost, x_train, y_train, m, n_samples, z, lr, epochs, lmb):
    ##
    w_bayes = torch.randn(1, x_train.shape[1], requires_grad=True)
    b_bayes = torch.randn(1, requires_grad=True)
    ##
    for epoch in range(epochs):
        epoch += 1
        wgrad = torch.zeros(1, x_train.shape[1])
        bgrad = torch.zeros(1)
        sample = m.sample(torch.Size([n_samples, len(y_train)]))
        ### Forma cutre. Vectorizar !!
        for i in range(n_samples):
            c_d = sample[i].t()[0]
            loss = learner_cost(w_bayes, b_bayes, x_train, y_train, lmb, c_d, z)
            loss.backward()
            wgrad += w_bayes.grad
            bgrad += b_bayes.grad
            w_bayes.grad.zero_()
            b_bayes.grad.zero_()
        ####
        wgrad /= n_samples
        bgrad /= n_samples

        with torch.no_grad():
            w_bayes -= wgrad * lr
            b_bayes -= bgrad * lr
    #
    return(w_bayes, b_bayes)
