import numpy as np
import torch
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

## Flatten learner cost
def learner_cost_flatten(w, x, y, lmb):
    x = x.view(-1,w.shape[1]-1)
    weights = w[0,:-1].view(1,-1)
    bias = w[0,-1]
    return torch.sum( (x @ weights.t() + bias - y)**2 ) +  lmb * weights @ weights.t()

## Flatten attacker cost
def attacker_cost_flatten(w, x, x_old, c_d, z):
    weights = w[0,:-1].view(1,-1)
    bias = w[0,-1]
    x = x.view(x_old.shape[0],-1)
    ##
    diff = x_old - x
    return  torch.sum( c_d*(x @ weights.t() + bias)**2 )  +  torch.sum(diff**2)

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

## Train Nash defense using backward method
def train_nash_backward(learner_cost_flatten, attacker_cost_flatten, x_clean, y_train,
    lmb, c_d, z, epochs=300, out_lr=10e-6, in_lr=0.01):

    ## Initialize parameters
    w = torch.randn(1, x_clean.shape[1] + 1, requires_grad=True)
    x = torch.randn(x_clean.shape[0]*x_clean.shape[1],
        requires_grad=True)
    ##
    gm = lambda w, x: attacker_cost_flatten(w, x, x_clean, c_d, z)
    fm = lambda w, x: learner_cost_flatten(w, x, y_train, lmb)
    ##
    xt = torch.zeros(int(epochs), x.shape[0])

    for i in range(epochs):
        # We nee to compute the total derivative of f wrt x
        ##
        for j in range(epochs):
            grad_x = torch.autograd.grad(gm(w,x), x, create_graph=True)[0]
            new_x = x - in_lr * grad_x
            x = Variable(new_x, requires_grad=True)
            xt[j] = x
        ## CHECK WITH ANALYTICAL SOLUTION
        ###
        alpha = -torch.autograd.grad(fm(w,x), x, retain_graph=True)[0]
        gr = torch.zeros_like(w)
        ###
        for j in range(epochs-1,-1,-1):
            x_tmp = Variable(xt[j], requires_grad=True)
            grad_x, = torch.autograd.grad( gm(w,x_tmp), x_tmp, create_graph=True )
            loss = x_tmp - in_lr * grad_x
            loss = loss@alpha
            aux1 = torch.autograd.grad(loss, w, retain_graph=True)[0]
            aux2 = torch.autograd.grad(loss, x_tmp)[0]
            gr -= aux1
            alpha = aux2

        grad_w = torch.autograd.grad(fm(w,x), w)[0]
        ##
        w = w - out_lr * (grad_w + gr)
        #if i%10 == 0:
        #    print( 'epoch {}, loss {}'.format(i,fm(w,x)) )
    return w
