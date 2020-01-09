import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch.autograd import Variable
from torch.autograd import grad
from contextlib import contextmanager
from timeit import default_timer
#from utils import *

@contextmanager
def timer(tag=''):
    with open('time_{}.log'.format(tag), 'a') as f:
        start = default_timer() # abrir fichero
        try:
            yield
        finally:
            end = default_timer() # cescribir y cerrar
            f.write(str(end - start) + '\n')
            print("[{}] Elapsed time (s): {:.6f}".format(tag, end - start))

def stand(x, mean, std):
    x = x - mean
    x = x/std
    return(x)

def predict(X, w):
    weights = w[0,:-1].view(1,-1)
    bias = w[0,-1]
    return( X @ weights.t() + bias )

def mse(y, pred, w, lmb):
    diff = y - pred
    return( torch.sum(diff * diff) / diff.numel() + lmb*w @ w.t() )

def rmse(y, pred):
    return torch.sqrt( torch.mean( (pred - y )**2 ) )

def create_train_test(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    x_train = np.asarray(x_train,dtype=np.float32)
    y_train = np.asarray(y_train,dtype=np.float32).reshape(-1,1)
    x_test = np.asarray(x_test,dtype=np.float32)
    y_test = np.asarray(y_test,dtype=np.float32).reshape(-1,1)
    ### to torch
    x_train = Variable( torch.from_numpy(x_train) )
    y_train = Variable( torch.from_numpy(y_train) )
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    x_train_renom = stand(x_train, x_train.mean(dim=0), x_train.std(dim=0))
    x_test_renom = stand(x_test, x_train.mean(dim=0), x_train.std(dim=0))
    return(x_train_renom, y_train, x_test_renom, y_test)

def train_rr(X, y, params):
    w = torch.randn(1, X.shape[1] + 1, requires_grad=True)
    epochs = params["epochs_rr"]
    lr = params["lr_rr"]
    lmb = params["lmb"]
    for epoch in range(epochs):
        epoch += 1
        preds = predict(X, w)
        loss = mse(preds, y, w, lmb)
        loss.backward()
        with torch.no_grad():
            w -= w.grad * lr
            w.grad.zero_()
    return(w)

# Exact solution to the attacker problem
def attack(X, w, c_d, z):
    weights = w[0,:-1].view(1,-1)
    bias = w[0,-1]
    ##
    p1 = ( 1/c_d + weights @ weights.t() )**(-1)
    p1 = torch.diag( p1.squeeze(1) )
    p2 = ( X @ weights.t() - (z - bias) ) @ weights
    out = X - p1 @ p2
    return(out)

def learner_cost_flatten(X, y, w, params):
    X = X.view( -1, w.shape[1]-1 )
    weights = w[0,:-1].view(1,-1)
    bias = w[0,-1]
    return torch.sum( (X @ weights.t() + bias - y)**2 ) +  params["lmb"] * weights @ weights.t()

def attacker_cost_flatten(X, X_clean, w, params):
    c_d = params["c_d_train"]
    z = params["z_train"]
    weights = w[0,:-1].view(1,-1)
    bias = w[0,-1]
    X = X.view(X_clean.shape[0],-1)
    ##
    diff = X_clean - X
    return  torch.sum( c_d*(X @ weights.t() + bias - z)**2 )  +  torch.sum(diff**2)


def compute_backwar_derivative(X_clean, y, w, params):
    ##
    S = params["inner_epochs"]
    ilr = params["inner_lr"]
    ##
    gm = lambda w, X: attacker_cost_flatten(X, X_clean, w, params)
    fm = lambda w, X: learner_cost_flatten(X, y, w, params)
    X = torch.randn(X_clean.shape[0]*X_clean.shape[1], requires_grad=True)
    ##
    Xt = torch.zeros(int(S), X.shape[0])
    ## Inner loop
    for j in range(S):
        grad_X = torch.autograd.grad( gm(w,X) , X, create_graph=True )[0]
        new_X = X - ilr*grad_X
        X = Variable(new_X, requires_grad=True)
        Xt[j] = X ## Store for later usage
    ########
    alpha = -torch.autograd.grad( fm(w,X), X, retain_graph=True )[0]
    gr = torch.zeros_like(w)
    ########
    for j in range(S-1,-1,-1):
        X_tmp = Variable(Xt[j], requires_grad=True)
        grad_X, = torch.autograd.grad( gm(w, X_tmp), X_tmp, create_graph=True )
        loss = X_tmp - ilr*grad_X
        loss = loss@alpha ## To compute Hessian Vector Product
        aux1 = torch.autograd.grad(loss, w, retain_graph=True)[0]
        aux2 = torch.autograd.grad(loss, X_tmp)[0]
        gr -= aux1
        alpha = aux2

    grad_w = torch.autograd.grad( fm(w, X), w )[0]
    ##
    return grad_w + gr

def train_nash_rr_test(X_clean, y, params, verbose = False):
    lr = params["outer_lr"]
    T = params["outer_epochs"]
    w = torch.randn(1, X_clean.shape[1] + 1, requires_grad=True)
    X = torch.randn(X_clean.shape[0]*X_clean.shape[1], requires_grad=True)
    fm = lambda w, X: learner_cost_flatten(X, y, w, params)

    for i in range(T):
        grad = compute_backwar_derivative(X_clean, y, w, params)
        w = w - lr*grad
        if verbose:
            if i%10 == 0:
                print( 'epoch {}, loss {}'.format(i,fm(w,X)) )
    return w

def train_nash_rr(X_clean, y, params, verbose = False):
    lr = params["outer_lr"]
    ilr = params["inner_lr"]
    T = params["outer_epochs"]
    S = params["inner_epochs"]
    ##
    X = torch.randn(X_clean.shape[0]*X_clean.shape[1], requires_grad=True)
    w = torch.randn(1, X_clean.shape[1] + 1, requires_grad=True)
    ##
    gm = lambda w, X: attacker_cost_flatten(X, X_clean, w, params)
    fm = lambda w, X: learner_cost_flatten(X, y, w, params)
    ##
    Xt = torch.zeros(int(S), X.shape[0])
    ##
    for i in range(T):
        ## Inner loop
        for j in range(S):
            grad_X = torch.autograd.grad( gm(w,X) , X, create_graph=True )[0]
            new_X = X - ilr*grad_X
            X = Variable(new_X, requires_grad=True)
            Xt[j] = X ## Store for later usage
        ########
        alpha = -torch.autograd.grad( fm(w,X), X, retain_graph=True )[0]
        gr = torch.zeros_like(w)
        ########
        for j in range(S-1,-1,-1):
            X_tmp = Variable(Xt[j], requires_grad=True)
            grad_X, = torch.autograd.grad( gm(w, X_tmp), X_tmp, create_graph=True )
            loss = X_tmp - ilr*grad_X
            loss = loss@alpha ## To compute Hessian Vector Product
            aux1 = torch.autograd.grad(loss, w, retain_graph=True)[0]
            aux2 = torch.autograd.grad(loss, X_tmp)[0]
            gr -= aux1
            alpha = aux2

        grad_w = torch.autograd.grad( fm(w, X), w )[0]
        ##
        w = w - lr*(grad_w + gr)
        if verbose:
            if i%10 == 0:
                print( 'epoch {}, loss {}'.format(i,fm(w,X)) )
    return w





if __name__ == '__main__':

    #c_d =
    #z = torch.zeros([len(y_train),1])
    # Read wine dataset
    data = pd.read_csv("data/winequality-white.csv", sep = ";")
    X = data.loc[:, data.columns != "quality"]
    y = data.quality
    ##
    pca = PCA(n_components=X.shape[1], svd_solver='full')
    pca.fit(X)
    X = pca.fit_transform(X)
    ##
    X_train, y_train, X_test, y_test = create_train_test(X,y)
    ## Parameters
    params = {
        "epochs_rr"    : 1000,
        "lr_rr"        : 0.01,
        "lmb"          : 0.0,
        "c_d_train"    : torch.ones([len(y_train), 1])*0.5,
        "z_train"      : torch.zeros([len(y_train),1]),
        "c_d_test"    : torch.ones([len(y_test), 1])*0.5,
        "z_test"      : torch.zeros([len(y_test),1]),
        "outer_lr"     : 10e-6,
        "inner_lr"     : 0.01,
        "outer_epochs" : 350,
        "inner_epochs" : 100
    }
    ##
    with timer(tag='a'):
        w_rr = train_rr(X_train, y_train, params)
    ##
    X_test_attacked = attack(X_test, w_rr, params["c_d_test"], params["z_test"])
    pred_attacked =  predict(X_test_attacked, w_rr)
    pred_clean    =  predict(X_test, w_rr)
    #
    print("RR clean test RMSE: ",   rmse( y_test, pred_clean ) )
    print("RR attacked test RMSE: ", rmse( y_test, pred_attacked ) )
    ##
    with timer(tag='b'):
        w_nash = train_nash_rr(X_train, y_train, params)
    ##
    X_test_attacked = attack(X_test, w_nash, params["c_d_test"], params["z_test"])
    pred_attacked =  predict(X_test_attacked, w_nash)
    pred_clean    =  predict(X_test, w_nash)
    #
    print("Nash clean test RMSE: ",   rmse( y_test, pred_clean ) )
    print("Nash attacked test RMSE: ", rmse( y_test, pred_attacked ) )
    ##
