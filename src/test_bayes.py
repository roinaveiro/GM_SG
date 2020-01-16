import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.autograd import Variable
from torch.autograd import grad
from nash_advreg import *

data = pd.read_csv("../data/winequality-white.csv", sep = ";")
X = data.loc[:, data.columns != "quality"]
y = data.quality
##
pca = PCA(n_components=X.shape[1], svd_solver='full')
pca.fit(X)
X = pca.fit_transform(X)
##
MEAN = 0.5
VAR = 0.01
m = torch.distributions.Gamma(torch.tensor([MEAN**2/VAR]), torch.tensor([MEAN/VAR])) ## shape, rate
##
X_train, y_train, X_test, y_test = create_train_test(X,y)
params = {
                "epochs_rr"    : 1000,
                "lr_rr"        : 0.01,
                "lmb"          : 0.0,
                "c_d_train"    : torch.ones([len(y_train), 1]) * MEAN,
                "z_train"      : torch.zeros([len(y_train),1]),#.to("cuda"),
                "c_d_test"     : torch.ones([len(y_test), 1]) * MEAN,
                "z_test"       : torch.zeros([len(y_test),1]),#.to("cuda"),
                "outer_lr"     : 10e-6,
                "inner_lr"     : 0.01,
                "outer_epochs" : 200,
                "inner_epochs" : 100,
                "n_samples"    : 10,
                "prior"        : m  
            }

##
c_d_train = params["prior"].sample(torch.Size([params["n_samples"], len(y_train)]))#.to("cuda")
##
print("hola")
##
w_rr = train_rr(X_train, y_train, params)
w_nash = train_nash_rr(X_train, y_train, params)
w_bayes = train_bayes_rr_test(X_train, y_train, c_d_train, params, verbose = True)
##

c_d_test = params["prior"].sample(torch.Size([1, len(y_test)]))[0]
##
X_test_attacked = attack(X_test, w_rr, c_d_test, params["z_test"])
pred_attacked =  predict(X_test_attacked, w_rr)
pred_clean    =  predict(X_test, w_rr)
#
rmse_raw_clean = rmse( y_test, pred_clean )
rmse_raw_at    = rmse( y_test, pred_attacked )
#
##
X_test_attacked = attack(X_test, w_nash, c_d_test, params["z_test"])
pred_attacked =  predict(X_test_attacked, w_nash)
pred_clean    =  predict(X_test, w_nash)
#
rmse_nash_clean = rmse( y_test, pred_clean )
rmse_nash_at    = rmse( y_test, pred_attacked )
##
X_test_attacked = attack(X_test, w_bayes, c_d_test, params["z_test"])
pred_attacked =  predict(X_test_attacked, w_bayes)
pred_clean    =  predict(X_test, w_bayes)
#
rmse_bayes_clean = rmse( y_test, pred_clean )
rmse_bayes_at    = rmse( y_test, pred_attacked )
#####
#####
print("____Non-Strategic Defender____")
###
print( "Loss Clean test set: ", rmse_raw_clean )
###
print( "Loss attacked test set: ", rmse_raw_at )
###
print("\n____Strategic Bayes Defender____")
###
print( "Loss Bayes Clean test set: ", rmse_bayes_clean )
###
print( "Loss Bayes attacked test set: ", rmse_bayes_at )
###
print("\n____Strategic Nash Defender____")
###
print( "Loss Nash Clean test set: ", rmse_nash_clean )
###
print( "Loss Nash attacked test set: ", rmse_nash_at )
###
