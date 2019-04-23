import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch.autograd import Variable
from torch.autograd import grad
from nash_advreg import *

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
    start = 0.01
    stop = 1.0
    grid_size = 10
    MEAN_GRID = np.logspace(np.log10(start), np.log10(stop), num=grid_size)
    N_EXP = 10 # For hold-out validation
    ##
    rmse_raw_clean = np.zeros(N_EXP)
    rmse_nash_clean = np.zeros(N_EXP)
    rmse_raw_at = np.zeros(N_EXP)
    rmse_nash_at = np.zeros(N_EXP)
    ##
    for MEAN in MEAN_GRID:
        for i in range(N_EXP):
            X_train, y_train, X_test, y_test = create_train_test(X,y)
            ## Parameters
            params = {
                "epochs_rr"    : 1000,
                "lr_rr"        : 0.01,
                "lmb"          : 0.0,
                "c_d_train"    : torch.ones([len(y_train), 1]) * MEAN,
                "z_train"      : torch.zeros([len(y_train),1]),
                "c_d_test"     : torch.ones([len(y_test), 1]) * MEAN,
                "z_test"       : torch.zeros([len(y_test),1]),
                "outer_lr"     : 10e-6,
                "inner_lr"     : 0.01,
                "outer_epochs" : 350,
                "inner_epochs" : 100
            }
            ##
            with timer(tag='raw'):
                w_rr = train_rr(X_train, y_train, params)
            ##
            X_test_attacked = attack(X_test, w_rr, params["c_d_test"], params["z_test"])
            pred_attacked =  predict(X_test_attacked, w_rr)
            pred_clean    =  predict(X_test, w_rr)
            #
            rmse_raw_clean[i] = rmse( y_test, pred_clean )
            rmse_raw_at[i]    = rmse( y_test, pred_attacked )
            #
            # print("RR clean test RMSE: ",   rmse( y_test, pred_clean ) )
            # print("RR attacked test RMSE: ", rmse( y_test, pred_attacked ) )
            ##
            with timer(tag='nash'):
                w_nash = train_nash_rr(X_train, y_train, params)
            ##
            X_test_attacked = attack(X_test, w_nash, params["c_d_test"], params["z_test"])
            pred_attacked =  predict(X_test_attacked, w_nash)
            pred_clean    =  predict(X_test, w_nash)
            #
            rmse_nash_clean[i] = rmse( y_test, pred_clean )
            rmse_nash_at[i]    = rmse( y_test, pred_attacked )
            #
            # print("Nash clean test RMSE: ",   rmse( y_test, pred_clean ) )
            # print("Nash attacked test RMSE: ", rmse( y_test, pred_attacked ) )
            ##
        df = pd.DataFrame({"EXP":range(N_EXP), "raw_cleandata":rmse_raw_clean,
         "raw_atdata":rmse_raw_at, "nash_rawdata":rmse_nash_clean, "nash_atdata":rmse_nash_at})

        name = "results/exp1/"+"mean"+str(MEAN)+".csv"
        df.to_csv(name, index=False)
