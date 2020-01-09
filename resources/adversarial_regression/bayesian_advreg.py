import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
from utils import *

N_EXP = 10 # For hold-out validation
q = 0.33 # Size of test set
LMB = 0.0 # Shrinkage parameter
EPOCHS = 1000
LR = 10e-6
start = 0.01
stop = 1.0
grid_size = 10
MEAN_GRID = np.logspace(np.log10(start), np.log10(stop), num=grid_size)
VAR_GRID = np.logspace(np.log10(start), np.log10(stop), num=grid_size)
N_SAMPLES = 10

# Read the dataset
data = pd.read_csv("data/winequality-white.csv", sep = ";")
X = data.loc[:, data.columns != "quality"]
y = data.quality

# Extract first components using PCA
pca = PCA(n_components=11, svd_solver='full')
pca.fit(X)
x = pca.fit_transform(X)
rmse_raw_clean = np.zeros(N_EXP)
rmse_nash_clean = np.zeros(N_EXP)
rmse_bayes_clean = np.zeros(N_EXP)
rmse_raw_at = np.zeros(N_EXP)
rmse_nash_at = np.zeros(N_EXP)
rmse_bayes_at = np.zeros(N_EXP)

for MEAN in MEAN_GRID:
    for VAR in VAR_GRID:
        for i in range(N_EXP):
            ### This should be in loop
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=q)
            x_train = np.asarray(x_train,dtype=np.float32)
            y_train = np.asarray(y_train,dtype=np.float32).reshape(-1,1)
            x_test = np.asarray(x_test,dtype=np.float32) #un poco trampa
            y_test = np.asarray(y_test,dtype=np.float32).reshape(-1,1)
            ### to torch
            x_train = Variable( torch.from_numpy(x_train) )
            y_train = Variable( torch.from_numpy(y_train) )
            x_test = torch.from_numpy(x_test)
            y_test = torch.from_numpy(y_test)
            ### standarize
            x_train_renom = stand(x_train, x_train.mean(dim=0), x_train.std(dim=0))
            x_test_renom = stand(x_test, x_train.mean(dim=0), x_train.std(dim=0))

            Z = torch.zeros([len(y_train),1])
            M = torch.distributions.Gamma(torch.tensor([MEAN**2/VAR]), torch.tensor([MEAN/VAR])) ## shape, rate
            w, b = train_model(model, mse, x_train_renom, y_train, lr=0.01, epochs=1000, lmb=LMB)
            #
            w_nash, b_nash = train_nash_model(learner_cost, x_train_renom, y_train, mean=MEAN,
             z=Z, lr=LR, epochs=EPOCHS, lmb=LMB)
            #
            w_bayes, b_bayes = train_bayes_model(learner_cost, x_train_renom, y_train,
             m=M, n_samples=N_SAMPLES, z=Z, lr=LR, epochs=EPOCHS, lmb=LMB)
            ##
            # Compute metrics
            sample_test = M.sample(torch.Size([len(y_test)]))
            c_d = sample_test.t()[0]
            Z_TEST = torch.zeros([len(y_test),1])
            #
            pred_clean = model(x_test_renom, w, b)
            pred_at = model(attack(w, b, x_test_renom, c_d, Z_TEST), w, b)
            rmse_raw_clean[i] = rmse(pred_clean, y_test)
            rmse_raw_at[i] = rmse(pred_at, y_test)
            #
            pred_clean = model(x_test_renom, w_nash, b_nash)
            pred_at = model(attack(w_nash, b_nash, x_test_renom, c_d, Z_TEST), w_nash, b_nash)
            rmse_nash_clean[i] = rmse(pred_clean, y_test)
            rmse_nash_at[i] = rmse(pred_at, y_test)
            #
            pred_clean = model(x_test_renom, w_bayes, b_bayes)
            pred_at = model(attack(w_bayes, b_bayes, x_test_renom, c_d, Z_TEST), w_bayes, b_bayes)
            rmse_bayes_clean[i] = rmse(pred_clean, y_test)
            rmse_bayes_at[i] = rmse(pred_at, y_test)
            print("Experiment: ", i)

        df =  pd.DataFrame({"EXP":range(N_EXP), "raw_cleandata":rmse_raw_clean,
         "raw_atdata":rmse_raw_at, "nash_rawdata":rmse_nash_clean, "nash_atdata":rmse_nash_at,
          "bayes_rawdata":rmse_bayes_clean, "bayes_atdata":rmse_bayes_at})
        #
        name = "results/"+"mean"+str(MEAN)+"var"+str(VAR)+".csv"
        df.to_csv(name, index=False)
