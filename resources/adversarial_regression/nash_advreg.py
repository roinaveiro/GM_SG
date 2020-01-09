import pandas as pd
import numpy as np
import torch
from utils import *
from torch.autograd import Variable
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

## PARAMETERS FOR THE EXPERIMENT
N_EXP = 1 # For hold-out validation
q = 0.33 # Size of test set
LMB = 0.0 # Shrinkage parameter
EPOCHS = 100
OUT_LR = 10e-6
IN_LR = 0.01
start = 0.5
stop = 1.0
grid_size = 1
MEAN_GRID = np.logspace(np.log10(start), np.log10(stop), num=grid_size)

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
rmse_raw_at = np.zeros(N_EXP)
rmse_nash_at = np.zeros(N_EXP)


for MEAN in MEAN_GRID:
    for i in range(N_EXP):
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
        w, b = train_model(model, mse, x_train_renom, y_train, lr=0.01, epochs=1000, lmb=LMB)
        #
        w_nash, b_nash = train_nash_model(learner_cost, x_train_renom, y_train, mean=MEAN,
            z=Z, lr=OUT_LR, epochs=EPOCHS, lmb=LMB)
        #
        w_bw = train_nash_backward(learner_cost_flatten, attacker_cost_flatten,
            x_train_renom, y_train, LMB, MEAN, Z, epochs=EPOCHS, out_lr=OUT_LR, in_lr=IN_LR)
        #
        print(w_bw)
        print(w_nash, b_nash)
        # Compute metrics
        Z_TEST = torch.zeros([len(y_test),1])
        #
        pred_clean = model(x_test_renom, w, b)
        pred_at = model(attack(w, b, x_test_renom, MEAN, Z_TEST), w, b)
        rmse_raw_clean[i] = rmse(pred_clean, y_test)
        rmse_raw_at[i] = rmse(pred_at, y_test)
        #
        pred_clean = model(x_test_renom, w_nash, b_nash)
        pred_at = model(attack(w_nash, b_nash, x_test_renom, MEAN, Z_TEST), w_nash, b_nash)
        rmse_nash_clean[i] = rmse(pred_clean, y_test)
        rmse_nash_at[i] = rmse(pred_at, y_test)
        #
        w_nash_bw = w_bw[0][:-1]
        b_nash_bw = w_bw[0][-1]
        pred_clean = model(x_test_renom, w_nash_bw, b_nash_bw)
        pred_at = model(attack(w_nash_bw, b_nash_bw, x_test_renom, MEAN, Z_TEST), w_nash, b_nash)
        rmse_nash_bw_clean[i] = rmse(pred_clean, y_test)
        rmse_nash_bw_at[i] = rmse(pred_at, y_test)
        #
        print("Experiment: ", i)

    df =  pd.DataFrame({"EXP":range(N_EXP), "raw_cleandata":rmse_raw_clean,
     "raw_atdata":rmse_raw_at, "nash_rawdata":rmse_nash_clean, "nash_atdata":rmse_nash_at,
      "nash_bw_rawdata":rmse_nash_bw_clean, "nash_bw_atdata":rmse_nash_bw_at})
    #
    name = "results/"+"mean"+str(MEAN)+".csv"
    df.to_csv(name, index=False)
