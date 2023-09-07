import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
import sys


from src.theory import y_pred_from_full_connectivity, y_corrected_quad, y_0_quad
from src.correlation_functions import rate, mean_by_region



if len(sys.argv) < 2:
    f = open("../results/most_recent.txt", "r")
    dirname = f.read()
else:
    dirname = sys.argv[1]

with open(dirname+"/index_dict.pkl", "rb") as file:
    index_dict = pkl.load(file)

with open(dirname + "/param_dict.pkl", "rb") as file:
    param_dict = pkl.load(file)

b = param_dict["b"]
N = param_dict["N"]
h_min = param_dict["h_min"]
h_max = param_dict["h_max"]
n_h = param_dict["n_h"]
g_ii = param_dict["g_ii"]
h_range = np.linspace(h_min, h_max, n_h)


region_list = ['' for i in range(N)]
inputs_quad = []
for key in index_dict:
    start = min(index_dict[key])
    end = max(index_dict[key])
    region_list[start:end+1] = (end+1-start)*[key]


region_list = len(h_range) * region_list

h_list = [h for h in h_range for i in range(N)]




pred_rates_l = []
pred_rates_q = []
tree_level_rates_q = []
for h in h_range:
    with open(dirname + "/W_l_h={}.pkl".format(h), "rb") as file:
            W_l = pkl.load(file)

    with open(dirname + "/W_q_h={}.pkl".format(h), "rb") as file:
            W_q = pkl.load(file)

    y0 = b*np.ones(N)
 
    y_l = y_pred_from_full_connectivity(W_l, y0, index_dict)
    y_q_0 = y_0_quad(W_q,y0)
    correction = 0#y_corrected_quad(W_q,y0, y_q_0)
    y_q = y_q_0  + correction 
    inputs_quad.extend(W_q @ y_q + b) 
    print(np.max(np.linalg.eigvals(2*np.diag(y_q)@W_q)))


    pred_rates_q.extend(y_q)
    pred_rates_l.extend(y_l)
    tree_level_rates_q.extend(y_q_0)

df = pd.read_csv(dirname+"/lin_vs_quad.csv")
df["pred_rate_l"] = pred_rates_l
df["pred_rate_q"] = pred_rates_q
df["tree_level_rate"] = tree_level_rates_q
df["input"] = inputs_quad

df.to_csv(dirname+"/lin_vs_quad.csv")
