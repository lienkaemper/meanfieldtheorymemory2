import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
import sys


from src.theory import y_0_quad, loop_correction, cor_pred
from src.correlation_functions import sum_by_region
from src.generate_connectivity import  hippo_weights




if len(sys.argv) < 2:
    f = open("../results/most_recent.txt", "r")
    dirname = f.read()
else:
    dirname = sys.argv[1]

with open(dirname+"/index_dict.pkl", "rb") as file:
    index_dict = pkl.load(file)

with open(dirname+"/adjacency.pkl", "rb") as file:
    A = pkl.load(file)

with open(dirname + "/param_dict.pkl", "rb") as file:
    param_dict = pkl.load(file)



N = param_dict["N"]
Ns = np.array([len(R) for R in index_dict.values()])
h_min = param_dict["h_min"]
h_max = param_dict["h_max"]

g_ii = param_dict["g_ii"]
J0 = param_dict["J"]
g = param_dict["g"]


b_small = param_dict["b"]
b = np.ones(N)
for i, key in enumerate(index_dict.keys()):
    b[index_dict[key]] = b_small[i]

n_h = 10
h_range = np.linspace(h_min, h_max, n_h)

region_list = list(index_dict.keys())

region_i_list = []
region_j_list = []
for i in range(6):
    for j in range(6):
        if i <= j:
            region_i_list.append(region_list[i])
            region_j_list.append(region_list[j])
        

region_list = len(h_range) * region_list
region_i_list = len(h_range) * region_i_list
region_j_list = len(h_range) * region_j_list

h_list = [h for h in h_range for i in range(6)]
h_list_cor = [h for h in h_range for i in range(int(6*(6-1)/2+6))]



pred_rates = []
pred_cors = []
regions = []
reduced_rates = []
for h in h_range:
    J =  hippo_weights(index_dict, A, h,h, g, J0, i_plast = 1, g_ii = g_ii)
    #J_small = macro_weights(J = J0, h3 = h, h1 = h, g = g, g_ii = g_ii)
    J_small = sum_by_region(J, index_dict=index_dict)
    #y_q = y_0_quad(J,b)
    
    y_q_red = y_0_quad(J_small, b_small)
    correction = np.real(loop_correction(J_small,  y_q_red, b_small))
    y_corrected = y_q_red + correction 
        
    pred_rates.extend(y_q_red)
    reduced_rates.extend(y_corrected)


    J_lin =J_small* (2*(J_small@y_corrected + b_small))[...,None]
    pred_cors_mat = cor_pred(J_lin, Ns, y_corrected)
    pred_cors.extend( pred_cors_mat[np.triu_indices(6)])


print(len(reduced_rates))
print(len(pred_rates))

rate_df = pd.DataFrame({"tree_rate":pred_rates, "h" : h_list, "region": region_list ,  "corr_rate": reduced_rates})
cor_df = pd.DataFrame({"pred_cor": pred_cors, "region_i" : region_i_list, "region_j": region_j_list, "h": h_list_cor})
print(rate_df)
print(cor_df)
rate_df.to_csv("../results/fig_1_data/pred_rates.csv")
cor_df.to_csv("../results/fig_1_data/pred_cors.csv")
