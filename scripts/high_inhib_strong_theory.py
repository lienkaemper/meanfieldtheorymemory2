import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
import sys


from src.theory import y_pred_from_full_connectivity, y_corrected_quad, y_0_quad, covariance_full, find_iso_rate
from src.correlation_functions import rate, mean_by_region, cov_to_cor
from src.generate_connectivity import excitatory_only, gen_adjacency, hippo_weights, macro_weights




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
h_min = param_dict["h_min"]
h_max = param_dict["h_max"]

g_ii = param_dict["g_ii"]
J0 = param_dict["J"]
g = param_dict["g"]

b_small = param_dict["b"]
print(param_dict)
b = np.ones(N)
for i, key in enumerate(index_dict.keys()):
    b[index_dict[key]] = b_small[i]

n_h = 5
h_range = np.linspace(h_min, h_max, n_h)

region_list = ['' for i in range(N)]
for key in index_dict:
    start = min(index_dict[key])
    end = max(index_dict[key])
    region_list[start:end+1] = (end+1-start)*[key]

region_i_list = []
region_j_list = []
for i in range(N):
    for j in range(N):
        if i < j:
            region_i_list.append(region_list[i])
            region_j_list.append(region_list[j])
        

region_list = len(h_range) * region_list
region_i_list = len(h_range) * region_i_list
region_j_list = len(h_range) * region_j_list

h_list = [h for h in h_range for i in range(N)]
h_list_cor = [h for h in h_range for i in range(int(N*(N-1)/2))]



pred_rates = []
pred_cors = []
reduced_rates = []
tree_level_rates = []
regions = []


for h in h_range:
    J =  hippo_weights(index_dict, A, h,h, g, J0,  g_ii = g_ii)
    y_q = y_0_quad(J,b)
    #y_q_corrected = y_corrected_quad(J,  y_q, b)
    pred_rates.extend(y_q)

    J_small = macro_weights(J = J0, h3 = h, h1 = h, g = g, g_ii = g_ii)
    y_q_red = y_0_quad(J_small, b_small)
    y_red_corrected= y_corrected_quad(J_small,  y_q_red, b_small)
    y_q_red_long = np.zeros(N)
    y_tree_level = np.zeros(N)
    for i, key in enumerate(index_dict):
        y_q_red_long[index_dict[key]] = y_red_corrected[i]
        y_tree_level[index_dict[key]] = y_q_red[i]
    reduced_rates.extend(y_q_red_long)
    tree_level_rates.extend(y_tree_level)


    J_lin =J* (2*(J@y_q+b))[...,None]
    pred_covs = covariance_full(J_lin, y_q)
    pred_cors_mat = cov_to_cor(pred_covs)
    pred_cors.extend( pred_cors_mat[np.triu_indices(N,1)])


rate_df = pd.DataFrame({"pred_rate":pred_rates, "reduced_rate": reduced_rates, "tree_rate": tree_level_rates, "h" : h_list, "region": region_list })
cor_df = pd.DataFrame({"pred_cor": pred_cors, "region_i" : region_i_list, "region_j": region_j_list, "h": h_list_cor})

rate_df.to_csv("../results/strong_sim_data/pred_rates.csv")
cor_df.to_csv("../results/strong_sim_data/pred_cors.csv")
