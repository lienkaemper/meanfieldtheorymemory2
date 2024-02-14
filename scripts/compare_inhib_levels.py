import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os

from src.simulation import sim_glm_pop
from src.theory import y_0_quad,  cor_pred
from src.correlation_functions import rate, two_pop_correlation, mean_pop_correlation
from src.generate_connectivity import gen_adjacency, hippo_weights, macro_weights


# generate adjacency matrix 
N_E =60
N_I = 15
cells_per_region =np.array([N_E, N_E, N_I,  N_E, N_E, N_I])
N = np.sum(cells_per_region)
pEE = .2
pIE = .8
pII = .8
pEI = .8

macro_connectivity = np.array([
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pIE, pIE, pII, pIE, pIE, pII],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pIE, pIE, pII, pIE, pIE, pII]])
             
A, index_dict = gen_adjacency(cells_per_region, macro_connectivity)

with open("../results/compare_inhib_levels/index.pkl", "wb") as f:
    pkl.dump(obj = index_dict, file = f)


#simulation parameters 
dt = 0.02
tstop = 50000


b_small = np.array([.5, .5, .7, .5, .5, .7])  #without excitability
J0 = .2
g_ii = 1
g_min = 1
g_max = 5
n_g = 5
h_h = 2
gs = np.linspace(g_min, g_max, n_g)

J_baseline = macro_weights(J = J0, h3 = 1, h1 = 1, g = g_min)

y_baseline = y_0_quad(J_baseline, b_small)

print("baseline: {}".format(y_baseline[3]))
ys_pred_engram = []
ys_sim_engram = []
ys_pred_non_engram = []
ys_sim_non_engram = []
h_list = []
g_list = []
cors_ee = []
cors_en = []
cors_nn = []
sim_cors_ee = []
sim_cors_en = []
sim_cors_nn = []

for trial in range(1):
    for g in gs:
        #b_iso = find_iso_rate_input(target_rate= y_baseline[3], J = J0, h = 1, g = g, g_ii = g_ii, b = b_small, b0_min = 0, b0_max = 1)
        b_iso = 0
        for h in [1,2]:
            h_list.append(h)
            g_list.append(g)
            J_new =  macro_weights(J = J0, h3 = h, h1 = h, g = g)
            J =  hippo_weights(index_dict, A, h3 = h, h1 = h, g = g, J = J0,  g_ii =     g_ii)
            y_new = y_0_quad(J_new, b_iso+b_small)
            ys_pred_engram.append(y_new[3])
            ys_pred_non_engram.append(y_new[4])

            gain =  2*(J_new@y_new+b_iso + b_small)
            J_lin =J_new* gain[...,None]
            pred_cors = cor_pred( J = J_lin, Ns = cells_per_region, y = y_new)
            cors_ee.append(pred_cors[3,3])
            cors_en.append(pred_cors[3,4])
            cors_nn.append(pred_cors[4,4])
            b = np.concatenate([b_iso+b_small[i]*np.ones(cells_per_region[i]) for i in range(6)])
            v, spktimes = sim_glm_pop(J=J,  E=b, dt = dt, tstop=tstop,  v_th = 0, maxspikes = tstop * N, p = 2)
            with open("../results/compare_inhib_levels/spktimes_g={}h={}.pkl".format(g,h), "wb") as f :
                pkl.dump(obj = spktimes, file = f)
            neurons = index_dict['CA1E']
            rates = [rate(spktimes, i, dt, tstop) for i in neurons]
            mean_rate = np.mean(rates)
            ys_sim_engram.append(mean_rate)
            neurons = index_dict['CA1P']
            rates = [rate(spktimes, i, dt, tstop) for i in neurons]
            mean_rate = np.mean(rates)
            ys_sim_non_engram.append(mean_rate)

            engram_cells = index_dict["CA1E"]
            non_engram_cells = index_dict["CA1P"]

            sim_cors_ee.append(mean_pop_correlation(spktimes, engram_cells, dt, tstop))
            sim_cors_en.append(two_pop_correlation(spktimes, engram_cells, non_engram_cells, dt, tstop))
            sim_cors_nn.append(mean_pop_correlation(spktimes, non_engram_cells, dt, tstop))


df = pd.DataFrame({"g" : g_list, "h" : h_list, "pred_rate_engram" : ys_pred_engram, "pred_rate_non_engram" : ys_pred_non_engram,  
                                               "sim_rate_engram" : ys_sim_engram, "sim_rate_non_engram" : ys_sim_non_engram, 
                                               "pred_cor_engram_vs_engram": cors_ee,  "pred_cor_non_engram_vs_non_engram": cors_nn,"pred_cor_engram_vs_non_engram": cors_en, 
                                               "sim_cor_engram_vs_engram": sim_cors_ee,  "sim_cor_non_engram_vs_non_engram": sim_cors_nn,"sim_cor_engram_vs_non_engram": sim_cors_en})

df = df.groupby(["g", "h"]).mean()
print(df)
df = df.reset_index()
print(df.columns)

# Pivot the DataFrame
pivoted_df = df.pivot(index='g', columns='h', values=["pred_rate_engram", "pred_rate_non_engram", "sim_rate_engram" , "sim_rate_non_engram", 
"pred_cor_engram_vs_engram",  "pred_cor_non_engram_vs_non_engram" , "pred_cor_engram_vs_non_engram", 
"sim_cor_engram_vs_engram",  "sim_cor_non_engram_vs_non_engram" , "sim_cor_engram_vs_non_engram"])

# Flatten the multi-level columns
pivoted_df.columns = [f'{col[0]}_h={col[1]}' for col in pivoted_df.columns]

# Reset the index
df= pivoted_df.reset_index()

columns_to_process = [
    'pred_rate_engram_h=1', 'pred_rate_engram_h=2',
    'pred_rate_non_engram_h=1', 'pred_rate_non_engram_h=2',
    'sim_rate_engram_h=1', 'sim_rate_engram_h=2', 
    'sim_rate_non_engram_h=1', 'sim_rate_non_engram_h=2', 
    'pred_cor_engram_vs_engram_h=1', 'pred_cor_engram_vs_engram_h=2', 
    'pred_cor_engram_vs_non_engram_h=1', 'pred_cor_engram_vs_non_engram_h=2',
     "pred_cor_non_engram_vs_non_engram_h=1",  "pred_cor_non_engram_vs_non_engram_h=2" , 
     "sim_cor_engram_vs_engram_h=1",  "sim_cor_engram_vs_engram_h=2",  
     "sim_cor_non_engram_vs_non_engram_h=1" , "sim_cor_non_engram_vs_non_engram_h=2" , 
     "sim_cor_engram_vs_non_engram_h=1",  "sim_cor_engram_vs_non_engram_h=2"
]

for column in columns_to_process:
    if column.endswith('_h=1'):
        # Extract the corresponding column with h=2
        column_h2 = column.replace('_h=1', '_h=2')
        
        # Create a new column name for the ratio
        ratio_column_name = column.replace('_h=1', '_ratio')
        
        # Calculate the ratio and add it to the DataFrame
        df[ratio_column_name] = df[column_h2] / df[column]

with open("../results/compare_inhib_levels/df.pkl", "wb") as f:
    pkl.dump(obj = df, file = f)

print(df)