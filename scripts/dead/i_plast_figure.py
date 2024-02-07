import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os

from src.simulation import sim_glm_pop
from src.theory import y_pred_full, covariance_full,  y_0_quad,  find_iso_rate, y_corrected_quad, find_iso_rate_input, cor_pred
from src.correlation_functions import rate, mean_by_region, tot_cross_covariance_matrix, two_pop_correlation, mean_pop_correlation, cov_to_cor
from src.plotting import raster_plot, abline
from src.generate_connectivity import excitatory_only, gen_adjacency, hippo_weights, macro_weights
from src.plotting import raster_plot


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
g = 1
g_ii = 1

hi3_min = 1
hi3_max = 2

hi1_min = 1
hi1_max = 2

n_points = 50

hi1_list = []
hi3_list = []
g_list = []
h_list = []
C_list_ca3 = []
C_list_ca1 = []
r_list_ca3 = []
r_list_ca1 = []

for hi3 in np.linspace(hi3_min, hi3_max, n_points):
    for h in [1, 2]:
        J = macro_weights(J = J0, h3 = h, h1 = h, g = g, h_i = 1, g_ii = g_ii, h_i_ca3=(hi3)*(h-1)+(2-h))
        r = y_0_quad(J, b_small)
        gain =  2*(J@r+ b_small)
        J_lin =J* gain[...,None]
        pred_cors = cor_pred( J = J_lin, Ns = cells_per_region, y0 = r)
        r_list_ca3.append(r[0])
        r_list_ca1.append(r[3])
        C_list_ca1.append(pred_cors[3,3])
        C_list_ca3.append(pred_cors[0,0])
        hi3_list.append(hi3)
        g_list.append(g)
        h_list.append(h)

CA3_df = pd.DataFrame({"g" : g_list, "h" : h_list, "hi3": hi3_list,"rate (CA1)" : r_list_ca1, "cor (CA1)": C_list_ca1, "rate (CA3)" : r_list_ca3, "cor (CA3)": C_list_ca3})

colnames = ["rate (CA1)", "cor (CA1)", "rate (CA3)", "cor (CA3)"]
pivoted_df = CA3_df.pivot(index=['g', 'hi3'], columns='h', values=colnames)

# Flatten the multi-level columns
pivoted_df.columns = [f'{col[0]}_h={col[1]}' for col in pivoted_df.columns]

# Reset the index
CA3_df= pivoted_df.reset_index()

columns_to_process = [colname + "_h=1" for colname in colnames] +  [colname + "_h=2" for colname in colnames]
for column in columns_to_process:
    if column.endswith('_h=1'):
        # Extract the corresponding column with h=2
        column_h2 = column.replace('_h=1', '_h=2')
        
        # Create a new column name for the ratio
        ratio_column_name = column.replace('_h=1', '_ratio')
        
        # Calculate the ratio and add it to the DataFrame
        CA3_df[ratio_column_name] = CA3_df[column_h2] / CA3_df[column]

with open("../results/compare_inhib_levels/i_plast_df_CA3.pkl", "wb") as f:
    pkl.dump(obj = CA3_df, file = f)


hi3_fixed = hi3_max
ind = n_points
for i,r in enumerate(CA3_df["rate (CA3)_ratio"]):
    if r < 1:
        hi3_fixed = CA3_df["hi3"][i]
        ind = i
        break


hi1_list = []
hi3_list = []
g_list = []
h_list = []
C_list_ca3 = []
C_list_ca1 = []
r_list_ca3 = []
r_list_ca1 = []

for hi1 in np.linspace(hi1_min, hi1_max, n_points):
    for h in [1, 2]:
        J = macro_weights(J = J0, h3 = h, h1 = h, g = g, h_i = (hi1)*(h-1)+(2-h), g_ii = g_ii, h_i_ca3=(hi3_fixed)*(h-1)+(2-h))
        r = y_0_quad(J, b_small)
        gain =  2*(J@r+ b_small)
        J_lin =J* gain[...,None]
        pred_cors = cor_pred( J = J_lin, Ns = cells_per_region, y0 = r)
        r_list_ca3.append(r[0])
        r_list_ca1.append(r[3])
        C_list_ca1.append(pred_cors[3,3])
        C_list_ca3.append(pred_cors[0,0])
        hi1_list.append(hi1)
        g_list.append(g)
        h_list.append(h)


CA1_df = pd.DataFrame({"g" : g_list, "h" : h_list, "hi1": hi1_list,"rate (CA1)" : r_list_ca1, "cor (CA1)": C_list_ca1, "rate (CA3)" : r_list_ca3, "cor (CA3)": C_list_ca3})

colnames = ["rate (CA1)", "cor (CA1)", "rate (CA3)", "cor (CA3)"]
pivoted_df = CA1_df.pivot(index=['g', 'hi1'], columns='h', values=colnames)

# Flatten the multi-level columns
pivoted_df.columns = [f'{col[0]}_h={col[1]}' for col in pivoted_df.columns]

# Reset the index
CA1_df= pivoted_df.reset_index()

columns_to_process = [colname + "_h=1" for colname in colnames] +  [colname + "_h=2" for colname in colnames]
for column in columns_to_process:
    if column.endswith('_h=1'):
        # Extract the corresponding column with h=2
        column_h2 = column.replace('_h=1', '_h=2')
        
        # Create a new column name for the ratio
        ratio_column_name = column.replace('_h=1', '_ratio')
        
        # Calculate the ratio and add it to the DataFrame
        CA1_df[ratio_column_name] = CA1_df[column_h2] / CA3_df[column]

with open("../results/compare_inhib_levels/i_plast_df_CA1.pkl", "wb") as f:
    pkl.dump(obj = CA1_df, file = f)




fig, axs = plt.subplots(2,2, figsize = (6,8))
sns.lineplot(data = CA3_df, x = "hi3", y = "rate (CA1)_ratio", ax = axs[0,0], label = "CA1", color = "black")
sns.lineplot(data = CA3_df, x = "hi3", y = "rate (CA3)_ratio", ax = axs[0,0], label = "CA3", color = "gray")
axs[0,0].scatter(x = [hi3_fixed, hi3_fixed], y = [ CA3_df["rate (CA3)_ratio"][ind],  CA3_df["rate (CA1)_ratio"][ind]], color = "red")
axs[0,0].set_title("rate ratio")
axs[0,0].legend()
axs[0,0].set_xlabel("inhibitory engram strength onto CA3")
axs[0,0].set_ylabel("rate ratio")

axs[0,1].scatter(x = [hi3_fixed, hi3_fixed], y = [ CA3_df["cor (CA3)_ratio"][ind],  CA3_df["cor (CA1)_ratio"][ind]], color = "red")
sns.lineplot(data = CA3_df, x = "hi3", y = "cor (CA1)_ratio", ax = axs[0,1], label = "CA1", color = "black")
sns.lineplot(data = CA3_df, x = "hi3", y = "cor (CA3)_ratio", ax = axs[0,1], label = "CA3", color = "gray")
axs[0,1].set_title("correlation ratio")
axs[0,1].legend()
axs[0,1].set_xlabel("inhibitory engram strength onto CA3")
axs[0,1].set_ylabel("correlation ratio")

hi1_fixed = hi1_max
ind = n_points
for i,r in enumerate(CA1_df["rate (CA1)_ratio"]):
    if r < 1:
        hi1_fixed = CA1_df["hi1"][i]
        ind = i
        break

sns.lineplot(data = CA1_df, x = "hi1", y = "rate (CA1)_ratio", ax = axs[1,0], label = "CA1", color = "black")
sns.lineplot(data = CA1_df, x = "hi1", y = "rate (CA3)_ratio", ax = axs[1,0], label = "CA3", color = "gray")
axs[1,0].scatter(x = [hi1_fixed, hi1_fixed], y = [ CA1_df["rate (CA3)_ratio"][ind],  CA1_df["rate (CA1)_ratio"][ind]], color = "red")
axs[1,0].set_title("rate ratio")
axs[1,0].legend()
axs[1,0].set_xlabel("inhibitory engram strength onto CA1")
axs[1,0].set_ylabel("rate ratio")

axs[1,1].scatter(x = [hi1_fixed, hi1_fixed], y = [ CA1_df["cor (CA3)_ratio"][ind],  CA1_df["cor (CA1)_ratio"][ind]], color = "red")
sns.lineplot(data = CA1_df, x = "hi1", y = "cor (CA1)_ratio", ax = axs[1,1], label = "CA1", color = "black")
sns.lineplot(data = CA1_df, x = "hi1", y = "cor (CA3)_ratio", ax = axs[1,1], label = "CA3", color = "gray")
axs[1,1].set_title("correlation ratio")
axs[1,1].legend()
axs[1,1].set_xlabel("inhibitory engram strength onto CA1")
axs[1,1].set_ylabel("correlation ratio")

plt.tight_layout()
plt.savefig("../results/inhibitory_plasticity_fig.pdf")
plt.show()