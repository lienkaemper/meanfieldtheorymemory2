import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os

from src.simulation import sim_glm_pop
from src.theory import y_pred_full, covariance_full,  y_0_quad,  find_iso_rate, y_corrected_quad, find_iso_rate_input, cor_pred, loop_correction, find_iso_rate_ca3
from src.theory import CA1_internal_cov_offdiag, CA1_inherited_cov, CA3_internal_cov, CA3_E_from_E, CA3_E_from_N, CA3_E_from_I
from src.correlation_functions import rate, mean_by_region, tot_cross_covariance_matrix, two_pop_correlation, mean_pop_correlation, cov_to_cor, sum_by_region
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


b_small = np.array([.4, .4, .5, .4, .4, .5])   #without excitability
J0 = .2
g_ii = 1
g_min = 1
g_max = 3
n_g = 10
h_h = 2
gs = np.linspace(g_min, g_max, n_g)

J =  hippo_weights(index_dict, A, h3 = 1, h1 = 1, g = 1, J = J0,  g_ii =  g_ii)
J_baseline = sum_by_region(J, index_dict=index_dict)


y_baseline = y_0_quad(J_baseline, b_small)
ys_pred_engram = []
ys_pred_non_engram = []
h_list = []
g_list = []
cors_ee = []
cors_en = []
cors_nn = []

CA1_internal = []
CA1_inherited = []
CA3 = [] 
from_CA3I = []
from_CA3E = []
from_CA3N =[]

for g in gs:
    J_small =macro_weights(J=J0, h3 = 1, h1 =1, g =g)
    b_iso =find_iso_rate_input(target_rate= y_baseline[3], J = J_small, b = b_small, b0_min = 0, b0_max = 1, n_points=1000, plot=False)
    for h in [1,2]:
        h_i1, h_i3  = find_iso_rate_ca3(y_baseline[3],y_baseline[0], h=h, J0 = J0, g=g, g_ii=g_ii, b = b_iso, h_i_min = 1, h_i_max = 4, type = "quadratic", n_points = 1000)
        h_list.append(h)
        g_list.append(g)
        J =  hippo_weights(index_dict, A, h3 = h, h1 = h, g = g, J = J0,  g_ii = 1, i_plast = h_i1, i_plast_3=h_i3)
        J_small = sum_by_region(J, index_dict=index_dict)
       
        y_q_red = y_0_quad(J_small, b_iso)
        correction = np.real(loop_correction(J_small,  y_q_red, b_iso))
        y_corrected = y_q_red + correction 

        ys_pred_engram.append(y_corrected[3])
        ys_pred_non_engram.append(y_corrected[4])

        gain =  2*(J_small@y_corrected+ b_iso)
        J_lin =J_small* gain[...,None]
        D = np.linalg.inv(np.eye(6) - J_lin)
        pred_cors = cor_pred( J = J_lin , Ns = cells_per_region, y0 = y_corrected)
        cors_ee.append(pred_cors[3,3])
        cors_en.append(pred_cors[3,4])
        cors_nn.append(pred_cors[4,4])
        CA1_internal.append(CA1_internal_cov_offdiag(J = J_small, r = y_corrected, b = b_iso, N = cells_per_region)[0,0])
        CA1_inherited.append(CA1_inherited_cov(J = J_small, r = y_corrected, b = b_iso, N = cells_per_region)[0,0])
        CA3.append(CA3_internal_cov(J = J_small, r = y_corrected, b = b_iso, N = cells_per_region)[0,0])

        from_CA3I.append(CA3_E_from_I(J = J_small, r = y_corrected, b = b_iso, N = cells_per_region))
        from_CA3E.append(CA3_E_from_E(J = J_small, r = y_corrected, b = b_iso, N = cells_per_region)) 
        from_CA3N.append(CA3_E_from_N(J = J_small, r = y_corrected, b = b_iso, N = cells_per_region))  



decomp_df = pd.DataFrame({"g" : g_list, "h" : h_list, "CA1_internal" :  CA1_internal, "CA1_inherited" : CA1_inherited, "CA3" : CA3, 
"from_CA3I" :from_CA3I, "from_CA3E" : from_CA3E, "from_CA3N" : from_CA3N })

with open("../results/compare_inhib_levels/decomposition_df.pkl", "wb") as f:
    pkl.dump(obj = decomp_df, file = f)

df = pd.DataFrame({"g" : g_list, "h" : h_list, "pred_rate_engram" : ys_pred_engram, "pred_rate_non_engram" : ys_pred_non_engram,  
                                               "pred_cor_engram_vs_engram": cors_ee,  "pred_cor_non_engram_vs_non_engram": cors_nn,"pred_cor_engram_vs_non_engram": cors_en})


with open("../results/compare_inhib_levels/raw_theory_df.pkl", "wb") as f:
    pkl.dump(obj = df, file = f)

# Pivot the DataFrame
pivoted_df = df.pivot(index='g', columns='h', values=["pred_rate_engram", "pred_rate_non_engram",
"pred_cor_engram_vs_engram",  "pred_cor_non_engram_vs_non_engram" , "pred_cor_engram_vs_non_engram"])

# Flatten the multi-level columns
pivoted_df.columns = [f'{col[0]}_h={col[1]}' for col in pivoted_df.columns]

# Reset the index
df= pivoted_df.reset_index()

columns_to_process = [
    'pred_rate_engram_h=1', 'pred_rate_engram_h=2',
    'pred_rate_non_engram_h=1', 'pred_rate_non_engram_h=2',
    'pred_cor_engram_vs_engram_h=1', 'pred_cor_engram_vs_engram_h=2', 
    'pred_cor_engram_vs_non_engram_h=1', 'pred_cor_engram_vs_non_engram_h=2',
     "pred_cor_non_engram_vs_non_engram_h=1",  "pred_cor_non_engram_vs_non_engram_h=2" , 
]

for column in columns_to_process:
    if column.endswith('_h=1'):
        # Extract the corresponding column with h=2
        column_h2 = column.replace('_h=1', '_h=2')
        
        # Create a new column name for the ratio
        ratio_column_name = column.replace('_h=1', '_ratio')
        
        # Calculate the ratio and add it to the DataFrame
        df[ratio_column_name] = df[column_h2] / df[column]

with open("../results/compare_inhib_levels/theory_df.pkl", "wb") as f:
    pkl.dump(obj = df, file = f)

print(df)