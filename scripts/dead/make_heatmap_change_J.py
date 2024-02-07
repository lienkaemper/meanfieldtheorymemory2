import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os

from src.simulation import sim_glm_pop
from src.theory import y_pred_full, covariance_full,  y_0_quad, find_iso_rate, y_pred, cor_pred, length_1_cor
from src.correlation_functions import rate, mean_by_region, tot_cross_covariance_matrix, two_pop_correlation, mean_pop_correlation, cov_to_cor
from src.plotting import raster_plot, abline
from src.generate_connectivity import excitatory_only, gen_adjacency, hippo_weights, macro_weights

plt.style.use('paper_style.mplstyle')
cmap = 'viridis'

h_min = 1
h_max = 2

N_E =60
N_I = 15
cells_per_region =np.array([N_E, N_E, N_I,  N_E, N_E, N_I])
b = [.5, .5, 1, .5, .5, 1]  #without excitability
#b = [.6, .5, .7, .6, .5, .7] #with excitability


N = np.sum(cells_per_region)
J0 = .2

h_min = 1
h_max = 2


g_min = 1
g_max = 4
J0_min = .1
J0_max = 1

n_points = 41
gs = np.linspace(g_min, g_max, n_points)
J0s = np.linspace(J0_min,J0_max, n_points)
y_before =  np.zeros((n_points, n_points))
y_after =  np.zeros((n_points, n_points))
C_ee_before = np.zeros((n_points, n_points))
C_ee_after = np.zeros((n_points, n_points))
C_ee_before_approx = np.zeros((n_points, n_points))
C_ee_after_approx = np.zeros((n_points, n_points))
stability = np.zeros((n_points, n_points))
for i, g in enumerate(gs): 
    for j, J0 in enumerate(J0s): 
        G_before = macro_weights(J = J0, h3 = h_min, h1 = h_min, g = g, g_ii = 1)
        G_after =  macro_weights(J = J0, h3 = h_max, h1 = h_max, g = g, g_ii = 1)
        stability[i,j] = np.max(np.linalg.eigvals(np.eye(6) - G_after)) > 0.01
        if stability[i,j] == True:
            #rate before
            y = y_0_quad(G_before,  b)
            y_before[i,j] = y[3]

        
            G_lin =G_before* (2*(G_before@y+b))[...,None]
            #correlations before
            C = cor_pred(G_lin, cells_per_region, y )
            C_ee_before[i,j] = C[3,3]
            C_ee_before_approx[i,j] = length_1_cor(G_lin, cells_per_region, y)[3,3]


            #rate after
            y = y_0_quad(G_after,  b)
            y_after[i, j] = y[3]
            G_lin =G_after* (2*(G_after@y+b))[...,None]
            #correlations after
            C = cor_pred(G_lin, cells_per_region, y )
            C_ee_after[i,j] = C[3,3]
            C_ee_after_approx[i,j] = length_1_cor(G_lin, cells_per_region, y)[3,3]
        


delta_cor = C_ee_after/C_ee_before
delta_cor_approx = C_ee_after_approx/C_ee_before_approx



fig, ax = plt.subplots()
ax.scatter(np.array(y_before), np.array(y_after))
plt.show()

with open("../results/fig_5_data/delta_cor.pkl", "wb") as file:
    pkl.dump(delta_cor, file)
    



fig, axs = plt.subplots(1, 2, figsize=(5, 2.5))
cs = axs[0].imshow(delta_cor, origin="lower", extent = (J0_min, J0_max, g_min, g_max), aspect = .75, vmax = 8)
plt.colorbar(cs, ax = axs[0])
cs = axs[1].imshow(delta_cor_approx, origin="lower", extent = (J0_min, J0_max, g_min, g_max),  aspect = .75, vmax = 4)
plt.colorbar(cs, ax = axs[1])

axs[0].set_xlabel("J0")
axs[0].set_ylabel("g")
axs[1].set_xlabel("J0")
axs[1].set_ylabel("g")

plt.tight_layout()
plt.savefig("../results/fig_5_data/heatmap_delta_cor_quadratic.pdf")
plt.show()

