import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os

from src.simulation import sim_glm_pop
from src.theory import y_pred_full, covariance_full,  y_0_quad, find_iso_rate, y_pred, cor_pred, length_1_cor, find_iso_rate_ca3
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
b = [.5, .5, .7, .5, .5, .7]  #without excitability
#b = [.6, .5, .7, .6, .5, .7] #with excitability


N = np.sum(cells_per_region)
J0 = .2

h_min = 1
h_max = 2
h_i_min =1
h_i_max = 6

g_min = 0.5
g_max = 4
g_ii_min = 0
g_ii_max = 2

n_points = 41
gs = np.linspace(g_min, g_max, n_points)
g_iis = np.linspace(g_ii_min, g_ii_max, n_points)



y_before =  np.zeros((n_points, n_points))
y_after =  np.zeros((n_points, n_points))
C_ee_before = np.zeros((n_points, n_points))
C_ee_after = np.zeros((n_points, n_points))
C_ee_after_plast = np.zeros((n_points, n_points))
stability = np.zeros((n_points, n_points))
h_is= np.zeros((n_points, n_points))

y_before3 =  np.zeros((n_points, n_points))
y_after3 =  np.zeros((n_points, n_points))
C_ee_before3 = np.zeros((n_points, n_points))
C_ee_after3 = np.zeros((n_points, n_points))
C_ee_after_plast3 = np.zeros((n_points, n_points))
h_is3= np.zeros((n_points, n_points))
for i, g in enumerate(gs): 
    for j, g_ii in enumerate(g_iis): 
        G_before = macro_weights(J = J0, h1 = h_min, h3 = h_min, g = g, h_i = 1, g_ii = g_ii)

        #rate before
        y = y_0_quad(G_before,  b)
        baselines = y[3], y[0]
        y_before[i,j] = y[3]
        y_before3[i, j] = y[0]

    
        G_lin =G_before* (2*(G_before@y+b))[...,None]
        #correlations before
        C = cor_pred(G_lin, cells_per_region, y )
        C_ee_before[i,j] = C[3,3]
        C_ee_before3[i,j] = C[0,0]

        #connectivity after learning: without plasticity 
        G_after = macro_weights(J = J0, h1 = h_max, h3 = h_max ,g=g, h_i=1, g_ii=g_ii)

        #rate after: without plasticity 
        y = y_0_quad(G_after,  b)
        y_after[i, j] = y[3]
        y_after3[i, j] = y[0]
        G_lin =G_after* (2*(G_after@y+b))[...,None]

        #correlations after: without plasticity 
        C = cor_pred(G_lin, cells_per_region, y )
        C_ee_after[i,j] = C[3,3]
        C_ee_after3[i,j] = C[0,0]
        stability[i,j] = np.max(np.linalg.eigvals(np.eye(6) - G_lin)) > 0

        #connectivity after learning: with plasticity 
        h_i1, hi_3 = find_iso_rate_ca3(baselines[0], baselines[1],  h_max, J0, g, g_ii, b, h_i_min, h_i_max, "quadratic")
        h_is[i,j] = h_i1
        h_is3[i,j] = hi_3
        G_after_plast = macro_weights(J = J0, h1 = h_max, h3 = h_max ,g=g, h_i=h_i1, h_i_ca3=hi_3,  g_ii=g_ii)

        #rate after: with plasticity 
        y = y_0_quad(G_after_plast,  b)
        G_lin =G_after_plast* (2*(G_after_plast@y+b))[...,None]
        #correlations after: with plasticity 
        C = cor_pred(G_lin, cells_per_region, y )
        C_ee_after_plast[i,j] = C[3,3]
        C_ee_after_plast3[i,j] = C[0,0]
        stability[i,j] = np.max(np.linalg.eigvals(np.eye(6) - G_lin)) > 0

print(h_is)
g = 3
g_ii = 0.25

delta_cor = C_ee_after/C_ee_before
delta_cor_plast = C_ee_after_plast/C_ee_before
delta_rate = y_after/y_before

delta_cor3 = C_ee_after3/C_ee_before3
delta_cor_plast3 = C_ee_after_plast3/C_ee_before3
delta_rate3 = y_after3/y_before3
# ind = np.argmax(delta_cor)
# i, j = np.unravel_index(ind, (n_points, n_points))
# print(gs[i], g_iis[j])
# print(delta_cor[i, j])

fig, ax = plt.subplots()
ax.scatter(np.array(y_before), np.array(y_after))
plt.show()

with open("../results/fig_5_data/delta_cor.pkl", "wb") as file:
    pkl.dump(delta_cor, file)
    
with open("../results/fig_5_data/delta_rate.pkl", "wb") as file:
    pkl.dump(delta_rate, file)


#plot without inhibitiory plasticity 
fig, axs = plt.subplots(2, 4, figsize=(9,6))
cs = axs[0,0].imshow(delta_cor, origin="lower", extent = (g_ii_min, g_ii_max, g_min, g_max), vmin = 1, vmax = 5)
plt.colorbar(cs, ax = axs[0,0])
axs[0,0].set_xlabel("g_ii")
axs[0,0].set_ylabel("g")
axs[0,0].scatter(x = [1, g_ii], y = [1, g], color = "red")
axs[0,0].title.set_text('Static inhibition ')

#plot with inhibitory plasticity 
cs = axs[0,1].imshow(delta_cor_plast, origin="lower", extent = (g_ii_min, g_ii_max, g_min, g_max), vmin = 1, vmax = 5)
plt.colorbar(cs, ax = axs[0,1])
axs[0,1].set_xlabel("g_ii")
axs[0,1].set_ylabel("g")
axs[0,1].scatter(x = [1, g_ii], y = [1, g], color = "red")
axs[0,1].title.set_text('Plastic inhibition ')

#rates_without 
cs = axs[0,2].imshow(delta_rate, origin="lower", extent = (g_ii_min, g_ii_max, g_min, g_max))
plt.colorbar(cs, ax = axs[0,2])
axs[0,2].set_xlabel("g_ii")
axs[0,2].set_ylabel("g")
axs[0,2].scatter(x = [1, g_ii], y = [1, g], color = "red")
axs[0,2].title.set_text('rates')

#inhibitory plasticity needed 
cs = axs[0,3].imshow(h_is, origin="lower", extent = (g_ii_min, g_ii_max, g_min, g_max))
plt.colorbar(cs, ax = axs[0,3])
axs[0,3].set_xlabel("g_ii")
axs[0,3].set_ylabel("g")
axs[0,3].scatter(x = [1, g_ii], y = [1, g], color = "red")
axs[0,3].title.set_text('inhibitory plasticity needed')

# now in CA3

#plot without inhibitiory plasticity 
cs = axs[1,0].imshow(delta_cor3, origin="lower", extent = (g_ii_min, g_ii_max, g_min, g_max), vmin = 1, vmax = 5)
plt.colorbar(cs, ax = axs[1,0])
axs[1,0].set_xlabel("g_ii")
axs[1,0].set_ylabel("g")
axs[1,0].scatter(x = [1, g_ii], y = [1, g], color = "red")
axs[1,0].title.set_text('Static inhibition ')

#plot with inhibitory plasticity 
cs = axs[1,1].imshow(delta_cor_plast3, origin="lower", extent = (g_ii_min, g_ii_max, g_min, g_max), vmin = 1, vmax = 5)
plt.colorbar(cs, ax = axs[1,1])
axs[1,1].set_xlabel("g_ii")
axs[1,1].set_ylabel("g")
axs[1,1].scatter(x = [1, g_ii], y = [1, g], color = "red")
axs[1,1].title.set_text('Plastic inhibition ')

#rates_without 
cs = axs[1,2].imshow(delta_rate3, origin="lower", extent = (g_ii_min, g_ii_max, g_min, g_max))
plt.colorbar(cs, ax = axs[1,2])
axs[1,2].set_xlabel("g_ii")
axs[1,2].set_ylabel("g")
axs[1,2].scatter(x = [1, g_ii], y = [1, g], color = "red")
axs[1,2].title.set_text('rates')

#inhibitory plasticity needed 
cs = axs[1,3].imshow(h_is3, origin="lower", extent = (g_ii_min, g_ii_max, g_min, g_max))
plt.colorbar(cs, ax = axs[1,3])
axs[1,3].set_xlabel("g_ii")
axs[1,3].set_ylabel("g")
axs[1,3].scatter(x = [1, g_ii], y = [1, g], color = "red")
axs[1,3].title.set_text('inhibitory plasticity needed')


plt.tight_layout()
plt.savefig("../results/heatmaps/heatmap_delta_cor_quadratic.pdf")
plt.show()

