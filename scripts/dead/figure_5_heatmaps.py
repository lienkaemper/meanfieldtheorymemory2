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
C_ee_before_approx = np.zeros((n_points, n_points))
C_ee_after_approx = np.zeros((n_points, n_points))
stability = np.zeros((n_points, n_points))
h_is= np.zeros((n_points, n_points))
for i, g in enumerate(gs): 
    for j, g_ii in enumerate(g_iis): 
        G_before = macro_weights(J0, h_min,h_min ,g, h_i_min, g_ii)

        #rate before
        y = y_0_quad(G_before,  b)
        y_before[i,j] = y[3]

    
        G_lin =G_before* (2*(G_before@y+b))[...,None]
        #correlations before
        C = cor_pred(G_lin, cells_per_region, y )
        C_ee_before[i,j] = C[3,3]
        C_ee_before_approx[i,j] = length_1_cor(G_lin, cells_per_region, y)[3,3]


        h_i = find_iso_rate(y[3], h_max, J0, g, g_ii, b, h_i_min, h_i_max, "quadratic")
        h_is[i,j] = h_i
        G_after = macro_weights(J0, h_max,h_max ,g, h_i, g_ii)

        #rate after
        y = y_0_quad(G_after,  b)
        y_after[i, j] = y[3]
        G_lin =G_after* (2*(G_after@y+b))[...,None]
        #correlations after
        C = cor_pred(G_lin, cells_per_region, y )
        C_ee_after[i,j] = C[3,3]
        C_ee_after_approx[i,j] = length_1_cor(G_lin, cells_per_region, y)[3,3]
        stability[i,j] = np.max(np.linalg.eigvals(np.eye(6) - G_lin)) > 0

g = 3
g_ii = 0.25

delta_cor = C_ee_after/C_ee_before
delta_cor_approx = C_ee_after_approx/C_ee_before_approx
ind = np.argmax(delta_cor)
i, j = np.unravel_index(ind, (n_points, n_points))
print(gs[i], g_iis[j])
print(delta_cor[i, j])

fig, ax = plt.subplots()
ax.scatter(np.array(y_before), np.array(y_after))
plt.show()

with open("../results/fig_5_data/delta_cor.pkl", "wb") as file:
    pkl.dump(delta_cor, file)
    



fig, axs = plt.subplots(1, 2, figsize=(6, 3))
cs = axs[0].imshow(delta_cor, origin="lower", extent = (g_ii_min, g_ii_max, g_min, g_max))
plt.colorbar(cs, ax = axs[0])
cs = axs[1].imshow(delta_cor_approx, origin="lower", extent = (g_ii_min, g_ii_max, g_min, g_max))
plt.colorbar(cs, ax = axs[1])

axs[0].set_xlabel("g_ii")
axs[0].set_ylabel("g")
axs[1].set_xlabel("g_ii")
axs[1].set_ylabel("g")

axs[0].scatter(x = [1, g_ii], y = [1, g], color = "red")
axs[1].scatter(x = [1, g_ii], y = [1, g], color = "red")
plt.savefig("../results/heatmaps/heatmap_delta_cor_quadratic.pdf")
plt.show()

quit()


g = 3
g_ii = 0.25
h_max = 2
h_i_max = 1.5

n_points = 101
hs = np.linspace(1, h_max, n_points)
h_is = np.linspace(1,h_i_max, n_points)
rates = np.zeros((n_points, n_points))
rates_p  = np.zeros((n_points, n_points))
for i, h in enumerate(hs):
    for j, h_i in enumerate(h_is):
        G = macro_weights(J0, h, h, g, h_i, g_ii = g_ii)
        y = y_0_quad(G, b)
        rates[j,i] = y[3]
        rates_p[j,i] = y[4]

norm_value = rates[0, 0]
rates = rates/norm_value

with open("../results/fig_5_data/norm_rates.pkl", "wb") as file:
    pkl.dump(rates, file)

matched_h_i_ind = np.argmin(np.abs(rates - rates[0,0] ) , axis = 0)
matched_h_i_l = h_is[matched_h_i_ind]

overall_min = np.min(rates)
overall_max = np.max(rates)

levels_more = np.linspace(overall_min, overall_max, 50)
H, H_I = np.meshgrid(hs, h_is)


fig, ax = plt.subplots(figsize=(10,10))
ax.contour(H, H_I, rates, levels = [rates[0,0]])
cs = ax.imshow(rates, origin = "lower", extent = (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max, cmap = cmap)
#ax.plot(hs, matched_h_i_l)
ax.set_title("engram rates \n h_i = {}".format(matched_h_i_l[-1]))
ax.set_ylabel("h_i")
ax.set_xlabel("h")
fig.colorbar(cs)
plt.savefig("../results/heatmaps/rates_linear_heatmap_g={}J={}g_ii={}.pdf".format(g, J0, g_ii))
plt.show()