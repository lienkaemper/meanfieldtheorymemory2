import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os

from src.simulation import sim_glm_pop
from src.theory import y_pred_full, covariance_full,  y_0_quad,  find_iso_rate, y_corrected_quad, find_iso_rate_input
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


#simulation parameters 
dt = 0.02
tstop = 5000


b_small = np.array([.5, .5, .7, .5, .5, .7])  #without excitability
J0 = .2
g_ii = 1
g_min = 1
g_max = 5
n_g = 5
gs = np.linspace(g_min, g_max, n_g)

J_baseline = macro_weights(J = J0, h3 = 1, h1 = 1, g = g_min)

y_baseline = y_0_quad(J_baseline, b_small)

print("baseline: {}".format(y_baseline[3]))
ys = np.zeros(n_g)
for i, g in enumerate(gs):
    b_iso = find_iso_rate_input(target_rate= y_baseline[3], J = J0, h = 1, g = g, g_ii = g_ii, b = b_small, b0_min = 0, b0_max = 1)
    J_new =  macro_weights(J = J0, h3 = 1, h1 = 1, g = g)
    J =  hippo_weights(index_dict, A, h3 = 1, h1 = 1, g = g, J = J0,  g_ii =     g_ii)
    y_new = y_0_quad(J_new, b_iso+b_small)
    ys[i] = y_new[5]
    b = np.concatenate([b_iso+b_small[i]*np.ones(cells_per_region[i]) for i in range(6)])
    v, spktimes = sim_glm_pop(J=J,  E=b, dt = dt, tstop=tstop,  v_th = 0, maxspikes = tstop * N, p = 2)
    neurons = index_dict['CA1E']
    rates = [rate(spktimes, i, dt, tstop) for i in neurons]
    mean_rate = np.mean(rates)
    print("g = {}, pred_rate = {}, sim_rate = {}, b = {}".format(g, y_new[3], mean_rate, b_iso ))


plt.plot(ys)
plt.show()