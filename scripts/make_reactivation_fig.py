import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os

from src.simulation import sim_glm_pop
from src.theory import y_pred_full, covariance_full,  y_0_quad, find_iso_rate
from src.correlation_functions import rate, mean_by_region, tot_cross_covariance_matrix, two_pop_correlation, mean_pop_correlation, cov_to_cor
from src.plotting import raster_plot, abline
from src.generate_connectivity import excitatory_only, gen_adjacency, hippo_weights, macro_weights

plt.style.use('paper_style.mplstyle')




N_E = 60
N_I = 15
cells_per_region =np.array([N_E, N_E, N_I,  N_E, N_E, N_I])
b_small = [.5, .5, .7, .5, .5, .7]


N = np.sum(cells_per_region)
b = np.concatenate([b_small[i]*np.ones(cells_per_region[i]) for i in range(6)])


J0 = .2



dt = 0.02
tstop = 40
tstim = 20


pEE = .2
pIE = .8
pII =.8
pEI = .8





macro_connectivity = np.array([
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pIE, pIE, pII, pIE, pIE, pII],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pIE, pIE, pII, pIE, pIE, pII]])


parameters = ["b", "J", "pEE", "pII", "pEI", "N", "dt", "tstop"]
values = [b_small, J0,  pEE, pII, pEI, N, dt, tstop]
param_df = pd.DataFrame({"parameter": parameters, "value": values})
dirname = "../results/reactivation/" + "_".join([p + str(v) for (p,v) in zip(param_df.parameter, param_df.value)])

if not os.path.isdir(dirname):
    os.mkdir(dirname)

text_file = open("../results/most_recent.txt", "w")
text_file.write(dirname)
text_file.close()

A, index_dict = gen_adjacency(cells_per_region, macro_connectivity)
yticks = [r[0] for r in index_dict.values()]

with open(dirname + "/index_dict.pkl", "wb") as file:
    pkl.dump(index_dict, file)

with open(dirname + "/adjacency.pkl", "wb") as file:
    pkl.dump(A, file)

with open(dirname + "/param_dict.pkl", "wb") as file:
    pkl.dump(dict(zip(parameters, values)), file)

min_stim = 0
max_stim = 1
n_stim = 50
stims = np.linspace(min_stim, max_stim, n_stim)

fig, axs = plt.subplots(2,2, figsize = (7, 4))


# #before learning, low_inhib model
# g = 1
# g_ii = 1
# h = 1
# J =  hippo_weights(index_dict, A, h,h, g, J0, g_ii = g_ii)
# pred_rates = y_0_quad(J, b)
# max_rate = np.max(pred_rates)
# maxspikes = int(np.floor(N*max_rate*tstop ))
# gc.collect()
# v, spktimes = sim_glm_pop(J=J,  E=b, dt = dt, tstop=tstop, tstim = tstim,  Estim=b_stim, v_th = 0, maxspikes = maxspikes, p = 2)
# raster_plot(spktimes, range(N),0, tstop, yticks = yticks, ax = axs[0,0])


#after learning, low_inhib model

b_stim= np.copy(b)
b_stim[:int(N_E/2)] += .25
g = 1
g_ii = 1
h = 2
J =  hippo_weights(index_dict, A, h,h, g, J0, g_ii = g_ii)
pred_rates = y_0_quad(J, b)
max_rate = np.max(pred_rates)
maxspikes = int(np.floor(N*max_rate*tstop ))
gc.collect()
v, spktimes = sim_glm_pop(J=J,  E=b, dt = dt, tstop=tstop, tstim = tstim,  Estim=b_stim, v_th = 0, maxspikes = maxspikes, p = 2)
raster_plot(spktimes, range(N),0, tstop, yticks = yticks,ax = axs[0,0])
axs[0,0].set_title(f"g={g}, g_II={g_ii}")



result = np.zeros((6, n_stim))
for i, stim in enumerate(stims):
    b_stim= np.copy(b)
    b_stim[:int(N_E/2)] += stim
    rates = y_0_quad(J, b_stim)
    result[:, i] = mean_by_region(rates, index_dict)


axs[1,0].plot(stims, result[3, :].T, color = "#F37343", label = "Engram")
axs[1,0].plot(stims, result[4, :].T, color = "#06ABC8", label = "Non-engram")
axs[1,0].set_xlabel("stimulus strength")
axs[1,0].set_ylabel("rate")

tstop = 100
tstim = 50
b_stim= np.copy(b)
b_stim[:int(N_E/2)] += .5


# #before learning, high_inhib model
# g = 4
# g_ii = 1
# h = 1
# J =  hippo_weights(index_dict, A, h,h, g, J0, g_ii = g_ii)
# pred_rates = y_0_quad(J, b)
# max_rate = np.max(pred_rates)
# maxspikes = int(np.floor(N*max_rate*tstop ))
# gc.collect()
# v, spktimes = sim_glm_pop(J=J,  E=b, dt = dt, tstop=tstop, tstim = tstim,  Estim=b_stim, v_th = 0, maxspikes = maxspikes, p = 2)
# raster_plot(spktimes, range(N),0, tstop, yticks = yticks, ax = axs[1,0])

#after learning, low_inhib model

g = 3
g_ii = 1
h = 2
J =  hippo_weights(index_dict, A, h,h, g, J0, g_ii = g_ii)
pred_rates = y_0_quad(J, b)
max_rate = np.max(pred_rates)
maxspikes = int(np.floor(N*max_rate*tstop ))
gc.collect()
v, spktimes = sim_glm_pop(J=J,  E=b, dt = dt, tstop=tstop, tstim = tstim,  Estim=b_stim, v_th = 0, maxspikes = maxspikes, p = 2)
raster_plot(spktimes, range(N),0, tstop, yticks = yticks, ax = axs[0,1])
axs[0,1].set_title(f"g={g}, g_II={g_ii}")


result = np.zeros((6, n_stim))
for i, stim in enumerate(stims):
    b_stim= np.copy(b)
    b_stim[:int(N_E/2)] += stim
    rates = y_0_quad(J, b_stim)
    result[:, i] = mean_by_region(rates, index_dict)

axs[1,1].plot(stims, result[3, :].T, color = "#F37343", label = "Engram")
axs[1,1].plot(stims, result[4, :].T, color = "#06ABC8", label = "Non-engram")
axs[1,1].set_xlabel("stimulus strength")
axs[1,1].set_ylabel("rate")
plt.tight_layout()
plt.savefig("../results/reactivation/figure.pdf")
plt.show() 


