import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os

from src.simulation import sim_glm_pop
from src.theory import  y_0_quad,  find_iso_rate_input, cor_pred, find_iso_rate_ca3
from src.correlation_functions import  mean_by_region, sum_by_region
from src.plotting import raster_plot
from src.generate_connectivity import gen_adjacency, hippo_weights

plt.style.use('paper_style.mplstyle')


N_E = 60
N_I = 15
cells_per_region =np.array([N_E, N_E, N_I,  N_E, N_E, N_I])
b_small = np.array([.4, .4, .5, .4, .4, .5]) 


N = np.sum(cells_per_region)
b = np.concatenate([b_small[i]*np.ones(cells_per_region[i]) for i in range(6)])


J0 = .2



dt = 0.02
tstop = 100
tstim = 50


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

J =  hippo_weights(index_dict, A, h3 = 1, h1 = 1, g = 1, J = J0,  g_ii =  1)
J_baseline = sum_by_region(J, index_dict=index_dict)
y_baseline = y_0_quad(J_baseline, b_small)


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

fig, axs = plt.subplots(3,2, figsize = (7, 6))


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





J_small = sum_by_region(J, index_dict=index_dict)
result = np.zeros((6, n_stim))
result_cors = np.zeros((3,n_stim))
for i, stim in enumerate(stims):
    b_stim= np.copy(b)
    b_stim[:int(N_E/2)] += stim
    rates = y_0_quad(J, b_stim)
    result[:, i] = mean_by_region(rates, index_dict)
    pred_cors = cor_pred( J = J_small , Ns = cells_per_region, y =result[:,i])
    result_cors[:, i] = np.array([pred_cors[3,3], pred_cors[3,4], pred_cors[4,4]])

axs[1,0].plot(stims, result[3, :].T, color = "#F37343", label = "Engram")
axs[1,0].plot(stims, result[4, :].T, color = "#06ABC8", label = "Non-engram")
axs[1,0].set_xlabel("stimulus strength")
axs[1,0].set_ylabel("rate")

axs[2,0].plot(stims, result_cors[0, :], color = "#F37343", label = "Engram-Engram")
axs[2,0].plot(stims, result_cors[1, :], label = "Engram vs. non-engram", color = "#FEC20E")
axs[2,0].plot(stims, result_cors[2, :], color = "#06ABC8", label = "Non-engram-Non-engram")

tstop = 100
tstim = 50




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


J =  hippo_weights(index_dict, A, h3 = 1, h1 = 1, g = g, J = J0,  g_ii =  g_ii)
J_small = sum_by_region(J, index_dict=index_dict)
b_iso =find_iso_rate_input(target_rate= y_baseline[3], J = J_small, b = b_small, b0_min = 0, b0_max = .5, n_points=1000, plot = False)
h_i1, h_i3  = find_iso_rate_ca3(y_baseline[3],y_baseline[4], h=h, J0 = J0, g=g, g_ii=g_ii, b = b_iso, h_i_min = 1, h_i_max = 4, type = "quadratic", n_points = 1000)
b = np.concatenate([b_iso[i]*np.ones(cells_per_region[i]) for i in range(6)])
b_stim= np.copy(b)
b_stim[:int(N_E/2)] += .8


h = 2
J =  hippo_weights(index_dict, A, h,h, i_plast=h_i1, i_plast_3=h_i3, g=g, J=J0, g_ii = g_ii)
pred_rates = y_0_quad(J, b)
max_rate = np.max(pred_rates)
maxspikes = int(np.floor(N*max_rate*tstop ))
gc.collect()
v, spktimes = sim_glm_pop(J=J,  E=b, dt = dt, tstop=tstop, tstim = tstim,  Estim=b_stim, v_th = 0, maxspikes = maxspikes, p = 2)
raster_plot(spktimes, range(N),0, tstop, yticks = yticks, ax = axs[0,1])
axs[0,1].set_title(f"g={g}, g_II={g_ii}")

J_small = sum_by_region(J, index_dict=index_dict)
result = np.zeros((6, n_stim))
result_cors = np.zeros((3,n_stim))
for i, stim in enumerate(stims):
    b_stim= np.copy(b)
    b_stim[:int(N_E/2)] += stim
    rates = y_0_quad(J, b_stim)
    result[:, i] = mean_by_region(rates, index_dict)
    pred_cors = cor_pred( J = J_small , Ns = cells_per_region, y=result[:,i])
    result_cors[:, i] = np.array([pred_cors[3,3], pred_cors[3,4], pred_cors[4,4]])

axs[1,1].plot(stims, result[3, :].T, color = "#F37343", label = "Engram")
axs[1,1].plot(stims, result[4, :].T, color = "#06ABC8", label = "Non-engram")
axs[1,1].set_xlabel("stimulus strength")
axs[1,1].set_ylabel("rate")

axs[2, 1].plot(stims, result_cors[0, :], color = "#F37343", label = "Engram-Engram")
axs[2, 1].plot(stims, result_cors[1, :], label = "Engram vs. non-engram", color = "#FEC20E")
axs[2, 1].plot(stims, result_cors[2, :], color = "#06ABC8", label = "Non-engram-Non-engram")

plt.tight_layout()
plt.savefig("../results/reactivation/figure.pdf")
plt.show() 


