import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
#from  datetime import datetime



from src.generate_connectivity import hippo_weights, gen_adjacency, macro_weights

from src.simulation import sim_glm_pop
from src.theory import y_pred_from_full_connectivity, y_corrected_quad, y_0_quad, y_pred
from src.correlation_functions import rate, mean_by_region


# def raster_plot(spktimes, neurons, t_start, t_stop):
#     df = pd.DataFrame(spktimes, columns = ["time", "neuron"])
#     df = df[(df["time"] < t_stop) &( df["time"] > t_start) ]
#     df = df[df["neuron"].isin(neurons)]
#     fig, ax = plt.subplots()
#     s = 1000
#     sns.scatterplot(data = df, x = "time", y = "neuron", marker = "|" , s = s/len(neurons), ax = ax, hue = "neuron",  palette = ["black"])
#     plt.legend([],[], frameon=False)
#     return fig, ax


h_min = 1
h_max = 2
n_h = 5

h_range = np.linspace(h_min, h_max, n_h)


cells_per_region = 20*np.array([1, 1, 1, 1, 1, 1])
N = np.sum(cells_per_region)
b = .5
g = 2
J = .2
g_ii = .5

dt = 0.02
tstop = 2*10**5# simulation time
#tstop = 1000

n_points = 101
h_is = np.linspace(1,4, n_points)
rates = np.zeros((n_points, n_h))
for i, h in enumerate(h_range):
    for j, h_i in enumerate(h_is):
        G = macro_weights(J, h, h, g, h_i, g_ii= g_ii)
        y = y_pred(G,  b)
        rates[j,i] = y[3]

before_val = y_pred(macro_weights(J, 1, 1, g, 1, g_ii = g_ii), b)[3]
matched_h_i_l = h_is[np.argmin(np.abs(rates - before_val ) , axis = 0)]

rates = np.zeros((n_points, n_h))
for i, h in enumerate(h_range):
    for j, h_i in enumerate(h_is):
        G = macro_weights(J, h, h, g, h_i, g_ii = g_ii)
        y = y_0_quad(G,  b)
        rates[j,i] = y[3]

before_val =y_0_quad(macro_weights(J, 1, 1, g, 1, g_ii = g_ii), b)[3]
matched_h_i_q = h_is[np.argmin(np.abs(rates - before_val ) , axis = 0)]

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


parameters = ["b", "g", "J", "g_ii", "pEE", "pII", "pEI", "N", "h_min", "h_max", "n_h", "dt", "tstop"]
values = [b, g, J, g_ii, pEE, pII, pEI, N, h_min, h_max, n_h, dt, tstop]
param_df = pd.DataFrame({"parameter": parameters, "value": values})
dirname = "../results/" + "_".join([p + str(v) for (p,v) in zip(param_df.parameter, param_df.value)])

if not os.path.isdir(dirname):
    os.mkdir(dirname)

text_file = open("../results/most_recent.txt", "w")
text_file.write(dirname)
text_file.close()

A, index_dict = gen_adjacency(cells_per_region, macro_connectivity)

with open(dirname + "/index_dict.pkl", "wb") as file:
    pkl.dump(index_dict, file)

with open(dirname + "/adjacency.pkl", "wb") as file:
    pkl.dump(A, file)

with open(dirname + "/param_dict.pkl", "wb") as file:
    pkl.dump(dict(zip(parameters, values)), file)


param_df.to_csv(dirname+"/parameters.csv")

for i,h in enumerate(h_range):
    W_l =  hippo_weights(index_dict, A, h,h, g, J, i_plast = matched_h_i_l[i])
    with open(dirname + "/W_l_h={}.pkl".format(h), "wb") as file:
            pkl.dump(W_l, file)

    W_q =  hippo_weights(index_dict, A, h,h, g, J, i_plast = matched_h_i_q[i])
    with open(dirname + "/W_q_h={}.pkl".format(h), "wb") as file:
            pkl.dump(W_q, file)

    y0 = b*np.ones(N)

    y = y_pred_from_full_connectivity(W_l, b, index_dict)
    maxspikes = int(np.max(y) * N * tstop)


    gc.collect()
    _, spktimes_q = sim_glm_pop(J=W_q, E=b, dt = dt, tstop=tstop, tstim=0, Estim=0, v_th = 0, maxspikes = maxspikes, p = 2)


    _, spktimes_l = sim_glm_pop(J=W_l, E=b, dt = dt, tstop=tstop, tstim=0, Estim=0, v_th = 0, maxspikes = maxspikes, p = 1)


    with open(dirname + "/spikes_q_h={}.pkl".format(h), "wb") as file:
        pkl.dump(spktimes_q, file)

    with open(dirname + "/spikes_l_h={}.pkl".format(h), "wb") as file:
        pkl.dump(spktimes_l, file)
