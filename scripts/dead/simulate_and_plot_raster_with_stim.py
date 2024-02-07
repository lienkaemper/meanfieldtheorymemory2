import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os


plt.style.use('poster_style.mplstyle')

from src.generate_connectivity import hippo_weights, gen_adjacency, macro_weights

from src.simulation import sim_glm_pop
from src.theory import y_pred_from_full_connectivity, y_corrected_quad, y_0_quad, y_pred
from src.correlation_functions import rate, mean_by_region


def raster_plot(spktimes, neurons, t_start, t_stop):
    df = pd.DataFrame(spktimes, columns = ["time", "neuron"])
    df = df[(df["time"] < t_stop) &( df["time"] > t_start) ]
    df = df[df["neuron"].isin(neurons)]
    fig, ax = plt.subplots()
    s = 2000
    sns.scatterplot(data = df, x = "time", y = "neuron", marker = "|" , s = s/len(neurons), ax = ax, hue = "neuron",  palette = ["black"])
    plt.legend([],[], frameon=False)
    return fig, ax


h_min = 1
h_max = 2
n_h = 2

h_range = np.linspace(h_min, h_max, n_h)

N_E = 160
N_I = 40
N_stim = 80
cells_per_region =np.array([N_E, N_E, N_I,  N_E, N_E, N_I])
b_small = .5
N = np.sum(cells_per_region)
b = .5*np.ones(N)


Ns = np.array([160, 160, 40, 160, 160, 40])
Ns = np.array([160, 160, 40, 160, 160, 40])
b[0:N_E] += .1
b[2*N_E + N_I: 3*N_E + N_I] += .1
b[2*N_E: 2*N_E + N_I] += .2
b[4*N_E+N_I:] += .2
g = 3
J = .2
g_ii = .25

b_stim = np.copy(b)
b_stim[0:N_stim] += 1


dt = 0.02
tstop = 100
tstim = 50

n_points = 101
h_is = np.linspace(1,2, n_points)
rates = np.zeros((n_points, n_h))
for i, h in enumerate(h_range):
    for j, h_i in enumerate(h_is):
        G = macro_weights(J, h, h, g, h_i, g_ii= g_ii)
        y = y_pred(G,  b_small)
        rates[j,i] = y[3]

matched_h_i_l = h_is[np.argmin(np.abs(rates - rates[0,0] ) , axis = 0)]
print(matched_h_i_l)

rates = np.zeros((n_points, n_h))
for i, h in enumerate(h_range):
    for j, h_i in enumerate(h_is):
        G = macro_weights(J, h, h, g, h_i, g_ii = g_ii)
        y = y_0_quad(G,  b_small)
        rates[j,i] = y[3]

matched_h_i_q = h_is[np.argmin(np.abs(rates - rates[0,0] ) , axis = 0)]
print(matched_h_i_q)
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
values = [b_small, g, J, g_ii, pEE, pII, pEI, N, h_min, h_max, n_h, dt, tstop]
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
    W_l =  hippo_weights(index_dict, A, h,h, g, J, i_plast = matched_h_i_l [i], g_ii = g_ii)
    #plt.imshow(W_l)
    #plt.title("linear")
    #plt.show()
    with open(dirname + "/W_l_h={}.pkl".format(h), "wb") as file:
            pkl.dump(W_l, file)

    W_q =  hippo_weights(index_dict, A, h,h, g, J, i_plast = matched_h_i_q[i], g_ii = g_ii)
    with open(dirname + "/W_q_h={}.pkl".format(h), "wb") as file:
            pkl.dump(W_q, file)
   # plt.imshow(W_q)
   # plt.title("quadratic")
   # plt.show()
    y0 = b*np.ones(N)

    y = y_pred_from_full_connectivity(W_l, b, index_dict)
    maxspikes = int(np.max(y) * N * tstop)


    gc.collect()
    _, spktimes_q = sim_glm_pop(J=W_q, E=b, dt = dt, tstop=tstop, tstim=tstim, Estim=b_stim, v_th = 0, maxspikes = maxspikes, p = 2)

    with open(dirname + "/spikes_q_h={}.pkl".format(h), "wb") as file:
        pkl.dump(spktimes_q, file)


    fig, ax = raster_plot(spktimes_q, range(N), 0, tstop)
    plt.savefig("../results/raster_spikes_q_stim_h={}.pdf".format(h))
    plt.savefig("../results/raster_spikes_q_stim_h={}.png".format(h))    
    plt.show()

    _, spktimes_l = sim_glm_pop(J=W_l, E=b, dt = dt, tstop=tstop,  tstim=tstim, Estim=b_stim, v_th = 0, maxspikes = maxspikes, p = 1)

    with open(dirname + "/spikes_l_h={}.pkl".format(h), "wb") as file:
        pkl.dump(spktimes_l, file)

    fig, ax = raster_plot(spktimes_l, range(N), 0, tstop)
    plt.savefig("../results/raster_spikes_l_stim_h={}.pdf".format(h))
    plt.savefig("../results/raster_spikes_l_stim_h={}.png".format(h))
    plt.show()


