import numpy as np
import params_model_1
import params_model_2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc


from src.generate_connectivity import hippo_weights, gen_adjacency
from src.simulation import sim_glm_pop
from src.theory import y_pred_from_full_connectivity, y_corrected_quad, y_0_quad
from src.correlation_functions import rate


def raster_plot(spktimes, neurons, t_start, t_stop):
    df = pd.DataFrame(spktimes, columns = ["time", "neuron"])
    df = df[(df["time"] < t_stop) &( df["time"] > t_start) ]
    df = df[df["neuron"].isin(neurons)]
    fig, ax = plt.subplots()
    s = 1000
    sns.scatterplot(data = df, x = "time", y = "neuron", marker = "|" , s = s/len(neurons), ax = ax, hue = "neuron",  palette = ["black"])
    plt.legend([],[], frameon=False)
    return fig, ax


cells_per_region = 5*np.array([1, 1, 1, 1, 1, 1])
N = np.sum(cells_per_region)
b = 1
g = 1
J = .1
h = 1

pEE = .2
pIE = .8
pII =.8
pEI = .8
import time

macro_connectivity = np.array([
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pIE, pIE, pII, pIE, pIE, pII],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pIE, pIE, pII, pIE, pIE, pII]])

A, index_dict = gen_adjacency(cells_per_region, macro_connectivity)
W =  hippo_weights(index_dict, A, h,h, g, J)
print(np.max(np.linalg.eigvals(W)))
y0 = b*np.ones(N)


dt = 0.02
tstop = 1000
maxspikes = 100000

_, spktimes_q = sim_glm_pop(J=W, E=b, dt = dt, tstop=tstop, tstim=0, Estim=0, v_th = 0, maxspikes = maxspikes, p = 2)
_, spktimes_l = sim_glm_pop(J=W, E=b, dt = dt, tstop=tstop, tstim=0, Estim=0, v_th = 0, maxspikes = maxspikes, p = 1)

fig, ax = raster_plot(spktimes_q,  range(N), 1, tstop)
plt.show()
fig, ax = raster_plot(spktimes_l,  range(N), 1, tstop)
plt.show()

y_l = y_pred_from_full_connectivity(W, y0, index_dict)
print("prediction: y_l")
y_q_0 = y_0_quad(W,y0)
y_q = y_corrected_quad(W,y0, y_q_0)
print("prediction: y_q")

rates_q = np.array([rate(spktimes_q, i,  tstop = tstop) for i in range(N)])
rates_l = np.array([rate(spktimes_l, i, tstop = tstop) for i in range(N)])

plt.figure()
plt.scatter(y_l, rates_l)
plt.title("linear")
plt.show()

plt.figure()
plt.scatter(y_q, rates_q)
plt.title("quadratic")
plt.show()

plt.figure()
plt.scatter(y_q_0, rates_q)
plt.title("quadratic tree level ")
plt.show()