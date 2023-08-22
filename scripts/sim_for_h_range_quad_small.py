import numpy as np
import params_model_1
import params_model_2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
#from  datetime import datetime



from src.generate_connectivity import hippo_weights, gen_adjacency
from src.simulation import sim_glm_pop
from src.theory import y_pred_from_full_connectivity, y_corrected_quad, y_0_quad
from src.correlation_functions import rate, mean_by_region


def raster_plot(spktimes, neurons, t_start, t_stop):
    df = pd.DataFrame(spktimes, columns = ["time", "neuron"])
    df = df[(df["time"] < t_stop) &( df["time"] > t_start) ]
    df = df[df["neuron"].isin(neurons)]
    fig, ax = plt.subplots()
    s = 1000
    sns.scatterplot(data = df, x = "time", y = "neuron", marker = "|" , s = s/len(neurons), ax = ax, hue = "neuron",  palette = ["black"])
    plt.legend([],[], frameon=False)
    return fig, ax


h_range = [1, 1.25, 1.5, 1.75, 2]

cells_per_region = 20*np.array([1, 1, 1, 1, 1, 1])
N = np.sum(cells_per_region)
b = 1
g = 6
J = .25

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

A, index_dict = gen_adjacency(cells_per_region, macro_connectivity)

rates_l = []
rates_q = []

pred_rates_l = []
pred_rates_q = []

region_list = ['' for i in range(N)]
inputs_quad = []
for key in index_dict:
    start = min(index_dict[key])
    end = max(index_dict[key])
    region_list[start:end+1] = (end+1-start)*[key]

region_list = len(h_range) * region_list

h_list = [h for h in h_range for i in range(N)]


param_df = pd.DataFrame({"parameter": ["b", "g", "J", "pEE", "pII", "pEI"], "value": [b, g, J, pEE, pII, pEI]})

dirname = "../results/" + "_".join([p + str(v) for (p,v) in zip(param_df.parameter, param_df.value)])

if not os.path.isdir(dirname):
    os.mkdir(dirname)

text_file = open("../results/most_recent.txt", "w")
text_file.write(dirname)
text_file.close()

param_df.to_csv(dirname+"/parameters")

for h in h_range:
    W =  hippo_weights(index_dict, A, h,h, g, J)
  
    y0 = b*np.ones(N)


    dt = 0.02
    tstop = 5000
    maxspikes = 10**12

    _, spktimes_q = sim_glm_pop(J=W, E=b, dt = dt, tstop=tstop, tstim=0, Estim=0, v_th = 0, maxspikes = maxspikes, p = 2)
    raster_plot(spktimes_q, range(N),0, 1000)
    plt.title("quadratic")
    plt.savefig(dirname + "/quad_raster_plot_h={}.pdf".format(h))

    _, spktimes_l = sim_glm_pop(J=W, E=b, dt = dt, tstop=tstop, tstim=0, Estim=0, v_th = 0, maxspikes = maxspikes, p = 1)
    raster_plot(spktimes_l, range(N),0, 1000)
    plt.title("linear")
    plt.savefig(dirname + "/linear_raster_plot_h={}.pdf".format(h))




 
    y_l = y_pred_from_full_connectivity(W, y0, index_dict)
    y_q_0 = y_0_quad(W,y0)
    correction = y_corrected_quad(W,y0, y_q_0)
    y_q = y_q_0  + correction 
    inputs_quad.extend(W @ y_q + b) 
    print(np.max(np.linalg.eigvals(2*np.diag(y_q)@W)))
    

    rates_q.extend([rate(spktimes_q, i,  tstop = tstop) for i in range(N)])
    rates_l.extend([rate(spktimes_l, i, tstop = tstop) for i in range(N)])

    pred_rates_q.extend(y_q)
    pred_rates_l.extend(y_l)


df = pd.DataFrame({"region": region_list, "h": h_list, "rate_l": rates_l, "rate_q": rates_q, "pred_rate_l": pred_rates_l, "pred_rate_q": pred_rates_q, "input": inputs_quad})
df.to_csv(dirname+"/lin_vs_quad.csv")
