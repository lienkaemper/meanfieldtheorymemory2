import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os




from src.simulation import sim_glm_pop
from src.theory import y_pred_full, covariance_full, y_pred, y_0_quad
from src.correlation_functions import rate, mean_by_region, tot_cross_covariance_matrix
from src.plotting import raster_plot, abline
from src.generate_connectivity import excitatory_only, gen_adjacency, macro_weights,  hippo_weights

h_min = 1
h_max = 2
n_h = 2


h_range = np.linspace(h_min, h_max, n_h)

N_E =60
N_I = 15
cells_per_region =np.array([N_E, N_E, N_I,  N_E, N_E, N_I])
b_small = .5
N = np.sum(cells_per_region)
b = .5*np.ones(N)
b[0:N_E] += .1
b[2*N_E + N_I: 3*N_E + N_I] += .1
b[2*N_E: 2*N_E + N_I] += .2
b[4*N_E+N_I:] += .2
g = 3
J0 = .2
g_ii = .25




dt = 0.02
tstop = 100
tstim = 50

n_points = 101
h_is = np.linspace(1,2, n_points)
rates = np.zeros((n_points, n_h))
for i, h in enumerate(h_range):
    for j, h_i in enumerate(h_is):
        G = macro_weights(J0, h, h, g, h_i, g_ii= g_ii)
        y = y_pred(G,  b_small)
        rates[j,i] = y[3]

matched_h_i_l = h_is[np.argmin(np.abs(rates - rates[0,0] ) , axis = 0)]
print(matched_h_i_l)

rates = np.zeros((n_points, n_h))

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
values = [b_small, g, J0, g_ii, pEE, pII, pEI, N, h_min, h_max, n_h, dt, tstop]
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
hs = np.linspace(1, 2, n_h)
rate_df = pd.DataFrame({"h":[], "i": [], "rate":[], "rate_pred":[]})
cor_df = pd.DataFrame({"h": [], "i": [], "j": [], "correlation": [], "covariance" : [], "correlation_pred": [], "covariance_pred" : []})
for k, h in enumerate(hs):
    CA1_cells = [i for i in index_dict["CA1E"]]+ [j for j in index_dict["CA1P"]]
    J =  hippo_weights(index_dict, A, h,h, g, J0, i_plast = matched_h_i_l [k], g_ii = g_ii)
   # plt.imshow(J)
    #plt.show()
    dt = 0.02
    tstop = 2*10**5
    rates_pred = y_0_quad(J, b)
    max_rate = np.max(rates_pred)
    maxspikes = int(np.floor(N*max_rate*tstop ))
    gc.collect()
    v, spktimes = sim_glm_pop(J=J,  E=b, dt = dt, tstop=tstop,  v_th = 0, maxspikes = maxspikes, p = 2)
    raster_plot(spktimes, range(N), 500, 700)
    plt.savefig("../results/raster_ext_only_h={}.pdf".format(h))
    #plt.show()
    rates = [rate(spktimes, i, dt, tstop) for i in CA1_cells] 
    rates_pred = y_0_quad(J, b)
    covariances = tot_cross_covariance_matrix(spktimes, CA1_cells, dt, tstop)
    J_lin =J* (2*(J@rates_pred+b))[...,None]
    covariances_pred = (np.linalg.inv(np.eye(N) -  J_lin)@np.diag(rates_pred)@ np.linalg.inv(np.eye(N) -  J_lin).T)[np.ix_(CA1_cells,CA1_cells)]
    #covariances_pred = covariance_full(J_lin, rates_pred)[np.ix_(CA1_cells,CA1_cells)]
    rates_pred = rates_pred[CA1_cells]
    correlations =  (1/np.sqrt(np.diag(covariances))) *covariances* (1/(np.sqrt(np.diag(covariances))))[...,None]
    correlations_pred = (1/np.sqrt(np.diag(covariances_pred))) *covariances_pred* (1/(np.sqrt(np.diag(covariances_pred))))[...,None]
    i_list = []
    j_list = []
    for i in CA1_cells:
        for j in CA1_cells:
            i_list.append(i)
            j_list.append(j)
    rate_df = pd.concat([rate_df, pd.DataFrame({"h": h, "i": CA1_cells, "rate": rates, "rate_pred": rates_pred})])
    cor_df = pd.concat([cor_df, pd.DataFrame({"h": h, "i": i_list, "j": j_list, "correlation": correlations.flatten(), "covariance": covariances.flatten(), "correlation_pred" : correlations_pred.flatten(), "covariance_pred": covariances_pred.flatten()})])

rate_df['region'] = rate_df['i'].apply(lambda x: 'engram' if x in index_dict["CA1E"] else 'non-engram')
cor_df['region_i'] = cor_df['i'].apply(lambda x: 'engram' if  x in index_dict["CA1E"] else 'non-engram')
cor_df['region_j'] = cor_df['j'].apply(lambda x: 'engram' if  x in index_dict["CA1E"] else 'non-engram')

rate_df.to_csv("../results/inhib_also_rates.csv")
cor_df.to_csv("../results/inhib_also__corrs.csv")
