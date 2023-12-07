import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os

from src.simulation import sim_glm_pop
from src.theory import y_pred_full, covariance_full,  y_0_quad
from src.correlation_functions import rate, mean_by_region, tot_cross_covariance_matrix, two_pop_correlation, mean_pop_correlation, cov_to_cor
from src.plotting import raster_plot, abline
from src.generate_connectivity import excitatory_only, gen_adjacency, hippo_weights, macro_weights

plt.style.use('paper_style.mplstyle')

h_min = 1
h_max = 2
n_h = 5
trials = 5


h_range = np.linspace(h_min, h_max, n_h)

N_E =60
N_I = 15
cells_per_region =np.array([N_E, N_E, N_I,  N_E, N_E, N_I])
b_small = [.5, .5, .7, .5, .5, .7]


N = np.sum(cells_per_region)
b = np.concatenate([b_small[i]*np.ones(cells_per_region[i]) for i in range(6)])
g = 1
J0 = .2
g_ii = 1




dt = 0.02
tstop = 50000
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


parameters = ["b", "g", "J", "g_ii", "pEE", "pII", "pEI", "N", "h_min", "h_max", "n_h", "dt", "tstop"]
values = [b_small, g, J0, g_ii, pEE, pII, pEI, N, h_min, h_max, n_h, dt, tstop]
param_df = pd.DataFrame({"parameter": parameters, "value": values})
dirname = "../results/fig_1_data/" + "_".join([p + str(v) for (p,v) in zip(param_df.parameter, param_df.value)])

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



mean_rates = []
mean_pred_rates = []
regions = []
hs_list = []

full_rates = []
full_rate_regions = []
full_rate_hs = []
is_list = []

cors = []
pred_cors = []
regions_i = []
regions_j = []
cor_hs = []
for i in range(trials):
    for k, h in enumerate(h_range):

        J =  hippo_weights(index_dict, A, h,h, g, J0, i_plast = 1, g_ii = g_ii)
    # plt.imshow(J)
        #plt.show()
        dt = 0.02

        pred_rates = y_0_quad(J, b)
        J_lin =J* (2*(J@pred_rates+b))[...,None]
        pred_covs = covariance_full(J_lin, pred_rates)
        pred_cors_mat = cov_to_cor(pred_covs)

        max_rate = np.max(pred_rates)
        maxspikes = int(np.floor(N*max_rate*tstop ))
        gc.collect()
        v, spktimes = sim_glm_pop(J=J,  E=b, dt = dt, tstop=tstop,  v_th = 0, maxspikes = maxspikes, p = 2)

        #raster_plot(spktimes, range(2*N_engram, 4*N_engram),0, tstop, yticks = [2*N_engram, 3*N_engram, 4*N_engram])
        #plt.savefig("../results/fig_1_data/raster_ext_only_h={}.pdf".format(h))
    # plt.show()
        with open("../results/fig_1_data/spikes_h={}ext_only_low_inhib.pkl".format(h), "wb") as file:
            pkl.dump(spktimes, file)
    

        for region in index_dict:
            if (region == "CA1E" )or (region == "CA1P" ):
                neurons = index_dict[region]
                rates = [rate(spktimes, i, dt, tstop) for i in neurons]
                full_rates.extend(rates)
                mean_rate = np.mean(rates)
                mean_pred_rate = np.mean(pred_rates[neurons])
                mean_rates.append(mean_rate)
                mean_pred_rates.append(mean_pred_rate)
                regions.append(region)
                full_rate_regions.extend(N_E *[region])
                hs_list.append(h)
                full_rate_hs.extend(N_E *[h])
                is_list.extend(neurons)

        for region_i in index_dict:
            for region_j in index_dict:
                if ((region_i == "CA1E" )or (region_i== "CA1P" )) and ((region_j == "CA1E" )or (region_j == "CA1P" )):
                    if region_i == region_j:
                        neurons= index_dict[region_i]
                        cor = mean_pop_correlation(spktimes, neurons, dt, tstop)
                        cors.append(cor)
                        cor_hs.append(h)
                        pred_cors.append(np.mean(pred_cors_mat[np.ix_(neurons, neurons)][np.triu_indices(len(neurons),1)]))
                        regions_i.append(region_i)
                        regions_j.append(region_j)

                    if region_i < region_j: 
                        neurons_i = index_dict[region_i]
                        neurons_j = index_dict[region_j]
                        cor = two_pop_correlation(spktimes, neurons_i, neurons_j, dt, tstop)
                        cors.append(cor)
                        cor_hs.append(h)
                        pred_cors.append(np.mean(pred_cors_mat[np.ix_(neurons_i, neurons_j)]))
                        regions_i.append(region_i)
                        regions_j.append(region_j)


rate_df = pd.DataFrame({"region": regions, "h": hs_list, "rate": mean_rates, "pred_rates": mean_pred_rates})
rate_df.to_csv("../results/fig_1_data/rate_df_low_inhib.csv")

full_rate_df = pd.DataFrame({"region": full_rate_regions, "h": full_rate_hs, "rate": full_rates, "i": is_list})
full_rate_df.to_csv("../results/fig_1_data/full_rate_df_low_inhib.csv")

cor_df = pd.DataFrame({"region_i": regions_i, "region_j": regions_j, "h": cor_hs, "correlation": cors, "pred_correlation": pred_cors})
cor_df.to_csv("../results/fig_1_data/cor_df_low_inhib.csv")