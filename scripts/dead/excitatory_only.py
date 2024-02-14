import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os

from src.simulation import sim_glm_pop
from src.theory import  covariance_full,  y_0_quad
from src.correlation_functions import rate, mean_by_region, tot_cross_covariance_matrix, two_pop_correlation, mean_pop_correlation, cov_to_cor
from src.plotting import raster_plot, abline
from src.generate_connectivity import excitatory_only

plt.style.use('paper_style.mplstyle')

p = 0.1
N_engram = 50
N = 4*N_engram 

J0 = 1.1/(N*p)

y = .2*np.ones(N)
# y[0:N_engram] += .1
# y[2*N_engram:3*N_engram] += .1


n_h = 2
hs = np.linspace(1, 2, n_h)
rate_df = pd.DataFrame({"h":[], "i": [], "rate":[], "rate_pred":[]})
cor_df = pd.DataFrame({"h": [], "i": [], "j": [], "correlation": [], "covariance" : [], "correlation_pred": [], "covariance_pred" : []})

A = np.random.rand(4*N_engram, 4*N_engram) < p 
with open("../results/fig_1_data/adjacency.pkl", "wb") as file:
    pkl.dump(A, file)

index_dict = {"CA3E": range(N_engram), "CA3P": range(N_engram, 2*N_engram), "CA1E": range(2*N_engram, 3*N_engram), "CA1P": range(3*N_engram, 4*N_engram)}
with open("../results/fig_1_data/index.pkl", "wb") as file:
    pkl.dump(index_dict, file)

tstop = 5*10**4

mean_rates = []
mean_pred_rates = []
regions = []
hs_list = []

cors = []
pred_cors = []
regions_i = []
regions_j = []
cor_hs = []

for h in hs:

    J = excitatory_only(index_dict, A, h, J0)
   # plt.imshow(J)
    #plt.show()
    dt = 0.02

    pred_rates = y_0_quad(J, y)
    J_lin =J* (2*(J@pred_rates+y))[...,None]
    pred_covs = covariance_full(J_lin, pred_rates)
    pred_cors_mat = cov_to_cor(pred_covs)

    max_rate = np.max(pred_rates)
    maxspikes = int(np.floor(N*max_rate*tstop ))
    gc.collect()
    v, spktimes = sim_glm_pop(J=J,  E=y, dt = dt, tstop=tstop,  v_th = 0, maxspikes = maxspikes, p = 2)

    #raster_plot(spktimes, range(2*N_engram, 4*N_engram),0, tstop, yticks = [2*N_engram, 3*N_engram, 4*N_engram])
    #plt.savefig("../results/fig_1_data/raster_ext_only_h={}.pdf".format(h))
   # plt.show()
    with open("../results/fig_1_data/spikes_h={}ext_only.pkl".format(h), "wb") as file:
        pkl.dump(spktimes, file)
   

    for region in index_dict:
        if region[0:3] == "CA1":
            neurons = index_dict[region]
            rates = [rate(spktimes, i, dt, tstop) for i in neurons]
            mean_rate = np.mean(rates)
            mean_pred_rate = np.mean(pred_rates[neurons])
            mean_rates.append(mean_rate)
            mean_pred_rates.append(mean_pred_rate)
            regions.append(region)
            hs_list.append(h)

    for region_i in index_dict:
        for region_j in index_dict:
            if (region_i[0:3] == "CA1" ) and (region_j[0:3] == "CA1"):
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
rate_df.to_csv("../results/fig_1_data/rate_df.csv")

cor_df = pd.DataFrame({"region_i": regions_i, "region_j": regions_j, "h": cor_hs, "correlation": cors, "pred_correlation": pred_cors})
cor_df.to_csv("../results/fig_1_data/cor_df.csv")