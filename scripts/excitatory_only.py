import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os




from src.simulation import sim_glm_pop
from src.theory import y_pred_full, covariance_full
from src.correlation_functions import rate, mean_by_region, tot_cross_covariance_matrix
from src.plotting import raster_plot, abline
from src.generate_connectivity import excitatory_only

p = 0.1
N_engram = 25
N = 4*N_engram 

J0 = 1/(N*p)

y = np.ones(N)
y[0:N_engram] += .5
y[2*N_engram:3*N_engram] += .5

n_h = 2
hs = np.linspace(1, 2, n_h)
rate_df = pd.DataFrame({"h":[], "i": [], "rate":[], "rate_pred":[]})
cor_df = pd.DataFrame({"h": [], "i": [], "j": [], "correlation": [], "covariance" : [], "correlation_pred": [], "covariance_pred" : []})
for h in hs:
    A = np.random.rand(4*N_engram, 4*N_engram) < p 
    index_dict = {"CA3E": range(N_engram), "CA3P": range(N_engram, 2*N_engram), "CA1E": range(2*N_engram, 3*N_engram), "CA1P": range(3*N_engram, 4*N_engram)}
    J = excitatory_only(index_dict, A, h, J0)
   # plt.imshow(J)
    #plt.show()
    dt = 0.02
    tstop = 2*10**5
    rate_pred = np.linalg.inv(np.eye(N) - J) @np.ones(N)
    max_rate = np.max(rate_pred)
    maxspikes = int(np.floor(N*max_rate*tstop ))
    v, spktimes = sim_glm_pop(J=J,  E=y, dt = dt, tstop=tstop,  v_th = 0, maxspikes = maxspikes, p = 1)
    raster_plot(spktimes, range(2*N_engram, 4*N_engram), 500, 700, yticks = [2*N_engram, 3*N_engram, 4*N_engram])
    plt.savefig("../results/raster_ext_only_h={}.pdf".format(h))
    rates = [rate(spktimes, i, dt, tstop) for i in range(2*N_engram, 4*N_engram)]
    rates_pred = y_pred_full(J, y)
    covariances = tot_cross_covariance_matrix(spktimes, range(2*N_engram, 4*N_engram), dt, tstop)
    covariances_pred = covariance_full(J, rates_pred)[2*N_engram:4*N_engram, 2*N_engram:4*N_engram]
    rates_pred = rates_pred[2*N_engram:4*N_engram]
    correlations =  (1/np.sqrt(np.diag(covariances))) *covariances* (1/(np.sqrt(np.diag(covariances))))[...,None]
    correlations_pred = (1/np.sqrt(np.diag(covariances_pred))) *covariances_pred* (1/(np.sqrt(np.diag(covariances_pred))))[...,None]
    i_list = []
    j_list = []
    for i in range(2*N_engram, 4*N_engram):
        for j in range(2*N_engram, 4*N_engram):
            i_list.append(i)
            j_list.append(j)
    rate_df = pd.concat([rate_df, pd.DataFrame({"h": h, "i": range(2*N_engram, 4*N_engram), "rate": rates, "rate_pred": rates_pred})])
    cor_df = pd.concat([cor_df, pd.DataFrame({"h": h, "i": i_list, "j": j_list, "correlation": correlations.flatten(), "covariance": covariances.flatten(), "correlation_pred" : correlations_pred.flatten(), "covariance_pred": covariances_pred.flatten()})])

rate_df['region'] = rate_df['i'].apply(lambda x: 'engram' if x <= 3*N_engram else 'non-engram')
cor_df['region_i'] = cor_df['i'].apply(lambda x: 'engram' if x <=  3*N_engram else 'non-engram')
cor_df['region_j'] = cor_df['j'].apply(lambda x: 'engram' if x <=  3*N_engram else 'non-engram')

rate_df.to_csv("../results/excitatory_only_rates.csv")
cor_df.to_csv("../results/excitatory_only_corrs.csv")
