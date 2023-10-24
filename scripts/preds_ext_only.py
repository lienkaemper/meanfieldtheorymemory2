import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
import sys



from src.theory import y_pred_full, covariance_full,  y_0_quad
from src.correlation_functions import rate, mean_by_region, tot_cross_covariance_matrix
from src.plotting import raster_plot, abline
from src.generate_connectivity import excitatory_only

with open("../results/fig_1_data/adjacency.pkl", "rb") as file:
    A = pkl.load(file)

with open("../results/fig_1_data/index.pkl", "rb") as file:
    index_dict = pkl.load(file)

p = 0.1
N_engram = 50
N = 4*N_engram 

J0 = 1.1/(N*p)

y = .2*np.ones(N)
y[0:N_engram] += .1
y[2*N_engram:3*N_engram] += .1

h_range = np.linspace(1,2,5)
pred_rates = []
pred_rates_full = []
correlations = []
i_s_c = []
j_s_c = []
i_s_r = []

for h in h_range:
    J = excitatory_only(index_dict, A, h, J0)
    y_q_0 = y_0_quad(J,y)
    pred_rates_full.extend(y_q_0)
    covariances_pred = covariance_full(J, y_q_0)
    correlations_pred = (1/np.sqrt(np.diag(covariances_pred))) *covariances_pred* (1/(np.sqrt(np.diag(covariances_pred))))[...,None]
    correlations.extend(correlations_pred.flatten())
    i_list_c = []
    j_list_c = []
    for i in range(N):
        for j in range(N):
            i_list_c.append(i)
            j_list_c.append(j)
    i_s_c.extend(i_list_c)
    j_s_c.extend(j_list_c)
    i_s_r.extend([i for i in range(N)])



h_list = [h for h in h_range for i in range(N)]
rate_df = pd.DataFrame({"h":h_list, "i": i_s_r, "rate_full":pred_rates_full})
rate_df['region'] = rate_df['i'].apply(lambda x: 'engram' if x <= 3*N_engram else 'non-engram')

rate_df.to_csv("../results/fig_1_data/rate_df.csv")

h_list_cor = [h for h in h_range for i in range(N**2)]


cor_df = pd.DataFrame({"h":h_list_cor, "cor_full":correlations, "i": i_s_c, "j": j_s_c})
cor_df['region_i'] = cor_df['i'].apply(lambda x: 'engram' if x <=  3*N_engram else 'non-engram')
cor_df['region_j'] = cor_df['j'].apply(lambda x: 'engram' if x <=  3*N_engram else 'non-engram')
cor_df.to_csv("../results/fig_1_data/cor_df.csv")


