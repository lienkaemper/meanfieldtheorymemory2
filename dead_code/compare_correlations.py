import sys
import os 
import numpy as np
import params
from src.correlation_functions import  rate, tot_pop_autocovariance, tot_cross_covariance_matrix, mean_by_region
from src.theory import  y_pred, overall_cor_pred, overall_cor_from_full_connectivity, y_pred_from_full_connectivity, length_1_full, C_pred_off_diag
import matplotlib.pyplot as plt
from src.generate_connectivity import hippo_weights, macro_weights
import pickle as pkl
import pandas as pd
from tqdm import tqdm


with open("results/index_dict_after.pkl", "rb") as f:
    index_dict = pkl.load(f)

with open("results/before_learning_W.pkl", "rb") as f:
    W_before = pkl.load(f)

with open("results/after_learning_W.pkl", "rb") as f:
    W_after = pkl.load(f)

with open("results/before_learning_spikes.pkl", "rb") as f:
    spktimes_before = pkl.load(f)

with open("results/after_learning_spikes.pkl", "rb") as f:
    spktimes_after = pkl.load(f)



par = params.params()
N = par.N
tau = par.tau
b = par.b
gain = par.gain
trans = par.trans
tstop = par.tstop
dt = .02 * tau  # Euler step
p = par.macro_connectivity
h3_before = par.h3_before
h3_after = par.h3_after
h1_before = par.h1_before
h1_after = par.h1_after
g = par.g
J = par.J
Ns = par.cells_per_region
y0 = par.b[0]
y_0 = y0 * np.ones(N)

J_before = macro_weights(J, h3_before, h1_before, g)
J_after = macro_weights(J, h3_before, h1_before, g)


C_pred_before = C_pred_off_diag(J_before, Ns, p, y0)
C_before_data = tot_cross_covariance_matrix(spktimes_before, range(N), dt, tstop)
C_mean_before_data = mean_by_region(C_before_data, index_dict)

C_pred_after = C_pred_off_diag(J_after, Ns, p, y0)
C_after_data = tot_cross_covariance_matrix(spktimes_after, range(N), dt, tstop)
C_mean_after_data = mean_by_region(C_after_data, index_dict)



df = pd.DataFrame(columns = ["i", "j", "pair_group", "session", "rate_i", "rate_j", "covariance"])
CA1E = index_dict["CA1E"]
CA1P = index_dict["CA1P"]

for i in tqdm(CA1E):
    for j in CA1E:
        if i!= j:
            df = df.append({"i" : i, 
                            "j" : j, 
                            "pair_group" : "Tagged vs Tagged",
                            "session" : "before", 
                            "rate_i" : rate(spktimes_before, i, dt, tstop),
                            "rate_j" :rate(spktimes_before, j ,dt, tstop),
                            "covariance" : C_before_data[i,j]
                            }, ignore_index = True)
            df = df.append({"i" : i, 
                            "j" : j, 
                            "pair_group" : "Tagged vs Tagged",
                            "session" : "after", 
                            "rate_i" : rate(spktimes_after, i, dt, tstop),
                            "rate_j" :rate(spktimes_after, j ,dt, tstop),
                            "covariance" : C_after_data[i,j]
                            }, ignore_index = True)

for i in tqdm(CA1E):
    for j in CA1P:
        df = df.append({"i" : i, 
                        "j" : j, 
                        "pair_group" : "Tagged vs Non-tagged",
                        "session" : "before", 
                        "rate_i" : rate(spktimes_before, i, dt, tstop),
                        "rate_j" :rate(spktimes_before, j ,dt, tstop),
                        "covariance" : C_before_data[i,j]
                        }, ignore_index = True)
        df = df.append({"i" : i, 
                        "j" : j, 
                        "pair_group" : "Tagged vs Non-tagged",
                        "session" : "after", 
                        "rate_i" : rate(spktimes_after, i, dt, tstop),
                        "rate_j" :rate(spktimes_after, j ,dt, tstop),
                        "covariance" : C_after_data[i,j]
                        }, ignore_index = True)

for i in tqdm(CA1P):
    for j in CA1P:
        if i != j :
            df = df.append({"i" : i, 
                            "j" : j, 
                            "pair_group" : "Non-tagged vs Non-tagged",
                            "session" : "before", 
                            "rate_i" : rate(spktimes_before, i, dt, tstop),
                            "rate_j" :rate(spktimes_before, j ,dt, tstop),
                            "covariance" : C_before_data[i,j]
                            }, ignore_index = True)
            df = df.append({"i" : i, 
                            "j" : j, 
                            "pair_group" : "Non-tagged vs Non-tagged",
                            "session" : "after", 
                            "rate_i" : rate(spktimes_after, i, dt, tstop),
                            "rate_j" :rate(spktimes_after, j ,dt, tstop),
                            "covariance" : C_after_data[i,j]
                            }, ignore_index = True)


df["correlation"] = df["covariance"]/np.sqrt((df["rate_i"] * df["rate_j"]))
df.to_csv("results/pairwise_covariances_from_sim.csv")