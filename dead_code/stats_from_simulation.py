import sys
import os 
import numpy as np
import params
from src.correlation_functions import  rate, tot_pop_autocovariance, tot_cross_covariance_matrix
from src.theory import  y_pred, overall_cor_pred, overall_cor_from_full_connectivity, y_pred_from_full_connectivity, length_1_full, C_pred_off_diag
import matplotlib.pyplot as plt
from src.generate_connectivity import hippo_weights, macro_weights
import pickle as pkl

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
#p = par.p
h3_before = par.h3_before
h3_after = par.h3_after
h1_before = par.h1_before
h1_after = par.h1_after
g = par.g
J = par.J
Ns = par.cells_per_region
y0 = par.b[0]
y_0 = y0 * np.ones(N)
p_mat = par.macro_connectivity




def analyze_simulation(index_dict, spktimes, dt, tstop):
    full_rates = []
    mean_rates =  np.array([])
    rate_stds = np.array([])
    pop_variances = np.array([])
    cov_mats = []
    mean_offdiags = np.array([])
    for region in index_dict.keys():
        inds = index_dict[region]
        rates_loc = np.array([])
        for ind in inds:
            r = rate(spktimes, ind, dt, tstop)
            rates_loc = np.append(rates_loc, r)
        full_rates.append(rates_loc)
        mean_rate = np.mean(rates_loc)
        rate_std = np.std(rates_loc)
        rate_stds = np.append(rate_stds, rate_std)
        mean_rates = np.append(mean_rates, mean_rate)
        pop_variance = tot_pop_autocovariance(spktimes, inds, dt, tstop)
        pop_variances = np.append(pop_variances, pop_variance)
        C = tot_cross_covariance_matrix(spktimes, inds, dt, tstop)
        cov_mats.append(C)
        mean_offdiag = np.mean(C[np.triu_indices(C.shape[0], k=1)])
        mean_offdiags = np.append(mean_offdiags, mean_offdiag)
    return {"all_rates" : full_rates,
            "mean_rates" : mean_rates, 
            "rate_stds": rate_stds, 
            "pop_variances": pop_variances, 
            "cov_mats": cov_mats, 
            "mean_offdiags": mean_offdiags}






def predict_from_theory(W, index_dict, Ns, j_scalar, h3, h1, g, p_mat, y0):
    result = {}
    J = macro_weights(j_scalar, h3, h1, g)
    result["y_pred_val"] = y_pred(J, Ns,  p_mat, y0)
    result["y_pred_summary"], result["y_pred_full"] =  y_pred_from_full_connectivity(W, y_0, index_dict)
   # result["C"] = overall_cor_pred(J, Ns, p_mat, y0)
    result["C_summary"], result["offdiag"], result["C_full"], result["C_blocks"] = overall_cor_from_full_connectivity(W, y0, index_dict)
    result["length_1"], result["length_1_blocks"] = length_1_full(W, y0, index_dict)
    result["C_pred"] = C_pred_off_diag(J, Ns, p_mat, y0)
    return result 

pred_before = predict_from_theory(W_before, index_dict, Ns, J, h3_before, h1_before, g, p_mat, y0)
pred_after = predict_from_theory(W_after, index_dict, Ns, J, h3_after, h1_after, g, p_mat, y0)


sim_data_before = analyze_simulation(index_dict, spktimes_before, dt, tstop)
sim_data_after  = analyze_simulation(index_dict, spktimes_after, dt, tstop)







with open("results/data_before.pkl", "wb") as f:
    pkl.dump(sim_data_before, f)

with open("results/data_after.pkl", "wb") as f:
    pkl.dump(sim_data_after, f)

with open("results/theory_before.pkl", "wb") as f:
    pkl.dump(pred_before, f)

with open("results/theory_after.pkl", "wb") as f:
    pkl.dump(pred_after, f)




