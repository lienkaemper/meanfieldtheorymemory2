import sys
import os 
import numpy as np
import params
from src.correlation_functions import  rate, tot_pop_autocovariance, tot_cross_covariance_matrix
from src.theory import  y_pred, overall_cor_pred, overall_cor_from_full_connectivity, y_pred_from_full_connectivity
import matplotlib.pyplot as plt
from src.generate_connectivity import hippo_weights, macro_weights
import pickle as pkl

with open("results/index_dict_before.pkl", "rb") as f:
    index_dict = pkl.load(f)

with open("results/data_before.pkl", "rb") as f:
    sim_data_before = pkl.load(f)

with open("results/data_after.pkl", "rb") as f:
    sim_data_after = pkl.load(f)

with open("results/theory_before.pkl", "rb") as f:
    pred_before = pkl.load(f)

with open("results/theory_after.pkl", "rb") as f:
    pred_after = pkl.load(f)


#plot rates 

width = 0.35
labels = index_dict.keys()

rates_before = sim_data_before["mean_rates"]
y_pred_before = pred_before["y_pred_val"]
stds_before = sim_data_before["rate_stds"]
fig, ax = plt.subplots()
x = np.arange(len(rates_before))
ax.bar(x = x-width/4,  height= rates_before, yerr = stds_before, width = width/4, label = "simulation before")
ax.bar(x = x-width/2, height = y_pred_before, width = width/4, label = "prediction before")

rates_after = sim_data_after["mean_rates"]
y_pred_after = pred_after["y_pred_val"]
stds_after = sim_data_after["rate_stds"]
ax.bar(x = x+width/4,  height= rates_after, yerr = stds_after, width = width/4, label = "simulation after")
ax.bar(x = x+width/2, height = y_pred_after, width = width/4, label = "prediction after")


ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("mean rate")
ax.legend()

plt.title("rates")
plt.savefig("results/rates_before_and_after.pdf")
plt.show()


pop_variance_before = sim_data_before["pop_variances"]
pop_variance_after = sim_data_after["pop_variances"]
pred_pop_variance_before = np.diag(pred_before["C_summary"])
pred_pop_variance_after = np.diag(pred_after["C_summary"])



fig, ax = plt.subplots()
x = np.arange(len(pop_variance_before))

ax.bar(x = x-width/4,  height= pred_pop_variance_before, width = width/4, label = "prediction before")
ax.bar(x = x-width/2,  height= pop_variance_before, width = width/4, label = "simulation before")


ax.bar(x = x+width/4,  height= pred_pop_variance_after, width = width/4, label = "prediction after")
ax.bar(x = x+width/2,  height= pop_variance_after, width = width/4, label = "simulation after")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("population variance")
ax.legend()

plt.title("population variance")
plt.savefig("results/population_variance_before_and_after.pdf")
plt.show()

off_diagonal_before = sim_data_before["mean_offdiags"]
off_diagonal_after = sim_data_after["mean_offdiags"]
pred_off_diagonal_before = np.diag(pred_before["offdiag"])
pred_off_diagonal_after = np.diag(pred_after["offdiag"])




fig, ax = plt.subplots()
x = np.arange(len(pop_variance_before))

ax.bar(x = x-width/4,  height= pred_off_diagonal_before, width = width/4, label = "prediction before")
ax.bar(x = x-width/2,  height= off_diagonal_before, width = width/4, label = "simulation before")


ax.bar(x = x+width/4,  height= pred_off_diagonal_after, width = width/4, label = "prediction after")
ax.bar(x = x+width/2,  height= off_diagonal_after, width = width/4, label = "simulation after")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("mean offdiagonal covariance")
ax.legend()

plt.title("mean off diagonal covariance")
plt.savefig("results/off_diagonal_covariance_variance_before_and_after.pdf")
plt.show()


all_covariance_before = np.concatenate([C.flatten() for C in sim_data_before["cov_mats"]])
all_covariance_after = np.concatenate([C.flatten() for C in sim_data_after["cov_mats"]])
pred_all_covariances_before = np.concatenate([C.flatten() for C in pred_before["length_1_blocks"]])
pred_all_covariances_after = np.concatenate([C.flatten() for C in pred_after["length_1_blocks"]])

fig, ax = plt.subplots()
ax.scatter(x = all_covariance_before, y = pred_all_covariances_before )
ax.scatter(x = all_covariance_after, y = pred_all_covariances_after )
ax.plot(np.linspace(0, np.max(all_covariance_after)), np.linspace(0, np.max(all_covariance_after)))
ax.set_xlabel("simulation")
ax.set_ylabel("theory")
plt.savefig("results/theory_vs_sim.pdf")
plt.show()


all_rates_before = sim_data_before["all_rates"]
all_rates_after = sim_data_after["all_rates"]
pred_rates_before = pred_before["y_pred_full"]
pred_rates_after = pred_after["y_pred_full"]

fig, ax = plt.subplots()
ax.scatter(x = all_rates_before , y = pred_rates_before )
ax.scatter(x = all_rates_after , y = pred_rates_after )
ax.plot(np.linspace(0, np.max(all_rates_after)), np.linspace(0, np.max(all_rates_after)))
ax.set_xlabel("simulation")
ax.set_ylabel("theory")
plt.savefig("results/theory_vs_sim_rates.pdf")
plt.show()



