import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
import networkx as nx
import sys

from src.simulation import sim_glm_pop
from src.theory import y_pred_full, covariance_full,  y_0_quad
from src.correlation_functions import rate, mean_by_region, tot_cross_covariance_matrix, create_pop_spike_train
from src.plotting import raster_plot, abline
from src.generate_connectivity import excitatory_only, gen_adjacency


plt.style.use('paper_style.mplstyle')

if len(sys.argv) < 2:
    f = open("../results/most_recent.txt", "r")
    dirname = f.read()
else:
    dirname = sys.argv[1]

with open(dirname+"/index_dict.pkl", "rb") as file:
    index_dict = pkl.load(file)

with open(dirname + "/param_dict.pkl", "rb") as file:
    param_dict = pkl.load(file)


################################








CA1E = index_dict["CA1E"]
CA1P = index_dict["CA1P"]
dt_spktrain = 1

CA1_neurons = list(CA1E) + list(CA1P)
N = param_dict["N"]
tstop = 500


rate_df = pd.read_csv("../results/fig_1_data/full_rate_df_low_inhib.csv")
baseline_rate = np.mean(rate_df[rate_df["h"] == 1]["rate"])


norm_rate_df = rate_df.copy()
norm_rate_df["rate"] = rate_df["rate"]/baseline_rate

result_df = norm_rate_df.pivot(index=['region', "i"], columns=['h'], values=['rate']).reset_index()
colnames = [f'rate (h = {col})' for col in result_df.columns.get_level_values(1)]
colnames[0] = "region"
colnames[1] = "i"
result_df.columns = colnames
print(result_df.head())


fig, ax = plt.subplots(figsize = (2,2))
sns.scatterplot(data =result_df, x = "rate (h = 1.0)", y = "rate (h = 2.0)", hue = "region", ax = ax)
ax.set_xlim(0.5, 2.5)
ax.set_ylim(0.5, 2.5)
plt.savefig("../results/fig_1_data/rate_scatterplot.pdf")
plt.show()
