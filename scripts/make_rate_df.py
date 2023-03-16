import sys
import numpy as np
import params_model_1
import params_model_2
from src.theory import  y_pred,  C_pred_off_diag, J_eff
import matplotlib.pyplot as plt
from src.generate_connectivity import  macro_weights
import seaborn as sb
import pandas as pd

showfigs = False
if len(sys.argv) > 1:
    if sys.argv[1] == "show":
        showfigs = True

plt.rcParams['pdf.fonttype'] = 42 

for i, params in enumerate([params_model_1, params_model_2]):
  
    data_df = pd.read_csv("results/pairwise_covariances_from_sim_{}.csv".format(i))
    data_df["correlation"] = [complex(x).real for x in data_df["correlation"]]
    data_df["covariance"] = [complex(x).real for x in data_df["covariance"]]

    sim_data = data_df.loc[data_df["source"] == "data", :]
    print(sim_data.head())
    sim_data =sim_data[["i", "rate_i", "h", "type_i"]]
    print(sim_data.head())
    g_df = sim_data.groupby(["type_i", "i", "h"])
    print("here")
    mean_g_df = g_df.agg(np.mean)
    print(mean_g_df.head)
    mean_g_df.to_csv("results/rates_only_df_{}.csv".format(i))