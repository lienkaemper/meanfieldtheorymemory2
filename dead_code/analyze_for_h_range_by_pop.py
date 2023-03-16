import sys
import os 
import numpy as np
import params
from src.correlation_functions import  rate, tot_pop_autocovariance, tot_cross_covariance_matrix, mean_by_region, two_pop_covariance, tot_cross_covariance, pop_rate
from src.theory import  y_pred, overall_cor_pred, cor_from_full_connectivity, y_pred_from_full_connectivity, length_1_full, C_pred_off_diag
import matplotlib.pyplot as plt
from src.generate_connectivity import hippo_weights, macro_weights
import pickle as pkl
import pandas as pd
from tqdm import tqdm


with open("results/index_dict.pkl", "rb") as f:
    index_dict = pkl.load(f)

par = params.params()
N = par.N
b = par.b
dt=par.dt
tstop = par.tstop
p = par.macro_connectivity
h_vals = par.h_range
g = par.g
J = par.J
Ns = par.cells_per_region
y0 = par.b[0]
y_0 = y0 * np.ones(N)



cor_df = pd.DataFrame(columns = ["pair_group", "h", "covariance", "region"])
rate_df = pd.DataFrame(columns = ["group", "h", "rate", "region"])
CA1E = index_dict["CA1E"]
CA1P = index_dict["CA1P"]

CA3E = index_dict["CA3E"]
CA3P = index_dict["CA3P"]

for h in h_vals:
   
    with open("results/W_h={}.pkl".format(h), "rb") as f:
        W = pkl.load(f)


    with open("results/spikes_h={}.pkl".format(h), "rb") as f:
        spktimes = pkl.load(f)
    
    print(h)

    print(spktimes.shape)
    print(spktimes[1:10,:])

    #CA1 
    tagged = pd.DataFrame({"pair_group": ["Tagged vs Tagged"], 
                            "h" : [h], 
                            "covariance" : [np.real((tot_pop_autocovariance(spktimes, CA1E, dt, tstop) - sum([ tot_cross_covariance(spktimes, i, i, dt, tstop ) for i in CA1E]))/(len(CA1E) * (len(CA1E)-1)))], 
                            "region": ["CA1"]})
    non_tagged = pd.DataFrame({"pair_group": ["Non-tagged vs Non-tagged"], 
                                "h" : [h], 
                                "covariance" : [np.real((tot_pop_autocovariance(spktimes, CA1P, dt, tstop) - sum([ tot_cross_covariance(spktimes, i, i, dt, tstop ) for i in CA1P]))/(len(CA1P) * (len(CA1P)-1)))],
                                 "region": ["CA1"]})
    mixed = pd.DataFrame({"pair_group": ["Tagged vs Non-tagged"], 
                         "h" : [h], 
                        "covariance" : [np.real(two_pop_covariance(spktimes, CA1P, CA1E, dt, tstop)/(len(CA1P)*len(CA1E)))], 
                         "region": ["CA1"]})
    cor_df = pd.concat([cor_df, tagged, non_tagged, mixed], ignore_index= True )

    tagged = pd.DataFrame({"group": ["Tagged"],
                            "h" : [h],
                            "rate": [pop_rate(spktimes, CA1E, dt, tstop)/len(CA1E)],
                             "region": ["CA1"]})             
    non_tagged = pd.DataFrame({"group": ["Non-tagged"],
                            "h" : [h],
                            "rate": [pop_rate(spktimes, CA1P, dt, tstop)/len(CA1P)],
                            "region": ["CA1"]})
    rate_df = pd.concat([rate_df, tagged, non_tagged],ignore_index= True )

    #CA3
    tagged = pd.DataFrame({"pair_group": ["Tagged vs Tagged"], 
                            "h" : [h], 
                            "covariance" : [np.real((tot_pop_autocovariance(spktimes, CA3E, dt, tstop) - sum([ tot_cross_covariance(spktimes, i, i, dt, tstop ) for i in CA3E]))/(len(CA3E) * (len(CA3E)-1)))], 
                            "region": ["CA3"]})
    non_tagged = pd.DataFrame({"pair_group": ["Non-tagged vs Non-tagged"], 
                                "h" : [h], 
                                "covariance" : [np.real((tot_pop_autocovariance(spktimes, CA3P, dt, tstop) - sum([ tot_cross_covariance(spktimes, i, i, dt, tstop ) for i in CA3P]))/(len(CA3P) * (len(CA3P)-1)))],
                                 "region": ["CA3"]})
    mixed = pd.DataFrame({"pair_group": ["Tagged vs Non-tagged"], 
                         "h" : [h], 
                        "covariance" : [np.real(two_pop_covariance(spktimes, CA3P, CA3E, dt, tstop)/(len(CA3P)*len(CA3E)))], 
                         "region": ["CA3"]})
    cor_df = pd.concat([cor_df, tagged, non_tagged, mixed], ignore_index= True )

    tagged = pd.DataFrame({"group": ["Tagged"],
                            "h" : [h],
                            "rate": [pop_rate(spktimes, CA3E, dt, tstop)/len(CA3E)],
                             "region": ["CA3"]})            
    non_tagged = pd.DataFrame({"group": ["Non-tagged"],
                            "h" : [h],
                            "rate": [pop_rate(spktimes, CA3P, dt, tstop)/len(CA3P)],
                            "region": ["CA3"]})
    rate_df = pd.concat([rate_df, tagged, non_tagged],ignore_index= True )

cor_df.to_csv("results/population_cor_df.csv")
rate_df.to_csv("results/population_rate_df.csv")