import sys
import os 
import numpy as np
import params_model_1
import params_model_2
from src.correlation_functions import  rate, tot_pop_autocovariance, tot_cross_covariance_matrix, mean_by_region, tot_cross_covariance
from src.theory import  y_pred, overall_cor_pred, cor_from_full_connectivity, y_pred_from_full_connectivity, length_1_full, C_pred_off_diag
import matplotlib.pyplot as plt
from src.generate_connectivity import hippo_weights, macro_weights
import pickle as pkl
import pandas as pd
from tqdm import tqdm
import itertools

#for vers, params in enumerate([params_model_1, params_model_2]):
for params in [params_model_2]:
    vers = 1
    with open("results/index_dict_{}.pkl".format(vers), "rb") as f:
        index_dict = pkl.load(f)
    par = params.params()
    N = par.N
    b = par.b
    dt =par.dt
    tstop = par.tstop
    p = par.macro_connectivity
    h_vals = par.h_range

    g = par.g
    J = par.J
    Ns = par.cells_per_region
    y0 = par.b[0]
    y_0 = y0 * np.ones(N)

    df = pd.DataFrame(columns = ["i", "j", "pair_group", "h", "rate_i", "rate_j", "covariance", "source"])
    CA1E = index_dict["CA1E"]
    CA1P = index_dict["CA1P"]
    for h in h_vals:
    
        with open("results/W_h={}i={}.pkl".format(h,vers), "rb") as f:
            W = pkl.load(f)


        with open("results/spikes_h={}i={}.pkl".format(h,vers), "rb") as f:
            spktimes = pkl.load(f)
        
    # C = tot_cross_covariance_matrix(spktimes, range(N), dt, tstop)
        C_theory = cor_from_full_connectivity(W, y0, index_dict)
        Y_theory = y_pred_from_full_connectivity(W, y0, index_dict)

        print(h)
        for i, j in tqdm(itertools.product(CA1E, CA1E)):
            if i > j:
                df_row = pd.DataFrame({"i" : [i,i], 
                                "j" : [j,j], 
                                "pair_group" : ["Tagged vs Tagged", "Tagged vs Tagged"] ,
                                "type_i": ["Tagged", "Tagged"],
                                "type_j": ["Tagged", "Tagged"],
                                "h" : [h,h], 
                                "rate_i" : [rate(spktimes, i, dt, tstop), Y_theory[i]],
                                "rate_j" : [rate(spktimes, j ,dt, tstop), Y_theory[j]],
                                "covariance" : [tot_cross_covariance(spktimes, i, j, dt, tstop), C_theory[i,j]],
                                "source" : ["data", "theory"]
                                })
                df = pd.concat([df, df_row])

        for i, j in tqdm(itertools.product(CA1E, CA1P)):
                    df_row = pd.DataFrame({"i" : [i,i], 
                                    "j" : [j,j], 
                                    "pair_group" : [ "Tagged vs Non-tagged",  "Tagged vs Non-tagged"] ,
                                    "type_i": ["Tagged", "Tagged"],
                                    "type_j": ["Non-tagged", "Non-tagged"],
                                    "h" : [h,h], 
                                    "rate_i" : [rate(spktimes, i, dt, tstop), Y_theory[i]],
                                    "rate_j" : [rate(spktimes, j ,dt, tstop), Y_theory[j]],
                                    "covariance" : [tot_cross_covariance(spktimes, i, j, dt, tstop), C_theory[i,j]],
                                    "source" : ["data", "theory"]
                                    })
                    df = pd.concat([df, df_row])

        for i, j in tqdm(itertools.product(CA1P, CA1P)):
                if i > j :
                    df_row = pd.DataFrame({"i" : [i,i], 
                                    "j" : [j,j], 
                                    "pair_group" : [  "Non-tagged vs Non-tagged",  "Non-tagged vs Non-tagged"] ,
                                    "type_i": ["Non-tagged", "Non-tagged"],
                                    "type_j": ["Non-tagged", "Non-tagged"],
                                    "h" : [h,h], 
                                    "rate_i" : [rate(spktimes, i, dt, tstop), Y_theory[i]],
                                    "rate_j" : [rate(spktimes, j ,dt, tstop), Y_theory[j]],
                                    "covariance" : [tot_cross_covariance(spktimes, i, j, dt, tstop), C_theory[i,j]],
                                    "source" : ["data", "theory"]
                                    })
                    df = pd.concat([df, df_row])


    df["correlation"] = df["covariance"]/np.sqrt((df["rate_i"] * df["rate_j"]))
    df.to_csv("results/pairwise_covariances_from_sim_{}.csv".format(vers))