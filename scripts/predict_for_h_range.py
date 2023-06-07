import numpy as np
import params_model_1
import params_model_2
from src.theory import  y_pred,  C_pred_off_diag, J_eff, cor_from_full_connectivity, y_pred_from_full_connectivity
import matplotlib.pyplot as plt
from src.generate_connectivity import  macro_weights
import seaborn as sb
import pandas as pd
import pickle as pkl
from src.generate_connectivity import hippo_weights
from src.correlation_functions import mean_by_region


for i, params in enumerate([params_model_1, params_model_2]):


    par = params.params()
    N = par.N
    b = par.b
    tstop = par.tstop
    #dt = .02 * tau  # Euler step
    #p = par.p


    g = par.g
    J = par.J
    Ns = par.cells_per_region
    y0 = par.b[0]
    p_mat = par.macro_connectivity


    H_range = np.linspace(1, 1.75)

    hue_order = ["Tagged vs Tagged", "Tagged vs Non-tagged", "Non-tagged vs Non-tagged"]
    cor_df = pd.DataFrame(columns=["relative strength", "correlation", "pair_group", "source"])
    rate_df = pd.DataFrame(columns=["relative strength", "rate", "type"])

    with open("results/adjacency_{}.pkl".format(i), "rb") as f:
        A =  pkl.load(f)

    with open("results/index_dict_{}.pkl".format(i), "rb") as f:
        index_dict =  pkl.load(f)

    for h in H_range:
        G = macro_weights(J, h, h, g)
        C = C_pred_off_diag(G, Ns, p_mat, y0)
        y = y_pred(G, Ns, p_mat, y0)
        W =  hippo_weights(index_dict, A, h,h, g, J)
        y_full =y_pred_from_full_connectivity(W, y0, index_dict)
       # print(np.min(y_full))
        Cov_full = cor_from_full_connectivity(W, y0, index_dict)
        Cor_full = np.diag(np.sqrt(1/y_full)) @ Cov_full @ np.diag(np.sqrt(1/y_full))
        Cor_full = mean_by_region(Cor_full, index_dict)
        Cov_full =  mean_by_region(Cov_full, index_dict)
        y_full = mean_by_region(y_full, index_dict)
        cor_df_rows = pd.DataFrame({"relative strength":[h, h, h, h, h, h], 
                        "covariance": [C[3,3],C[3,4], C[4,4], C[0,0], C[0,1], C[1,1]],
                        "correlation": [C[3,3]/y[3],C[3,4]/(np.sqrt(y[3]*y[4])), C[4,4]/y[4], C[0,0]/y[0],C[0,1]/(np.sqrt(y[0]*y[1])), C[1,1]/y[1]],
                        "pair_group": ["Tagged vs Tagged",  "Tagged vs Non-tagged", "Non-tagged vs Non-tagged", "Tagged vs Tagged",  "Tagged vs Non-tagged", "Non-tagged vs Non-tagged"],
                        "region": ["CA1", "CA1", "CA1", "CA3", "CA3", "CA3"], 
                        "source": ["reduced", "reduced", "reduced", "reduced", "reduced", "reduced"], })
        cor_df_rows_full = pd.DataFrame({"relative strength":[h, h, h, h, h, h], 
                        "covariance": [Cov_full[3,3],Cov_full[3,4], Cov_full[4,4], Cov_full[0,0], Cov_full[0,1], Cov_full[1,1]],
                        "correlation": [Cor_full[3,3],Cor_full[3,4], Cor_full[4,4], Cor_full[0,0],Cor_full[0,1], Cor_full[1,1]],
                        "pair_group": ["Tagged vs Tagged",  "Tagged vs Non-tagged", "Non-tagged vs Non-tagged", "Tagged vs Tagged",  "Tagged vs Non-tagged", "Non-tagged vs Non-tagged"],
                        "region": ["CA1", "CA1", "CA1", "CA3", "CA3", "CA3"], 
                        "source": ["full", "full", "full", "full", "full", "full"], })
        cor_df = pd.concat([cor_df, cor_df_rows, cor_df_rows_full], ignore_index = True)
        rate_df_rows = pd.DataFrame({"relative strength":[h, h, h, h], 
                        "rate":[y[3], y[4], y[0], y[1] ],
                        "type": ["Tagged", "Non-tagged", "Tagged", "Non-tagged"],
                        "region": ["CA1", "CA1", "CA3", "CA3"],
                        "source": ["reduced", "reduced", "reduced", "reduced"]})
        rate_df_rows_full = pd.DataFrame({"relative strength":[h, h, h, h], 
                        "rate":[y_full[3], y_full[4], y_full[0], y_full[1] ],
                        "type": ["Tagged", "Non-tagged", "Tagged", "Non-tagged"],
                        "region": ["CA1", "CA1", "CA3", "CA3"],
                        "source": ["full", "full", "full", "full"]})
        rate_df = pd.concat([rate_df, rate_df_rows, rate_df_rows_full], ignore_index= True)
    rate_df.to_csv("results/rates_pred_by_h_{}.csv".format(i))
    cor_df.to_csv("results/correlations_pred_by_h_{}.csv".format(i))








