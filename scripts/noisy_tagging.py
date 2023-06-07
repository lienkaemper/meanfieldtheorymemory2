import sys
import numpy as np
import params_model_1
import params_model_2
from src.theory import  y_pred,  C_pred_off_diag, J_eff, rates_with_noisy_tagging, cor_with_noisy_tagging
import matplotlib.pyplot as plt
from src.generate_connectivity import  macro_weights
import seaborn as sb
import pandas as pd
from tabulate import tabulate

showfigs = False
if len(sys.argv) > 1:
    if sys.argv[1] == "show":
        showfigs = True

plt.rcParams['pdf.fonttype'] = 42 

for i, params in enumerate([params_model_1, params_model_2]):
    par = params.params()
    p_E = 0.5
    p_P = 0.5


    y0 = 0.1

    p_fp = 0.2
    p_fn = 0.2

    
    cor_df = pd.read_csv("results/correlations_pred_by_h_{}.csv".format(i))
    cor_df = cor_df[cor_df["region"] == "CA1"]


    rate_df = pd.read_csv("results/rates_pred_by_h_{}.csv".format(i))
    rate_df = rate_df[rate_df["region"] == "CA1"]

    pivot_rate_df = rate_df.pivot_table(values='rate', index=['relative strength', 'source'], columns='type').reset_index()

  
    pivot_cor_df = cor_df.pivot_table(values = 'correlation', index=['relative strength', 'source'], columns = 'pair_group').reset_index()



    pivot_cor_df["noise"] = False
    def lambdafunc(x): 
        result = cor_with_noisy_tagging(x['Tagged vs Tagged'],
                                           x['Tagged vs Non-tagged'],
                                           x['Non-tagged vs Non-tagged'], p_E, p_P, p_fp, p_fn  )
        return pd.Series([True, result[0], result[1], result[2]])

    noisy_cor_df =  pivot_cor_df.copy()    

    noisy_cor_df[['noise', 'Tagged vs Tagged', 
    'Tagged vs Non-tagged', 
    'Non-tagged vs Non-tagged']] =  pivot_cor_df.apply(lambdafunc, axis=1)

    noisy_cor_df = pd.concat([noisy_cor_df, pivot_cor_df])

    print(noisy_cor_df.head())

    def lambdafunc(x): 
        result = rates_with_noisy_tagging(x['Tagged'],
                                           x['Non-tagged'], p_E, p_P, p_fp, p_fn  )
        return pd.Series([True, result[0], result[1]])

    pivot_rate_df["noise"] = False
    noisy_rate_df =  pivot_rate_df.copy()
    noisy_rate_df[['noise', 'Tagged', 
    'Non-tagged']] =  pivot_rate_df.apply(lambdafunc, axis=1)
    noisy_rate_df = pd.concat([noisy_rate_df, pivot_rate_df])



    #print(pivot_cor_df.head(10))
    melted_rate_df = pd.melt(noisy_rate_df, id_vars= ["relative strength", "source", "noise"], value_vars=['Non-tagged', 'Tagged'], var_name='Tag', value_name='rate')
    melted_rate_df.to_csv("results/noisy_rate_df_{}.csv".format(i))

    print(noisy_cor_df.head())
    melted_cor_df = pd.melt(noisy_cor_df , id_vars= ["relative strength", "source", "noise"], value_vars=['Tagged vs Tagged', 'Tagged vs Non-tagged',  'Non-tagged vs Non-tagged'], var_name='Tag', value_name='cor')
    melted_cor_df.to_csv("results/noisy_cor_df_{}.csv".format(i))








   