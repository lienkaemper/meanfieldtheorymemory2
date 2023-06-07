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

    print(pivot_cor_df.head())
    print(pivot_cor_df.columns)

    lambdafunc = lambda x: pd.Series(cor_with_noisy_tagging(x['Tagged vs Tagged'],
                                           x['Tagged vs Non-tagged'],
                                           x['Non-tagged vs Non-tagged'], p_E, p_P, p_fp, p_fn  ))
    pivot_cor_df[['Tagged vs Tagged noisy', 
    'Tagged vs Non-tagged noisy', 
    'Non-tagged vs Non-tagged noisy']] =  pivot_cor_df.apply(lambdafunc, axis=1)

    lambdafunc = lambda x: pd.Series(rates_with_noisy_tagging(x['Tagged'],
                                           x['Non-tagged'], p_E, p_P, p_fp, p_fn  ))
    pivot_rate_df[['Tagged noisy', 
    'Non-tagged noisy']] =  pivot_rate_df.apply(lambdafunc, axis=1)


    #print(pivot_cor_df.head(10))
    melted_rate_df = pd.melt(pivot_rate_df, id_vars= ["relative strength", "source"], value_vars=['Non-tagged', 'Tagged', 'Tagged noisy', 'Non-tagged noisy'], var_name='Tag', value_name='Value')
    melted_rate_df.to_csv("results/noisy_rate_df.csv")

    print(pivot_cor_df.head())
    melted_cor_df = pd.melt(pivot_cor_df, id_vars= ["relative strength", "source"], value_vars=['Tagged vs Tagged', 'Tagged vs Non-tagged',  'Non-tagged vs Non-tagged', 'Tagged vs Tagged noisy', 'Tagged vs Non-tagged noisy', 'Non-tagged vs Non-tagged noisy'], var_name='Tag', value_name='Value')
    melted_cor_df.to_csv("results/noisy_cor_df.csv")








   