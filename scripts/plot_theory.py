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
    par = params.params()
    N = par.N
    b = par.b
    tstop = par.tstop
    g = par.g
    J = par.J
    Ns = par.cells_per_region
    y0 = par.b[0]
    p_mat = par.macro_connectivity
    N = np.sum(Ns)

    y0 = 0.1


    
    cor_df = pd.read_csv("results/noisy_cor_df_{}.csv".format(i))
    cor_df = cor_df[cor_df["region"] == "CA1"]
    rate_df = pd.read_csv("results/noisy_rate_df_{}.csv".format(i))

    hue_order = ["Tagged vs Tagged", "Tagged vs Non-tagged", "Non-tagged vs Non-tagged"]
    
    fig, ax = plt.subplots()
    
    sb.lineplot(data = cor_df, 
                    x = "relative strength", 
                    y = "correlation", 
                    hue = 'pair_group',  
                    hue_order = hue_order,  
                    style = "source",
                    ax = ax)

    ax.set_title("correlations from martrix: model {}".format(i))             
    plt.savefig("results/correlation_theory_only_fig_matrix_{}.pdf".format(i))
    plt.ylim(0, 0.012)

    if showfigs:
        plt.show()

    
   


    hue_order = ["Tagged", "Non-tagged"]
    
    fig, ax = plt.subplots()
    
    sb.lineplot(data = rate_df[rate_df["region"] == "CA1"], x = "relative strength", y = "rate", hue = 'type', style = "source",ax =ax, hue_order = hue_order)
    plt.gca().set_ylim(bottom=0)
    plt.savefig("results/rate_theory_only_fig_{}.pdf".format(i))

    if showfigs:
        plt.show()
