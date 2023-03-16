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

plt.rcParams["text.usetex"] = True
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
    print(p_mat)
    N = np.sum(Ns)

    y0 = 0.1

    data_df = pd.read_csv("results/pairwise_covariances_from_sim_{}.csv".format(i))
    cor_df = pd.read_csv("results/correlations_pred_by_h_{}.csv".format(i))
    rate_df = pd.read_csv("results/rates_pred_by_h_{}.csv".format(i))

    hue_order = ["Tagged vs Tagged", "Tagged vs Non-tagged", "Non-tagged vs Non-tagged"]
    sim_cor_df = pd.read_csv("results/population_cor_df.csv")
    fig, ax = plt.subplots()
    sb.scatterplot(data = sim_cor_df[sim_cor_df["region"] == "CA1"], 
                    x = "h", y = "covariance", 
                    hue = "pair_group", 
                    hue_order = hue_order, 
                    marker="$\circ$", 
                    s = 500, 
                    ax = ax)
    sb.lineplot(data = cor_df[cor_df["region"] == "CA1"], 
                    x = "relative strength", 
                    y = "covariance", 
                    hue = 'pair_group',  
                    style = "source",
                    hue_order = hue_order,  
                    ax = ax)
    ax.set_title("correlations from simulation")           
    plt.savefig("results/correlation_theory_fig_simulation_CA1.pdf")
    if showfigs:
        plt.show()

    fig, ax = plt.subplots()
    sb.scatterplot(data = sim_cor_df[sim_cor_df["region"] == "CA3"], 
                    x = "h", y = "covariance", 
                    hue = "pair_group", 
                    hue_order = hue_order, 
                    marker="$\circ$", 
                    s = 500, 
                    ax = ax)
    sb.lineplot(data = cor_df[cor_df["region"] == "CA3"], 
                    x = "relative strength", 
                    y = "covariance", 
                    hue = 'pair_group',
                    style = "source",  
                    hue_order = hue_order,  
                    ax = ax)
    ax.set_title("correlations from simulation")           
    plt.savefig("results/correlation_theory_fig_simulation_CA3.pdf")
    if showfigs:
        plt.show()

    sim_rate_df = pd.read_csv("results/population_rate_df.csv")
    fig, ax = plt.subplots()
    sb.scatterplot(data = sim_rate_df[sim_rate_df["region"] == "CA1"], 
                    x = "h", y = "rate",
                    hue = "group", 
                    marker="$\circ$", 
                    s = 500, 
                    ax = ax)
    sb.lineplot(data = rate_df[rate_df["region"] == "CA1"], x = "relative strength", y = "rate", hue = 'type', style = "source",ax =ax)
    plt.savefig("results/rate_theory_figCA1.pdf")
    if showfigs:
        plt.show()

    fig, ax = plt.subplots()
    sb.scatterplot(data = sim_rate_df[sim_rate_df["region"] == "CA3"], 
                    x = "h", y = "rate",
                    hue = "group", 
                    marker="$\circ$", 
                    s = 500, 
                    ax = ax)
    sb.lineplot(data = rate_df[rate_df["region"] == "CA3"], x = "relative strength", y = "rate", hue = 'type',style = "source", ax =ax)
    plt.savefig("results/rate_theory_fig.pdf")
    if showfigs:
        plt.show()
