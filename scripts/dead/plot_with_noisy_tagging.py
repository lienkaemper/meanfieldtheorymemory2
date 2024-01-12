import sys
import numpy as np
import params_model_1
import params_model_2
from src.theory import  y_pred,  C_pred_off_diag, J_eff
import matplotlib.pyplot as plt
from src.generate_connectivity import  macro_weights
import seaborn as sb
import pandas as pd
import pickle as pkl

showfigs = False
if len(sys.argv) > 1:
    if sys.argv[1] == "show":
        showfigs = True

p_fp = 0.2
p_fn = 0.2


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

    
    with open("results/index_dict_{}.pkl".format(i), "rb") as f:
        index_dict = pkl.load(f)

    noisy_index_dict = {'CA3I' : np.array(index_dict['CA3I']), 
    'CA3E' : np.array(index_dict['CA3E']), 
    'CA3P' : np.array(index_dict['CA3P']),
    'CA1I' : np.array(index_dict['CA1I'])}

    n_E_orig = len(index_dict['CA1E'])
    n_P_orig = len(index_dict['CA1P'])

    E_tags = (np.random.rand(n_E_orig) > p_fn)
    E_non_tags = np.invert(E_tags)
    P_tags = (np.random.rand(n_P_orig) < p_fp)
    P_non_tags =  np.invert(P_tags)
    
    true_pos = np.array(index_dict['CA1E'])[E_tags]
    false_pos = np.array(index_dict['CA1P'])[P_tags]

    true_neg= np.array(index_dict['CA1P'])[P_non_tags]
    false_neg = np.array(index_dict['CA1E'])[E_non_tags]

   
    noisy_tagged_E = np.concatenate((true_pos, false_pos))
    noisy_tagged_P = np.concatenate((true_neg, false_neg))

    noisy_index_dict['CA1E'] = noisy_tagged_E
    noisy_index_dict['CA1P'] = noisy_tagged_P





    
    y0 = 0.1

    data_df = pd.read_csv("results/pairwise_covariances_from_sim_{}.csv".format(i))
    data_df["correlation"] = [complex(x).real for x in data_df["correlation"]]
    data_df["covariance"] = [complex(x).real for x in data_df["covariance"]]

    noisy_df = data_df.copy()
    #noisy_df['noisy_tag_i']  = ''
    #noisy_df['noisy_tag_j']  = ''
    #noisy_df['noisy_pair_group'] = ''

    def noisy_pair(i,j):
        if i in noisy_index_dict["CA1E"]:
            if j in noisy_index_dict["CA1E"]:
               return  "Tagged vs Tagged"
            else:
               return "Tagged vs Non-tagged"
        else:
            if j in noisy_index_dict["CA1E"]:
                return  "Tagged vs Non-tagged"
            else:
                return "Non-tagged vs Non-tagged"

    def noisy_label(i):
        if i in noisy_index_dict["CA1E"]:
            return "Tagged"
        else:
            return "Non-tagged"
    


    noisy_df['noisy_pair_group'] = noisy_df.apply(lambda x: noisy_pair(x["i"], x["j"]), axis=1)
    noisy_df['noisy_type_i'] = noisy_df.apply(lambda x: noisy_label(x["i"]), axis=1)
    noisy_df['noisy_type_j'] = noisy_df.apply(lambda x: noisy_label(x["j"]), axis=1)

    


   

    cor_df = pd.read_csv("results/correlations_pred_by_h_{}.csv".format(i))
    cor_df = cor_df[cor_df["region"] == "CA1"]
    rate_df = pd.read_csv("results/rates_pred_by_h_{}.csv".format(i))

    hue_order = ["Tagged vs Tagged", "Tagged vs Non-tagged", "Non-tagged vs Non-tagged"]
    pred_data = data_df.loc[data_df["source"] == "theory", :]
    g_df = pred_data.groupby(["pair_group", "h"])
    mean_data_df = g_df.aggregate(np.mean)

    noisy_pred_data = noisy_df.loc[noisy_df["source"] == "theory", :]
    noisy_g_df = noisy_pred_data.groupby(["noisy_pair_group", "h"])
    noisy_mean_data_df = noisy_g_df.aggregate(np.mean)

    fig, ax = plt.subplots()
    sb.scatterplot(data = mean_data_df, 
                    x = "h", y = "correlation", 
                    hue = "pair_group", 
                    hue_order = hue_order, 
                    marker="$\circ$", 
                    s = 500, 
                    ax = ax)
    sb.scatterplot(data = noisy_mean_data_df, 
                x = "h", y = "correlation", 
                hue = "noisy_pair_group", 
                hue_order = hue_order, 
                marker="$N$", 
                s = 500, 
                ax = ax)

    sb.lineplot(data = cor_df, 
                    x = "relative strength", 
                    y = "correlation", 
                    hue = 'pair_group',  
                    hue_order = hue_order,  
                    style = "source",
                    ax = ax)

    ax.set_title("correlations from martrix: model {}".format(i))             
    plt.savefig("results/correlation_theory_fig_matrix_{}.pdf".format(i))
    plt.ylim(0, 0.012)

    if showfigs:
        plt.show()

    sim_data = data_df.loc[data_df["source"] == "data", :]
    g_df = sim_data.groupby(["pair_group", "h"])
    mean_data_df = g_df.aggregate(np.mean)

    noisy_sim_data = noisy_df.loc[noisy_df["source"] == "data", :]
    noisy_g_df = noisy_pred_data.groupby(["noisy_pair_group", "h"])
    noisy_mean_data_df = noisy_g_df.aggregate(np.mean)

    fig, ax = plt.subplots()
    sb.scatterplot(data = mean_data_df, 
                    x = "h", y = "correlation", 
                    hue = "pair_group", 
                    hue_order = hue_order, 
                    marker="$\circ$", 
                    s = 500, 
                    ax = ax)
    sb.scatterplot(data = noisy_mean_data_df, 
            x = "h", y = "correlation", 
            hue = "noisy_pair_group", 
            hue_order = hue_order, 
            marker="$N$", 
            s = 500, 
            ax = ax)

    sb.lineplot(data = cor_df, 
                    x = "relative strength", 
                    y = "correlation", 
                    hue = 'pair_group',  
                    hue_order = hue_order,  
                    style = "source",
                    ax = ax)
    plt.ylim(0, 0.012)

    ax.set_title("correlations from simulation: model {}".format(i))           

    plt.savefig("results/correlation_theory_fig_simulation_{}.pdf".format(i))

    if showfigs:
        plt.show()

    fig, ax = plt.subplots()
    ax.scatter(x = data_df.loc[data_df["source"] == "data", :]["rate_i"], y =data_df.loc[data_df["source"] == "theory", :]["rate_i"])
    ax.plot(np.linspace(0, 0.5), np.linspace(0, 0.5))
    plt.savefig("results/compare_matrix_sim_rate{}.pdf".format(i))
    if showfigs:
        plt.show()

    fig, ax = plt.subplots()
    ax.scatter(x = data_df.loc[data_df["source"] == "data", :]["covariance"], y =data_df.loc[data_df["source"] == "theory", :]["covariance"])
    ax.plot(np.linspace(0, 0.5), np.linspace(0, 0.5))
    plt.savefig("results/compare_matrix_sim_cov{}.pdf".format(i))
    if showfigs:
        plt.show()

    hue_order = ["Tagged", "Non-tagged"]
    sim_data = data_df.loc[data_df["source"] == "data", :]
    g_df = sim_data.groupby(["type_i", "i", "h"])
    mean_g_df = g_df.aggregate(np.mean)
    g_df = mean_g_df.groupby(["type_i", "h"])
    mean_data_df = g_df.aggregate(np.mean)

    noisy_sim_data = noisy_df.loc[noisy_df["source"] == "data", :]
    g_df = noisy_sim_data.groupby(["noisy_type_i", "i", "h"])
    mean_g_df = g_df.aggregate(np.mean)
    g_df = mean_g_df.groupby(["noisy_type_i", "h"])
    mean_noisy_data_df = g_df.aggregate(np.mean)
    
    #print(mean_data_df)
    fig, ax = plt.subplots()
    sb.scatterplot(data = mean_data_df, 
                    x = "h", y = "rate_i", 
                    hue = "type_i", 
                    marker="$\circ$", 
                    s = 500, 
                    ax = ax,
                    hue_order = hue_order)
    sb.scatterplot(data = mean_noisy_data_df, 
                    x = "h", y = "rate_i", 
                    hue = "noisy_type_i", 
                    marker="$N$", 
                    s = 500, 
                    ax = ax,
                    hue_order = hue_order)
    sb.lineplot(data = rate_df[rate_df["region"] == "CA1"], x = "relative strength", y = "rate", hue = 'type', style = "source",ax =ax, hue_order = hue_order)
    plt.gca().set_ylim(bottom=0)
    plt.savefig("results/rate_theory_fig_{}.pdf".format(i))

    if showfigs:
        plt.show()
