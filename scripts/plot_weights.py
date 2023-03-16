
import numpy as np
import params_model_1
import params_model_2
from src.theory import  y_pred,  C_pred_off_diag, J_eff
import matplotlib.pyplot as plt
from src.generate_connectivity import  macro_weights
import seaborn as sb
import pandas as pd
from matplotlib import colors
import pickle as pkl

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


    h_vals = par.h_range

    hue_order = ["Tagged vs Tagged", "Tagged vs Non-tagged", "Non-tagged vs Non-tagged"]
    cor_df = pd.DataFrame(columns=["relative strength", "correlation", "pair_group"])
    rate_df = pd.DataFrame(columns=["relative strength", "rate", "type"])
    plt.rcParams["text.usetex"] = True
    plt.rcParams['pdf.fonttype'] = 42 
    for h in h_vals:

        with open("results/W_h={}i={}.pkl".format(h,i), "rb") as f:
            W = pkl.load(f)

        divnorm=colors.TwoSlopeNorm(vmin=np.min(W), vcenter=0., vmax=np.max(W))
        plt.figure()
        plt.imshow(W, cmap = "bwr", norm = divnorm)
        plt.colorbar(ticks = np.unique(W.flatten()))
        plt.savefig("results/weights_h={}_{}.pdf".format(h,i))

        Gmax =  macro_weights(J, 2, 2, g)
        G = macro_weights(J, h, h, g)
        J_e  = J_eff(G, Ns, p_mat)
        divnorm=colors.TwoSlopeNorm(vmin=np.min(Gmax), vcenter=0., vmax=np.max(Gmax))
        plt.figure()
        plt.imshow(J_e, cmap = "bwr", norm = divnorm)
        plt.colorbar(ticks = J_e.flatten())
        plt.savefig("results/eff_weights_h={}_{}.pdf".format(h,i))
