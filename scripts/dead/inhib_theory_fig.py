import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os


from src.theory import y_0_quad
from src.generate_connectivity import macro_weights

def fp_and_lin(J0, g, h, b, N,  p = 2):
    J = macro_weights(J = J0, h3 = h, h1 = h, g = g)
    if p == 2:
        r = y_0_quad(J, b)
        gain =  2*(J@r+b)
        J_lin =J* gain[...,None]
    else: 
        J_lin = J
        r = np.linalg.inv(np.eye(6) - J_lin)@ b
    return r, J_lin
   


def CA3_prop(J0, g, h, b, N, nterms = None, p = 2):
    _, J_lin = fp_and_lin(J0, g, h, b, N, p)
    J_CA3 = J_lin[:3, :3]
    if nterms != None:
        Delta = np.identity(3)
        for n in range(1,nterms+1):
            Delta += np.linalg.matrix_power(J_CA3, n)      
    else:
        Delta = np.linalg.inv(np.identity(3) - J_CA3)
    return Delta


def CA1_prop(J0, g, h, b, N, nterms = None, p = 2):
    _, J_lin = fp_and_lin(J0, g, h, b, N, p)
    J_CA1 = J_lin[3:, 3:]
    if nterms != None:
        Delta = np.identity(3)
        for n in range(1,nterms+1):
            Delta += np.linalg.matrix_power(J_CA1, n)
    else:
        Delta = np.linalg.inv(np.identity(3) - J_CA1)
    return Delta


def CA1_internal_cov(J0, g, h, b, N, nterms = None, p = 2):
    r, J_lin = fp_and_lin(J0, g, h, b, N, p)
    R = np.diag(r/N)
    R1 = R[3:, 3:]
    D_11 = CA1_prop(J0, g, h, b, N, nterms =nterms, p = p)
    return D_11 @ R1 @ D_11.T

def CA1_internal_cov_offdiag(J0, g, h, b, N, nterms = None, p = 2):
    r, J_lin = fp_and_lin(J0, g, h, b, N, p)
    R = np.diag(r/N)
    R1 = R[3:, 3:]
    D_11 = CA1_prop(J0, g, h, b, N, nterms =nterms, p = p)
    return D_11 @ R1 @ D_11.T - R1

def CA3_internal_cov(J0, g, h, b, N, nterms = None, p = 2):
    r, J_lin = fp_and_lin(J0, g, h, b, N, p)
    R = np.diag(r/N)
    R3 = R[:3, :3]
    J_CA3 = J_lin[:3, :3]
    D_33 = CA3_prop(J0, g, h, b, N, nterms =nterms, p = p)
    return (D_33 @ R3 @ D_33.T)


def CA1_inherited_cov(J0, g, h, b, N, nterms = None, p = 2):
    C_33 = CA3_internal_cov(J0, g, h, b, N, nterms = nterms, p = p)
    _, J_lin = fp_and_lin(J0, g, h, b, N, p)
    J_13 = J_lin[3:, :3]
    D_11 = CA1_prop(J0, g, h, b, N, nterms = nterms, p = p)
    return (D_11 @ J_13 @ C_33 @ J_13.T @ D_11.T)



def CA3_E_from_E(J0, g, h, b, N, nterms = None, p = 2):
    r, J_lin = fp_and_lin(J0, g, h, b, N, p)
    R = np.diag(r/N)
    R3 = R[:3, :3]
    J_CA3 = J_lin[:3, :3]
    D_33 = CA3_prop(J0, g, h, b, N, nterms =nterms, p = p)
    return D_33[0,0]**2 * R3[0,0]

def CA3_E_from_N(J0, g, h, b, N, nterms = None, p = 2):
    r, J_lin = fp_and_lin(J0, g, h, b, N, p)
    R = np.diag(r/N)
    R3 = R[:3, :3]
    J_CA3 = J_lin[:3, :3]
    D_33 = CA3_prop(J0, g, h, b, N, nterms =nterms, p = p)
    return D_33[0,1]**2 * R3[1,1]

def CA3_E_from_I(J0, g, h, b, N, nterms = None, p = 2):
    r, J_lin = fp_and_lin(J0, g, h, b, N, p)
    R = np.diag(r/N)
    R3 = R[:3, :3]
    J_CA3 = J_lin[:3, :3]
    D_33 = CA3_prop(J0, g, h, b, N, nterms =nterms, p = p)
    return D_33[0,2]**2 * R3[2,2]

J0 = .2
g_min = 1
g_max = 6
n_g = 10
gs = np.linspace(g_min, g_max, n_g)
b = np.array([.5, .5, .7, .5, .5, .7])
N_E =60
N_I = 15
Ns =np.array([N_E, N_E, N_I,  N_E, N_E, N_I])
p= 1
nterms = 10


internal_before = np.array([CA1_internal_cov_offdiag(J0=J0, g=g, h=1,b=b, N=Ns, p = p)[0,0] for g in gs])
inherited_before = np.array([CA1_inherited_cov(J0=J0, g=g, h=1,b=b, N=Ns, p = p)[0,0] for g in gs])
ca3_before =  np.array([CA3_internal_cov(J0=J0, g=g, h=1,b=b, N=Ns, p =1)[0,0] for g in gs])
total_before = internal_before  + inherited_before

internal_after = np.array([CA1_internal_cov_offdiag(J0=J0, g=g, h=2,b=b, N=Ns, p = p)[0,0] for g in gs])
inherited_after = np.array([CA1_inherited_cov(J0=J0, g=g, h=2,b=b, N=Ns, p = p)[0,0] for g in gs])
#inherited_after_fixed = np.array([CA1_inherited_cov_ca3_fixed(J0=J0, g=g, h=2,b=b, N=Ns, p = p) for g in gs])

ca3_after =  np.array([CA3_internal_cov(J0=J0, g=g, h=2,b=b, N=Ns, p = p)[0,0] for g in gs])

total_after = internal_after + inherited_after


fig, axs = plt.subplots(1,3, sharey=True, sharex=True, figsize = (5,2.5))
axs[0].plot(gs, internal_before, color ="gray", label = "before")
axs[0].plot(gs, internal_after, color ="black", label = "after")
axs[0].legend()
axs[0].set_title("Internally generated")
fig.supxlabel("Inhibitory strength g")
fig.supylabel("Covariance")

axs[1].plot(gs, inherited_before, color ="gray", label = "before")
axs[1].plot(gs, inherited_after, color ="black", label = "after")
#axs[1].plot(gs, inherited_after_fixed, color ="black", linestyle = "--", label = "after, CA3 fixed")
axs[1].set_title("From CA3")
axs[1].legend()

axs[2].plot(gs, inherited_before + internal_before , color ="gray", label = "before")
axs[2].plot(gs, inherited_after + internal_after, color ="black", label = "after")
#axs[1].plot(gs, inherited_after_fixed, color ="black", linestyle = "--", label = "after, CA3 fixed")
axs[2].set_title("Total")
axs[2].legend()
fig.suptitle("CA1, Engram-Engram covariance")
plt.tight_layout()
plt.savefig("../results/CA1_cov_sources.pdf")
plt.show()

# fig, ax = plt.subplots(figsize = (3,3))
# axs[3].plot(gs, ca3_before, color ="gray", label = "before")
# ax.plot(gs, ca3_after, color ="black", label = "after")
# ax.legend()
# fig.suptitle("CA3, Engram-Engram covariance")
# fig.supxlabel("Inhibitory strength g")
# fig.supylabel("Covariance")
# plt.tight_layout()  
# plt.show()


fig, axs = plt.subplots(1, 4, figsize = (7,2.75), sharey = True, sharex = True )

I_before = np.array([CA3_E_from_I(J0 = .2, g = g, h = 1, b=b, N=Ns, p = p) for g in gs])
I_after = np.array([CA3_E_from_I(J0 = .2, g = g, h = 2, b=b, N=Ns, p = p) for g in gs])
I_before_approx = np.array([CA3_E_from_I(J0 = .2, g = g, h = 1, b=b, N=Ns, p = p, nterms = nterms) for g in gs])
I_after_approx = np.array([CA3_E_from_I(J0 = .2, g = g, h = 2, b=b, N=Ns, p = p, nterms = nterms) for g in gs])

E_before= np.array([CA3_E_from_E(J0 = .2, g = g, h = 1, b=b, N=Ns, p = p) for g in gs])
E_after = np.array([CA3_E_from_E(J0 = .2, g = g, h = 2, b=b, N=Ns, p = p) for g in gs])
E_before_approx= np.array([CA3_E_from_E(J0 = .2, g = g, h = 1, b=b, N=Ns, p = p, nterms = nterms) for g in gs])
E_after_approx= np.array([CA3_E_from_E(J0 = .2, g = g, h = 2, b=b, N=Ns, p = p, nterms = nterms) for g in gs])

N_before =  np.array([CA3_E_from_N(J0 = .2, g = g, h = 1, b=b, N=Ns, p = p) for g in gs])
N_after = np.array([CA3_E_from_N(J0 = .2, g = g, h = 2, b=b, N=Ns, p = p) for g in gs])
N_before_approx =  np.array([CA3_E_from_N(J0 = .2, g = g, h = 1, b=b, N=Ns, p = p, nterms = nterms) for g in gs])
N_after_approx = np.array([CA3_E_from_N(J0 = .2, g = g, h = 2, b=b, N=Ns, p = p, nterms = nterms) for g in gs])

tot_before=  np.array([CA3_internal_cov(J0=J0, g=g, h=1,b=b, N=Ns, p = p)[0,0] for g in gs])
tot_after =  np.array([CA3_internal_cov(J0=J0, g=g, h=2,b=b, N=Ns, p = p)[0,0] for g in gs])

axs[0].set_ylabel("covariance")
axs[0].plot(gs,E_before + N_before + I_before, label = "before", color = "gray")
axs[0].plot(gs, E_after + N_after + I_after, label = "after", color = "black")
axs[0].plot(gs,E_before_approx + N_before_approx + I_before_approx, label = "before", color = "gray", linestyle ="--")
axs[0].plot(gs, E_after_approx + N_after_approx + I_after_approx, label = "after", color = "black", linestyle = "--")
axs[0].set_title("Total")


axs[1].plot(gs, E_before, label = "before",color = "gray")
axs[1].plot(gs, E_after, label = "after", color = "black")
axs[1].plot(gs, E_before_approx, label = "before",color = "gray", linestyle = "--")
axs[1].plot(gs, E_after_approx, label = "after", color = "black", linestyle = "--")
axs[1].set_title("Engram")


axs[2].plot(gs, N_before, label = "before", color = "gray")
axs[2].plot(gs, N_after, label = "after", color = "black")
axs[2].plot(gs, N_before_approx, label = "before", color = "gray", linestyle = "--")
axs[2].plot(gs, N_after_approx, label = "after", color = "black", linestyle = "--")
axs[2].set_title("Non-engram")

axs[3].plot(gs, I_before, label = "before", color = "gray")
axs[3].plot(gs, I_after, label = "after", color = "black")
axs[3].plot(gs, I_before_approx, label = "before", color = "gray", linestyle = "--")
axs[3].plot(gs, I_after_approx, label = "after", color = "black", linestyle = "--")
axs[3].set_title("Inhibitory")



fig.supxlabel("Inhibitory strength g")
fig.suptitle("CA3 engram-engram covariance")
plt.tight_layout()
plt.legend()
plt.savefig("../results/CA3_cov_sources.pdf")
plt.show()