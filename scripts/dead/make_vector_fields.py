import sys
import numpy as np
import params
from src.theory import  y_pred,  J_eff
import matplotlib.pyplot as plt
from src.generate_connectivity import  macro_weights


showfigs = False
if len(sys.argv) > 1:
    if sys.argv[1] == "show":
        showfigs = True

plt.rcParams["text.usetex"] = True
plt.rcParams['pdf.fonttype'] = 42 

par = params.params()
b = par.b
tstop = par.tstop
#dt = .02 * tau  # Euler step
#p = par.p
h3_before = par.h3_before
h3_after = par.h3_after
h1_before = par.h1_before
h1_after = par.h1_after
g = par.g
J = par.J
Ns = par.cells_per_region
y0 = par.b[0]
p_mat = par.macro_connectivity


H_range = np.linspace(1, 2)
y0 = 0.1


# make vector field plots 
G_before = macro_weights(J, h3_before, h1_before, g)
J_eff_before = J_eff(G_before, Ns, p_mat)
G_after = macro_weights(J, h3_after, h1_after, g)
J_eff_after = J_eff(G_after, Ns, p_mat)

x = np.linspace(0, 0.5)
y = np.linspace(0, 0.5)

xv, yv = np.meshgrid(x, y)
xv = xv.flatten()
yv = yv.flatten()
fp_before =  y_pred(G_before, Ns, p_mat, y0)
fp_after =  y_pred(G_after, Ns, p_mat, y0)


def arrow(J, xv, yv, xi, yi, fp, y0):
    N = len(fp)
    z = np.zeros((len(xv), N))
    for i, (x, y) in enumerate(zip(xv, yv)):
        input = fp
        input[xi] = x
        input[yi] = y
        output = (-np.eye(N) + J) @ input + y0
        z[i, :] = output
    return z

fig, axs = plt.subplots(1,2)
z = arrow(J_eff_before, xv, yv, 3, 4, fp_before, y0)
axs[0].quiver(xv, yv, z[:,3], z[:,4])
axs[0].set_aspect('equal', adjustable='box')
axs[0].set_xlabel("tagged rate")
axs[0].set_ylabel("non-tagged rate")


z = arrow(J_eff_after, xv, yv, 3, 4, fp_after, y0)
axs[1].quiver(xv, yv, z[:,3], z[:,4])
axs[1].set_aspect('equal', adjustable='box')
axs[1].set_xlabel("tagged rate")
axs[1].set_ylabel("non-tagged rate")

plt.savefig("results/vector_field_CA1.pdf")
if showfigs:
    plt.show()

fig, axs = plt.subplots(1,2)
z = arrow(J_eff_before, xv, yv, 0, 1, fp_before, y0)
axs[0].quiver(xv, yv, z[:,0], z[:,1])
axs[0].set_aspect('equal', adjustable='box')
axs[0].set_xlabel("tagged rate")
axs[0].set_ylabel("non-tagged rate")


z = arrow(J_eff_after, xv, yv, 0, 1, fp_after, y0)
axs[1].quiver(xv, yv, z[:,0], z[:,1])
axs[1].set_aspect('equal', adjustable='box')
axs[1].set_xlabel("tagged rate")
axs[1].set_ylabel("non-tagged rate")

plt.savefig("results/vector_field_CA3.pdf")
if showfigs:
    plt.show()
