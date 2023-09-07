import numpy as np
from src.theory import  y_pred,  C_pred_off_diag, y_0_quad,  cor_pred
import matplotlib.pyplot as plt


def macro_weights(J, h3, h1, g, h_i =1, g_ii = 1, w_31 = 1):
    return J*np.array([[ h3, 1, -h_i*g, 0, 0, 0], #CA3E
                        [1,  1, -g, 0, 0, 0], #CA3P
                        [1,  1, -g_ii*g, 0, 0, 0],  #CA3I
                        [h1, 1,  0, 0, 0, -h_i*g], #CA1E 
                        [1,  1,  0, 0, 0, -g],  #CAIP
                        [w_31,  w_31,  0, 1, 1, -g_ii*g]]) #CA1I

def find_iso_rate(y, h, J, g, h_i_min, h_i_max,type, n_points = 100):
    h_is = np.linspace(h_i_min, h_i_max, n_points)
    if type == "linear":
        ys =  np.array([y_pred(macro_weights(J, h,h ,g, h_i, g_ii),  b)[3] for h_i in h_is])
    elif type == "quadratic": 
        ys =  np.array([y_0_quad(macro_weights(J, h,h ,g, h_i, g_ii),  b, steps = 100)[3] for h_i in h_is])
    h_i = h_is[np.argmin(np.abs(ys - y))]
    return h_i 

J = 0.2
b= [.6, .5, 1, .6, .5, 1]

Ns = np.array([160, 160, 40, 160, 160, 40])
w_31 = 1
h_min = 1
h_max = 2
h_i_min =1
h_i_max = 4

g_min = 0
g_max = 5
g_ii_min = 0
g_ii_max = 2

n_points = 41
n_points_h_i = 201

gs = np.linspace(g_min, g_max, n_points)
g_iis = np.linspace(g_ii_min, g_ii_max, n_points)

y_before =  np.zeros((n_points, n_points))
y_after =  np.zeros((n_points, n_points))
C_ee_before = np.zeros((n_points, n_points))
C_ep_before = np.zeros((n_points, n_points))
C_pp_before = np.zeros((n_points, n_points))
C_ee_after = np.zeros((n_points, n_points))
C_ep_after = np.zeros((n_points, n_points))
C_pp_after = np.zeros((n_points, n_points))
for i, g in enumerate(gs): 
    for j, g_ii in enumerate(g_iis): 
        G_before = macro_weights(J, h_min,h_min ,g, h_i_min, g_ii)

        #rate before
        y = y_pred(G_before,  b)
        y_before[i,j] = y[3]

        #correlations before
        C = cor_pred(G_before, Ns, y )
        C_ee_before[i,j] = C[3,3]
        C_ep_before[i,j] = C[3,4]
        C_pp_after[i,j]  = C[4,4]

        #find iso-rate curve 
        h_i = find_iso_rate(y[3], h_max, J, g, h_i_min, h_i_max, type = "linear")

        G_after = macro_weights(J, h_max,h_max ,g, h_i, g_ii)

        #rate after
        y = y_pred(G_after,  b)
        y_after[i, j] = y[3]

        #correlations after
        C = cor_pred(G_after, Ns, y )
        C_ee_after[i,j] = C[3,3]
        C_ep_after[i,j] = C[3,4]
        C_pp_after[i,j] = C[4,4]


       
delta_cor = C_ee_after/C_ee_before
ind = np.argmin(delta_cor)
i, j = np.unravel_index(ind, (n_points, n_points))
print(gs[i], g_iis[j])
print(delta_cor[i,j])


# fig, ax = plt.subplots()
# ax.scatter(np.array(y_before), np.array(y_after))
# plt.show()

fig, ax = plt.subplots()
cs = ax.imshow(delta_cor, origin="lower", extent = (g_ii_min, g_ii_max, g_min, g_max))
ax.set_xlabel("g_ii")
ax.set_ylabel("g")
fig.colorbar(cs)
plt.savefig("../results/heatmaps/heatmap_delta_cor_linear.pdf")

plt.show()

n_points_h_i = 51
y_before =  np.zeros((n_points, n_points))
y_after =  np.zeros((n_points, n_points))
C_ee_before = np.zeros((n_points, n_points))
C_ep_before = np.zeros((n_points, n_points))
C_pp_before = np.zeros((n_points, n_points))
C_ee_after = np.zeros((n_points, n_points))
C_ep_after = np.zeros((n_points, n_points))
C_pp_after = np.zeros((n_points, n_points))
for i, g in enumerate(gs): 
    for j, g_ii in enumerate(g_iis): 
        G_before = macro_weights(J, h_min,h_min ,g, h_i_min, g_ii)

        #rate before
        y = y_0_quad(G_before,  b)
        y_before[i,j] = y[3]

        #correlations before
        C = cor_pred(G_before, Ns, y )
        C_ee_before[i,j] = C[3,3]
        C_ep_before[i,j] = C[3,4]
        C_pp_after[i,j]  = C[4,4]

        #find iso-rate curve 
        h_i = find_iso_rate(y[3], h_max, J, g, h_i_min, h_i_max, type = "quadratic")

        G_after = macro_weights(J, h_max,h_max ,g, h_i, g_ii)

        #rate after
        y = y_0_quad(G_after,  b)
        y_after[i, j] = y[3]

        #correlations after
        C = cor_pred(G_after, Ns, y )
        C_ee_after[i,j] = C[3,3]
        C_ep_after[i,j] = C[3,4]
        C_pp_after[i,j] = C[4,4]


       
delta_cor = C_ee_after/C_ee_before
ind = np.argmin(delta_cor)
i, j = np.unravel_index(ind, (n_points, n_points))
print(gs[i], g_iis[j])
print(delta_cor[i, j])

fig, ax = plt.subplots()
ax.scatter(np.array(y_before), np.array(y_after))
plt.show()

fig, ax = plt.subplots()
cs = ax.imshow(delta_cor, origin="lower", extent = (g_ii_min, g_ii_max, g_min, g_max))
ax.set_xlabel("g_ii")
ax.set_ylabel("g")
fig.colorbar(cs)
plt.savefig("../results/heatmaps/heatmap_delta_cor_quadratic.pdf")

plt.show()