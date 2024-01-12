import numpy as np
from src.theory import  y_pred,  C_pred_off_diag, y_0_quad,  cor_pred
import matplotlib.pyplot as plt

cmap = 'viridis'
plt.style.use('poster_style.mplstyle')


def macro_weights(J, h3, h1, g, h_i =1, g_ii = 1, w_31 = 1):
    return J*np.array([[ h3, 1, -h_i*g, 0, 0, 0], #CA3E
                        [1,  1, -g, 0, 0, 0], #CA3P
                        [1,  1, -g_ii*g, 0, 0, 0],  #CA3I
                        [h1, 1,  0, 0, 0, -h_i*g], #CA1E 
                        [1,  1,  0, 0, 0, -g],  #CAIP
                        [w_31,  w_31,  0, 1, 1, -g_ii*g]]) #CA1I

def find_iso_rate(y, h, J, g, h_i_min, h_i_max,type, n_points = 200):
    h_is = np.linspace(h_i_min, h_i_max, n_points)
    for h_i in h_is: 
        if type == "linear":
            y_h =  y_pred(macro_weights(J, h,h ,g, h_i, g_ii),  b)[3]
        elif type == "quadratic": 
            y_h = y_0_quad(macro_weights(J, h,h ,g, h_i, g_ii),  b, steps = 50)[3]
        if y_h <= y:
            return h_i
    return h_is[n_points-1]

J = 0.2
b= [.6, .5, .7, .6, .5, .7]

Ns = np.array([160, 160, 40, 160, 160, 40])
w_31 = 1



w_31 = 1
h_min = 1
h_max = 2
h_i_min =1
h_i_max = 6

g_min = 0.5
g_max = 4
g_ii_min = 0
g_ii_max = 2

n_points = 41
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
stability = np.zeros((n_points, n_points))
h_is= np.zeros((n_points, n_points))
for i, g in enumerate(gs): 
    for j, g_ii in enumerate(g_iis): 
        G_before = macro_weights(J, h_min,h_min ,g, h_i_min, g_ii)

        #rate before
        y = y_0_quad(G_before,  b)
        y_before[i,j] = y[3]

    
        G_lin =G_before* (2*(G_before@y+b))[...,None]
        #correlations before
        C = cor_pred(G_lin, Ns, y )
        C_ee_before[i,j] = C[3,3]
        C_ep_before[i,j] = C[3,4]
        C_pp_after[i,j]  = C[4,4]

        #find iso-rate curve 
        h_i = find_iso_rate(y[3], h_max, J, g, h_i_min, h_i_max, type = "quadratic", n_points=200)
        h_is[i,j] = h_i
        G_after = macro_weights(J, h_max,h_max ,g, h_i, g_ii)

        #rate after
        y = y_0_quad(G_after,  b)
        y_after[i, j] = y[3]
        G_lin =G_after* (2*(G_after@y+b))[...,None]
        #correlations after
        C = cor_pred(G_lin, Ns, y )
        C_ee_after[i,j] = C[3,3]
        C_ep_after[i,j] = C[3,4]
        C_pp_after[i,j] = C[4,4]
        stability[i,j] = np.max(np.linalg.eigvals(np.eye(6) - G_lin)) > 0

g = 3
g_ii = 0.25

delta_cor = C_ee_after/C_ee_before
ind = np.argmax(delta_cor)
i, j = np.unravel_index(ind, (n_points, n_points))
print(gs[i], g_iis[j])
print(delta_cor[i, j])

fig, ax = plt.subplots()
ax.scatter(np.array(y_before), np.array(y_after))
plt.show()




fig, ax = plt.subplots(figsize=(7,7))
cs = ax.imshow(delta_cor, origin="lower", extent = (g_ii_min, g_ii_max, g_min, g_max))
ax.set_xlabel("g_ii")
ax.set_ylabel("g")
fig.colorbar(cs)
ax.scatter(x = [g_ii], y = [g], color = "red")
ax.set_aspect(.75)
plt.savefig("../results/heatmaps/heatmap_delta_cor_quadratic.pdf")

plt.show()



J = 0.2
b= [.6, .5, .7, .6, .5, .7]
g = 3
g_ii = 0.25
Ns = np.array([160, 160, 40, 160, 160, 40])
w_31 = 1
h_max = 2
h_i_max = 1.5

n_points = 101
hs = np.linspace(1, h_max, n_points)
h_is = np.linspace(1,h_i_max, n_points)
rates = np.zeros((n_points, n_points))
rates_p  = np.zeros((n_points, n_points))
corrs_ee= np.zeros((n_points, n_points))
corrs_ep  = np.zeros((n_points, n_points))
corrs_pp  = np.zeros((n_points, n_points))
for i, h in enumerate(hs):
    for j, h_i in enumerate(h_is):
        G = macro_weights(J, h, h, g, h_i, g_ii = g_ii , w_31 = w_31)
        y = y_0_quad(G, b)
        C = cor_pred(G,Ns, y )
        rates[j,i] = y[3]
        rates_p[j,i] = y[4]
        corrs_ee[j,i] = C[3,3]
        corrs_ep[j,i] = C[3,4]
        corrs_pp[j,i] = C[4,4]


matched_h_i_ind = np.argmin(np.abs(rates - rates[0,0] ) , axis = 0)
matched_h_i_l = h_is[matched_h_i_ind]

overall_min = np.min(rates)
overall_max = np.max(rates)

levels_more = np.linspace(overall_min, overall_max, 50)
H, H_I = np.meshgrid(hs, h_is)


fig, ax = plt.subplots(figsize=(10,10))
ax.contour(H, H_I, rates, levels = [rates[0,0]])
cs = ax.imshow(rates, origin = "lower", extent = (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max, cmap = cmap)
ax.set_title("engram\nrates")
ax.set_ylabel("h_i")
ax.set_xlabel("h")
fig.colorbar(cs)
plt.savefig("../results/heatmaps/rates_linear_heatmap_g={}J={}g_ii={}w_31={}.pdf".format(g, J, g_ii, w_31))
plt.show()

