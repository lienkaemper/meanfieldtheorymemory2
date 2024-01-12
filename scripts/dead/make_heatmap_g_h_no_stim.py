import numpy as np
from src.theory import  y_pred,  C_pred_off_diag, y_0_quad,  cor_pred
import matplotlib.pyplot as plt

cmap = 'magma'
plt.style.use('poster_style.mplstyle')


def macro_weights(J, h3, h1, g, h_i =1, g_ii = 1, w_31 = 1):
    return J*np.array([[ h3, 1, -h_i*g, 0, 0, 0], #CA3E
                        [1,  1, -g, 0, 0, 0], #CA3P
                        [1,  1, -g_ii*g, 0, 0, 0],  #CA3I
                        [h1, 1,  0, 0, 0, -h_i*g], #CA1E 
                        [1,  1,  0, 0, 0, -g],  #CAIP
                        [w_31,  w_31,  0, 1, 1, -g_ii*g]]) #CA1I



J = 0.2
b= [.6, .5, 1, .6, .5, 1]
g = 3
g_ii = 0.5
Ns = np.array([160, 160, 40, 160, 160, 40])
w_31 = 1
h_max = 2
h_i_max = 1.25

n_points = 201
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
        y = y_pred(G,  b)
        C = cor_pred(G,Ns, y )
        Cov = C_pred_off_diag(G, Ns,1, y)
        rates[j,i] = y[3]
        rates_p[j,i] = y[4]
        corrs_ee[j,i] = C[3,3]
        corrs_ep[j,i] = C[3,4]
        corrs_pp[j,i] = C[4,4]



matched_h_i_ind = np.argmin(np.abs(rates - rates[0,0] ) , axis = 0)
matched_h_i_l = h_is[matched_h_i_ind]

overall_min = min(np.min(rates), np.min(rates_p))
overall_max = max(np.max(rates), np.max(rates_p))

levels_more = np.linspace(overall_min, overall_max, 50)
H, H_I = np.meshgrid(hs, h_is)


fig, axs = plt.subplots(1,2, sharey= True, figsize=(20,10))
axs[0].contour(H, H_I, rates, levels = [rates[0,0]])
cs = axs[0].imshow(rates, origin = "lower", extent = (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max, cmap = cmap)
axs[0].set_aspect(2)
axs[0].set_title("engram\nrates")

axs[1].contour(H, H_I, rates, levels = [rates[0,0]])
axs[1].imshow(rates_p, origin = "lower",  extent = (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max, cmap = cmap)
axs[1].set_aspect(2)
axs[1].set_title("non-engram\nrates")
fig.colorbar(cs, ax=axs.ravel().tolist())

fig.supxlabel("h: engram strength")
fig.supylabel("g_i: inhibitory engram strength ")
plt.savefig("../results/heatmaps/rates_linear_heatmap_g={}J={}g_ii={}w_31={}.pdf".format(g, J, g_ii, w_31))
plt.show()
 


overall_min = min(np.min(corrs_ee), np.min(corrs_ep), np.min(corrs_pp))
overall_max = max(np.max(corrs_ee), np.max(corrs_ep), np.max(corrs_pp))

levels_more = np.linspace(overall_min, overall_max, 50)
H, H_I = np.meshgrid(hs, h_is)
fig, axs = plt.subplots(1,3, sharey = True, figsize=(50,10) )
axs[0].contour(H, H_I, rates, levels = [rates[0,0]])
cs = axs[0].imshow(corrs_ee, origin = "lower", extent =  (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max, cmap = cmap)
axs[0].set_aspect(2)
axs[0].set_title("engram/\nengram\n correlation")
axs[1].contour(H, H_I, rates, levels = [rates[0,0]])
axs[1].imshow(corrs_ep, origin = "lower", extent =  (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max, cmap = cmap)
axs[1].set_aspect(2)
axs[1].set_title("engram/\nnon-engram\n correlation")
axs[2].contour(H, H_I, rates, levels = [rates[0,0]])
axs[2].imshow(corrs_pp, origin = "lower", extent = (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max, cmap = cmap)
axs[2].set_aspect(2)
axs[2].set_title("non-engram\n/non-engram\n correlation")
cbar = fig.colorbar(cs, ax=axs.ravel().tolist())
fig.supxlabel("h: engram strength")
fig.supylabel("g_i: inhibitory engram strength ")
plt.savefig("../results/heatmaps/corrs_linear_heatmap_g={}J={}g_ii={}w_31={}.pdf".format(g, J, g_ii, w_31))
plt.show()

fig, ax = plt.subplots( figsize=(20,10))
corrs_ee_iso = corrs_ee[matched_h_i_ind, range(n_points)]
corrs_ep_iso = corrs_ep[matched_h_i_ind, range(n_points)]
corrs_pp_iso = corrs_pp[matched_h_i_ind, range(n_points)]


ax.plot(hs, corrs_ee_iso, label = "engram/engram")
ax.plot(hs, corrs_ep_iso, label = "engram/non-engram")
ax.plot(hs, corrs_pp_iso, label = "non-engram/non-engram")


plt.legend()
plt.savefig("../results/heatmaps/corrs_linear_line_g={}J={}g_ii={}w_31={}.pdf".format(g, J, g_ii, w_31))

plt.show()



