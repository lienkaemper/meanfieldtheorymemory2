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


J = 0.2
b= [.6, .5, 1, .6, .5, 1]
#b = .1 * np.ones(6)
b_stim =[1,  .5, 1, .6, .5, 1]
g = 3
g_ii = 0.5
Ns = np.array([160, 160, 40, 160, 160, 40])
w_31 = 1
h_max = 2
h_i_max = 1.25

n_points = 41
hs = np.linspace(1, h_max, n_points)
h_is = np.linspace(1,h_i_max, n_points)
rates = np.zeros((n_points, n_points))
rates_p  = np.zeros((n_points, n_points))
corrs_ee= np.zeros((n_points, n_points))
corrs_ep  = np.zeros((n_points, n_points))
corrs_pp  = np.zeros((n_points, n_points))
rates_stim = np.zeros((n_points, n_points))
rates_p_stim  = np.zeros((n_points, n_points))
corrs_ee_stim = np.zeros((n_points, n_points))
corrs_ep_stim  = np.zeros((n_points, n_points))
corrs_pp_stim  = np.zeros((n_points, n_points))
for i, h in enumerate(hs):
    for j, h_i in enumerate(h_is):
        G = macro_weights(J, h, h, g, h_i, g_ii = g_ii , w_31 = w_31)
        y = y_pred(G,  b)
        y_stim = y_pred(G, b_stim)
        C = cor_pred(G,Ns, y )
        C_stim = cor_pred(G,Ns, y_stim )
        Cov = C_pred_off_diag(G, Ns,1, y)
        Cov_stim = C_pred_off_diag(G, Ns, 1, y_stim)
        rates[j,i] = y[3]
        rates_p[j,i] = y[4]
        corrs_ee[j,i] = C[3,3]
        corrs_ep[j,i] = C[3,4]
        corrs_pp[j,i] = C[4,4]
        rates_stim[j,i] = y_stim[3]
        rates_p_stim[j,i] = y_stim[4]
        corrs_ee_stim[j,i] = C_stim[3,3]
        corrs_ep_stim[j,i] = C_stim[3,4]
        corrs_pp_stim[j,i] = C_stim[4,4]



matched_h_i_ind = np.argmin(np.abs(rates - rates[0,0] ) , axis = 0)
matched_h_i_l = h_is[matched_h_i_ind]

overall_min = min(np.min(rates), np.min(rates_p), np.min(rates_stim), np.min(rates_p_stim))
overall_max = max(np.max(rates), np.max(rates_p), np.max(rates_stim), np.max(rates_p_stim))

levels_more = np.linspace(overall_min, overall_max, 50)
H, H_I = np.meshgrid(hs, h_is)
fig, axs = plt.subplots(2,2, sharey= True)
axs[0,0].contour(H, H_I, rates, levels = [rates[0,0]])
axs[0,0].plot(hs, matched_h_i_l, color = "red")
axs[0,0].imshow(rates, origin = "lower", extent = (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[0,0].set_aspect(2)

axs[0,1].contour(H, H_I, rates, levels = [rates[0,0]])
axs[0,1].imshow(rates_p, origin = "lower",  extent = (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[0,1].set_aspect(2)

axs[1,0].contour(H, H_I, rates, levels = [rates[0,0]])
axs[1,0].imshow(rates_stim, origin = "lower", extent =  (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[1,0].set_aspect(2)

axs[1,1].contour(H, H_I, rates, levels = [rates[0,0]])
cs = axs[1,1].imshow(rates_p_stim, origin = "lower",  extent = (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[1,1].set_aspect(2)

fig.colorbar(cs, ax=axs.ravel().tolist())

fig.supxlabel("h: engram strength")
fig.supylabel("g_i: inhibitory engram strength ")
plt.savefig("../results/heatmaps/rates_linear_heatmap_g={}J={}g_ii={}w_31={}.pdf".format(g, J, g_ii, w_31))
plt.show()

fig, ax = plt.subplots()
rates_iso = rates[matched_h_i_ind, range(n_points)]
rates_p_iso = rates_p[matched_h_i_ind, range(n_points)]
rates_iso_stim = rates_stim[matched_h_i_ind, range(n_points)]
rates_p_iso_stim = rates_p_stim[matched_h_i_ind, range(n_points)]
ax.plot(hs, rates_iso, label = "Engram, spontaneous")
ax.plot(hs, rates_p_iso, label = "Non-engram, spontaneous")
ax.plot(hs, rates_iso_stim, label = "Engram, stimulus evoked")
ax.plot(hs, rates_p_iso_stim, label = "Non-engram, stimulus evoked")
plt.legend()
plt.savefig("../results/heatmaps/rates_iso_line_g={}J={}g_ii={}w_31={}.pdf".format(g, J, g_ii, w_31))
plt.show()
print(rates_iso)

fig, ax = plt.subplots()
rates_const = rates[0, range(n_points)]
rates_p_const= rates_p[0, range(n_points)]
rates_const_stim = rates_stim[0, range(n_points)]
rates_p_const_stim = rates_p_stim[0, range(n_points)]
ax.plot(hs, rates_const, label = "Engram, spontaneous")
ax.plot(hs, rates_p_const, label = "Non-engram, spontaneous")
ax.plot(hs, rates_const_stim, label = "Engram, stimulus evoked")
ax.plot(hs, rates_p_const_stim, label = "Non-engram, stimulus evoked")
plt.legend()
plt.savefig("../results/heatmaps/rates_const_line_g={}J={}g_ii={}w_31={}.pdf".format(g, J, g_ii, w_31))
plt.show()


overall_min = min(np.min(corrs_ee), np.min(corrs_ep), np.min(corrs_pp), np.min(corrs_ee_stim), np.min(corrs_ep_stim), np.min(corrs_pp_stim))
overall_max = max(np.max(corrs_ee), np.max(corrs_ep), np.max(corrs_pp), np.max(corrs_ee_stim), np.max(corrs_ep_stim), np.max(corrs_pp_stim))

levels_more = np.linspace(overall_min, overall_max, 50)
H, H_I = np.meshgrid(hs, h_is)
fig, axs = plt.subplots(2,3, sharey = True )
axs[0,0].contour(H, H_I, rates, levels = [rates[0,0]])
axs[0,0].imshow(corrs_ee, origin = "lower", extent =  (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[0,0].set_aspect(2)
axs[0,1].contour(H, H_I, rates, levels = [rates[0,0]])
axs[0,1].imshow(corrs_ep, origin = "lower", extent =  (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[0,1].set_aspect(2)
axs[0,2].contour(H, H_I, rates, levels = [rates[0,0]])
axs[0,2].imshow(corrs_pp, origin = "lower", extent = (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[0,2].set_aspect(2)
axs[1,0].contour(H, H_I, rates, levels = [rates[0,0]])
axs[1,0].imshow(corrs_ee_stim, origin = "lower", extent =  (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[1,0].set_aspect(2)
axs[1,1].contour(H, H_I, rates, levels = [rates[0,0]])
axs[1,1].imshow(corrs_ep_stim, origin = "lower", extent =  (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[1,1].set_aspect(2)
axs[1,2].contour(H, H_I, rates, levels = [rates[0,0]])
cs = axs[1,2].imshow(corrs_pp_stim, origin = "lower", extent =  (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[1,2].set_aspect(2)
cbar = fig.colorbar(cs, ax=axs.ravel().tolist())
fig.supxlabel("h: engram strength")
fig.supylabel("g_i: inhibitory engram strength ")
plt.savefig("../results/heatmaps/corrs_linear_heatmap_g={}J={}g_ii={}w_31={}.pdf".format(g, J, g_ii, w_31))
plt.show()

fig, ax = plt.subplots()
corrs_ee_iso = corrs_ee[matched_h_i_ind, range(n_points)]
corrs_ep_iso = corrs_ep[matched_h_i_ind, range(n_points)]
corrs_pp_iso = corrs_pp[matched_h_i_ind, range(n_points)]
corrs_ee_iso_stim = corrs_ee_stim[matched_h_i_ind, range(n_points)]
corrs_ep_iso_stim = corrs_ep_stim[matched_h_i_ind, range(n_points)]
corrs_pp_iso_stim = corrs_pp_stim[matched_h_i_ind, range(n_points)]

ax.plot(hs, corrs_ee_iso, label = "Engram- Engram, spontaneous")
ax.plot(hs, corrs_ep_iso, label = "Engram- Non-engram, spontaneous")
ax.plot(hs, corrs_pp_iso, label = "Non-engram- Non-engram,spontaneous")

ax.plot(hs, corrs_ee_iso_stim, label = "Engram- Engram, stimulus evoked")
ax.plot(hs, corrs_ep_iso_stim, label = "Engram- Non-engram, stimulus evoked")
ax.plot(hs, corrs_pp_iso_stim, label = "Non-engram- Non-engram, stimulus evoked")

plt.legend()
plt.savefig("../results/heatmaps/corrs_linear_line_g={}J={}g_ii={}w_31={}.pdf".format(g, J, g_ii, w_31))

plt.show()

fig, ax = plt.subplots()
corrs_ee_const = corrs_ee[0, range(n_points)]
corrs_ep_const = corrs_ep[0, range(n_points)]
corrs_pp_const = corrs_pp[0, range(n_points)]
corrs_ee_const_stim = corrs_ee_stim[0, range(n_points)]
corrs_ep_const_stim = corrs_ep_stim[0, range(n_points)]
corrs_pp_const_stim = corrs_pp_stim[0, range(n_points)]

ax.plot(hs, corrs_ee_const, label = "Engram- Engram, spontaneous")
ax.plot(hs, corrs_ep_const, label = "Engram- Non-engram, spontaneous")
ax.plot(hs, corrs_pp_const, label = "Non-engram- Non-engram,spontaneous")

ax.plot(hs, corrs_ee_const_stim, label = "Engram- Engram, stimulus evoked")
ax.plot(hs, corrs_ep_const_stim, label = "Engram- Non-engram, stimulus evoked")
ax.plot(hs, corrs_pp_const_stim, label = "Non-engram- Non-engram, stimulus evoked")

plt.legend()
plt.savefig("../results/correlation_engram_baseline_input.pdf")
plt.show()


n_points = 41
hs = np.linspace(1, h_max, n_points)
h_is = np.linspace(1,h_i_max, n_points)
rates = np.zeros((n_points, n_points))
rates_p  = np.zeros((n_points, n_points))
corrs_ee= np.zeros((n_points, n_points))
corrs_ep  = np.zeros((n_points, n_points))
corrs_pp  = np.zeros((n_points, n_points))
rates_stim = np.zeros((n_points, n_points))
rates_p_stim  = np.zeros((n_points, n_points))
corrs_ee_stim = np.zeros((n_points, n_points))
corrs_ep_stim  = np.zeros((n_points, n_points))
corrs_pp_stim  = np.zeros((n_points, n_points))
for i, h in enumerate(hs):
    for j, h_i in enumerate(h_is):
        G = macro_weights(J, h, h, g, h_i, g_ii = g_ii , w_31 = w_31)
        y = y_0_quad(G, b)
        y_stim = y_0_quad(G, b_stim)
        C = cor_pred(G,Ns, y )
        C_stim = cor_pred(G,Ns, y_stim )
        rates[j,i] = y[3]
        rates_p[j,i] = y[4]
        corrs_ee[j,i] = C[3,3]
        corrs_ep[j,i] = C[3,4]
        corrs_pp[j,i] = C[4,4]
        rates_stim[j,i] = y_stim[3]
        rates_p_stim[j,i] = y_stim[4]
        corrs_ee_stim[j,i] = C_stim[3,3]
        corrs_ep_stim[j,i] = C_stim[3,4]
        corrs_pp_stim[j,i] = C_stim[4,4]

matched_h_i_ind = np.argmin(np.abs(rates - rates[0,0] ) , axis = 0)
matched_h_i_l = h_is[matched_h_i_ind]


overall_min = min(np.min(rates), np.min(rates_p), np.min(rates_stim), np.min(rates_p_stim))
overall_max = max(np.max(rates), np.max(rates_p), np.max(rates_stim), np.max(rates_p_stim))

levels_more = np.linspace(overall_min, overall_max, 50)
H, H_I = np.meshgrid(hs, h_is)
fig, axs = plt.subplots(2,2, sharey= True)
axs[0,0].contour(H, H_I, rates, levels = [rates[0,0]])
axs[0,0].plot(hs, matched_h_i_l, color = "red")
axs[0,0].imshow(rates, origin = "lower", extent = (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[0,0].set_aspect(2)

axs[0,1].contour(H, H_I, rates, levels = [rates[0,0]])
axs[0,1].imshow(rates_p, origin = "lower",  extent = (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[0,1].set_aspect(2)

axs[1,0].contour(H, H_I, rates, levels = [rates[0,0]])
axs[1,0].imshow(rates_stim, origin = "lower", extent =  (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[1,0].set_aspect(2)

axs[1,1].contour(H, H_I, rates, levels = [rates[0,0]])
cs = axs[1,1].imshow(rates_p_stim, origin = "lower",  extent = (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[1,1].set_aspect(2)

fig.colorbar(cs, ax=axs.ravel().tolist())

fig.supxlabel("h: engram strength")
fig.supylabel("g_i: inhibitory engram strength ")
plt.savefig("../results/heatmaps/rates_quadratic_heatmap_g={}J={}g_ii={}w_31={}.pdf".format(g, J, g_ii, w_31))
plt.show()

fig, ax = plt.subplots()
rates_iso = rates[matched_h_i_ind, range(n_points)]
rates_p_iso = rates_p[matched_h_i_ind, range(n_points)]
rates_iso_stim = rates_stim[matched_h_i_ind, range(n_points)]
rates_p_iso_stim = rates_p_stim[matched_h_i_ind, range(n_points)]
ax.plot(hs, rates_iso, label = "Engram, spontaneous")
ax.plot(hs, rates_p_iso, label = "Non-engram, spontaneous")
ax.plot(hs, rates_iso_stim, label = "Engram, stimulus evoked")
ax.plot(hs, rates_p_iso_stim, label = "Non-engram, stimulus evoked")
plt.legend()
plt.savefig("../results/heatmaps/rates_iso_line_quadratic_g={}J={}g_ii={}w_31={}.pdf".format(g, J, g_ii, w_31))

plt.show()


overall_min = min(np.min(corrs_ee), np.min(corrs_ep), np.min(corrs_pp), np.min(corrs_ee_stim), np.min(corrs_ep_stim), np.min(corrs_pp_stim))
overall_max = max(np.max(corrs_ee), np.max(corrs_ep), np.max(corrs_pp), np.max(corrs_ee_stim), np.max(corrs_ep_stim), np.max(corrs_pp_stim))

levels_more = np.linspace(overall_min, overall_max, 50)
H, H_I = np.meshgrid(hs, h_is)
fig, axs = plt.subplots(2,3, sharey = True )
axs[0,0].contour(H, H_I, rates, levels = [rates[0,0]])
axs[0,0].imshow(corrs_ee, origin = "lower", extent =  (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[0,0].set_aspect(2)
axs[0,1].contour(H, H_I, rates, levels = [rates[0,0]])
axs[0,1].imshow(corrs_ep, origin = "lower", extent =  (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[0,1].set_aspect(2)
axs[0,2].contour(H, H_I, rates, levels = [rates[0,0]])
axs[0,2].imshow(corrs_pp, origin = "lower", extent = (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[0,2].set_aspect(2)
axs[1,0].contour(H, H_I, rates, levels = [rates[0,0]])
axs[1,0].imshow(corrs_ee_stim, origin = "lower", extent =  (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[1,0].set_aspect(2)
axs[1,1].contour(H, H_I, rates, levels = [rates[0,0]])
axs[1,1].imshow(corrs_ep_stim, origin = "lower", extent =  (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[1,1].set_aspect(2)
axs[1,2].contour(H, H_I, rates, levels = [rates[0,0]])
cs = axs[1,2].imshow(corrs_pp_stim, origin = "lower", extent =  (1,h_max, 1, h_i_max), vmin = overall_min, vmax = overall_max)
axs[1,2].set_aspect(2)
cbar = fig.colorbar(cs, ax=axs.ravel().tolist())
fig.supxlabel("h: engram strength")
fig.supylabel("g_i: inhibitory engram strength ")
plt.savefig("../results/heatmaps/corrs_quadratic_heatmap_g={}J={}g_ii={}w_31={}.pdf".format(g, J, g_ii, w_31))

plt.show()

fig, ax = plt.subplots()
corrs_ee_iso = corrs_ee[matched_h_i_ind, range(n_points)]
corrs_ep_iso = corrs_ep[matched_h_i_ind, range(n_points)]
corrs_pp_iso = corrs_pp[matched_h_i_ind, range(n_points)]
corrs_ee_iso_stim = corrs_ee_stim[matched_h_i_ind, range(n_points)]
corrs_ep_iso_stim = corrs_ep_stim[matched_h_i_ind, range(n_points)]
corrs_pp_iso_stim = corrs_pp_stim[matched_h_i_ind, range(n_points)]

ax.plot(hs, corrs_ee_iso, label = "Engram- Engram, spontaneous")
ax.plot(hs, corrs_ep_iso, label = "Engram- Non-engram, spontaneous")
ax.plot(hs, corrs_pp_iso, label = "Non-engram- Non-engram,spontaneous")

ax.plot(hs, corrs_ee_iso_stim, label = "Engram- Engram, stimulus evoked")
ax.plot(hs, corrs_ep_iso_stim, label = "Engram- Non-engram, stimulus evoked")
ax.plot(hs, corrs_pp_iso_stim, label = "Non-engram- Non-engram, stimulus evoked")

plt.legend()
plt.savefig("../results/heatmaps/corrs_iso_line_quadratic_g={}J={}g_ii={}w_31={}.pdf".format(g, J, g_ii, w_31))

plt.show()