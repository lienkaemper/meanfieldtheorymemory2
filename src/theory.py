import numpy as np
from collections import namedtuple
from scipy.optimize import fsolve
import itertools 
import matplotlib.pyplot as plt

from src.generate_connectivity import macro_weights

def J_eff(J, Ns, p):
    J_eff = J.copy()
   # J_eff *= p
    #J_eff = J_eff @ np.diag(Ns)
    return J_eff 

def y_pred(J, y0):
    N = J.shape[0]
    if len(np.shape(y0)) == 0:
        y = y0 * np.ones(N)
    elif len(y0) == N:
        y = y0
    else:
        raise Exception("input y0 must either be a scalar, or match shape of matrix J")
    y = np.linalg.inv(np.identity(N) -J) @y
    y = np.maximum(y, 0)
    return y 


def y_pred_from_full_connectivity(W, y0, index_dict):
    N = W.shape[0]
    B = np.linalg.inv(np.identity(N) - W)
    if len(np.shape(y0)) == 0:
        y = y0 * np.ones(N)
    elif len(y0) == N:
        y = y0
    else:
        raise Exception("input y0 must either be a scalar, or match shape of matrix J")
    y = B @ y
    y = np.maximum(y, 0)
    return y

def y_pred_full(W, y0):
    N = W.shape[0]
    B = np.linalg.inv(np.identity(N) - W)
    if len(np.shape(y0)) == 0:
        y = y0 * np.ones(N)
    elif len(y0) == N:
        y = y0
    else:
        raise Exception("input y0 must either be a scalar, or match shape of matrix J")
    y = B @ y
    y = np.maximum(y, 0)
    return y

def y_0_quad(W, y0, steps = 1000,  dt = 0.1):
    N = W.shape[0]
    #y = y_pred_from_full_connectivity(W, y0, 0)
    v = np.random.rand(N)
    for i in range(steps):
        v  = v + dt*(-v +  W @ np.maximum(0,v )**2+y0)
    y = v**2
    return y



def y_corrected_quad(W,  y_0, b):
    N = W.shape[0]
  
    W_lin = W * (2*(W@y_0+b))[...,None]
    E, V = np.linalg.eig(W_lin)
    Vinv = np.linalg.inv(V)
    WV = W_lin @ V
    D = np.linalg.inv(np.eye(N) - W_lin)
    EE = np.zeros((N,N))
    for m,l in itertools.product(range(N), range(N)):
        EE[m, l] = 1/(2 - E[m] - E[l])
    M = np.einsum("k, ij, jl, lk, jm, mk, lm -> i", y_0, D, WV, Vinv, WV, Vinv, EE)
    return y_0 + (1/(2*np.pi))*M

def loop_correction(W,  y_0, b):
    N = W.shape[0]
  
    W_lin = W * (2*(W@y_0+b))[...,None]
    E, V = np.linalg.eig(W_lin)
    Vinv = np.linalg.inv(V)
    WV = W_lin @ V
    D = np.linalg.inv(np.eye(N) - W_lin)
    EE = np.zeros((N,N))
    for m,l in itertools.product(range(N), range(N)):
        EE[m, l] = 1/(2 - E[m] - E[l])
    M = np.einsum("k, ij, jl, lk, jm, mk, lm -> i", y_0, D, WV, Vinv, WV, Vinv, EE)
    return (1/(2*np.pi))*M

def c_ij_pred(J, Ns, p, y):
    #y = y_pred(J, Ns, p, y0)
    C_off_diag = np.zeros((len(Ns), len(Ns)))
    C_diag = np.zeros(len(Ns))
    C_off_diag = C_pred_off_diag(J, Ns, p, y)
    for i in range(len(Ns)):
        C_diag[i] = C_off_diag[i, i] + y[i]

    C_pair = namedtuple("C_pair", "off_diag diag") 
    result = C_pair(C_off_diag, C_diag)
    return result

def cor_pred(J, Ns, y0):
    off_diag, diag  = c_ij_pred(J, Ns, 1, y0)
    return (1/np.sqrt(diag)) *off_diag * (1/(np.sqrt(diag)))[...,None]

def length_1_full(W, y0, index_dict):
    N = W.shape[0]
    B = np.linalg.inv(np.identity(N) - W)
    if len(np.shape(y0)) == 0:
        y = y0 * np.ones(N)
    elif len(y0) == N:
        y = y0
    else:
        raise Exception("input y0 must either be a scalar, or match shape of matrix J")
    y = B @ y
    Y = np.diag(y)
    C =  Y + W @ Y + Y @ W.T + W @ Y @ W.T
    return C

def length_1_offdiag_cov(W, Ns, y):
    N = W.shape[0]
    y_norm = y/N
    Y = np.diag(y_norm)
    Cov =   W @ Y + Y @ W.T + W @ Y @ W.T
    return Cov

def length_1_cor(W, Ns, y):
    Cov = length_1_offdiag_cov(W, Ns, y)
    var = np.diag(Cov) + y
    return (1/np.sqrt(var)) * Cov * (1/(np.sqrt(var)))[...,None]


def overall_cor_pred(J, Ns, p, y0):
    C = np.zeros((len(Ns), len(Ns)))
    C_off_diag, C_diag = c_ij_pred(J, Ns, p, y0)
    for i in range(len(Ns)):
        for j in range(len(Ns)):
            if i != j:
                C[i,j] = C_off_diag[i,j]*Ns[i] * Ns[j]
            else:
                C[i,j] = C_off_diag[i,j] * Ns[i] * (Ns[j] -1) + Ns[i] * C_diag[i]
    return C

def cor_from_full_connectivity(W, y0, index_dict):
    N = W.shape[0]
    if len(np.shape(y0)) == 0:
        y = y0 * np.ones(N)
    elif len(y0) == N:
        y = y0
    else:
        raise Exception("input y0 must either be a scalar, or match shape of matrix W")
    n = W.shape[0]
    B = np.linalg.inv(np.identity(n) - W)
    y = B @ y
    C = B @ np.diag(y) @ B.T
    return C

def covariance_full(W, y):
    n = W.shape[0]
    B = np.linalg.inv(np.identity(n) - W)
    C = B @ np.diag(y) @ B.T
    return C

#this is the version we actually use 
def C_pred_off_diag(J, Ns, p, y0):
    N = J.shape[0]
    if len(np.shape(y0)) == 0:
        y = y0 * np.ones(N)
    elif len(y0) == N:
        y = y0
    else:
        raise Exception("input y0 must either be a scalar, or match shape of matrix J")
    D = np.linalg.inv(np.identity(N) - J_eff(J, Ns, p))
   # y =D @ y 
    Y = np.diag(y)
    return D @ Y @ (D @ np.diag(1/Ns)).T - np.diag(1/Ns) * Y

def rates_with_noisy_tagging(r_E, r_P, p_E, p_P, p_FP, p_FN):
    p_TP = 1- p_FN #true positive: probability tagged given engram
    p_TN = 1 - p_FP #true negativeL probability not tagged given not engram 
    p_EgT = p_E*p_TP/(p_E*p_TP + p_P*p_FP) #probability engram given tagged 
    p_EgNT = p_E*p_FP/(p_E*p_FP + p_P*p_TN) #probability engram givne non-tagged 
    p_PgT = 1 -  p_EgT  #probability non engram given tagged 
    p_PgNT =  1 -  p_EgNT #probability non engram given non-tagged 

    r_tagged =  p_EgT*r_E +  p_PgT *r_P 
    r_non_tagged = p_EgNT*r_E + p_PgNT *r_P
    return r_tagged, r_non_tagged

def cor_with_noisy_tagging(c_EE, c_PE, c_PP, p_E, p_P, p_FP,  p_FN):
    p_TP = 1- p_FN #true positive: probability tagged given engram
    p_TN = 1 - p_FP #true negativeL probability not tagged given not engram 
    p_EgT = p_E*p_TP/(p_E*p_TP + p_P*p_FP) #probability engram given tagged 
    p_EgNT = p_E*p_FP/(p_E*p_FP + p_P*p_TN) #probability engram givne non-tagged 
    p_PgT = 1 -  p_EgT  #probability non engram given tagged 
    p_PgNT =  1 -  p_EgNT #probability non engran given non-tagged 

    c_TT = (p_EgT**2)*c_EE + (2*p_EgT*p_PgT) *c_PE  + (p_PgT**2)*c_PP
    c_NT = (p_EgNT*p_EgT)*c_EE + (p_PgNT*p_EgT + p_EgNT*p_PgT)*c_PE + (p_PgT*p_PgNT)*c_PP
    c_NN = (p_EgNT**2)*c_EE + (2*p_EgNT*p_PgNT) *c_PE  + (p_PgNT**2)*c_PP
    return c_TT, c_NT, c_NN

def find_iso_rate(y, h, J, g, g_ii, b, h_i_min, h_i_max,type, n_points = 200):
    h_is = np.linspace(h_i_min, h_i_max, n_points)
    y_hs = np.zeros(n_points)
    for i, h_i in enumerate(h_is): 
        if type == "linear":
            y_h =  y_pred(macro_weights(J, h,h ,g, h_i, g_ii),  b)[3]
            y_hs[i] = y_h
        elif type == "quadratic": 
            y_h = y_0_quad(macro_weights(J, h,h ,g, h_i, g_ii),  b, steps = 500)[3]
            y_hs[i] = y_h
        if y_h <= y:
            return h_i

    return h_is[n_points-1]


def find_iso_rate_ca3(yca1, yca3, h, J0, g, g_ii, b, h_i_min, h_i_max,type, n_points = 200):
    h_is = np.linspace(h_i_min, h_i_max, n_points)
    y_hs = np.zeros(n_points)
    #first, match ca3
    for i, h_i3 in enumerate(h_is): 
        J = macro_weights(J=J0, h1 = h,h3 = h ,g = g, h_i= 1, g_ii = g_ii, h_i_ca3= h_i3)
        if type == "linear":
            y_h =  y_pred( J,  b)[0]
            y_hs[i] = y_h
        elif type == "quadratic": 
            y_h = y_0_quad(J,  b, steps = 500)[0]
            y_hs[i] = y_h
        if y_h <= yca3:
            for i, h_i1 in enumerate(h_is): 
                J = macro_weights(J=J0, h1 = h,h3 = h ,g = g, h_i= h_i1, g_ii = g_ii, h_i_ca3= h_i3)
                if type == "linear":
                    y_h =  y_pred( J,  b)[3]
                    y_hs[i] = y_h
                elif type == "quadratic": 
                    y_h = y_0_quad(J,  b, steps = 500)[3]
                    y_hs[i] = y_h
                if y_h <= yca1:
                    return(h_i1, h_i3)
            return(h_is[n_points-1, h_i3])
    return (h_is[n_points-1], h_is[n_points-1])


def find_iso_rate_input(target_rate, J, b, b0_min = .1, b0_max = 5, p=2, n_points = 200, plot = True):
    b0s =  np.linspace(b0_min, b0_max, n_points)
    y_hs = np.zeros(n_points)
    for i, b0 in enumerate(b0s): 
        b_new = np.copy(b)
        b_new[2] += b0/2
        b_new[5] += b0/2
        b_new[0:2] += b0
        b_new[3:5] += b0 
        if p == 1:
            y_h =  y_pred(J,  b_new)[3]
            y_hs[i] = y_h
        elif p== 2: 
            y =  y_0_quad(J,  b_new, steps = 500)
            correction = np.real(loop_correction(J,  y, b_new))
            y_corrected = y + correction 
            y_h = y_corrected[3]
            y_hs[i] = y_h
        if y_h >= target_rate:
            if plot:
                plt.plot(y_hs[:i], color = "black")
                plt.hlines(y = [target_rate], xmin = 0, xmax = i, color = "red")
                plt.show()
            print(f"from loop {i}")
            return b_new
    
    print("at return")
    if plot:  
        plt.plot(y_hs, color = "black")
        plt.hlines(y = [target_rate], xmin = 0, xmax = n_points, color = "red")
        plt.show()


    b_new = np.copy(b)
    b_new[0:2] +=  b0s[n_points-1]
    b_new[3:5] +=  b0s[n_points-1]

    return b_new


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