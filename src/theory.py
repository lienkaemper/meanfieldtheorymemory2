import numpy as np
from collections import namedtuple

def J_eff(J, Ns, p):
    J_eff = J.copy()
   # J_eff *= p
    #J_eff = J_eff @ np.diag(Ns)
    return J_eff 

def y_pred(J, Ns, p, y0):
    N = J.shape[0]
    if len(np.shape(y0)) == 0:
        y = y0 * np.ones(N)
    elif len(y0) == N:
        y = y0
    else:
        raise Exception("input y0 must either be a scalar, or match shape of matrix J")
    y = np.linalg.inv(np.identity(N) - J_eff(J, Ns, p)) @y
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

def c_ij_pred(J, Ns, p, y0):
    y = y_pred(J, Ns, p, y0)
    C_off_diag = np.zeros((len(Ns), len(Ns)))
    C_diag = np.zeros(len(Ns))
    for i in range(len(Ns)):
        for j in range(len(Ns)):
            C_off_diag[i,j] = p*J[i,j] * y[j] + p*J[j, i] * y[i] + sum([y[k] * J[i, k] * J[j, k] * Ns[k]*p**2 for k in range(len(Ns))])
    for i in range(len(Ns)):
        C_diag[i] = C_off_diag[i, i] + y[i]
    C_pair = namedtuple("C_pair", "off_diag diag") 
    result = C_pair(C_off_diag, C_diag)
    return result

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

def C_pred_off_diag(J, Ns, p, y0):
    N = J.shape[0]
    if len(np.shape(y0)) == 0:
        y = y0 * np.ones(N)
    elif len(y0) == N:
        y = y0
    else:
        raise Exception("input y0 must either be a scalar, or match shape of matrix J")
    D = np.linalg.inv(np.identity(N) - J_eff(J, Ns, p))
    y =D @ y 
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