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
