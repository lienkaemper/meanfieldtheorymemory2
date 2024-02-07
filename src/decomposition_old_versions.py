import numpy as np
from collections import namedtuple
from scipy.optimize import fsolve
import itertools 
import matplotlib.pyplot as plt

from src.generate_connectivity import macro_weights
from src.theory import y_0_quad
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