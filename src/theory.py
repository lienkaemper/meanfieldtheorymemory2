import numpy as np
from collections import namedtuple
from scipy.optimize import fsolve
import itertools 
import matplotlib.pyplot as plt

from src.generate_connectivity import macro_weights


def y_pred(J, y0):
    ''' Given a connectivity matrix J and input y0, returns the tree-level rate in a linear model, 
        i.e. returns the rate y that solves y = J y + y0
    '''
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



def y_0_quad(W, y0, steps = 1000,  dt = 0.1):
    ''' Given a connectivity matrix W and input y0, returns the tree-level rate in a quadratic model, 
        i.e. returns the rate y that solves y = (W y + y0)**2
    '''
    N = W.shape[0]
    if len(np.shape(y0)) == 0:
        y = y0 * np.ones(N)
    elif len(y0) == N:
        y = y0
    else:
        raise Exception("input y0 must either be a scalar, or match shape of matrix J")
    v = np.random.rand(N)
    for i in range(steps):
        v  = v + dt*(-v +  W @ np.maximum(0,v )**2+y0)
    y = v**2
    return y


def loop_correction(W,  y_0, b):
    ''' Given a connectivity matrix W, tree level rate y_0 (outout of y_0_quad), and input b, 
    returns the one loop correction to the firing rates
    '''
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

def y_corrected_quad(W,  y_0, b):
    ''' Given a connectivity matrix W, tree level rate y_0 (outout of y_0_quad), and input b, 
    returns the estimate of the firing rates using the one-loop correction
    '''
    return y_0 + loop_correction(W, y_0,b)


def cor_pred(J, Ns, y):
    '''Computes the population-level correlation, i.e. the mean correlation, i.e. C[i,j] gives the mean 
    correlation between a neuron in population i and a distinct neuron in population j

    J: 6x6 regional connectivity matrix. 
    Ns: number of neurons in each region
    y: population mean rates  

     '''
    off_diag, diag  = c_ij_pred(J, Ns, y)
    C = (1/np.sqrt(diag)) *off_diag * (1/(np.sqrt(diag)))[...,None]
    return C

def c_ij_pred(J, Ns, y):
    '''Helper function for cor_pred. Computes the mean-field off-diagonal and diagonal covariance
    Inputs:  
    J: 6x6 regional connectivity matrix 
    Ns: number of neurons in each region
    y: population mean rates      
    '''
    C_off_diag = np.zeros((len(Ns), len(Ns)))
    C_diag = np.zeros(len(Ns))
    C_off_diag = C_pred_off_diag(J, Ns, y)
    for i in range(len(Ns)):
        C_diag[i] = C_off_diag[i, i] + y[i]
    C_pair = namedtuple("C_pair", "off_diag diag") 
    result = C_pair(C_off_diag, C_diag)
    return result

def C_pred_off_diag(J, Ns, y0):
    '''Helper function for cor_pred. Computes the mean-field off-diagonal covariance
    Inputs:  
    J: 6x6 regional connectivity matrix 
    Ns: number of neurons in each region
    y: population mean rates      
    '''
    N = J.shape[0]
    if len(np.shape(y0)) == 0:
        y = y0 * np.ones(N)
    elif len(y0) == N:
        y = y0
    else:
        raise Exception("input y0 must either be a scalar, or match shape of matrix J")
    D = np.linalg.inv(np.identity(N) - J)
   # y =D @ y 
    Y = np.diag(y)
    return D @ Y @ (D @ np.diag(1/Ns)).T - np.diag(1/Ns) * Y



def covariance_full(J, y):
    '''Neuron level (N x N) covariance. 
    J: Neuron level connectivity matrix
    y: Neuron level rates
    '''
    n = J.shape[0]
    B = np.linalg.inv(np.identity(n) - J)
    C = B @ np.diag(y) @ B.T
    return C



def find_iso_rate_ca3(yca1, yca3, h, J0, g, g_ii, b, h_i_min, h_i_max,type, n_points = 200):
    '''Computes the level of inhibitory plasticitiy needed to keep engram rates in CA1 and CA3 constant
    Inputs: 
    yca1: target rate in CA1 engram cells
    yca3: target rate in CA3 engram cells
    h: engram strength 
    J0: overall connectivity strenght 
    g: inhibition strength 
    g_ii: inhibition onto the inhibition
    h_i_min: minimum value of inhibitory plasticity
    h_i_max: maxium value of inhibitiory plasticity
    type: "linear" or "quadratic"
    n_points: number of grid points to use for search

    returns (h_i1, h_i3) : inhibitory plasticity onto CA1 Engram, CA3 engram. Engram inhibition is multiplied, i.e. 
    inhibition onto the engram cells is -g * h_i1 in CA1, -g * h_i3 in CA3 
    '''
    h_is = np.linspace(h_i_min, h_i_max, n_points)
    y_hs = np.zeros(n_points)
    #first, match ca3
    for i, h_i3 in enumerate(h_is): 
        J = macro_weights(J=J0, h1 = h,h3 = h ,g = g, h_i= 1, g_ii = g_ii, h_i_ca3= h_i3)
        if type == "linear":
            y_h =  y_pred( J,  b)[0]
            y_hs[i] = y_h
        elif type == "quadratic": 
            y_h = y_0_quad(J,  b)
            correction = np.real(loop_correction(J,  y_h, b))
            y_h += np.real(correction )
            y_hs[i] = y_h[0]
        if y_h[0] <= yca3:
            for i, h_i1 in enumerate(h_is): 
                J = macro_weights(J=J0, h1 = h,h3 = h ,g = g, h_i= h_i1, g_ii = g_ii, h_i_ca3= h_i3)
                if type == "linear":
                    y_h =  y_pred( J,  b)[3]
                    y_hs[i] = y_h
                elif type == "quadratic": 
                    y_h = y_0_quad(J,  b)
                    correction = np.real(loop_correction(J,  y_h, b))
                    y_h += np.real(correction )
                    y_hs[i] = y_h[3]
                if y_h[3] <= yca1:
                    return(h_i1, h_i3)
            return(h_is[n_points-1, h_i3])
    return (h_is[n_points-1], h_is[n_points-1])


def find_iso_rate_input_old(target_rate, J, b, b0_min = .1, b0_max = 5, p=2, n_points = 200, plot = False):
    '''Computes the level of increased input to excitatory neurons needed to bring excitatory firing rates 
    to a target rate
    target_rate: target rate of CA1 engram cells
    J: 6 by 6 connectivity matrix 
    b: basline input 
    b_0_min: minimum extra input for excitatory neurons
    b_0_max: maximum extra input for excitatory neurons
    p: degree (1 or 2)
    n_points: number of grid points to use for search
    plot: boolean, for whether to plot curve of rate vs. target rate (for debugging)

    returns b_new: input so that the engram rates match the target_rate 
    '''
    b0s =  np.linspace(b0_min, b0_max, n_points)
    y_hs = np.zeros(n_points)
    for i, b0 in enumerate(b0s): 
        b_new = np.copy(b)
        b_new[2] += 0
        b_new[5] += 0
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


def find_iso_rate_input(target_rate_1, target_rate_3, J, b, b0_min = .1, b0_max = 5, p=2, n_points = 200, plot = False):
    '''Computes the level of increased input to excitatory neurons needed to bring excitatory firing rates 
    to a target rate
    target_rate: target rate of CA1 engram cells
    J: 6 by 6 connectivity matrix 
    b: basline input 
    b_0_min: minimum extra input for excitatory neurons
    b_0_max: maximum extra input for excitatory neurons
    p: degree (1 or 2)
    n_points: number of grid points to use for search
    plot: boolean, for whether to plot curve of rate vs. target rate (for debugging)

    returns b_new: input so that the engram rates match the target_rate 
    '''
    b0s =  np.linspace(b0_min, b0_max, n_points)
    y_hs = np.zeros(n_points)
    for i, b0 in enumerate(b0s):  #outer loop: CA3
        b_new = np.copy(b)
        b_new[0:2] += b0
        #b_new[3:5] += b0 
        if p == 1:
            y_h =  y_pred(J,  b_new)[0]
            y_hs[i] = y_h
        elif p== 2: 
            y =  y_0_quad(J,  b_new, steps = 500)
            correction = np.real(loop_correction(J,  y, b_new))
            y_corrected = y + correction 
            y_h = y_corrected[0]
            y_hs[i] = y_h
        if y_h >= target_rate_3:
            for i, b0 in enumerate(b0s):  #outer loop: CA3
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
                if y_h >= target_rate_1:
                    print(f"from loop {i}")
                    return b_new
            return b_new
    print("at return")
    b_new = np.copy(b)
    b_new[0:2] +=  b0s[n_points-1]
    b_new[3:5] +=  b0s[n_points-1]

    return b_new

   


def CA3_prop(J, r, b, nterms = None):
    '''Propagator for CA3 only. In quadratic model
    Input: J: connectivity matrix (not already linearized)
           r: rate
           b: input
           '''
    gain =  2*(J@r+b)
    J_lin =J* gain[...,None]
    J_CA3 = J_lin[:3, :3]
    if nterms != None:
        Delta = np.identity(3)
        for n in range(1,nterms+1):
            Delta += np.linalg.matrix_power(J_CA3, n)      
    else:
        Delta = np.linalg.inv(np.identity(3) - J_CA3)
    return Delta


def CA1_prop(J, r, b, nterms = None):
    '''Propagator for CA1 only. In quadratic model 
    Input: J: connectivity matrix (not already linearized)
           r: rate
           b: input
           '''
    gain =  2*(J@r+b)
    J_lin =J* gain[...,None]
    J_CA1 = J_lin[3:, 3:]
    if nterms != None:
        Delta = np.identity(3)
        for n in range(1,nterms+1):
            Delta += np.linalg.matrix_power(J_CA1, n)
    else:
        Delta = np.linalg.inv(np.identity(3) - J_CA1)
    return Delta


def CA1_internal_cov(J, r, b, N, nterms = None):
    '''Internally generated population covariance in CA1. 
    Includes contributions from diagonal entries, ie. each neuron's variance driven by its rate. 
     In quadratic model 
    Input: J: connectivity matrix (not already linearized)
           r: rate
           b: input
           N: number of neurons in each region
           '''
    R = np.diag(r/N)
    R1 = R[3:, 3:]
    D_11 = CA1_prop(J, r, b, nterms =nterms)
    return D_11 @ R1 @ D_11.T

def CA1_internal_cov_offdiag(J, r, b, N, nterms = None):
    '''Internally generated population covariance in CA1. 
    Does not include contributions from diagonal entries, ie. each neuron's variance driven by its rate. 
     In quadratic model 
    Input: J: connectivity matrix (not already linearized)
           r: rate
           b: input
           N: number of neurons in each region
           '''
    R = np.diag(r/N)
    R1 = R[3:, 3:]
    D_11 = CA1_prop(J, r, b, nterms =nterms)
    return D_11 @ R1 @ D_11.T - R1

def CA3_internal_cov(J, r, b, N, nterms = None):
    '''Internally generated population covariance in CA3. 
    Includes contributions from diagonal entries, ie. each neuron's variance driven by its rate. 
     In quadratic model 
    Input: J: connectivity matrix (not already linearized)
           r: rate
           b: input
           N: number of neurons in each region
    '''
    R = np.diag(r/N)
    R3 = R[:3, :3]
    D_33 = CA3_prop(J, r, b, nterms =nterms)
    return (D_33 @ R3 @ D_33.T)


def CA1_inherited_cov(J, r, b, N, nterms = None):
    '''Covariance in CA1 inherited from CA3.
     In quadratic model 
    Input: J: connectivity matrix (not already linearized)
           r: rate
           b: input
           N: number of neurons in each region
    '''
    C_33 = CA3_internal_cov(J, r, b, N, nterms = nterms)
    gain =  2*(J@r+b)
    J_lin =J* gain[...,None]
    J_13 = J_lin[3:, :3]
    D_11 = CA1_prop(J, r, b,  nterms = nterms)
    return (D_11 @ J_13 @ C_33 @ J_13.T @ D_11.T)



def CA3_E_from_E(J, r, b, N, nterms = None):
    '''Covariance in CA3 engram cells originating from CA3 engram cells. 
    Input: J: connectivity matrix (not already linearized)
           r: rate
           b: input
           N: number of neurons in each region
    '''
    R = np.diag(r/N)
    R3 = R[:3, :3]
    D_33 = CA3_prop(J, r, b,  nterms =nterms)
    return D_33[0,0]**2 * R3[0,0]


def CA3_E_from_N(J, r, b, N, nterms = None):
    '''Covariance in CA3 engram cells originating from CA3 non-engram cells. 
    Input: J: connectivity matrix (not already linearized)
           r: rate
           b: input
           N: number of neurons in each region
    '''
    R = np.diag(r/N)
    R3 = R[:3, :3]
    D_33 = CA3_prop(J, r, b,  nterms =nterms)
    return D_33[0,1]**2 * R3[1,1]

def CA3_E_from_I(J, r, b, N, nterms = None):
    '''Covariance in CA3 engram cells originating from CA3 inhibitory cells. 
    Input: J: connectivity matrix (not already linearized)
           r: rate
           b: input
           N: number of neurons in each region
    '''
    R = np.diag(r/N)
    R3 = R[:3, :3]
    D_33 = CA3_prop(J, r, b,  nterms =nterms)
    return D_33[0,2]**2 * R3[2,2]