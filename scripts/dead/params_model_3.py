# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:04:07 2015

@author: gabeo

define a class to hold parameters
"""

import numpy as np

class params:
    def __init__(self):
        self.NCA3Ex = 80
        self.NCA3In = 20
        self.NCA1Ex = 80
        self.NCA1In = 20
        self.cells_per_region = 2*np.array([int(self.NCA3Ex/2), int(self.NCA3Ex/2), self.NCA3In, int(self.NCA1Ex/2), int(self.NCA1Ex/2), self.NCA1In])

        pEE = 0.1
        pIE = 0.1
        pII = 0.8
        pEI = 0.8

     

        self.macro_connectivity = np.array([
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pIE, pIE, pII, pIE, pIE, pII],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pIE, pIE, pII, pIE, pIE, pII]])
            

        self.g = 6
        self.N = np.sum( self.cells_per_region )
        self.J  = .25
        self.tstop = 4*10**5 # simulation time
        self.b = np.zeros((self.N,))
        self.dt = 0.02

        self.b[:] = .75 # for threshold-linear and threshold-quadratic transfer functions
        self.h_range = [1, 1.25, 1.5, 1.75]
        self.i_plast = 1
