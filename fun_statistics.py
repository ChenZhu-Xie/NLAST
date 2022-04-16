# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 00:42:22 2022

@author: Xcz
"""

import numpy as np
from fun_linear import Cal_kz

def U_Drop_n_sigma(U, n, is_energy):
    
    if is_energy == 0:
        U_amp = np.abs(U)
    else:
        U_amp = np.abs(U)**2
    # U_phase = np.angle(U)
    U_amp_mean = np.mean(U_amp)
    U_amp_std = np.std(U_amp)
    U_amp_trust = np.abs(U_amp - U_amp_mean) <= n*U_amp_std
    U = U * U_amp_trust.astype(np.int8)
    
    return U

def find_Kz(g, k):
    k_z, mesh_k_x_k_y = Cal_kz(g.shape[0], g.shape[1], k)
    g_energy = np.sum(np.abs(g)**2)
    k_z_weight = np.abs(g)**2 / g_energy
    K_z = np.sum(k_z_weight * k_z)
    return K_z