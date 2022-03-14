# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

import math
import numpy as np

#%%

def mesh_shift(Ix, Iy):
    
    nx, ny = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
    mesh_nx_ny = np.dstack((nx, ny))
    mesh_nx_ny_shift = mesh_nx_ny - (Ix // 2, Iy // 2)
    
    return mesh_nx_ny_shift

#%%
# 生成 r_shift

def Generate_r_shift(Ix = 0, Iy = 0, size_PerPixel = 0.77, 
                     theta_x = 1, theta_y = 0, ):
    
    mesh_Ix0_Iy0_shift = mesh_shift(Ix, Iy)
    r_shift = ( (mesh_Ix0_Iy0_shift[:, :, 0] * np.cos(theta_x / 180 * math.pi))**2 + (mesh_Ix0_Iy0_shift[:, :, 1] * np.cos(theta_y / 180 * math.pi))**2  + 0j )**0.5 * size_PerPixel
    
    return r_shift


def random_phase(Ix, Iy):
    
    return math.e**( (np.random.rand(Ix, Iy) * 2 * math.pi - math.pi) * 1j )