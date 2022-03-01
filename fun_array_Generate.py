# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

import numpy as np

#%%

def mesh_shift(Ix, Iy):
    
    nx, ny = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
    mesh_nx_ny = np.dstack((nx, ny))
    mesh_nx_ny_shift = mesh_nx_ny - (Ix // 2, Iy // 2)
    
    return mesh_nx_ny_shift