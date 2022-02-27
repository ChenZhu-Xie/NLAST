# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 21:37:19 2021

@author: Xcz
"""

#%%

import math
import numpy as np

#%%

def LN_n(lam, T, p = "e"):
    if p == "e":
        a = [0, 5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32e-2]
        b = [0, 2.860e-6, 4.700e-8, 6.113e-8, 1.516e-4]
    else:
        a = [0, 5.653, 0.1185, 0.2091, 89.61, 10.85, 1.97e-2]
        b = [0, 7.941e-7, 3.134e-8, -4.641e-9, 2.188e-6]
    F = (T - 24.5) * (T + 570.82)
    n = math.sqrt(a[1] + b[1]*F + (a[2] + b[2]*F) / (lam**2 - (a[3] + b[3]*F)**2) + (a[4] + b[4]*F) / (lam**2-a[5]**2) - a[6]*lam**2)
    return n

def KTP_n(lam, T, p = "z"):
    if p == "z" or p == "e":
        a = [0, 3.3134, 0.05694, 0.05657, 0, 0, 0.01682]
        b = [0, -1.1327e-7, 1.673e-7, -1.601e-8, 5.2833e-8]
    elif p == "y" or p == "o":
        a = [0, 3.0333, 0.04154, 0.04547, 0, 0, 0.01408]
        b = [0, -2.7261e-7, 1.7896e-7, 5.3168e-7, -3.4988e-7]
    else:
        a = [0, 3.0065, 0.03901, 0.04251, 0, 0, 0.01327]
        b = [0, -5.3580e-7, 2.8330e-7, 7.5693e-7, -3.9820e-7]
    F = T**2 - 400
    n = math.sqrt(a[1] + b[1]*F + (a[2] + b[2]*F) / (lam**2 - a[3] + b[3]*F) - (a[6] + b[4]*F)*lam**2)
    return n

# lam = 0.8
# T = 25
# print("LN_ne = {}".format(LN_n(lam, T, "e")))
# print("LN_no = {}".format(LN_n(lam, T, "o")))
# print("KTP_nz = {}".format(KTP_n(lam, T, "z")))
# print("KTP_ny = {}".format(KTP_n(lam, T, "y")))
# print("KTP_nx = {}".format(KTP_n(lam, T, "x")))
# print("KTP_ne = {}".format(KTP_n(lam, T, "e")))
# print("KTP_no = {}".format(KTP_n(lam, T, "o")))

#%%
# 计算 折射率、波矢

def Cal_n(size_PerPixel, 
          is_air, 
          lam, T, p = "e"):
    
    if is_air == 1:
        n = 1
    elif is_air == 0:
        n = LN_n(lam, T, p)
    else:
        n = KTP_n(lam, T, p)

    k = 2 * math.pi * size_PerPixel / (lam / 1000 / n) # lam / 1000 即以 mm 为单位
    
    return n, k

# 生成 kz 网格

def Cal_kz(Ix, Iy, k):
    
    nx, ny = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
    Mesh_nx_ny = np.dstack((nx, ny))
    Mesh_nx_ny_shift = Mesh_nx_ny - (Ix // 2, Iy // 2)
    Mesh_kx_ky_shift = np.dstack((2 * math.pi * Mesh_nx_ny_shift[:, :, 0] / Ix, 2 * math.pi * Mesh_nx_ny_shift[:, :, 1] / Iy))
    kz_shift = (k**2 - Mesh_kx_ky_shift[:, :, 0]**2 - Mesh_kx_ky_shift[:, :, 1]**2 + 0j )**0.5
    
    return kz_shift