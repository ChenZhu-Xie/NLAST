# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 00:42:22 2022

@author: Xcz
"""

import numpy as np
import math
from fun_linear import Cal_kz

def U_Drop_n_sigma(U, n, is_energy):
    
    U_amp = np.abs(U) if is_energy != 1 else np.abs(U) ** 2
    # U_phase = np.angle(U)
    U_amp_mean = np.mean(U_amp)
    U_amp_std = np.std(U_amp)
    U_amp_trust = np.abs(U_amp - U_amp_mean) <= n*U_amp_std
    U = U * U_amp_trust.astype(np.int8)
    
    return U

def find_Kxyz(g, k):
    k_z, mesh_k_x_k_y = Cal_kz(g.shape[0], g.shape[1], k)
    g_energy = np.sum(np.abs(g)**2)
    k_xyz_weight = np.abs(g)**2 / g_energy
    K_z = np.sum(k_xyz_weight * k_z)  #  g 点阵 的 坐标系 与 k_z 的 相同么 ？
    K_x, K_y = np.sum(k_xyz_weight * mesh_k_x_k_y[:,:,0]), np.sum(k_xyz_weight * mesh_k_x_k_y[:,:,1])
    return K_z, (K_x, K_y)

def find_data_1d_level(data_1d, level_percentage):
    data_covered_num = math.ceil(len(data_1d) * level_percentage) # 向上取整 以覆盖比 level_percentage 范围 更大的 数据
    real_level_percentage = data_covered_num / len(data_1d)
    # print(real_level_percentage)
    index = data_covered_num - 1
    level = sorted(data_1d)[index]
    return level, real_level_percentage