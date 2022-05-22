# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

import math
import numpy as np

#%%

def mesh_shift(Ix, Iy, *args):
    
    nx, ny = np.meshgrid(range(Iy), range(Ix))
    mesh_nx_ny = np.dstack((nx, ny))
    mesh_nx_ny_shift = mesh_nx_ny - (Iy // 2, Ix // 2)
    
    # print(len(arg))
    if len(args) >= 2: # 确保 额外 传入了 theta_x, theta_y 两个参数
    
        theta_x = args[0]
        theta_y = args[1]
        
        # print(mesh_nx_ny_shift[:, :, 0])
        # print(type(mesh_nx_ny_shift[0,0,0]))
        
        # mesh_nx_ny_shift[:, :, 0] = mesh_nx_ny_shift[:, :, 0] * np.cos(theta_x / 180 * math.pi) # python 像 go 一样，切片 只是引用，无法改变 被引用的 数组的 相应切片 的值？
        # mesh_nx_ny_shift[:, :, 1] = mesh_nx_ny_shift[:, :, 1] * np.cos(theta_y / 180 * math.pi) # 并不，确实是 改变了的，赋值是 深拷贝；引用是 浅拷贝
    
        # print(mesh_nx_ny_shift[:, :, 0])
        # print(type(mesh_nx_ny_shift[0,0,0])) # 破案了，tmd 是 int 的锅：右边 非 int 赋值给 左边 的 时候，左边的 数据类型 仍是 int，导致 右边的 计算结果 会强制转换为 int。
        # 额，不对，type(mesh_nx_ny_shift[0,0,0]) 总是 int，不是 数据类型 强制转换 成右侧的 int 的 缘故，
        # 而是 切片 并未引用 原来的，而是创建了 一块新的 内存区域，这样 并没有 改变 所引用的 原来的 区域的 数据
        # 而且 这里是 对数组 array 的 切片，不是 对 list 的；对 list 的 切片还是 list，然而 list 相当于 labview 的 簇，不能乘以 非整数，
        # 但 list 乘以 整数 也不是 每个元素 都乘以 整数，而是 将其 复制 几份后 并加入原来的 List
        
        mesh_nx_ny_shift = np.dstack((mesh_nx_ny_shift[:, :, 0] * np.cos(theta_x / 180 * math.pi), mesh_nx_ny_shift[:, :, 1] * np.cos(theta_y / 180 * math.pi)))
    
    return mesh_nx_ny_shift

#%%
# 生成 r_shift

def Generate_r_shift(Ix = 0, Iy = 0, size_PerPixel = 0.77, 
                     theta_x = 1, theta_y = 0, ):

    # print(Ix, Iy)
    mesh_nx_ny_shift = mesh_shift(Ix, Iy, 
                                  theta_x, theta_y)
    r_shift = ( mesh_nx_ny_shift[:, :, 0]**2 + mesh_nx_ny_shift[:, :, 1]**2  + 0j )**0.5 * size_PerPixel
    
    return r_shift

def random_phase(Ix, Iy):
    
    return math.e**( (np.random.rand(Ix, Iy) * 2 * math.pi - math.pi) * 1j )