# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 00:42:22 2022

@author: Xcz
"""

import os
import numpy as np
import scipy.stats
from scipy.io import savemat
import math
from fun_plot import plot_1d, plot_2d, plot_3d_XYZ, plot_3d_XYz

def pump_LG(file_full_name = "Grating.png", 
            #%%
            Ix = 0, Iy = 0, size_PerPixel = 0.77, 
            U1_0 = 0, w0 = 0, k = 0, z = 0, 
            #%%
            is_LG = 0, is_Gauss = 1, is_OAM = 1, 
            l = 1, p = 0, 
            theta_x = 1, theta_y = 0, 
            is_H_l = 0, is_H_theta = 0, 
            #%%
            is_save = 0, is_save_txt = 0, dpi = 100, 
            #%%
            cmap_2d = 'viridis', 
            #%%
            ticks_num = 6, is_contourf = 0, 
            is_title_on = 1, is_axes_on = 1, 
            is_mm = 1, is_propagation = 0, 
            #%%
            fontsize = 9, 
            font = {'family': 'serif',
                    'style': 'normal', # 'normal', 'italic', 'oblique'
                    'weight': 'normal',
                    'color': 'black', # 'black','gray','darkred'
                    }, 
            #%%
            is_self_colorbar = 0, is_colorbar_on = 1, 
            vmax = 1, vmin = 0):
    
    #%%
    file_name = os.path.splitext(file_full_name)[0]
    file_name_extension = os.path.splitext(file_full_name)[1]
    location = os.path.dirname(os.path.abspath(__file__))
    size_fig = Ix / dpi
    #%%
    # 将输入场 改为 LG 光束

    if is_LG == 1:
        # 将 实空间 输入场 变为 束腰 z = 0 处的 LG 光束
        
        Ix0, Iy0 = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        Mesh_Ix0_Iy0 = np.dstack((Ix0, Iy0))
        Mesh_Ix0_Iy0_shift = Mesh_Ix0_Iy0 - (Ix // 2, Iy // 2)
        r_shift = ( (Mesh_Ix0_Iy0_shift[:, :, 0] * np.cos(theta_x / 180 * math.pi))**2 + (Mesh_Ix0_Iy0_shift[:, :, 1] * np.cos(theta_y / 180 * math.pi))**2  + 0j )**0.5 * size_PerPixel
        
        C_LG_pl = ( 2/math.pi * math.factorial(p)/math.factorial( p + abs(l) ) )**0.5
        x = 2**0.5 * r_shift/w0
        LG_pl = C_LG_pl/w0 * x**abs(l) * scipy.special.genlaguerre(p, abs(l), True)(x**2)
        
        U1_0 = LG_pl

    #%%
    # 对输入场 引入 高斯限制

    if is_Gauss == 1 and is_LG == 0:
        # 将 实空间 输入场 变为 束腰 z = 0 处的 高斯光束
        
        Ix0, Iy0 = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        Mesh_Ix0_Iy0 = np.dstack((Ix0, Iy0))
        Mesh_Ix0_Iy0_shift = Mesh_Ix0_Iy0 - (Ix // 2, Iy // 2)
        r_shift = ( (Mesh_Ix0_Iy0_shift[:, :, 0] * np.cos(theta_x / 180 * math.pi))**2 + (Mesh_Ix0_Iy0_shift[:, :, 1] * np.cos(theta_y / 180 * math.pi))**2  + 0j )**0.5 * size_PerPixel
        
        U1_0 = np.power(math.e, - r_shift**2 / w0**2 )

    else:
        # 对 实空间 输入场 引入 高斯限制
        
        if (type(w0) == float or type(w0) == int) and w0 > 0: # 如果 传进来的 w0 既不是 float 也不是 int，或者 w0 <= 0，则表示 不对原图 引入 高斯限制
        
            Ix0, Iy0 = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
            Mesh_Ix0_Iy0 = np.dstack((Ix0, Iy0))
            Mesh_Ix0_Iy0_shift = Mesh_Ix0_Iy0 - (Ix // 2, Iy // 2)
            r_shift = ( (Mesh_Ix0_Iy0_shift[:, :, 0] * np.cos(theta_x / 180 * math.pi))**2 + (Mesh_Ix0_Iy0_shift[:, :, 1] * np.cos(theta_y / 180 * math.pi))**2  + 0j )**0.5 * size_PerPixel
            
            U1_0 = U1_0 * np.power(math.e, - r_shift**2 / w0**2 )

    #%%
    # 对输入场 引入 额外的 螺旋相位
        
    if is_OAM == 1 and is_Gauss == 0:
        # 高斯则 乘以 额外螺旋相位，非高斯 才直接 更改原场：高斯 已经 更改原场 了
        # 将输入场 在实空间 改为 纯相位 的 OAM
        
        Ix0, Iy0 = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        Mesh_Ix0_Iy0 = np.dstack((Ix0, Iy0))
        Mesh_Ix0_Iy0_shift = Mesh_Ix0_Iy0 - (Ix // 2, Iy // 2)
        U1_0 = np.power(math.e, l * np.arctan2(Mesh_Ix0_Iy0_shift[:, :, 0] * np.cos(theta_x / 180 * math.pi), Mesh_Ix0_Iy0_shift[:, :, 1] * np.cos(theta_y / 180 * math.pi)) * 1j)
        
    else:
        # 对输入场 引入 额外的 螺旋相位
        
        if is_H_l == 1:
            # 对 频谱空间 引入额外螺旋相位
            
            G = np.fft.fft2(U1_0)
            G_shift = np.fft.fftshift(G)
            
            n1_x0, n1_y0 = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
            Mesh_n1_x0_n1_y0 = np.dstack((n1_x0, n1_y0))
            Mesh_n1_x0_n1_y0_shift = Mesh_n1_x0_n1_y0 - (Ix // 2, Iy // 2)
            H_shift = np.power(math.e, l * np.arctan2(Mesh_n1_x0_n1_y0_shift[:, :, 0] * np.cos(theta_x / 180 * math.pi), Mesh_n1_x0_n1_y0_shift[:, :, 1] * np.cos(theta_y / 180 * math.pi)) * 1j)
            
            G_shift = G_shift * H_shift
            G = np.fft.ifftshift(G_shift)
            
            U1_0 = np.fft.ifft2(G)
        else:
            # 对 实空间 引入额外螺旋相位
            
            Ix0, Iy0 = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
            Mesh_Ix0_Iy0 = np.dstack((Ix0, Iy0))
            Mesh_Ix0_Iy0_shift = Mesh_Ix0_Iy0 - (Ix // 2, Iy // 2)
            U1_0 = U1_0 * np.power(math.e, l * np.arctan2(Mesh_Ix0_Iy0_shift[:, :, 0] * np.cos(theta_x / 180 * math.pi), Mesh_Ix0_Iy0_shift[:, :, 1] * np.cos(theta_y / 180 * math.pi)) * 1j)
            # θx 增大时，y = x 这个 45 度 的 线，会 越来越 偏向 x 轴 正向。

    #%%
    # 对输入场 引入 额外的 倾斜相位
    
    Kx, Ky = k * np.sin(theta_x / 180 * math.pi), k * np.sin(theta_y / 180 * math.pi) 

    if is_H_theta == 1:
        # 对 频谱空间 引入额外倾斜相位
        
        G = np.fft.fft2(U1_0)
        G_shift = np.fft.fftshift(G)

        n1_x0, n1_y0 = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        Mesh_n1_x0_n1_y0 = np.dstack((n1_x0, n1_y0))
        Mesh_n1_x0_n1_y0_shift = Mesh_n1_x0_n1_y0 - (Ix // 2, Iy // 2)
        
        # k_shift = (k**2 - Kx**2 - Ky**2 + 0j )**0.5
        H_shift = np.power(math.e, ( Kx * Mesh_n1_x0_n1_y0_shift[:, :, 0] + Ky * Mesh_n1_x0_n1_y0_shift[:, :, 1] ) * 1j)
        # 本该 H_shift 的 e 指数 的 相位部分，还要 加上 k_shift * i1_z0 的，不过这里 i1_z0 = i1_0 = 0，所以加了 等于没加
        
        G_shift = G_shift * H_shift
        G = np.fft.ifftshift(G_shift)
        
        U1_0 = np.fft.ifft2(G)

    else:
        # 对 实空间 引入额外倾斜相位
        
        Ix0, Iy0 = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        Mesh_Ix0_Iy0 = np.dstack((Ix0, Iy0))
        Mesh_Ix0_Iy0_shift = Mesh_Ix0_Iy0 - (Ix // 2, Iy // 2)
        
        # k_shift = (k**2 - Kx**2 - Ky**2 + 0j )**0.5
        U1_0 = U1_0 * np.power(math.e, ( Kx * Mesh_Ix0_Iy0_shift[:, :, 0] + Ky * Mesh_Ix0_Iy0_shift[:, :, 1] ) * 1j)
        # Mesh_Ix0_Iy0_shift[:, :, 0] 只与 第 2 个参数 有关，
        # 则 对于 第 1 个参数 而言，对于 不同的 第 1 个参数，都 引入了 相同的 倾斜相位。
        # 也就是 对于 同一列 的 不同的行，其 倾斜相位 是相同的
        # 因此 倾斜相位 也 只与 列 相关，也就是 只与 第 2 个参数 有关，所以 与 x 有关。
        
    #%%
    # 对输入场 引入 传播相位
    
    g = np.fft.fft2(U1_0)
    g_shift = np.fft.fftshift(g)
    
    z0 = z
    i_z0 = z0 / size_PerPixel

    nx, ny = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
    Mesh_nx_ny = np.dstack((nx, ny))
    Mesh_nx_ny_shift = Mesh_nx_ny - (Ix // 2, Iy // 2)
    Mesh_kx_ky_shift = np.dstack((2 * math.pi * Mesh_nx_ny_shift[:, :, 0] / Ix, 2 * math.pi * Mesh_nx_ny_shift[:, :, 1] / Iy))

    kz_shift = (k**2 - np.square(Mesh_kx_ky_shift[:, :, 0]) - np.square(Mesh_kx_ky_shift[:, :, 1]) + 0j )**0.5
    H_z0_shift = np.power(math.e, kz_shift * i_z0 * 1j)
    
    G_z0_shift = g_shift * H_z0_shift
    G_z0 = np.fft.ifftshift(G_z0_shift)
    U1_z0 = np.fft.ifft2(G_z0)
    
    #%%
    U1_0 = U1_z0

    U1_0_amp = np.abs(U1_0)
    # print("U1_0_amp.total_amp = {}".format(np.sum(U1_0)))
    # print("U1_0_amp.total_energy = {}".format(np.sum(U1_0**2)))
    U1_0_phase = np.angle(U1_0)

    print("AST - U1_0.total_energy = {}".format(np.sum(np.power(U1_0_amp, 2))))

    if is_save == 1:
        if not os.path.isdir("2. U1_0"):
            os.makedirs("2. U1_0")

    #%%
    #绘图：U1_0_amp

    U1_0_amp_address = location + "\\" + "2. U1_0" + "\\" + "2.1. AST - " + "U1_0_amp" + file_name_extension

    plot_2d(Ix, Iy, size_PerPixel, 0, 
            U1_0_amp, U1_0_amp_address, "U1_0_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            1, is_colorbar_on, vmax, vmin)

    #%%
    #绘图：U1_0_phase

    U1_0_phase_address = location + "\\" + "2. U1_0" + "\\" + "2.2. AST - " + "U1_0_phase" + file_name_extension

    plot_2d(Ix, Iy, size_PerPixel, 0, 
            U1_0_phase, U1_0_phase_address, "U1_0_phase", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, vmax, vmin)
    
    #%%
    # 储存 U1_0 到 txt 文件

    if is_save == 1:
        U1_0_full_name = "2. AST - U1_0" + (is_save_txt and ".txt" or ".mat")
        U1_0_txt_address = location + "\\" + "2. U1_0" + "\\" + U1_0_full_name
        np.savetxt(U1_0_txt_address, U1_0) if is_save_txt else savemat(U1_0_txt_address, {'U1_0':U1_0})
        
        #%%
        #再次绘图：U1_0_amp
    
        U1_0_amp_address = location + "\\" + "2." + "U1_0_amp" + file_name_extension
    
        plot_2d(Ix, Iy, size_PerPixel, 0, 
                U1_0_amp, U1_0_amp_address, "U1_0_amp", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, vmax, vmin)
    
        #再次绘图：U1_0_phase
    
        U1_0_phase_address = location + "\\" + "2." + "U1_0_phase" + file_name_extension
    
        plot_2d(Ix, Iy, size_PerPixel, 0, 
                U1_0_phase, U1_0_phase_address, "U1_0_phase", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, vmax, vmin)
    
    return U1_0