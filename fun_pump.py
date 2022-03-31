# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 00:42:22 2022

@author: Xcz
"""

import os
import math
import numpy as np
import scipy.stats
from scipy.io import savemat
from fun_os import U_dir, U_energy_print, U_plot, U_save
from fun_plot import plot_2d
from fun_array_Generate import mesh_shift, Generate_r_shift, random_phase
from fun_linear import Cal_kz

#%%
# 生成 束腰 z = 0 处的 HG 光束（但不包括 复振幅的 高斯分布）

def HG_without_Gauss_profile(Ix = 0, Iy = 0, size_PerPixel = 0.77, 
                             w0 = 0, 
                             m = 1, n = 0, 
                             theta_x = 1, theta_y = 0, ):
    
    mesh_Ix0_Iy0_shift = mesh_shift(Ix, Iy, 
                                    theta_x, theta_y)
    
    C_HG_mn = 1
    x, y = 2**0.5 * mesh_Ix0_Iy0_shift[:,:,0] * size_PerPixel/w0, 2**0.5 * mesh_Ix0_Iy0_shift[:,:,1] * size_PerPixel/w0
    M, N = [0]*m, [0]*n
    M.append(m), N.append(n)
    HG_mn = C_HG_mn * np.polynomial.hermite.hermval(x,M) * np.polynomial.hermite.hermval(y,N)
    
    return HG_mn

#%%
# 生成 束腰 z = 0 处的 LG 光束（但不包括 复振幅的 高斯分布）

def LG_without_Gauss_profile(Ix = 0, Iy = 0, size_PerPixel = 0.77, 
                             w0 = 0, 
                             l = 1, p = 0, 
                             theta_x = 1, theta_y = 0, ):
    
    r_shift = Generate_r_shift(Ix, Iy, size_PerPixel, 
                               theta_x, theta_y, )
    
    C_LG_pl = ( 2/math.pi * math.factorial(p)/math.factorial( p + abs(l) ) )**0.5
    x = 2**0.5 * r_shift/w0
    LG_pl = C_LG_pl/w0 * x**abs(l) * scipy.special.genlaguerre(p, abs(l), True)(x**2)
    
    return LG_pl    

#%%
# 生成 束腰 z = 0 处的 高斯光束

def Gauss(Ix = 0, Iy = 0, size_PerPixel = 0.77, 
          w0 = 0, 
          theta_x = 1, theta_y = 0, ):
    
    r_shift = Generate_r_shift(Ix, Iy, size_PerPixel, 
                               theta_x, theta_y, )
    
    if (type(w0) == float or type(w0) == np.float64 or type(w0) == int) and w0 > 0: # 如果 传进来的 w0 既不是 float 也不是 int，或者 w0 <= 0，则 图片为 1
        U = np.power(math.e, - r_shift**2 / w0**2 )
    else:
        U = np.ones((Ix,Iy),dtype=np.complex128)
        
    return U

#%%
# 对 输入场 引入 高斯限制

def Gauss_profile(Ix = 0, Iy = 0, size_PerPixel = 0.77, 
                  U = 0, w0 = 0, 
                  theta_x = 1, theta_y = 0, ):
    
    if (type(w0) == float or type(w0) == np.float64 or type(w0) == int) and w0 > 0: # 如果 传进来的 w0 既不是 float 也不是 int，或者 w0 <= 0，则表示 不对原图 引入 高斯限制
    
        r_shift = Generate_r_shift(Ix, Iy, size_PerPixel, 
                                   theta_x, theta_y, )
        
        U = U * np.power(math.e, - r_shift**2 / w0**2 )
        
    return U

#%%
# 生成 纯相位 OAM

def OAM(Ix = 0, Iy = 0, 
        l = 1, 
        theta_x = 1, theta_y = 0, ):
    
    mesh_Ix0_Iy0_shift = mesh_shift(Ix, Iy, 
                                    theta_x, theta_y)
    U = np.power(math.e, l * np.arctan2(mesh_Ix0_Iy0_shift[:, :, 0], mesh_Ix0_Iy0_shift[:, :, 1]) * 1j)
    
    return U

#%%
# 对 输入场 引入额外螺旋相位

def OAM_profile(Ix = 0, Iy = 0, 
                U = 0, 
                l = 1, 
                theta_x = 1, theta_y = 0, ):
    
    mesh_Ix0_Iy0_shift = mesh_shift(Ix, Iy, 
                                    theta_x, theta_y)
    U = U * np.power(math.e, l * np.arctan2(mesh_Ix0_Iy0_shift[:, :, 0], mesh_Ix0_Iy0_shift[:, :, 1]) * 1j)
    
    return U

# θx 增大时，y = x 这个 45 度 的 线，会 越来越 偏向 x 轴 正向。

#%%
# 对 输入场 的 频谱 引入额外螺旋相位

def OAM_profile_G(Ix = 0, Iy = 0, 
                  U = 0, 
                  l = 1, 
                  theta_x = 1, theta_y = 0, ):
    
    g = np.fft.fft2(U)
    g_shift = np.fft.fftshift(g)
    
    g_shift = OAM_profile(Ix, Iy, 
                          g_shift, 
                          l, 
                          theta_x, theta_y, )
    
    g = np.fft.ifftshift(g_shift)
    
    U = np.fft.ifft2(g)
    
    return U, g_shift

#%%
# 对 输入场 引入 额外的 倾斜相位

def incline_profile(Ix = 0, Iy = 0, 
                    U = 0, k = 0, 
                    theta_x = 1, theta_y = 0, ):
    
    Kx, Ky = k * np.sin(theta_x / 180 * math.pi), k * np.sin(theta_y / 180 * math.pi)
    
    mesh_Ix0_Iy0_shift = mesh_shift(Ix, Iy)
    # k_shift = (k**2 - Kx**2 - Ky**2 + 0j )**0.5
    U = U * np.power(math.e, ( Kx * mesh_Ix0_Iy0_shift[:, :, 0] + Ky * mesh_Ix0_Iy0_shift[:, :, 1] ) * 1j)
    
    return U
    
# mesh_Ix0_Iy0_shift[:, :, 0] 只与 第 2 个参数 有关，
# 则 对于 第 1 个参数 而言，对于 不同的 第 1 个参数，都 引入了 相同的 倾斜相位。
# 也就是 对于 同一列 的 不同的行，其 倾斜相位 是相同的
# 因此 倾斜相位 也 只与 列 相关，也就是 只与 第 2 个参数 有关，所以 与 x 有关。

#%%
# 对输入场 频域 引入 额外倾斜相位

def incline_profile_G(Ix = 0, Iy = 0, 
                      U = 0, k = 0, 
                      theta_x = 1, theta_y = 0, ):
    
    g = np.fft.fft2(U)
    g_shift = np.fft.fftshift(g)
    
    g_shift = incline_profile(Ix, Iy, 
                              g_shift, k, 
                              theta_x, theta_y)
    # 本该 H_shift 的 e 指数 的 相位部分，还要 加上 k_shift * i1_z0 的，不过这里 i1_z0 = i1_0 = 0，所以加了 等于没加
    g = np.fft.ifftshift(g_shift)
    
    U = np.fft.ifft2(g)
    
    return U, g_shift

#%%
# 对输入场 频域 引入 传播相位

def propagation_profile_G(Ix = 0, Iy = 0, size_PerPixel = 0.77, 
                          U = 0, k = 0, z = 0, ):
    
    g = np.fft.fft2(U)
    g_shift = np.fft.fftshift(g)
    
    kz_shift, mesh_kx_ky_shift = Cal_kz(Ix, Iy, k)
    i_z0 = z / size_PerPixel
    H_z0_shift = np.power(math.e, kz_shift * i_z0 * 1j)
    
    G_z0_shift = g_shift * H_z0_shift
    G_z0 = np.fft.ifftshift(G_z0_shift)
    U_z0 = np.fft.ifft2(G_z0)
    
    return U_z0, G_z0_shift

#%%
# 对输入场 空域 引入 传播相位

def propagation_profile_U(Ix = 0, Iy = 0, size_PerPixel = 0.77, 
                          U = 0, k = 0, z = 0, ):
    
    kz_shift, mesh_kx_ky_shift = Cal_kz(Ix, Iy, k)
    i_z0 = z / size_PerPixel
    H_z0_shift = np.power(math.e, kz_shift * i_z0 * 1j)
    
    U_z0 = U * H_z0_shift
    G_z0 = np.fft.ifft2(U_z0)
    G_z0_shift = np.fft.ifftshift(G_z0)
    
    return U_z0, G_z0_shift

#%%

def pump_LG(file_full_name = "Grating.png", 
            #%%
            Ix = 0, Iy = 0, size_PerPixel = 0.77, 
            U1_0 = 0, w0 = 0, k = 0, z = 0, 
            #%%
            is_LG = 0, is_Gauss = 1, is_OAM = 1, 
            l = 1, p = 0, 
            theta_x = 1, theta_y = 0, 
            is_random_phase = 0, 
            is_H_l = 0, is_H_theta = 0, is_H_random_phase = 0, 
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
            is_energy = 0, vmax = 1, vmin = 0, 
            is_print = 1, ):
    
    #%%
    
    # file_name = os.path.splitext(file_full_name)[0]
    img_name_extension = os.path.splitext(file_full_name)[1]
    size_fig = Ix / dpi
    #%%
    # 将输入场 改为 LG 光束

    if is_LG == 1:
        # 将 实空间 输入场 变为 束腰 z = 0 处的 LG 光束
        
        U1_0 = LG_without_Gauss_profile(Ix, Iy, size_PerPixel, 
                                        w0, 
                                        l, p, 
                                        theta_x, theta_y, )
    elif is_LG == 2:
        # 将 实空间 输入场 变为 束腰 z = 0 处的 HG 光束
        
        U1_0 = HG_without_Gauss_profile(Ix, Iy, size_PerPixel, 
                                        w0, 
                                        l, p, 
                                        theta_x, theta_y, )

    #%%
    # 对输入场 引入 高斯限制

    if is_Gauss == 1 and is_LG == 0:
        # 将 实空间 输入场 变为 束腰 z = 0 处的 高斯光束
            
        U1_0 = Gauss(Ix, Iy, size_PerPixel, 
                     w0, 
                     theta_x, theta_y, )

    else:
        # 对 实空间 输入场 引入 高斯限制
        
        U1_0 = Gauss_profile(Ix, Iy, size_PerPixel, 
                             U1_0, w0, 
                             theta_x, theta_y, )

    #%%
    # 对输入场 引入 额外的 螺旋相位
        
    if is_OAM == 1 and is_Gauss == 0:
        # 高斯则 乘以 额外螺旋相位，非高斯 才直接 更改原场：高斯 已经 更改原场 了
        # 将输入场 在实空间 改为 纯相位 的 OAM
        
        U1_0 = OAM(Ix, Iy, 
                   l, 
                   theta_x, theta_y, )
        
    elif is_LG != 2: # 只有 非厄米高斯时，l ≠ 0 时 才加 螺旋相位
        # 对输入场 引入 额外的 螺旋相位
        
        if is_H_l == 1:
            # 对 频谱空间 引入额外螺旋相位
            
            U1_0, G_z0_shift = OAM_profile_G(Ix, Iy, 
                                             U1_0, 
                                             l, 
                                             theta_x, theta_y, )
            
        else:
            # 对 实空间 引入额外螺旋相位
            
            U1_0 = OAM_profile(Ix, Iy, 
                               U1_0, 
                               l, 
                               theta_x, theta_y, )

    #%%
    # 对输入场 引入 额外的 倾斜相位

    if is_H_theta == 1:
        # 对 频谱空间 引入额外倾斜相位
        
        U1_0, G_z0_shift = incline_profile_G(Ix, Iy, 
                                             U1_0, k, 
                                             theta_x, theta_y, )

    else:
        # 对 实空间 引入额外倾斜相位
        
        U1_0 = incline_profile(Ix, Iy, 
                               U1_0, k, 
                               theta_x, theta_y)
        
    # U1_0 = U1_0**2
    #%%
    # 对输入场 引入 传播相位
    
    if is_H_l == 1:
        
        # 对输入场 的 频谱 引入 额外的 螺旋相位（纯相位）， 并在 频域传播 一定距离（纯相位），之后 返回空域（其实就是 在空域传播 / 乘以 传递函数）
        # 由于是两次 纯相位操作，不会 改变 频域 或 空域 的 总能量
        # 其他 总能量守恒，但改变 频谱能量分布 的 操作，如 希尔伯特变换，可能也行（不一定 加 螺旋相位 后，再 频域传播）
        # 或者 先后进行多次 不同的 能量守恒 操作 也行
        
        U_z0, G_z0_shift = propagation_profile_U(Ix, Iy, size_PerPixel, 
                                                 U1_0, k, z, )
        
    else:
        
        U_z0, G_z0_shift = propagation_profile_G(Ix, Iy, size_PerPixel, 
                                                 U1_0, k, z, )
    
    #%%
    # 对输入场 引入 随机相位
    
    if is_random_phase == 1:
    
        if is_H_random_phase == 1:
            
            # G_z0_shift *= random_phase(Ix, Iy)
            G_z0_shift = G_z0_shift * random_phase(Ix, Iy)
            G_z0 = np.fft.ifftshift(G_z0_shift)
            U_z0 = np.fft.ifft2(G_z0)
            
        else:
            
            # U_z0 *= random_phase(Ix, Iy)
            U_z0 = U_z0 * random_phase(Ix, Iy)
            G_z0 = np.fft.ifft2(U_z0)
            G_z0_shift = np.fft.ifftshift(G_z0)
            
    #%%
    #绘图：G1_0_amp
    
    folder_address = ''
    
    if is_save == 1:
        folder_address = U_dir("", "2. G1_0", 0, z, )
        
    U_amp_plot_address, U_phase_plot_address = U_plot("", folder_address, 1, 
                                                      G_z0_shift, "G1_0_shift", "AST", 
                                                      img_name_extension, 
                                                      #%%
                                                      [], 1, size_PerPixel,
                                                      is_save, dpi, size_fig,
                                                      cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                                                      fontsize, font,
                                                      1, is_colorbar_on, is_energy, vmax, vmin, 
                                                      #%%
                                                      z, )
    
    if is_save == 1:
        U_address = U_save("", folder_address, 1, 
                           G_z0_shift, "G1_0_shift", "AST", 
                           is_save_txt, z, )
    
    #%%
    
    if is_save == 1:
        folder_address = U_dir("", "2. U1_0", 0, z, )

    U_energy_print("", is_print, 1, 
                   U_z0, "U1_0", "AST", )
    
    U_amp_plot_address, U_phase_plot_address = U_plot("", folder_address, 1, 
                                                      U_z0, "U1_0", "AST", 
                                                      img_name_extension, 
                                                      #%%
                                                      [], 1, size_PerPixel,
                                                      is_save, dpi, size_fig,
                                                      cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                                                      fontsize, font,
                                                      1, is_colorbar_on, is_energy, vmax, vmin, 
                                                      #%%
                                                      z, )
    
    if is_save == 1:
        U_address = U_save("", folder_address, 1, 
                           U_z0, "U1_0", "AST", 
                           is_save_txt, z, )
    
    return U_z0, G_z0_shift

