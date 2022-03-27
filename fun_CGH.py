# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

#%%

import cv2
import math
import numpy as np
from scipy.io import loadmat
from fun_os import img_squared_bordered_Read
from fun_img_Resize import img_squared_Resize
from fun_plot import plot_2d
from fun_pump import pump_LG
from fun_SSI import Cal_IxIy
from fun_linear import Cal_n, Cal_kz
from fun_nonlinear import Cal_lc_SHG, Cal_GxGyGz

#%%

def Step_U(U, mode, 
           Duty_Cycle_x, Duty_Cycle_y, 
           is_positive_xy):
    
    if mode == 'x':
        return ( U > (2 * is_positive_xy - 1) * np.cos(Duty_Cycle_x * math.pi) ).astype(np.int8()) # uint8 会导致 之后 structure 和 modulation 也变成 无符号 整形，以致于 在 0 - 1 时 变成 255 而不是 -1...
    elif mode == 'y':
        return ( U > (2 * is_positive_xy - 1) * np.cos(Duty_Cycle_y * math.pi) ).astype(np.int8()) # uint8 会导致 之后 structure 和 modulation 也变成 无符号 整形，以致于 在 0 - 1 时 变成 255 而不是 -1...

#%%

def CGH(U, mode, 
        Duty_Cycle_x, Duty_Cycle_y, 
        is_positive_xy, 
        #%%
        Gx, Gy, 
        is_Gauss, l, 
        is_continuous, ):
    
    i1_x0, i1_y0 = np.meshgrid([i for i in range(U.shape[0])], [j for j in range(U.shape[1])])
    i1_x0_shift, i1_y0_shift = i1_x0 - U.shape[0] // 2, i1_y0 - U.shape[1] // 2
    if is_Gauss == 1 and l == 0:
        if mode == 'x*y' or mode == 'x+y':
            cgh = np.cos(Gx * i1_x0_shift)
            cgh_x = Step_U(cgh, 'x', 
                           Duty_Cycle_x, Duty_Cycle_y, 
                           is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
            cgh = np.cos(Gy * i1_y0_shift)
            cgh_y = Step_U(cgh, 'y', 
                           Duty_Cycle_x, Duty_Cycle_y, 
                           is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
            if mode == 'x*y':
                cgh = cgh_x * cgh_y
            else:
                cgh = np.mod(cgh_x + cgh_y, 2)
        elif mode == 'x':
            cgh = np.cos(Gx * i1_x0_shift)
            cgh = Step_U(cgh, 'x', 
                         Duty_Cycle_x, Duty_Cycle_y, 
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'y':
            cgh = np.cos(Gy * i1_y0_shift)
            cgh = Step_U(cgh, 'y', 
                         Duty_Cycle_x, Duty_Cycle_y, 
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'xy':
            cgh = np.cos(Gx * i1_x0_shift + Gy * i1_y0_shift)
            cgh = Step_U(cgh, 'x', 
                         Duty_Cycle_x, Duty_Cycle_y, 
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh # 在所有方向的占空比都认为是 Duty_Cycle_x
        return cgh
    else:
        if mode == 'x*y' or mode == 'x+y':
            cgh = np.cos(Gx * i1_x0_shift - ( np.angle(U) + math.pi )) - np.cos( np.arcsin( np.abs(U) / np.max( np.abs(U) ) ) )
            cgh_x = Step_U(cgh, 'x', 
                           Duty_Cycle_x, Duty_Cycle_y, 
                           is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
            cgh = np.cos(Gy * i1_y0_shift - ( np.angle(U) + math.pi )) - np.cos( np.arcsin( np.abs(U) / np.max( np.abs(U) ) ) )
            cgh_y = Step_U(cgh, 'y', 
                           Duty_Cycle_x, Duty_Cycle_y, 
                           is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
            if mode == 'x*y':
                cgh = cgh_x * cgh_y
            else:
                cgh = np.mod(cgh_x + cgh_y, 2)
        elif mode == 'x':
            cgh = np.cos(Gx * i1_x0_shift - ( np.angle(U) + math.pi )) - np.cos( np.arcsin( np.abs(U) / np.max( np.abs(U) ) ) )
            cgh = Step_U(cgh, 'x', 
                         Duty_Cycle_x, Duty_Cycle_y, 
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'y':
            cgh = np.cos(Gy * i1_y0_shift - ( np.angle(U) + math.pi )) - np.cos( np.arcsin( np.abs(U) / np.max( np.abs(U) ) ) )
            cgh = Step_U(cgh, 'y', 
                         Duty_Cycle_x, Duty_Cycle_y, 
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'xy':
            cgh = np.cos(Gx * i1_x0_shift + Gy * i1_y0_shift - ( np.angle(U) + math.pi )) - np.cos( np.arcsin( np.abs(U) / np.max( np.abs(U) ) ) )
            cgh = Step_U(cgh,'x', 
                         Duty_Cycle_x, Duty_Cycle_y, 
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
        return cgh

#%%

def structure_Generate_2D_CGH(U, mode, 
                              Duty_Cycle_x, Duty_Cycle_y, 
                              is_positive_xy, 
                              #%%
                              Gx, Gy, 
                              is_Gauss, l, 
                              is_continuous, 
                              #%%
                              is_target_far_field, is_transverse_xy, is_reverse_xy, ):
    
    if is_target_far_field == 0: # 如果 想要的 U1_0 是近场（晶体后端面）分布
        
        g = np.fft.fft2(U)
        g_shift = np.fft.fftshift(g)
        
        if is_transverse_xy == 1:
            structure = CGH(g_shift, mode, 
                            Duty_Cycle_x, Duty_Cycle_y, 
                            is_positive_xy, 
                            #%%
                            Gx, Gy, 
                            is_Gauss, l, 
                            is_continuous, ).T # 转置（沿 右下 对角线 翻转）
        else:
            structure = CGH(g_shift, mode, 
                            Duty_Cycle_x, Duty_Cycle_y, 
                            is_positive_xy, 
                            #%%
                            Gx, Gy, 
                            is_Gauss, l, 
                            is_continuous, )[::-1] # 上下翻转
            
    else: # 如果 想要的 U1_0 是远场分布
        if is_transverse_xy == 1:
            structure = CGH(U, mode, 
                            Duty_Cycle_x, Duty_Cycle_y, 
                            is_positive_xy, 
                            #%%
                            Gx, Gy, 
                            is_Gauss, l, 
                            is_continuous, ).T # 转置（沿 右下 对角线 翻转）
        else:
            structure = CGH(U, mode, 
                            Duty_Cycle_x, Duty_Cycle_y, 
                            is_positive_xy, 
                            #%%
                            Gx, Gy, 
                            is_Gauss, l, 
                            is_continuous, )[::-1] # 上下翻转
        
    if is_reverse_xy == 1:
        structure = 1 - structure

    return structure
        
def structure_Generate_2D_radial_G(Ix, Iy, 
                                   G, Duty_Cycle, 
                                   is_positive_xy, is_continuous, is_reverse_xy, ):
    
    ix, iy = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
    ix_shift, iy_shift = ix - Ix // 2, iy - Iy // 2
    
    cgh = np.cos( G * (ix_shift**2 + iy_shift**2)**0.5 )
    structure = Step_U(cgh, 'x', 
                       Duty_Cycle, Duty_Cycle, 
                       is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh # 在所有方向的占空比都认为是 Duty_Cycle_x
    
    if is_reverse_xy == 1:
        structure = 1 - structure
        
    return structure
    

def structure_chi2_Generate_2D(U1_name = "", 
                              img_full_name = "Grating.png", 
                              is_phase_only = 0, 
                              #%%
                              z_pump = 0, 
                              is_LG = 0, is_Gauss = 0, is_OAM = 0, 
                              l = 0, p = 0, 
                              theta_x = 0, theta_y = 0, 
                              is_random_phase = 0, 
                              is_H_l = 0, is_H_theta = 0, is_H_random_phase = 0, 
                              #%%
                              U1_0_NonZero_size = 1, w0 = 0.3, structure_size_Enlarge = 0.1, 
                              Duty_Cycle_x = 0.5, Duty_Cycle_y = 0.5, structure_xy_mode = 'x', Depth = 2, 
                              #%%
                              is_continuous = 1, is_target_far_field = 1, is_transverse_xy = 0, is_reverse_xy = 0, is_positive_xy = 1, is_no_backgroud = 1, 
                              #%%
                              lam1 = 0.8, is_air_pump = 0, is_air = 0, T = 25, 
                              Tx = 10, Ty = 10, Tz = "2*lc", 
                              mx = 0, my = 0, mz = 0, 
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
                              #%%
                              is_print = 1, ):

    
    
    #%%
    # 导入 方形，以及 加边框 的 图片
    
    img_name, img_name_extension, img_squared, size_PerPixel, size_fig, I1_x, I1_y, U1_0 = img_squared_bordered_Read(img_full_name, 
                                                                                                                     U1_0_NonZero_size, dpi, 
                                                                                                                     is_phase_only)

    #%%
    # 定义 调制区域 的 横向实际像素、调制区域 的 实际横向尺寸
    
    deff_structure_size_expect = U1_0_NonZero_size * ( 1 + structure_size_Enlarge )
    is_print and print("deff_structure_size_expect = {} mm".format(deff_structure_size_expect))

    Ix, Iy, deff_structure_size = Cal_IxIy(I1_x, I1_y, 
                                           deff_structure_size_expect, size_PerPixel, 
                                           is_print)

    #%%
    # 需要先将 目标 U1_0_NonZero = img_squared 给 放大 或 缩小 到 与 全息图（结构） 横向尺寸 Ix, Iy 相同，才能开始 之后的工作

    border_width, img_squared_resize_full_name, img_squared_resize = img_squared_Resize(img_name, img_name_extension, img_squared, 
                                                                                        Ix, Iy, I1_x, 
                                                                                        is_print, )
    
    if (type(U1_name) != str) or U1_name == "":
        #%%
        # U1_0 = U(x, y, 0) = img_squared_resize
        
        if is_phase_only == 1:
            U1_0 = np.power(math.e, (img_squared_resize.astype(np.complex128()) / 255 * 2 * math.pi - math.pi) * 1j) # 变成相位图
        else:
            U1_0 = img_squared_resize.astype(np.complex128)
        
        #%%
        # 预处理 输入场
        
        n1, k1 = Cal_n(size_PerPixel, 
                       is_air_pump, 
                       lam1, T, p = "e")
        
        U1_0 = pump_LG(img_squared_resize_full_name, 
                       Ix, Iy, size_PerPixel, 
                       U1_0, w0, k1, z_pump, 
                       is_LG, is_Gauss, is_OAM, 
                       l, p, 
                       theta_x, theta_y, 
                       is_random_phase, 
                       is_H_l, is_H_theta, is_H_random_phase, 
                       is_save, is_save_txt, dpi, 
                       cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                       fontsize, font, 
                       1, is_colorbar_on, is_energy, vmax, vmin, 
                       is_print, ) 

    else:
        
        #%%
        # 导入 方形，以及 加边框 的 图片
        
        U1_full_name = U1_name + (is_save_txt and ".txt" or ".mat")
        U1_0 = np.loadtxt(U1_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U1_full_name)['U'] # 加载 复振幅场
        
        U1_0 = cv2.resize(np.real(U1_0), (Ix, Iy), interpolation=cv2.INTER_AREA) + cv2.resize(np.imag(U1_0), (Ix, Iy), interpolation=cv2.INTER_AREA) * 1j
        # U1_0 必须 resize 为 Ix,Iy 大小； 
        # 但 cv2 、 skimage.transform 中 resize 都能处理 图片 和 float64，
        # 但似乎 没有东西 能直接 处理 complex128，但可 分别处理 实部和虚部，再合并为 complex128

    #%%

    n1, k1 = Cal_n(size_PerPixel, 
                   is_air, 
                   lam1, T, p = "e")
    
    k1_z_shift, mesh_k1_x_k1_y_shift = Cal_kz(I1_x, I1_y, k1)
    
    #%%
    # 线性 角谱理论 - 基波 begin

    g1 = np.fft.fft2(U1_0)
    g1_shift = np.fft.fftshift(g1)

    #%%

    lam2 = lam1 / 2

    n2, k2 = Cal_n(size_PerPixel, 
                   is_air, 
                   lam2, T, p = "e")
    
    k2_z_shift, mesh_k2_x_k2_y_shift = Cal_kz(I1_x, I1_y, k2)
    
    #%%

    dk, lc, Tz = Cal_lc_SHG(k1, k2, Tz, size_PerPixel, 
                            is_print = 0)
    # 尽管 Gz 在这里更新并不妥，因为 在该函数 外部，提供描边信息时， Tz 会 覆盖其值，因此 Gz 的值需要更新
    # 但在这里，我们并不需要 Gz 的值，而哪怕在外部，z 向的 structure Generate 的时候，需要的也只是更新后的 Tz，而不需要 Gz
    # 所以把下面这段 放进来 在这里 是 可以的。
    Gx, Gy, Gz = Cal_GxGyGz(mx, my, mz,
                            Tx, Ty, Tz, size_PerPixel, 
                            is_print)
    
    #%%
    # 开始生成 调制函数 structure 和 modulation = 1 - is_no_backgroud - Depth * structure，以及 structure_opposite = 1 - structure 及其 modulation

    if structure_xy_mode == "r":
        
        structure = structure_Generate_2D_radial_G(Ix, Iy, 
                                                   Gx, Duty_Cycle_x, 
                                                   is_positive_xy, is_continuous, is_reverse_xy, )
    else:

        structure = structure_Generate_2D_CGH(U1_0, structure_xy_mode, 
                                              Duty_Cycle_x, Duty_Cycle_y, 
                                              is_positive_xy, 
                                              #%%
                                              Gx, Gy, 
                                              is_Gauss, l, 
                                              is_continuous, 
                                              #%%
                                              is_target_far_field, is_transverse_xy, is_reverse_xy, )
    
    # vmax_structure, vmin_structure = 1, 0
    vmax_modulation, vmin_modulation = 1 - is_no_backgroud, -1 - is_no_backgroud

    # plot_2d([], 1, size_PerPixel,  
    #         structure, "χ2_structure" + img_name_extension, "χ2_structure", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         0, is_colorbar_on, 0, vmax_structure, vmin_structure)

    modulation = 1 - is_no_backgroud - Depth * structure
    modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (1 - is_no_backgroud, 1 - is_no_backgroud))

    plot_2d([], 1, size_PerPixel, 
            modulation_squared, "χ2_modulation_squared" + img_name_extension, "χ2_modulation_squared", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            0, is_colorbar_on, 0, vmax_modulation, vmin_modulation)

    #%%

    if mz != 0:
        structure_opposite = 1 - structure
    else:
        structure_opposite = structure
    
    # plot_2d([], 1, size_PerPixel,  
    #         structure_opposite, "χ2_structure_opposite" + img_name_extension, "χ2_structure_opposite", 
    #         is_save, dpi, size_fig,
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         0, is_colorbar_on, 0, vmax_structure, vmin_structure)
        
    modulation_opposite = 1 - is_no_backgroud - Depth * structure_opposite
    modulation_opposite_squared = np.pad(modulation_opposite, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (1 - is_no_backgroud, 1 - is_no_backgroud))

    plot_2d([], 1, size_PerPixel, 
            modulation_opposite_squared, "χ2_modulation_opposite_squared" + img_name_extension, "χ2_modulation_opposite_squared", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            0, is_colorbar_on, 0, vmax_modulation, vmin_modulation)
        
    return n1, k1, k1_z_shift, lam2, n2, k2, k2_z_shift, \
           dk, lc, Tz, Gx, Gy, Gz, \
           size_PerPixel, U1_0, g1_shift, \
           structure, structure_opposite, modulation, modulation_opposite, modulation_squared, modulation_opposite_squared
        
def structure_n1_Generate_2D(U1_name = "", 
                              img_full_name = "Grating.png", 
                              is_phase_only = 0, 
                              #%%
                              z_pump = 0, 
                              is_LG = 0, is_Gauss = 0, is_OAM = 0, 
                              l = 0, p = 0, 
                              theta_x = 0, theta_y = 0, 
                              is_random_phase = 0, 
                              is_H_l = 0, is_H_theta = 0, is_H_random_phase = 0, 
                              #%%
                              U1_0_NonZero_size = 1, w0 = 0.3, structure_size_Enlarge = 0.1, 
                              Duty_Cycle_x = 0.5, Duty_Cycle_y = 0.5, structure_xy_mode = 'x', Depth = 2, 
                              #%%
                              is_continuous = 1, is_target_far_field = 1, is_transverse_xy = 0, is_reverse_xy = 0, is_positive_xy = 1, 
                              #%%
                              lam1 = 0.8, is_air_pump = 0, is_air = 0, T = 25, 
                              Tx = 10, Ty = 10, Tz = "2*lc", 
                              mx = 0, my = 0, mz = 0, 
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
                              #%%
                              is_print = 1, ):

    #%%
    # 导入 方形，以及 加边框 的 图片
    
    img_name, img_name_extension, img_squared, size_PerPixel, size_fig, I1_x, I1_y, U1_0 = img_squared_bordered_Read(img_full_name, 
                                                                                                                     U1_0_NonZero_size, dpi, 
                                                                                                                     is_phase_only)

    #%%
    # 定义 调制区域 的 横向实际像素、调制区域 的 实际横向尺寸
    
    deff_structure_size_expect = U1_0_NonZero_size * ( 1 + structure_size_Enlarge )
    is_print and print("deff_structure_size_expect = {} mm".format(deff_structure_size_expect))

    Ix, Iy, deff_structure_size = Cal_IxIy(I1_x, I1_y, 
                                           deff_structure_size_expect, size_PerPixel, 
                                           is_print)

    #%%
    # 需要先将 目标 U1_0_NonZero = img_squared 给 放大 或 缩小 到 与 全息图（结构） 横向尺寸 Ix, Iy 相同，才能开始 之后的工作

    border_width, img_squared_resize_full_name, img_squared_resize = img_squared_Resize(img_name, img_name_extension, img_squared, 
                                                                                        Ix, Iy, I1_x, 
                                                                                        is_print, )

    if (type(U1_name) != str) or U1_name == "":
        #%%
        # U1_0 = U(x, y, 0) = img_squared_resize
        
        if is_phase_only == 1:
            U1_0 = np.power(math.e, (img_squared_resize.astype(np.complex128()) / 255 * 2 * math.pi - math.pi) * 1j) # 变成相位图
        else:
            U1_0 = img_squared_resize.astype(np.complex128)
        
        #%%
        # 预处理 输入场
        
        n1, k1 = Cal_n(size_PerPixel, 
                       is_air_pump, 
                       lam1, T, p = "e")
        
        U1_0 = pump_LG(img_squared_resize_full_name, 
                       Ix, Iy, size_PerPixel, 
                       U1_0, w0, k1, z_pump, 
                       is_LG, is_Gauss, is_OAM, 
                       l, p, 
                       theta_x, theta_y, 
                       is_random_phase, 
                       is_H_l, is_H_theta, is_H_random_phase, 
                       is_save, is_save_txt, dpi, 
                       cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                       fontsize, font, 
                       1, is_colorbar_on, is_energy, vmax, vmin, 
                       is_print, ) 

    else:
        
        #%%
        # 导入 方形，以及 加边框 的 图片
        
        U1_full_name = U1_name + (is_save_txt and ".txt" or ".mat")
        U1_0 = np.loadtxt(U1_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U1_full_name)['U'] # 加载 复振幅场
        
        U1_0 = cv2.resize(np.real(U1_0), (Ix, Iy), interpolation=cv2.INTER_AREA) + cv2.resize(np.imag(U1_0), (Ix, Iy), interpolation=cv2.INTER_AREA) * 1j
        # U1_0 必须 resize 为 Ix,Iy 大小； 
        # 但 cv2 、 skimage.transform 中 resize 都能处理 图片 和 float64，
        # 但似乎 没有东西 能直接 处理 complex128，但可 分别处理 实部和虚部，再合并为 complex128

    #%%

    n1, k1 = Cal_n(size_PerPixel, 
                   is_air, 
                   lam1, T, p = "e")

    k1_z_shift, mesh_k1_x_k1_y_shift = Cal_kz(I1_x, I1_y, k1)

    # %%
    # 线性 角谱理论 - 基波 begin

    g1 = np.fft.fft2(U1_0)
    g1_shift = np.fft.fftshift(g1)

    #%%

    lam2 = lam1 / 2

    n2, k2 = Cal_n(size_PerPixel, 
                   is_air, 
                   lam2, T, p = "e")

    k2_z_shift, mesh_k2_x_k2_y_shift = Cal_kz(I1_x, I1_y, k2)
    
    #%%

    dk, lc, Tz = Cal_lc_SHG(k1, k2, Tz, size_PerPixel, 
                            is_print = 0)

    Gx, Gy, Gz = Cal_GxGyGz(mx, my, mz,
                            Tx, Ty, Tz, size_PerPixel, 
                            is_print)
    
    #%%
    # 开始生成 调制函数 structure 和 modulation = n1 - Depth * structure，以及 structure_opposite = 1 - structure 及其 modulation

    if structure_xy_mode == "r":
        
        structure = structure_Generate_2D_radial_G(Ix, Iy, 
                                                   Gx, Duty_Cycle_x, 
                                                   is_positive_xy, is_continuous, is_reverse_xy, )
    else:

        structure = structure_Generate_2D_CGH(U1_0, structure_xy_mode, 
                                              Duty_Cycle_x, Duty_Cycle_y, 
                                              is_positive_xy, 
                                              #%%
                                              Gx, Gy, 
                                              is_Gauss, l, 
                                              is_continuous, 
                                              #%%
                                              is_target_far_field, is_transverse_xy, is_reverse_xy, )

    # vmax_structure, vmin_structure = 1, 0
    vmax_modulation, vmin_modulation = n1, n1 - Depth

    # plot_2d([], 1, size_PerPixel,  
    #         structure, "n1_structure" + img_name_extension, "n1_structure", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         0, is_colorbar_on, 0, vmax_structure, vmin_structure)

    modulation = n1 - Depth * structure
    modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (n1, n1))

    plot_2d([], 1, size_PerPixel, 
            modulation_squared, "n1_modulation_squared" + img_name_extension, "n1_modulation_squared", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            0, is_colorbar_on, 0, vmax_modulation, vmin_modulation)

    #%%
    
    if mz != 0:
        structure_opposite = 1 - structure
    else:
        structure_opposite = structure
    
    # plot_2d([], 1, size_PerPixel,  
    #         structure_opposite, "n1_structure_opposite" + img_name_extension, "n1_structure_opposite", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         0, is_colorbar_on, 0, vmax_structure, vmin_structure)

    modulation_opposite = n1 - Depth * structure_opposite
    modulation_opposite_squared = np.pad(modulation_opposite, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (n1, n1))

    plot_2d([], 1, size_PerPixel, 
            modulation_opposite_squared, "n1_modulation_opposite_squared" + img_name_extension, "n1_modulation_opposite_squared", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            0, is_colorbar_on, 0, vmax_modulation, vmin_modulation)
        
    return n1, k1, k1_z_shift, lam2, n2, k2, k2_z_shift, \
           dk, lc, Tz, Gx, Gy, Gz, \
           size_PerPixel, U1_0, g1_shift, \
           structure, structure_opposite, modulation, modulation_opposite, modulation_squared, modulation_opposite_squared