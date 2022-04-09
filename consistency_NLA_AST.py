# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:57 2021

@author: Xcz
"""

#%%

import os
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
from scipy.io import loadmat, savemat
from fun_os import U_Read
from fun_img_Resize import image_Add_black_border
from fun_plot import plot_2d
from b_1_AST import AST
from b_3_NLA import NLA
from fun_compare import U_compare

def consistency_NLA_AST(U1_name = "", 
                        img_full_name = "Grating.png", 
                        border_percentage = 0.3, 
                        is_phase_only = 0, 
                        #%%
                        z_pump = 0, 
                        is_LG = 0, is_Gauss = 0, is_OAM = 0, 
                        l = 0, p = 0, 
                        theta_x = 0, theta_y = 0, 
                        #%%
                        is_random_phase = 0, 
                        is_H_l = 0, is_H_theta = 0, is_H_random_phase = 0, 
                        # %%
                        # 生成横向结构
                        U1_name_Structure = '',
                        structure_size_Enlarge = 0.1,
                        is_phase_only_Structure = 0,
                        # %%
                        w0_Structure = 0, z_pump_Structure = 0,
                        is_LG_Structure = 0, is_Gauss_Structure = 0, is_OAM_Structure = 0, 
                        l_Structure = 0, p_Structure = 0, 
                        theta_x_Structure = 0, theta_y_Structure = 0,
                        # %%
                        is_random_phase_Structure = 0, 
                        is_H_l_Structure = 0, is_H_theta_Structure = 0, is_H_random_phase_Structure = 0, 
                        #%%
                        U1_0_NonZero_size = 1, w0 = 0.3,
                        z0 = 1, z0_new = 5, 
                        # %%
                        lam1=0.8, is_air_pump=0, is_air=0, T=25,
                        deff=30, is_fft = 1, fft_mode = 0, 
                        is_linear_convolution = 0,
                        #%%
                        Tx=10, Ty=10, Tz="2*lc",
                        mx=0, my=0, mz=0,
                        # %%
                        # 生成横向结构
                        Duty_Cycle_x = 0.5, Duty_Cycle_y = 0.5, Duty_Cycle_z = 0.5,
                        Depth = 2, structure_xy_mode = 'x', 
                        is_continuous = 0, is_target_far_field = 1, is_transverse_xy = 0, 
                        is_reverse_xy = 0, is_positive_xy = 1, is_no_backgroud = 0,
                        #%%
                        is_save = 0, is_save_txt = 0, dpi = 100, 
                        #%%
                        cmap_2d = 'viridis', 
                        #%%
                        ticks_num = 6, is_contourf = 0, 
                        is_title_on = 1, is_axes_on = 1, 
                        is_mm = 1,
                        #%%
                        fontsize = 9, 
                        font = {'family': 'serif',
                                'style': 'normal', # 'normal', 'italic', 'oblique'
                                'weight': 'normal',
                                'color': 'black', # 'black','gray','darkred'
                                }, 
                        #%%
                        is_colorbar_on = 1, is_energy = 1,
                        #%%
                        is_print = 1, is_contours = 1, n_TzQ = 1, 
                        Gz_max_Enhance = 1, match_mode = 1, ):
    
    # %%
    # 线性 惠更斯 菲涅尔 原理
    
    image_Add_black_border(img_full_name, 
                            border_percentage, 
                            is_print, )
    
    #%%
    # 先衍射 z0 后衍射 z0_new
    
    AST('', 
        img_full_name, 
        is_phase_only, 
        #%%
        z_pump, 
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        #%%
        is_random_phase, 
        is_H_l, is_H_theta, is_H_random_phase, 
        #%%
        U1_0_NonZero_size, w0,
        z0, 
        #%%
        lam1, is_air_pump, is_air, T, 
        #%%
        is_save, is_save_txt, dpi, 
        #%%
        cmap_2d, 
        #%%
        ticks_num, is_contourf, 
        is_title_on, is_axes_on, 
        is_mm, 
        #%%
        fontsize, font, 
        #%%
        is_colorbar_on, is_energy, 
        #%%
        is_print, )
    
    U1_name = "6. AST - U1_" + str(float('%.2g' % z0)) + "mm"
    # U1_full_name = U1_name + (is_save_txt and ".txt" or ".mat")
    # U1_short_name = U1_name.replace('6. AST - ', '')
    
    AST(U1_name, 
        img_full_name, 
        is_phase_only, 
        #%%
        z_pump, 
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        #%%
        is_random_phase, 
        is_H_l, is_H_theta, is_H_random_phase, 
        #%%
        U1_0_NonZero_size, w0,
        z0_new, 
        #%%
        lam1, is_air_pump, is_air, T, 
        #%%
        is_save, is_save_txt, dpi, 
        #%%
        cmap_2d, 
        #%%
        ticks_num, is_contourf, 
        is_title_on, is_axes_on, 
        is_mm, 
        #%%
        fontsize, font, 
        #%%
        is_colorbar_on, is_energy, 
        #%%
        is_print, )
    
    #%%
    # 直接衍射 Z0 = z0 + z0_new
    
    Z0 = z0 + z0_new
    
    AST("", 
        img_full_name, 
        is_phase_only, 
        #%%
        z_pump, 
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        #%%
        is_random_phase, 
        is_H_l, is_H_theta, is_H_random_phase, 
        #%%
        U1_0_NonZero_size, w0,
        Z0, 
        #%%
        lam1, is_air_pump, is_air, T, 
        #%%
        is_save, is_save_txt, dpi, 
        #%%
        cmap_2d, 
        #%%
        ticks_num, is_contourf, 
        is_title_on, is_axes_on, 
        is_mm, 
        #%%
        fontsize, font, 
        #%%
        is_colorbar_on, is_energy, 
        #%%
        is_print, )
    
    #%%
    # 非线性 惠更斯 菲涅尔 原理
    
    image_Add_black_border(img_full_name, 
                           border_percentage, 
                           is_print, )
    
    #%%
    # 先衍射 z0 后倍频 z0_NLA
    
    AST("", 
        img_full_name, 
        is_phase_only, 
        #%%
        z_pump, 
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        #%%
        is_random_phase, 
        is_H_l, is_H_theta, is_H_random_phase, 
        #%%
        U1_0_NonZero_size, w0,
        z0, 
        #%%
        lam1, is_air_pump, is_air, T, 
        #%%
        is_save, is_save_txt, dpi, 
        #%%
        cmap_2d, 
        #%%
        ticks_num, is_contourf, 
        is_title_on, is_axes_on, 
        is_mm, 
        #%%
        fontsize, font, 
        #%%
        is_colorbar_on, is_energy, 
        #%%
        is_print, )
    
    U1_name = "6. AST - U1_" + str(float('%.2g' % z0)) + "mm"
    # U1_full_name = U1_name + (is_save_txt and ".txt" or ".mat")
    # U1_short_name = U1_name.replace('6. AST - ', '')
    
    z0_NLA = z0_new
    
    NLA(U1_name, 
        img_full_name, 
        is_phase_only, 
        #%%
        z_pump, 
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        #%%
        is_random_phase, 
        is_H_l, is_H_theta, is_H_random_phase, 
        # %%
        # 生成横向结构
        U1_name_Structure,
        structure_size_Enlarge,
        is_phase_only_Structure,
        # %%
        w0_Structure, z_pump_Structure,
        is_LG_Structure, is_Gauss_Structure, is_OAM_Structure, 
        l_Structure, p_Structure, 
        theta_x_Structure, theta_y_Structure,
        # %%
        is_random_phase_Structure, 
        is_H_l_Structure, is_H_theta_Structure, is_H_random_phase_Structure, 
        #%%
        U1_0_NonZero_size, w0,
        z0_NLA, 
        #%%
        lam1, is_air_pump, is_air, T, 
        deff, is_fft, fft_mode, 
        is_linear_convolution,
        #%%
        Tx, Ty, Tz, 
        mx, my, mz, 
        # %%
        # 生成横向结构
        Duty_Cycle_x, Duty_Cycle_y, Duty_Cycle_z,
        Depth, structure_xy_mode, 
        is_continuous, is_target_far_field, is_transverse_xy, 
        is_reverse_xy, is_positive_xy, is_no_backgroud,
        #%%
        is_save, is_save_txt, dpi, 
        #%%
        cmap_2d, 
        #%%
        ticks_num, is_contourf, 
        is_title_on, is_axes_on, 
        is_mm, 
        #%%
        fontsize, font, 
        #%%
        is_colorbar_on, is_energy, 
        #%%
        is_print, is_contours, n_TzQ, 
        Gz_max_Enhance, match_mode, )
    
    U1_NLA_txt_name = "6. NLA - U2_" + str(float('%.2g' % z0_NLA)) + "mm"
    U1_NLA_txt_full_name = U1_NLA_txt_name + (is_save_txt and ".txt" or ".mat")
    # U1_NLA_txt_short_name = U1_NLA_txt_name.replace('6. NLA - ', '')
    U1_NLA_txt_short_name = U1_NLA_txt_name.replace('6. NLA - ', 'NLA - ')
    
    #%%
    # 先倍频 z0 后衍射 z0_AST
    
    NLA("", 
        img_full_name, 
        is_phase_only, 
        #%%
        z_pump, 
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        #%%
        is_random_phase, 
        is_H_l, is_H_theta, is_H_random_phase, 
        # %%
        # 生成横向结构
        U1_name_Structure,
        structure_size_Enlarge,
        is_phase_only_Structure,
        # %%
        w0_Structure, z_pump_Structure,
        is_LG_Structure, is_Gauss_Structure, is_OAM_Structure, 
        l_Structure, p_Structure, 
        theta_x_Structure, theta_y_Structure,
        # %%
        is_random_phase_Structure, 
        is_H_l_Structure, is_H_theta_Structure, is_H_random_phase_Structure, 
        #%%
        U1_0_NonZero_size, w0,
        z0, 
        #%%
        lam1, is_air_pump, is_air, T, 
        deff, is_fft, fft_mode,
        is_linear_convolution, 
        #%%
        Tx, Ty, Tz, 
        mx, my, mz, 
        # %%
        # 生成横向结构
        Duty_Cycle_x, Duty_Cycle_y, Duty_Cycle_z,
        Depth, structure_xy_mode, 
        is_continuous, is_target_far_field, is_transverse_xy, 
        is_reverse_xy, is_positive_xy, is_no_backgroud,
        #%%
        is_save, is_save_txt, dpi, 
        #%%
        cmap_2d, 
        #%%
        ticks_num, is_contourf, 
        is_title_on, is_axes_on, 
        is_mm, 
        #%%
        fontsize, font, 
        #%%
        is_colorbar_on, is_energy, 
        #%%
        is_print, is_contours, n_TzQ, Gz_max_Enhance, match_mode, )
    
    U2_txt_name = "6. NLA - U2_" + str(float('%.2g' % z0)) + "mm"
    # U2_txt_full_name = U2_txt_name + (is_save_txt and ".txt" or ".mat")
    # U2_txt_short_name = U2_txt_name.replace('6. NLA - ', '')
    
    z0_AST = z0_new
    
    AST(U2_txt_name, 
        img_full_name, 
        is_phase_only, 
        #%%
        z_pump, 
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        #%%
        is_random_phase, 
        is_H_l, is_H_theta, is_H_random_phase, 
        #%%
        U1_0_NonZero_size, w0,
        z0_AST, 
        #%%
        lam1, is_air_pump, is_air, T, 
        #%%
        is_save, is_save_txt, dpi, 
        #%%
        cmap_2d, 
        #%%
        ticks_num, is_contourf, 
        is_title_on, is_axes_on, 
        is_mm, 
        #%%
        fontsize, font, 
        #%%
        is_colorbar_on, is_energy, 
        #%%
        is_print, )
    
    U2_AST_txt_name = "6. AST - U2_" + str(float('%.2g' % z0_AST)) + "mm"
    U2_AST_txt_full_name = U2_AST_txt_name + (is_save_txt and ".txt" or ".mat")
    # U2_AST_txt_short_name = U2_AST_txt_name.replace('6. AST - ', '')
    U2_AST_txt_short_name = U2_AST_txt_name.replace('6. AST - ', 'AST - ')
    
    #%%
    # 直接倍频 Z0 = z0 + z0_new
    
    Z0 = z0 + z0_new
    
    NLA("", 
        img_full_name, 
        is_phase_only, 
        #%%
        z_pump, 
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        #%%
        is_random_phase, 
        is_H_l, is_H_theta, is_H_random_phase, 
        # %%
        # 生成横向结构
        U1_name_Structure,
        structure_size_Enlarge,
        is_phase_only_Structure,
        # %%
        w0_Structure, z_pump_Structure,
        is_LG_Structure, is_Gauss_Structure, is_OAM_Structure, 
        l_Structure, p_Structure, 
        theta_x_Structure, theta_y_Structure,
        # %%
        is_random_phase_Structure, 
        is_H_l_Structure, is_H_theta_Structure, is_H_random_phase_Structure, 
        #%%
        U1_0_NonZero_size, w0,
        Z0, 
        #%%
        lam1, is_air_pump, is_air, T, 
        deff, is_fft, fft_mode,
        is_linear_convolution, 
        #%%
        Tx, Ty, Tz, 
        mx, my, mz, 
        # %%
        # 生成横向结构
        Duty_Cycle_x, Duty_Cycle_y, Duty_Cycle_z,
        Depth, structure_xy_mode, 
        is_continuous, is_target_far_field, is_transverse_xy, 
        is_reverse_xy, is_positive_xy, is_no_backgroud,
        #%%
        is_save, is_save_txt, dpi, 
        #%%
        cmap_2d, 
        #%%
        ticks_num, is_contourf, 
        is_title_on, is_axes_on, 
        is_mm, 
        #%%
        fontsize, font, 
        #%%
        is_colorbar_on, is_energy, 
        #%%
        is_print, is_contours, n_TzQ, Gz_max_Enhance, match_mode, )
    
    U2_Z0_txt_name = "6. NLA - U2_" + str(float('%.2g' % Z0)) + "mm"
    # U2_Z0_txt_full_name = U2_Z0_txt_name + (is_save_txt and ".txt" or ".mat")
    # U2_Z0_txt_short_name = U2_Z0_txt_name.replace('6. NLA - ', '')
    
    # U2_Z0 = np.loadtxt(U2_Z0_txt_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U2_Z0_txt_full_name)['U'] # 加载 复振幅场
    
    img_name, img_name_extension, img_squared, size_PerPixel, size_fig, I2_x, I2_y, U2_Z0 = U_Read(U2_Z0_txt_name, img_full_name, 
                                                                                                   U1_0_NonZero_size, dpi, 
                                                                                                   is_save_txt, )
    
    #%%
    # 加和 U1_NLA 与 U2_AST = U2_Z0_Superposition
    
    U1_NLA = np.loadtxt(U1_NLA_txt_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U1_NLA_txt_full_name)['U'] # 加载 复振幅场
    U2_AST = np.loadtxt(U2_AST_txt_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U2_AST_txt_full_name)['U'] # 加载 复振幅场
    
    U2_Z0_Superposition = U1_NLA + U2_AST
    
    U2_Z0_Superposition_amp = np.abs(U2_Z0_Superposition)
    # print(np.max(U2_Z0_Superposition_amp))
    U2_Z0_Superposition_phase = np.angle(U2_Z0_Superposition)

    print("NLAST - U2_{}mm.total_energy = {}".format(Z0, np.sum(U2_Z0_Superposition_amp**2)))

    if is_save == 1:
        if not os.path.isdir("6. U2_" + str(float('%.2g' % Z0)) + "mm"):
            os.makedirs("6. U2_" + str(float('%.2g' % Z0)) + "mm")

    #%%
    #绘图：U2_Z0_Superposition_amp

    U2_Z0_Superposition_amp_address = "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "\\" + "6.1. NLAST - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_amp" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_abs" + img_name_extension

    plot_2d([], 1, size_PerPixel, 
            U2_Z0_Superposition_amp, U2_Z0_Superposition_amp_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    #%%
    #绘图：U2_Z0_Superposition_phase

    U2_Z0_Superposition_phase_address = "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "\\" + "6.2. NLAST - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_phase" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_angle" + img_name_extension

    plot_2d([], 1, size_PerPixel, 
            U2_Z0_Superposition_phase, U2_Z0_Superposition_phase_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_phase", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)
    
    #%%
    # 储存 U2_Z0_Superposition 到 txt 文件

    U2_Z0_Superposition_full_name = "6. NLAST - U2_" + str(float('%.2g' % Z0)) + "mm" + (is_save_txt and ".txt" or ".mat")
    if is_save == 1:
        U2_Z0_Superposition_txt_address = "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "\\" + U2_Z0_Superposition_full_name
        np.savetxt(U2_Z0_Superposition_txt_address, U2_Z0_Superposition) if is_save_txt else savemat(U2_Z0_Superposition_txt_address, {"U":U2_Z0_Superposition})

        #%%
        #再次绘图：U2_Z0_Superposition_amp
    
        U2_Z0_Superposition_amp_address = "6.1. NLAST - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_amp" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_abs" + img_name_extension
    
        plot_2d([], 1, size_PerPixel, 
                U2_Z0_Superposition_amp, U2_Z0_Superposition_amp_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_amp", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, is_energy, vmax, vmin)
        
        #再次绘图：U2_Z0_Superposition_phase
    
        U2_Z0_Superposition_phase_address = "6.2. NLAST - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_phase" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_angle" + img_name_extension
    
        plot_2d([], 1, size_PerPixel, 
                U2_Z0_Superposition_phase, U2_Z0_Superposition_phase_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_phase", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, 0, vmax, vmin)

    #%%
    # 储存 U2_Z0_Superposition 到 txt 文件

    # if is_save == 1:
    np.savetxt(U2_Z0_Superposition_full_name, U2_Z0_Superposition) if is_save_txt else savemat(U2_Z0_Superposition_full_name, {"U":U2_Z0_Superposition})
    
    #%%
    # 对比 U2_Z0_Superposition 与 U2_Z0 的 （绝对）误差
    
    U_compare(U2_Z0_Superposition, U2_Z0, Z0, 
              #%%
              img_name_extension, size_PerPixel, size_fig, 
              # %%
              is_save, is_save_txt, dpi, 
              #%%
              cmap_2d, 
              #%%
              ticks_num, is_contourf, 
              is_title_on, is_axes_on, is_mm,
              #%%
              fontsize, font, 
              #%%S
              is_colorbar_on, is_energy,
              #%%
              0 )
    
    #%%

if __name__ == '__main__':
    consistency_NLA_AST(U1_name = "",
                        img_full_name = "lena1.png",
                        border_percentage = 0.1,
                        is_phase_only = 0,
                        #%%
                        z_pump = 0,
                        is_LG = 0, is_Gauss = 0, is_OAM = 0,
                        l = 0, p = 0,
                        theta_x = 0, theta_y = 0,
                        # %%
                        is_random_phase = 0,
                        is_H_l = 0, is_H_theta = 0, is_H_random_phase = 0,
                        # %%
                        # 生成横向结构
                        U1_name_Structure = '',
                        structure_size_Enlarge = 0.1,
                        is_phase_only_Structure = 0,
                        # %%
                        w0_Structure = 0, z_pump_Structure = 0,
                        is_LG_Structure = 0, is_Gauss_Structure = 0, is_OAM_Structure = 0,
                        l_Structure = 0, p_Structure = 0,
                        theta_x_Structure = 0, theta_y_Structure = 0,
                        # %%
                        is_random_phase_Structure = 0,
                        is_H_l_Structure = 0, is_H_theta_Structure = 0, is_H_random_phase_Structure = 0,
                        #%%
                        U1_0_NonZero_size = 1, w0 = 0,
                        z0 = 3, z0_new = 5,
                        # %%
                        lam1=0.8, is_air_pump=0, is_air=0, T=25,
                        deff=30, is_fft = 1, fft_mode = 0,
                        is_linear_convolution = 0,
                        # %%
                        Tx=10, Ty=10, Tz="2*lc",
                        mx=0, my=0, mz=0,
                        # %%
                        # 生成横向结构
                        Duty_Cycle_x = 0.5, Duty_Cycle_y = 0.5, Duty_Cycle_z = 0.5,
                        Depth = 2, structure_xy_mode = 'x',
                        is_continuous = 0, is_target_far_field = 1, is_transverse_xy = 0,
                        is_reverse_xy = 0, is_positive_xy = 1, is_no_backgroud = 0,
                        #%%
                        is_save = 0, is_save_txt = 0, dpi = 100,
                        #%%
                        cmap_2d = 'viridis',
                        #%%
                        ticks_num = 6, is_contourf = 0,
                        is_title_on = 1, is_axes_on = 1, is_mm = 1,
                        #%%
                        fontsize = 9,
                        font = {'family': 'serif',
                                'style': 'normal', # 'normal', 'italic', 'oblique'
                                'weight': 'normal',
                                'color': 'black', # 'black','gray','darkred'
                                },
                        #%%
                        is_colorbar_on = 1, is_energy = 1,
                        #%%
                        is_print = 1, is_contours = 66, n_TzQ = 1, Gz_max_Enhance = 1, match_mode = 1, )

# 注意 colorbar 上的数量级