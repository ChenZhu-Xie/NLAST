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
from fun_plot import plot_1d, plot_2d, plot_3d_XYZ, plot_3d_XYz
from b_1_AST import AST
from b_3_NLA import NLA

def contours_NLA_AST(U1_name = "", 
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
                     z0_AST = 1, z0_NLA = 5, 
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
                     is_energy = 1, vmax = 1, vmin = 0, 
                     #%%
                     is_print = 1, is_contours = 1, n_TzQ = 1, 
                     Gz_max_Enhance = 1, match_mode = 1, ):
    
    #%%
    # 非线性 描边
    
    image_Add_black_border(img_full_name, 
                           border_percentage, 
                           is_print, )
    
    #%%
    # 先空气中 衍射 z0_AST，后晶体内 倍频 z0_NLA
    
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
        z0_AST, 
        #%%
        lam1, is_air_pump, 1, T, 
        #%%
        is_save, is_save_txt, dpi, 
        #%%
        cmap_2d, 
        #%%
        ticks_num, is_contourf, 
        is_title_on, is_axes_on, 
        is_mm, is_propagation, 
        #%%
        fontsize, font, 
        #%%
        is_self_colorbar, is_colorbar_on, 
        is_energy, vmax, vmin, 
        #%%
        is_print, )
    
    U1_name = "6. AST - U1_" + str(float('%.2g' % z0_AST)) + "mm"
    # U1_full_name = U1_name + ".txt"
    # U1_short_name = U1_name.replace('6. AST - ', '')
    
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
        is_mm, is_propagation, 
        #%%
        fontsize, font, 
        #%%
        is_self_colorbar, is_colorbar_on, 
        is_energy, vmax, vmin, 
        #%%
        is_print, is_contours, n_TzQ, 
        Gz_max_Enhance, match_mode, )
    
    U1_NLA_txt_name = "6. NLA - U2_" + str(float('%.2g' % z0_NLA)) + "mm"
    U1_NLA_txt_full_name = U1_NLA_txt_name + (is_save_txt and ".txt" or ".mat")
    # U1_NLA_txt_short_name = U1_NLA_txt_name.replace('6. NLA - ', '')
    U1_NLA_txt_short_name = U1_NLA_txt_name.replace('6. NLA - ', 'NLA - ')
    
    #%%
    # 先晶体内 倍频 z0_NLA，后空气中 衍射 z0_AST
    
    NLA('', 
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
        is_mm, is_propagation, 
        #%%
        fontsize, font, 
        #%%
        is_self_colorbar, is_colorbar_on, 
        is_energy, vmax, vmin, 
        #%%
        is_print, is_contours, n_TzQ, 
        Gz_max_Enhance, match_mode, )
    
    U2_txt_name = "6. NLA - U2_" + str(float('%.2g' % z0_NLA)) + "mm"
    # U2_txt_full_name = U2_txt_name + ".txt"
    # U2_txt_short_name = U2_txt_name.replace('6. NLA - ', '')
    
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
        lam1, is_air_pump, 1, T, 
        #%%
        is_save, is_save_txt, dpi, 
        #%%
        cmap_2d, 
        #%%
        ticks_num, is_contourf, 
        is_title_on, is_axes_on, 
        is_mm, is_propagation, 
        #%%
        fontsize, font, 
        #%%
        is_self_colorbar, is_colorbar_on, 
        is_energy, vmax, vmin, 
        #%%
        is_print, )
    
    U2_AST_txt_name = "6. AST - U2_" + str(float('%.2g' % z0_AST)) + "mm"
    # U2_AST_txt_full_name = U2_AST_txt_name + (is_save_txt and ".txt" or ".mat")
    # U2_AST_txt_short_name = U2_AST_txt_name.replace('6. AST - ', '')
    U2_AST_txt_short_name = U2_AST_txt_name.replace('6. AST - ', 'AST - ')
    
    #%%
    # 加和 U1_NLA 与 U2_AST = U2_Z0_Superposition
    
    Z0 = z0_AST + z0_NLA
    
    U1_NLA = np.loadtxt(U1_NLA_txt_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U1_NLA_txt_full_name)['U'] # 加载 复振幅场
    # U2_AST = np.loadtxt(U2_AST_txt_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U2_AST_txt_full_name)['U'] # 加载 复振幅场
    
    img_name, img_name_extension, img_squared, size_PerPixel, size_fig, I2_x, I2_y, U2_AST = U_Read(U2_AST_txt_name, img_full_name, 
                                                                                                    U1_0_NonZero_size, dpi, 
                                                                                                    is_save_txt, )
    
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
    
contours_NLA_AST(U1_name = "", 
                 img_full_name = "grating.png", 
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
                 U1_0_NonZero_size = 1, w0 = 0, 
                 z0_AST = 0.17, z0_NLA = 0.1, 
                 # %%
                 lam1=0.8, is_air_pump=0, is_air=0, T=25,
                 deff=30, is_fft = 1, fft_mode = 0, 
                 is_linear_convolution = 0,
                 #%%
                 Tx=10, Ty=10, Tz="2*lc",
                 mx=0, my=0, mz=0, # 可 mz = 1 查看匹配时 的 情况，理应效果 更好。
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
                 is_mm = 1, is_propagation = 0, 
                 #%%
                 fontsize = 6, 
                 font = {'family': 'serif',
                         'style': 'normal', # 'normal', 'italic', 'oblique'
                         'weight': 'normal',
                         'color': 'black', # 'black','gray','darkred'
                         }, 
                 #%%
                 is_self_colorbar = 1, is_colorbar_on = 1, 
                 is_energy = 0, vmax = 1, vmin = 0, 
                 #%%
                 is_print = 1, is_contours = 1, n_TzQ = 1, Gz_max_Enhance = 1, match_mode = 1, )

# 搭配 - 1
# U2_Z0_Superposition = U1_NLA - U2_AST
# lam1 = 0.8~0.80001，选择性极强；0.8001~0.8002 能让图片更清晰，能量分布与数量级与图片差不多；0.801 就差不多是 原图 2 倍能量了，类似相加；
# 0.80193 又回到了 描边；0.802 又回到了 能让图片 更清晰，并且又与 z0_AST 或 z0_NLA 无关，一直保持描边？不，0.5, 0.3 是描边，但 0.2, 0.1 又不是了
# z0_AST = 0.2、0.25、0.3、0.35、0.4、0.43、0.47、0.5, z0_NLA = 0.1
# z0_AST = 0.2, z0_NLA = 0.1、0.15、0.2、0.25、0.28、0.3
# z0_AST = 0.5, z0_NLA = 0.1、0.15、0.2、0.25、0.28、0.3

# 搭配 + 1
# U2_Z0_Superposition = U1_NLA + U2_AST
# lam1 = 0.80095，选择性极强
# z0_AST = 0.5， z0_NLA = 0.1、0.15、0.28、0.3 都行
# z0_AST = 0.2， z0_NLA = 0.1、0.15、0.3 都不行
# z0_AST = 0.17，z0_NLA = 0.1、0.2、0.3 都行

# 规律：如果 NLAST 的总能量 或 colorbar 的能量，倾向于 比 2 幅图的能量低 很多，则就 倾向于 描边了
# 只有 z0_AST 是 影响 描边的；z0_NLA 似乎 并不影响 描边，无论其是什么，都不影响
# 与是什么图，无关，能描边的参数，换其他图 也能描边；不能描的参数，换其他图也不能描
# 2 次 AST 如果是在晶体中，似乎 无论如何 都 无法描边