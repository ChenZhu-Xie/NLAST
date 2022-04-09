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
from fun_os import img_squared_bordered_Read
from fun_img_Resize import image_Add_black_border
from fun_plot import plot_1d, plot_2d, plot_3d_XYZ, plot_3d_XYz
from b_1_AST import AST
from B_3_NLA_SSI import NLA_SSI
from B_3_SFM_SSI import SFM_SSI

def contours_NLA_SSI_AST(U1_name = "", 
                         img_full_name = "lena.png", 
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
                         #%%
                         U1_0_NonZero_size = 1, w0 = 0, 
                         z0_AST = 0.03, z0_NLA = 0.01, deff_structure_sheet_expect = 1, is_energy_evolution_on = 1, 
                         #%%
                         lam1 = 0.8, is_air_pump = 0, is_air = 0, T = 25, 
                         deff = 30, 
                         #%%
                         Tx = 10, Ty = 10, Tz = "2*lc", 
                         mx = 0, my = 0, mz = 0, 
                         is_NLAST = 0, 
                         #%%
                         is_save = 0, is_save_txt = 0, dpi = 100, 
                         #%%
                         color_1d = 'b', cmap_2d = 'viridis', 
                         #%%
                         sample = 2, ticks_num = 6, is_contourf = 0, 
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
                         is_print = 1, is_contours = 1, n_TzQ = 1,
                         Gz_max_Enhance = 1, match_mode = 1,
                         #%%
                         is_NLA = 1, ):
    
    #%%
    # 非线性 惠更斯 菲涅尔 原理
    
    image_Add_black_border(img_full_name, 
                           border_percentage, 
                           is_print, )
    
    #%%
    # 路径设定
    
    img_name, img_name_extension, img_squared, size_PerPixel, size_fig, I2_x, I2_y, U1_0 = img_squared_bordered_Read(img_full_name, 
                                                                                                                     U1_0_NonZero_size, dpi, 
                                                                                                                     is_phase_only)
    
    #%%
    # 先衍射 z0_AST 后倍频 z0_NLA
    
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
        z0_AST, 
        #%%
        lam1, is_air_pump, 1, T, 
        #%%
        is_save, is_save_txt, dpi, 
        #%%
        cmap_2d, 
        #%%
        ticks_num, is_contourf, 
        is_title_on, is_axes_on, is_mm,
        #%%
        fontsize, font, 
        #%%
        is_colorbar_on, is_energy, 
        #%%
        is_print, )
    
    U1_name = "6. AST - U1_" + str(float('%.2g' % z0_AST)) + "mm"
    # U1_full_name = U1_name + (is_save_txt and ".txt" or ".mat")
    # U1_short_name = U1_name.replace('6. AST - ', '')
    
    arg = [ U1_name, 
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
            z0_NLA, 0, 1, 
            deff_structure_sheet_expect, 10, 
            0, 0, 0, 0, 
            #%%
            1, 0, 
            0, 0, is_energy_evolution_on, 
            #%%
            lam1, is_air_pump, is_air, T, 
            deff, 
            #%%
            Tx, Ty, Tz, 
            mx, my, mz, 
            is_NLAST, 
            #%%
            is_save, is_save_txt, dpi, 
            #%%
            color_1d, cmap_2d, 'rainbow', 
            10, -65, 2, 
            #%%
            sample, ticks_num, is_contourf, 
            is_title_on, is_axes_on, is_mm,
            #%%
            fontsize, font, 
            #%%
            is_colorbar_on, is_energy,
            #%%
            is_print, is_contours, n_TzQ, 
            Gz_max_Enhance, match_mode, ]
    
    if is_NLA == 1:
        NLA_SSI(*arg)
    else:
        SFM_SSI(*arg)
    
    U1_NLA_txt_name = "6. NLA - U2_" + str(float('%.2g' % z0_NLA)) + "mm" + "_SSI"
    U1_NLA_txt_full_name = U1_NLA_txt_name + (is_save_txt and ".txt" or ".mat")
    # U1_NLA_txt_short_name = U1_NLA_txt_name.replace('6. NLA - ', '')
    U1_NLA_txt_short_name = U1_NLA_txt_name.replace('6. NLA - ', 'NLA - ')
    
    #%%
    # 先倍频 z0_NLA 后衍射 z0_AST
    
    arg = [ '', 
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
            z0_NLA, 0, 1, 
            deff_structure_sheet_expect, 10, 
            0, 0, 0, 0, 
            #%%
            1, 0, 
            0, 0, is_energy_evolution_on, 
            #%%
            lam1, is_air_pump, is_air, T, 
            deff, 
            #%%
            Tx, Ty, Tz, 
            mx, my, mz, 
            is_NLAST, 
            #%%
            is_save, is_save_txt, dpi, 
            #%%
            color_1d, cmap_2d, 'rainbow', 
            10, -65, 2, 
            #%%
            sample, ticks_num, is_contourf, 
            is_title_on, is_axes_on, is_mm,
            #%%
            fontsize, font, 
            #%%
            is_colorbar_on, is_energy,
            #%%
            is_print, is_contours, n_TzQ, 
            Gz_max_Enhance, match_mode, ]
    
    if is_NLA == 1:
        NLA_SSI(*arg)
    else:
        SFM_SSI(*arg)
    
    U2_txt_name = "6. NLA - U2_" + str(float('%.2g' % z0_NLA)) + "mm" + "_SSI"
    # U2_txt_full_name = U2_txt_name + (is_save_txt and ".txt" or ".mat")
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
        is_title_on, is_axes_on, is_mm,
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
    # 加和 U1_NLA 与 U2_AST = U2_Z0_Superposition
    
    Z0 = z0_AST + z0_NLA
    
    U1_NLA = np.loadtxt(U1_NLA_txt_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U1_NLA_txt_full_name)['U'] # 加载 复振幅场
    U2_AST = np.loadtxt(U2_AST_txt_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U2_AST_txt_full_name)['U'] # 加载 复振幅场
    
    U2_Z0_Superposition = U1_NLA + U2_AST
    
    U2_Z0_Superposition_amp = np.abs(U2_Z0_Superposition)
    # print(np.max(U2_Z0_Superposition_amp))
    U2_Z0_Superposition_phase = np.angle(U2_Z0_Superposition)

    print("NLAST - U2_{}mm.total_energy = {}".format(Z0, np.sum(U2_Z0_Superposition_amp**2)))

    if is_save == 1:
        if not os.path.isdir("6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_SSI"):
            os.makedirs("6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_SSI")

    #%%
    #绘图：U2_Z0_Superposition_amp

    U2_Z0_Superposition_amp_address = "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_SSI" + "\\" + "6.1. NLAST - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_SSI" + "_Superposition_amp" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_abs" + img_name_extension

    plot_2d([], 1, size_PerPixel, 
            U2_Z0_Superposition_amp, U2_Z0_Superposition_amp_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_SSI" + "_Superposition_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    #%%
    #绘图：U2_Z0_Superposition_phase

    U2_Z0_Superposition_phase_address = "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_SSI" + "\\" + "6.2. NLAST - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_SSI" + "_Superposition_phase" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_angle" + img_name_extension

    plot_2d([], 1, size_PerPixel, 
            U2_Z0_Superposition_phase, U2_Z0_Superposition_phase_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_SSI" + "_Superposition_phase", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)
    
    #%%
    # 储存 U2_Z0_Superposition 到 txt 文件

    U2_Z0_Superposition_full_name = "6. NLAST - U2_" + str(float('%.2g' % Z0)) + "mm" + "_SSI" + (is_save_txt and ".txt" or ".mat")
    if is_save == 1:
        U2_Z0_Superposition_txt_address = "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_SSI" + "\\" + U2_Z0_Superposition_full_name
        np.savetxt(U2_Z0_Superposition_txt_address, U2_Z0_Superposition) if is_save_txt else savemat(U2_Z0_Superposition_txt_address, {"U":U2_Z0_Superposition})

        #%%
        #再次绘图：U2_Z0_Superposition_amp
    
        U2_Z0_Superposition_amp_address = "6.1. NLAST - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_SSI" + "_Superposition_amp" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_abs" + img_name_extension
    
        plot_2d([], 1, size_PerPixel, 
                U2_Z0_Superposition_amp, U2_Z0_Superposition_amp_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_SSI" + "_Superposition_amp", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, is_energy, vmax, vmin)
        
        #再次绘图：U2_Z0_Superposition_phase
    
        U2_Z0_Superposition_phase_address = "6.2. NLAST - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_SSI" + "_Superposition_phase" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_angle" + img_name_extension
    
        plot_2d([], 1, size_PerPixel, 
                U2_Z0_Superposition_phase, U2_Z0_Superposition_phase_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_SSI" + "_Superposition_phase", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, 0, vmax, vmin)

    #%%
    # 储存 U2_Z0_Superposition 到 txt 文件

    # if is_save == 1:
    np.savetxt(U2_Z0_Superposition_full_name, U2_Z0_Superposition) if is_save_txt else savemat(U2_Z0_Superposition_full_name, {"U":U2_Z0_Superposition})
    
#%%

if __name__ == '__main__':
    contours_NLA_SSI_AST(U1_name = "",
                         img_full_name = "grating.png",
                         border_percentage = 0.3,
                         is_phase_only = 0,
                         #%%
                         z_pump = 0,
                         is_LG = 0, is_Gauss = 0, is_OAM = 0,
                         l = 0, p = 0,
                         theta_x = 0, theta_y = 0,
                         is_random_phase = 0,
                         is_H_l = 0, is_H_theta = 0, is_H_random_phase = 0,
                         #%%
                         U1_0_NonZero_size = 1, w0 = 0,
                         z0_AST = 0.21, z0_NLA = 0.1, deff_structure_sheet_expect = 0.2, is_energy_evolution_on = 1,
                         #%%
                         lam1 = 0.8, is_air_pump = 0, is_air = 0, T = 25,
                         deff = 30,
                         Tx = 10, Ty = 10, Tz = "2*lc",
                         mx = 0, my = 0, mz = 0,
                         is_NLAST = 0,
                         #%%
                         is_save = 0, is_save_txt = 0, dpi = 100,
                         #%%
                         color_1d = 'b', cmap_2d = 'viridis',
                         #%%
                         sample = 2, ticks_num = 6, is_contourf = 0,
                         is_title_on = 1, is_axes_on = 1, is_mm = 1,
                         #%%
                         fontsize = 6,
                         font = {'family': 'serif',
                                 'style': 'normal', # 'normal', 'italic', 'oblique'
                                 'weight': 'normal',
                                 'color': 'black', # 'black','gray','darkred'
                                 },
                         #%%
                         is_colorbar_on = 1, is_energy = 1,
                         #%%
                         is_print = 1, is_contours = 66, n_TzQ = 1,
                         Gz_max_Enhance = 1, match_mode = 1,
                         #%%
                         is_NLA = 1, )

# 搭配 - 1
# U2_Z0_Superposition = U1_NLA - U2_AST
# deff_structure_sheet_expect = 0.06、 0.15
# lam1 = 0.8，选择性极强
# z0_AST = 0.27、0.3，注： 0.22、0.25 不行

# 搭配 - 2
# U2_Z0_Superposition = U1_NLA - U2_AST
# deff_structure_sheet_expect = 0.05
# lam1 = 0.8，选择性极强
# z0_AST = 0.27，注： 0.25、0.3 不行

# 搭配 + 1
# U2_Z0_Superposition = U1_NLA + U2_AST
# deff_structure_sheet_expect = 1、0.2
# lam1 = 0.8。选择性极强
# z0_AST = 0.2~0.22、0.23~0.25、0.28~0.33、0.35~0.38、0.4，注： 0.22、0.26~0.27、0.34、0.39 不行

# 规律：如果 NLAST 的总能量 或 colorbar 的能量，倾向于 比 2 幅图的能量低 很多，则就 倾向于 描边了
# 只有 z0_AST 是 影响 描边的；z0_NLA 似乎 并不影响 描边，无论其是什么，都不影响：当 z0_AST 能描边时，无论是什么 z0_NLA，都能描；但若 z0_AST 不能描，则 无论 z0_NLA 是什么都不能描
# 与是什么图，无关，能描边的参数，换其他图 也能描边；不能描的参数，换其他图也不能描
# 2 次 AST 如果是在晶体中，似乎 无论如何 都 无法描边
