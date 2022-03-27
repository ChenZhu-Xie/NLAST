# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:57 2021

@author: Xcz
"""

#%%

import os
import math
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
from scipy.io import savemat
from fun_img_Resize import image_Add_black_border
from fun_os import img_squared_bordered_Read, U_Read
from fun_plot import plot_2d
from fun_pump import pump_LG
from fun_linear import Cal_n
from b_1_AST import AST

def interference_AST__AST  (U1_name = "", 
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
                            #%%
                            U1_0_NonZero_size = 1, w0 = 0.3,
                            z0 = 1, z0_delay_expect = 5, 
                            #%%
                            lam1 = 0.8, is_air_pump = 0, is_air = 0, T = 25, 
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
                            is_print = 1, ):
    
    #%%
    
    image_Add_black_border(img_full_name, 
                           border_percentage, 
                           is_print, )
    
    #%%
    # 获取 size_PerPixel，方便 后续计算 n1, k1，以及 为了生成 U1_0 和 g1_shift
    
    if (type(U1_name) != str) or U1_name == "":
        
        #%%
        # 导入 方形，以及 加边框 的 图片
        
        img_name, img_name_extension, img_squared, size_PerPixel, size_fig, I1_x, I1_y, U1_0 = img_squared_bordered_Read(img_full_name, 
                                                                                                                         U1_0_NonZero_size, dpi, 
                                                                                                                         is_phase_only)
        
        #%%
        # 预处理 输入场
        
        n1, k1 = Cal_n(size_PerPixel, 
                       is_air_pump, 
                       lam1, T, p = "e")
        
        U1_0 = pump_LG(img_full_name, 
                       I1_x, I1_y, size_PerPixel, 
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
        # 导入 方形 的 图片，以及 U
        
        img_name, img_name_extension, img_squared, size_PerPixel, size_fig, I1_x, I1_y, U1_0 = U_Read(U1_name, img_full_name, 
                                                                                                      U1_0_NonZero_size, dpi, 
                                                                                                      is_save_txt, )
        
    #%%
    # 线性 角谱理论 - 基波 begin

    g1 = np.fft.fft2(U1_0)
    g1_shift = np.fft.fftshift(g1)
    
    #%%
    # 先以 1 衍射 z0_1 后 以 n 衍射 z0_n
    
    G1_z0_shift, U1_z0 =    AST('', 
                                img_full_name, 
                                is_phase_only, 
                                #%%
                                z_pump, 
                                is_LG, is_Gauss, is_OAM, 
                                l, p, 
                                theta_x, theta_y, 
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
                                is_mm, is_propagation, 
                                #%%
                                fontsize, font, 
                                #%%
                                is_self_colorbar, is_colorbar_on, 
                                is_energy, vmax, vmin, 
                                #%%
                                is_print, )
    
    # U1_name = "6. AST - U1_" + str(float('%.2g' % z0)) + "mm"
    # U1_full_name = U1_name + ".txt"
    # U1_short_name = U1_name.replace('6. AST - ', '')
    
    #%%
    
    #%%
    
    if U1_name.find("U2") != -1: # 如果找到了 U2 字样
        lam1 = lam1 / 2

    n1, k1 = Cal_n(size_PerPixel, 
                   is_air, 
                   lam1, T, p = "e")
    
    #%%
    
    z0_delay_min = math.pi / (k1 / size_PerPixel)
    
    is_print and print("z0_delay_min = {} mm".format(z0_delay_min))
    
    delay_min_nums = z0_delay_expect // z0_delay_min
    
    delay_min_nums_odd = delay_min_nums + np.mod(delay_min_nums + 1, 2)
    
    z0_delay = z0_delay_min * delay_min_nums_odd
    
    is_print and print("z0_delay = {} mm".format(z0_delay))
    
    z1 = z0 + z0_delay
    
    #%%
    
    G1_z1_shift, U1_z1 =    AST('', 
                                img_full_name, 
                                is_phase_only, 
                                #%%
                                z_pump, 
                                is_LG, is_Gauss, is_OAM, 
                                l, p, 
                                theta_x, theta_y, 
                                is_random_phase, 
                                is_H_l, is_H_theta, is_H_random_phase, 
                                #%%
                                U1_0_NonZero_size, w0,
                                z1, 
                                #%%
                                lam1, is_air_pump, is_air, T, 
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
    
    # U1_name = "6. AST - U1_" + str(float('%.2g' % z1)) + "mm"
    
    #%%
    # H
    
    U1_z0_superposition = U1_z0 + U1_z1
    
    G1_z0_superposition = np.fft.fft2(U1_z0_superposition)
    G1_z0_superposition_shift = np.fft.fftshift(G1_z0_superposition)
    
    # H1_z0_superposition_shift = G1_z0_superposition_shift / g1_shift
    H1_z0_superposition_shift = G1_z0_superposition_shift / G1_z0_shift
    
    H1_z0_superposition_shift_amp = np.abs(H1_z0_superposition_shift)
    H1_z0_superposition_shift_phase = np.angle(H1_z0_superposition_shift)

    if is_save == 1:
        if not os.path.isdir("4. H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition_shift"):
            os.makedirs("4. H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition_shift")

    #%%
    #绘图：H1_z0_superposition_shift_amp

    H1_z0_superposition_shift_amp_address = "4. H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition_shift" + "\\" + "4.1. AST - " + "H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition_shift_amp" + img_name_extension

    plot_2d([], 1, size_PerPixel, 
            H1_z0_superposition_shift_amp, H1_z0_superposition_shift_amp_address, "H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition_shift_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    #%%
    #绘图：H1_z0_superposition_shift_phase

    H1_z0_superposition_shift_phase_address = "4. H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition_shift" + "\\" + "4.2. AST - " + "H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition_shift_phase" + img_name_extension

    plot_2d([], 1, size_PerPixel, 
            H1_z0_superposition_shift_phase, H1_z0_superposition_shift_phase_address, "H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition_shift_phase", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)
    
    #%%
    # 储存 H1_z0_superposition_shift 到 txt 文件

    if is_save == 1:
        H1_z0_superposition_shift_full_name = "4. AST - H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition_shift" + (is_save_txt and ".txt" or ".mat")
        H1_z0_superposition_shift_txt_address = "4. H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition_shift" + "\\" + H1_z0_superposition_shift_full_name
        np.savetxt(H1_z0_superposition_shift_txt_address, H1_z0_superposition_shift) if is_save_txt else savemat(H1_z0_superposition_shift_txt_address, {'H':H1_z0_superposition_shift})
    
    #%%
    # G
    
    G1_z0_superposition_shift_amp = np.abs(G1_z0_superposition_shift)
    G1_z0_superposition_shift_phase = np.angle(G1_z0_superposition_shift)

    if is_save == 1:
        if not os.path.isdir("5. G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_shift"):
            os.makedirs("5. G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_shift")

    #%%
    #绘图：G1_z0_superposition_shift_amp

    G1_z0_superposition_shift_amp_address = "5. G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_shift" + "\\" + "5.1. AST - " + "G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_shift_amp" + img_name_extension

    plot_2d([], 1, size_PerPixel, 
            G1_z0_superposition_shift_amp, G1_z0_superposition_shift_amp_address, "G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_shift_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    #%%
    #绘图：G1_z0_superposition_shift_phase

    G1_z0_superposition_shift_phase_address = "5. G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_shift" + "\\" + "5.2. AST - " + "G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_shift_phase" + img_name_extension

    plot_2d([], 1, size_PerPixel, 
            G1_z0_superposition_shift_phase, G1_z0_superposition_shift_phase_address, "G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_shift_phase", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)
    
    #%%
    # 储存 G1_z0_superposition_shift 到 txt 文件

    if is_save == 1:
        G1_z0_superposition_shift_full_name = "5. AST - G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_shift" + (is_save_txt and ".txt" or ".mat")
        G1_z0_superposition_shift_txt_address = "5. G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_shift" + "\\" + G1_z0_superposition_shift_full_name
        np.savetxt(G1_z0_superposition_shift_txt_address, G1_z0_superposition_shift) if is_save_txt else savemat(G1_z0_superposition_shift_txt_address, {'G':G1_z0_superposition_shift})
    
    #%%
    # U
    
    U1_z0_superposition_amp = np.abs(U1_z0_superposition)
    U1_z0_superposition_phase = np.angle(U1_z0_superposition)
    
    print("AST - U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_{}mm_superposition.total_energy = {}".format(z0, np.sum(U1_z0_superposition_amp**2))) # print  不舍精度，save 却舍精度...

    if is_save == 1:
        if not os.path.isdir("6. U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition"):
            os.makedirs("6. U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition")

    #%%
    #绘图：U1_z0_superposition_amp

    U1_z0_superposition_amp_address = "6. U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "\\" + "6.1. AST - " + "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_amp" + img_name_extension

    plot_2d([], 1, size_PerPixel, 
            U1_z0_superposition_amp, U1_z0_superposition_amp_address, "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_amp", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    #%%
    #绘图：U1_z0_superposition_phase

    U1_z0_superposition_phase_address = "6. U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "\\" + "6.2. AST - " + "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_phase" + img_name_extension

    plot_2d([], 1, size_PerPixel, 
            U1_z0_superposition_phase, U1_z0_superposition_phase_address, "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_phase", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)
    
    #%%
    # 储存 U1_z0_superposition 到 txt 文件

    U1_z0_superposition_full_name = "6. AST - U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + (is_save_txt and ".txt" or ".mat")
    if is_save == 1:
        U1_z0_superposition_txt_address = "6. U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "\\" + U1_z0_superposition_full_name
        np.savetxt(U1_z0_superposition_txt_address, U1_z0_superposition) if is_save_txt else savemat(U1_z0_superposition_txt_address, {'U':U1_z0_superposition})

        #%%
        #再次绘图：U1_z0_superposition_amp
    
        U1_z0_superposition_amp_address = "6.1. AST - " + "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_amp" + img_name_extension
    
        plot_2d([], 1, size_PerPixel, 
                U1_z0_superposition_amp, U1_z0_superposition_amp_address, "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_amp", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, is_energy, vmax, vmin)
    
        #再次绘图：U1_z0_superposition_phase
    
        U1_z0_superposition_phase_address = "6.2. AST - " + "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_phase" + img_name_extension
    
        plot_2d([], 1, size_PerPixel, 
                U1_z0_superposition_phase, U1_z0_superposition_phase_address, "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_superposition" + "_phase", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, 0, vmax, vmin)

    #%%
    # 储存 U1_z0_superposition 到 txt 文件

    # if is_save == 1:
    np.savetxt(U1_z0_superposition_full_name, U1_z0_superposition) if is_save_txt else savemat(U1_z0_superposition_full_name, {'U':U1_z0_superposition})
    
    #%%
    
interference_AST__AST  (U1_name = "", 
                        img_full_name = "grating.png", 
                        border_percentage = 0.1, 
                        is_phase_only = 0, 
                        #%%
                        z_pump = 0, 
                        is_LG = 0, is_Gauss = 0, is_OAM = 0, 
                        l = 0, p = 0, 
                        theta_x = 0, theta_y = 0, 
                        is_random_phase = 0, 
                        is_H_l = 0, is_H_theta = 0, is_H_random_phase = 0, 
                        #%%
                        U1_0_NonZero_size = 1, w0 = 0, # 传递函数 是 等倾干涉图...
                        z0 = 0, z0_delay_expect = 0, #  z0 越大，描边能量不变，但会越糊；z0_delay_expect 越大，描边 能量越高，但也越糊
                        #%%
                        lam1 = 1, is_air_pump = 0, is_air = 0, T = 25, 
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
                        is_print = 1, )