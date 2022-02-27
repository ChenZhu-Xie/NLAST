# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:57 2021

@author: Xcz
"""

#%%

import os
import cv2
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
from a_Image_Add_Black_border import Image_Add_Black_border
from b_1_AST import AST
from b_3_NLA import NLA
from scipy.io import loadmat, savemat
from fun_plot import plot_1d, plot_2d, plot_3d_XYZ, plot_3d_XYz

def Consistency_NLA_AST(U1_txt_name = "", 
                        file_full_name = "Grating.png", 
                        border_percentage = 0.3, 
                        phase_only = 0, 
                        #%%
                        is_LG = 0, is_Gauss = 0, is_OAM = 0, 
                        l = 0, p = 0, 
                        theta_x = 0, theta_y = 0, 
                        is_H_l = 0, is_H_theta = 0, 
                        #%%
                        U1_0_NonZero_size = 1, w0 = 0.3,
                        z0 = 1, z0_new = 5, 
                        #%%
                        lam1 = 0.8, is_air_pump = 0, is_air = 0, T = 25, 
                        deff = 30, is_linear_convolution = 0, 
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
                        is_energy = 1, vmax = 1, vmin = 0):
    
    #%%
    # 线性 惠更斯 菲涅尔 原理
    
    # Image_Add_Black_border(file_full_name, border_percentage)
    
    # #%%
    # # 先衍射 z0 后衍射 z0_new
    
    # AST('', 
    #     file_full_name, 
    #     phase_only, 
    #     #%%
    #     is_LG, is_Gauss, is_OAM, 
    #     l, p, 
    #     theta_x, theta_y, 
    #     is_H_l, is_H_theta, 
    #     #%%
    #     U1_0_NonZero_size, w0,
    #     z0, 
    #     #%%
    #     lam1, is_air_pump, is_air, T, 
    #     #%%
    #     is_save, is_save_txt, dpi, 
    #     #%%
    #     cmap_2d, 
    #     #%%
    #     ticks_num, is_contourf, 
    #     is_title_on, is_axes_on, 
    #     is_mm, is_propagation, 
    #     #%%
    #     fontsize, font, 
    #     #%%
    #     is_self_colorbar, is_colorbar_on, 
    #     is_energy, vmax, vmin)
    
    # U1_txt_name = "6. AST - U1_" + str(float('%.2g' % z0)) + "mm"
    # # U1_txt_full_name = U1_txt_name + (is_save_txt and ".txt" or ".mat")
    # # U1_txt_short_name = U1_txt_name.replace('6. AST - ', '')
    
    # AST(U1_txt_name, 
    #     file_full_name, 
    #     phase_only, 
    #     #%%
    #     is_LG, is_Gauss, is_OAM, 
    #     l, p, 
    #     theta_x, theta_y, 
    #     is_H_l, is_H_theta, 
    #     #%%
    #     U1_0_NonZero_size, w0,
    #     z0, 
    #     #%%
    #     lam1, is_air_pump, is_air, T, 
    #     #%%
    #     is_save, is_save_txt, dpi, 
    #     #%%
    #     cmap_2d, 
    #     #%%
    #     ticks_num, is_contourf, 
    #     is_title_on, is_axes_on, 
    #     is_mm, is_propagation, 
    #     #%%
    #     fontsize, font, 
    #     #%%
    #     is_self_colorbar, is_colorbar_on, 
    #     is_energy, vmax, vmin)
    
    # #%%
    # # 直接衍射 Z0 = z0 + z0_new
    
    # Z0 = z0 + z0_new
    
    # AST('', 
    #     file_full_name, 
    #     phase_only, 
    #     #%%
    #     is_LG, is_Gauss, is_OAM, 
    #     l, p, 
    #     theta_x, theta_y, 
    #     is_H_l, is_H_theta, 
    #     #%%
    #     U1_0_NonZero_size, w0,
    #     Z0, 
    #     #%%
    #     lam1, is_air_pump, is_air, T, 
    #     #%%
    #     is_save, is_save_txt, dpi, 
    #     #%%
    #     cmap_2d, 
    #     #%%
    #     ticks_num, is_contourf, 
    #     is_title_on, is_axes_on, 
    #     is_mm, is_propagation, 
    #     #%%
    #     fontsize, font, 
    #     #%%
    #     is_self_colorbar, is_colorbar_on, 
    #     is_energy, vmax, vmin)
    
    #%%
    # 非线性 惠更斯 菲涅尔 原理
    
    Image_Add_Black_border(file_full_name, border_percentage)
    
    #%%
    # 先衍射 z0 后倍频 z0_NLA
    
    AST("", 
        file_full_name, 
        phase_only, 
        #%%
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        is_H_l, is_H_theta, 
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
        is_energy, vmax, vmin)
    
    U1_txt_name = "6. AST - U1_" + str(float('%.2g' % z0)) + "mm"
    # U1_txt_full_name = U1_txt_name + (is_save_txt and ".txt" or ".mat")
    # U1_txt_short_name = U1_txt_name.replace('6. AST - ', '')
    
    z0_NLA = z0_new
    
    NLA(U1_txt_name, 
        file_full_name, 
        phase_only, 
        #%%
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        is_H_l, is_H_theta, 
        #%%
        U1_0_NonZero_size, w0,
        z0_NLA, 
        #%%
        lam1, is_air_pump, is_air, T, 
        deff, is_linear_convolution, 
        Tx, Ty, Tz, 
        mx, my, mz, 
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
        is_energy, vmax, vmin)
    
    U1_NLA_txt_name = "6. NLA - U2_" + str(float('%.2g' % z0_NLA)) + "mm"
    U1_NLA_txt_full_name = U1_NLA_txt_name + (is_save_txt and ".txt" or ".mat")
    # U1_NLA_txt_short_name = U1_NLA_txt_name.replace('6. NLA - ', '')
    U1_NLA_txt_short_name = U1_NLA_txt_name.replace('6. NLA - ', 'NLA - ')
    
    #%%
    # 先倍频 z0 后衍射 z0_AST
    
    NLA("", 
        file_full_name, 
        phase_only, 
        #%%
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        is_H_l, is_H_theta, 
        #%%
        U1_0_NonZero_size, w0,
        z0, 
        #%%
        lam1, is_air_pump, is_air, T, 
        deff, is_linear_convolution, 
        Tx, Ty, Tz, 
        mx, my, mz, 
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
        is_energy, vmax, vmin)
    
    U2_txt_name = "6. NLA - U2_" + str(float('%.2g' % z0)) + "mm"
    # U2_txt_full_name = U2_txt_name + (is_save_txt and ".txt" or ".mat")
    # U2_txt_short_name = U2_txt_name.replace('6. NLA - ', '')
    
    z0_AST = z0_new
    
    AST(U2_txt_name, 
        file_full_name, 
        phase_only, 
        #%%
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        is_H_l, is_H_theta, 
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
        is_mm, is_propagation, 
        #%%
        fontsize, font, 
        #%%
        is_self_colorbar, is_colorbar_on, 
        is_energy, vmax, vmin)
    
    U2_AST_txt_name = "6. AST - U2_" + str(float('%.2g' % z0_AST)) + "mm"
    U2_AST_txt_full_name = U2_AST_txt_name + (is_save_txt and ".txt" or ".mat")
    # U2_AST_txt_short_name = U2_AST_txt_name.replace('6. AST - ', '')
    U2_AST_txt_short_name = U2_AST_txt_name.replace('6. AST - ', 'AST - ')
    
    #%%
    # 直接倍频 Z0 = z0 + z0_new
    
    Z0 = z0 + z0_new
    
    NLA("", 
        file_full_name, 
        phase_only, 
        #%%
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        is_H_l, is_H_theta, 
        #%%
        U1_0_NonZero_size, w0,
        Z0, 
        #%%
        lam1, is_air_pump, is_air, T, 
        deff, is_linear_convolution, 
        Tx, Ty, Tz, 
        mx, my, mz, 
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
        is_energy, vmax, vmin)
    
    U2_Z0_txt_name = "6. NLA - U2_" + str(float('%.2g' % Z0)) + "mm"
    U2_Z0_txt_full_name = U2_Z0_txt_name + (is_save_txt and ".txt" or ".mat")
    # U2_Z0_txt_short_name = U2_Z0_txt_name.replace('6. NLA - ', '')
    
    U2_Z0 = np.loadtxt(U2_Z0_txt_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U2_Z0_txt_full_name)['U'] # 加载 复振幅场
    
    U2_Z0_amp = np.abs(U2_Z0)
    U2_Z0_phase = np.angle(U2_Z0)
    
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
    # 路径设定
    
    size_fig = U2_Z0_Superposition.shape[0] / dpi
    
    file_name = os.path.splitext(file_full_name)[0]
    file_name_extension = os.path.splitext(file_full_name)[1]
    
    location = os.path.dirname(os.path.abspath(__file__))
    file_squared_address = location + "\\" + "1." + file_name + "_squared" + file_name_extension
    
    img_squared = cv2.imdecode(np.fromfile(file_squared_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
    
    size_PerPixel = U1_0_NonZero_size / img_squared.shape[0] # Unit: mm / 个 每个 像素点 的 尺寸，相当于 △x = △y = △z
    I2_x, I2_y = U2_Z0.shape[0], U2_Z0.shape[1]

    #%%
    #绘图：U2_Z0_Superposition_amp

    U2_Z0_Superposition_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "\\" + "6.1. NLAST - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_amp" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_abs" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, 0, 
            U2_Z0_Superposition_amp, U2_Z0_Superposition_amp_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    #%%
    #绘图：U2_Z0_Superposition_phase

    U2_Z0_Superposition_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "\\" + "6.2. NLAST - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_phase" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_angle" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, 0, 
            U2_Z0_Superposition_phase, U2_Z0_Superposition_phase_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_phase", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)
    
    #%%
    # 储存 U2_Z0_Superposition 到 txt 文件

    U2_Z0_Superposition_full_name = "6. NLAST - U2_" + str(float('%.2g' % Z0)) + "mm" + (is_save_txt and ".txt" or ".mat")
    if is_save == 1:
        U2_Z0_Superposition_txt_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "\\" + U2_Z0_Superposition_full_name
        np.savetxt(U2_Z0_Superposition_txt_address, U2_Z0_Superposition) if is_save_txt else savemat(U2_Z0_Superposition_txt_address, {"U":U2_Z0_Superposition})

        #%%
        #再次绘图：U2_Z0_Superposition_amp
    
        U2_Z0_Superposition_amp_address = location + "\\" + "6.1. NLAST - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_amp" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_abs" + file_name_extension
    
        plot_2d(I2_x, I2_y, size_PerPixel, 0, 
                U2_Z0_Superposition_amp, U2_Z0_Superposition_amp_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_amp", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, is_energy, vmax, vmin)
        
        #再次绘图：U2_Z0_Superposition_phase
    
        U2_Z0_Superposition_phase_address = location + "\\" + "6.2. NLAST - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_phase" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_angle" + file_name_extension
    
        plot_2d(I2_x, I2_y, size_PerPixel, 0, 
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
    # 对比 U2_Z0_Superposition 与 U2_Z0 的 绝对误差 1
    
    U2_Z0_Superposition_error = U2_Z0_Superposition - U2_Z0
    
    U2_Z0_Superposition_error_amp = np.abs(U2_Z0_Superposition_error)
    U2_Z0_Superposition_error_phase = np.angle(U2_Z0_Superposition_error)
    
    # print("Plus - U2_{}mm_Superposition_error.total_amp = {}".format(Z0, np.sum(U2_Z0_Superposition_error_amp)))
    # print("Plus - U2_{}mm_Superposition_error.total_energy = {}".format(Z0, np.sum(U2_Z0_Superposition_error_amp**2)))
    # print("Plus - U2_{}mm_Superposition_error.rsd = {}".format(Z0, np.std(U2_Z0_Superposition_error_amp) / np.mean(U2_Z0_Superposition_error_amp) ))

    if is_save == 1:
        if not os.path.isdir("6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error"):
            os.makedirs("6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error")

    #%%
    #绘图：U2_Z0_Superposition_error_amp

    U2_Z0_Superposition_error_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error" + "\\" + "6.1. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error" + "_amp" + " = " + "U2_Z0_Superposition_substract_U2_Z0" + "_abs" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, 0, 
            U2_Z0_Superposition_error_amp, U2_Z0_Superposition_error_amp_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error" + "_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    #%%
    #绘图：U2_Z0_Superposition_error_phase

    U2_Z0_Superposition_error_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error" + "\\" + "6.2. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error" + "_phase" + " = " + "U2_Z0_Superposition_substract_U2_Z0" + "_angle" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, 0, 
            U2_Z0_Superposition_error_phase, U2_Z0_Superposition_error_phase_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error" + "_phase", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)
    
    #%%
    # 储存 U2_Z0_Superposition_error 到 txt 文件

    U2_Z0_Superposition_error_full_name = "6. Plus - U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error" + (is_save_txt and ".txt" or ".mat")
    if is_save == 1:
        U2_Z0_Superposition_error_txt_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error" + "\\" + U2_Z0_Superposition_error_full_name
        np.savetxt(U2_Z0_Superposition_error_txt_address, U2_Z0_Superposition_error) if is_save_txt else savemat(U2_Z0_Superposition_error_txt_address, {"U":U2_Z0_Superposition_error})

        #%%
        #再次绘图：U2_Z0_Superposition_error_amp
    
        U2_Z0_Superposition_error_amp_address = location + "\\" + "6.1. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error" + "_amp" + " = " + "U2_Z0_Superposition_substract_U2_Z0" + "_abs" + file_name_extension
    
        plot_2d(I2_x, I2_y, size_PerPixel, 0, 
                U2_Z0_Superposition_error_amp, U2_Z0_Superposition_error_amp_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error" + "_amp", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, is_energy, vmax, vmin)
    
        #再次绘图：U2_Z0_Superposition_error_phase
    
        U2_Z0_Superposition_error_phase_address = location + "\\" + "6.2. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error" + "_phase" + " = " + "U2_Z0_Superposition_substract_U2_Z0" + "_angle" + file_name_extension
    
        plot_2d(I2_x, I2_y, size_PerPixel, 0, 
                U2_Z0_Superposition_error_phase, U2_Z0_Superposition_error_phase_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error" + "_phase", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, 0, vmax, vmin)

    #%%
    # 储存 U2_Z0_Superposition_error 到 txt 文件

    # if is_save == 1:
    np.savetxt(U2_Z0_Superposition_error_full_name, U2_Z0_Superposition_error) if is_save_txt else savemat(U2_Z0_Superposition_error_full_name, {"U":U2_Z0_Superposition_error})
    
    #%%
    # 对比 U2_Z0_Superposition 与 U2_Z0 的 绝对误差 2
    
    U2_Z0_Superposition_amp_error = U2_Z0_Superposition_amp - U2_Z0_amp
    U2_Z0_Superposition_phase_error = U2_Z0_Superposition_phase - U2_Z0_phase
    
    # print("Plus - U2_{}mm_Superposition_amp_error.total_amp = {}".format(Z0, np.sum(U2_Z0_Superposition_amp_error)))
    # print("Plus - U2_{}mm_Superposition_amp_error.total_energy = {}".format(Z0, np.sum(U2_Z0_Superposition_amp_error**2)))
    # print("Plus - U2_{}mm_Superposition_amp_error.rsd = {}".format(Z0, np.std(U2_Z0_Superposition_amp_error) / np.mean(U2_Z0_Superposition_amp_error) ))
    
    # print("Plus - U2_{}mm_Superposition_phase_error.total_phase = {}".format(Z0, np.sum(U2_Z0_Superposition_phase_error)))
    # print("Plus - U2_{}mm_Superposition_phase_error.total_energy = {}".format(Z0, np.sum(U2_Z0_Superposition_phase_error**2)))
    # print("Plus - U2_{}mm_Superposition_phase_error.rsd = {}".format(Z0, np.std(U2_Z0_Superposition_phase_error) / np.mean(U2_Z0_Superposition_phase_error) ))

    if is_save == 1:
        if not os.path.isdir("6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error"):
            os.makedirs("6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error")

    #%%
    #绘图：U2_Z0_Superposition_amp_error

    U2_Z0_Superposition_amp_error_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error" + "\\" + "6.1. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_amp_error" + " = " + "U2_Z0_Superposition_abs__substract__U2_Z0_abs" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, 0, 
            U2_Z0_Superposition_amp_error, U2_Z0_Superposition_amp_error_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_amp_error", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    #%%
    #绘图：U2_Z0_Superposition_phase_error

    U2_Z0_Superposition_phase_error_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error" + "\\" + "6.2. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_phase_error" + " = " + "U2_Z0_Superposition_angle__substract__U2_Z0_angle" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, 0, 
            U2_Z0_Superposition_phase_error, U2_Z0_Superposition_phase_error_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_phase_error", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)

    if is_save == 1:
        #%%
        #再次绘图：U2_Z0_Superposition_amp_error
    
        U2_Z0_Superposition_amp_error_address = location + "\\" + "6.1. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_amp_error" + " = " + "U2_Z0_Superposition_abs__substract__U2_Z0_abs" + file_name_extension
    
        plot_2d(I2_x, I2_y, size_PerPixel, 0, 
                U2_Z0_Superposition_amp_error, U2_Z0_Superposition_amp_error_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_amp_error", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, is_energy, vmax, vmin)
    
        #再次绘图：U2_Z0_Superposition_phase_error
    
        U2_Z0_Superposition_phase_error_address = location + "\\" + "6.2. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_phase_error" + " = " + "U2_Z0_Superposition_angle__substract__U2_Z0_angle" + file_name_extension
    
        plot_2d(I2_x, I2_y, size_PerPixel, 0, 
                U2_Z0_Superposition_phase_error, U2_Z0_Superposition_phase_error_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_phase_error", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, 0, vmax, vmin)
    
    # #%%
    # # 对比 U2_Z0_Superposition 与 U2_Z0 的 绝对误差 的 相对误差
    
    # U2_Z0_Superposition_error_relative = (U2_Z0_Superposition - U2_Z0) / U2_Z0
    
    # # 扔掉 amp 偏离 amp 均值 3 倍于 总体 标准差 以外 的 数据，保留 剩下的 3 倍 以内的 数据。
    # U2_Z0_Superposition_error_relative_amp = np.abs(U2_Z0_Superposition_error_relative)
    # U2_Z0_Superposition_error_relative_phase = np.angle(U2_Z0_Superposition_error_relative)
    # U2_Z0_Superposition_error_relative_amp_mean = np.mean(U2_Z0_Superposition_error_relative_amp)
    # U2_Z0_Superposition_error_relative_amp_std = np.std(U2_Z0_Superposition_error_relative_amp)
    # U2_Z0_Superposition_error_relative_amp_trust = np.abs(U2_Z0_Superposition_error_relative_amp - U2_Z0_Superposition_error_relative_amp_mean) <= 3*U2_Z0_Superposition_error_relative_amp_std
    # U2_Z0_Superposition_error_relative = U2_Z0_Superposition_error_relative * U2_Z0_Superposition_error_relative_amp_trust.astype(np.int8)
    
    # U2_Z0_Superposition_error_relative_amp = np.abs(U2_Z0_Superposition_error_relative)
    # U2_Z0_Superposition_error_relative_phase = np.angle(U2_Z0_Superposition_error_relative)
    
    # print("Plus - U2_{}mm_Superposition_error_relative.total_amp = {}".format(Z0, np.sum(U2_Z0_Superposition_error_relative_amp)))
    # print("Plus - U2_{}mm_Superposition_error_relative.total_energy = {}".format(Z0, np.sum(U2_Z0_Superposition_error_relative_amp**2)))
    # print("Plus - U2_{}mm_Superposition_error_relative.rsd = {}".format(Z0, np.std(U2_Z0_Superposition_error_relative_amp) / np.mean(U2_Z0_Superposition_error_relative_amp) ))

    # if is_save == 1:
    #     if not os.path.isdir("6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error_relative"):
    #         os.makedirs("6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error_relative")

    # #%%
    # #绘图：U2_Z0_Superposition_error_relative_amp

    # U2_Z0_Superposition_error_relative_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error_relative" + "\\" + "6.1. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error_relative" + "_amp" + " = " + "U2_Z0_Superposition_substract_U2_Z0" + "_abs" + file_name_extension

    # plot_2d(I2_x, I2_y, size_PerPixel, 0, 
    #         U2_Z0_Superposition_error_relative_amp, U2_Z0_Superposition_error_relative_amp_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error_relative" + "_amp", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         1, is_colorbar_on, is_energy, vmax, vmin)

    # #%%
    # #绘图：U2_Z0_Superposition_error_relative_phase

    # U2_Z0_Superposition_error_relative_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error_relative" + "\\" + "6.2. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error_relative" + "_phase" + " = " + "U2_Z0_Superposition_substract_U2_Z0" + "_angle" + file_name_extension

    # plot_2d(I2_x, I2_y, size_PerPixel, 0, 
    #         U2_Z0_Superposition_error_relative_phase, U2_Z0_Superposition_error_relative_phase_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error_relative" + "_phase", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         1, is_colorbar_on, 0, vmax, vmin)
    
    # #%%
    # # 储存 U2_Z0_Superposition_error_relative 到 txt 文件

    # U2_Z0_Superposition_error_relative_full_name = "6. Plus - U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error_relative" + (is_save_txt and ".txt" or ".mat")
    # if is_save == 1:
    #     U2_Z0_Superposition_error_relative_txt_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error_relative" + "\\" + U2_Z0_Superposition_error_relative_full_name
    #     np.savetxt(U2_Z0_Superposition_error_relative_txt_address, U2_Z0_Superposition_error_relative) if is_save_txt else savemat(U2_Z0_Superposition_error_relative_txt_address, {"U2_Z0_Superposition_error_relative":U2_Z0_Superposition_error_relative})

    #     #%%
    #     #再次绘图：U2_Z0_Superposition_error_relative_amp
    
    #     U2_Z0_Superposition_error_relative_amp_address = location + "\\" + "6.1. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error_relative" + "_amp" + " = " + "U2_Z0_Superposition_substract_U2_Z0" + "_abs" + file_name_extension
    
    #     plot_2d(I2_x, I2_y, size_PerPixel, 0, 
    #             U2_Z0_Superposition_error_relative_amp, U2_Z0_Superposition_error_relative_amp_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error_relative" + "_amp", 
    #             is_save, dpi, size_fig,  
    #             cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #             fontsize, font,
    #             1, is_colorbar_on, is_energy, vmax, vmin)
    
    #     #再次绘图：U2_Z0_Superposition_error_relative_phase
    
    #     U2_Z0_Superposition_error_relative_phase_address = location + "\\" + "6.2. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error_relative" + "_phase" + " = " + "U2_Z0_Superposition_substract_U2_Z0" + "_angle" + file_name_extension
    
    #     plot_2d(I2_x, I2_y, size_PerPixel, 0, 
    #             U2_Z0_Superposition_error_relative_phase, U2_Z0_Superposition_error_relative_phase_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_error_relative" + "_phase", 
    #             is_save, dpi, size_fig,  
    #             cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #             fontsize, font,
    #             1, is_colorbar_on, 0, vmax, vmin)

    # #%%
    # # 储存 U2_Z0_Superposition_error_relative 到 txt 文件

    # # if is_save == 1:
    # np.savetxt(U2_Z0_Superposition_error_relative_full_name, U2_Z0_Superposition_error_relative) if is_save_txt else savemat(U2_Z0_Superposition_error_relative_full_name, {"U":U2_Z0_Superposition_error_relative})
    
    # #%%
    # # 对比 U2_Z0_Superposition 与 U2_Z0 的 相对误差
    
    # U2_Z0_Superposition_relative_error = U2_Z0_Superposition / U2_Z0
    
    # U2_Z0_Superposition_relative_error_amp = np.abs(U2_Z0_Superposition_relative_error)
    # U2_Z0_Superposition_relative_error_phase = np.angle(U2_Z0_Superposition_relative_error)
    
    # print("Plus - U2_{}mm_Superposition_relative_error.total_amp = {}".format(Z0, np.sum(U2_Z0_Superposition_relative_error_amp)))
    # print("Plus - U2_{}mm_Superposition_relative_error.total_energy = {}".format(Z0, np.sum(U2_Z0_Superposition_relative_error_amp**2)))
    # print("Plus - U2_{}mm_Superposition_relative_error.rsd = {}".format(Z0, np.std(U2_Z0_Superposition_relative_error_amp) / np.mean(U2_Z0_Superposition_relative_error_amp) ))

    # if is_save == 1:
    #     if not os.path.isdir("6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_relative_error"):
    #         os.makedirs("6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_relative_error")

    # #%%
    # #绘图：U2_Z0_Superposition_relative_error_amp

    # U2_Z0_Superposition_relative_error_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_relative_error" + "\\" + "6.1. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_relative_error" + "_amp" + " = " + "U2_Z0_Superposition_substract_U2_Z0" + "_abs" + file_name_extension

    # plot_2d(I2_x, I2_y, size_PerPixel, 0, 
    #         U2_Z0_Superposition_relative_error_amp, U2_Z0_Superposition_relative_error_amp_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_relative_error" + "_amp", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         1, is_colorbar_on, is_energy, vmax, vmin)

    # #%%
    # #绘图：U2_Z0_Superposition_relative_error_phase

    # U2_Z0_Superposition_relative_error_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_relative_error" + "\\" + "6.2. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_relative_error" + "_phase" + " = " + "U2_Z0_Superposition_substract_U2_Z0" + "_angle" + file_name_extension

    # plot_2d(I2_x, I2_y, size_PerPixel, 0, 
    #         U2_Z0_Superposition_relative_error_phase, U2_Z0_Superposition_relative_error_phase_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_relative_error" + "_phase", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         1, is_colorbar_on, 0, vmax, vmin)
    
    # #%%
    # # 储存 U2_Z0_Superposition_relative_error 到 txt 文件

    # U2_Z0_Superposition_relative_error_full_name = "6. Plus - U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_relative_error" + (is_save_txt and ".txt" or ".mat")
    # if is_save == 1:
    #     U2_Z0_Superposition_relative_error_txt_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_relative_error" + "\\" + U2_Z0_Superposition_relative_error_full_name
    #     np.savetxt(U2_Z0_Superposition_relative_error_txt_address, U2_Z0_Superposition_relative_error) if is_save_txt else savemat(U2_Z0_Superposition_relative_error_txt_address, {"U2_Z0_Superposition_relative_error":U2_Z0_Superposition_relative_error})

    #     #%%
    #     #再次绘图：U2_Z0_Superposition_relative_error_amp
    
    #     U2_Z0_Superposition_relative_error_amp_address = location + "\\" + "6.1. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_relative_error" + "_amp" + " = " + "U2_Z0_Superposition_substract_U2_Z0" + "_abs" + file_name_extension
    
    #     plot_2d(I2_x, I2_y, size_PerPixel, 0, 
    #             U2_Z0_Superposition_relative_error_amp, U2_Z0_Superposition_relative_error_amp_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_relative_error" + "_amp", 
    #             is_save, dpi, size_fig,  
    #             cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #             fontsize, font,
    #             1, is_colorbar_on, is_energy, vmax, vmin)
    
    #     #再次绘图：U2_Z0_Superposition_relative_error_phase
    
    #     U2_Z0_Superposition_relative_error_phase_address = location + "\\" + "6.2. Plus - " + "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_relative_error" + "_phase" + " = " + "U2_Z0_Superposition_substract_U2_Z0" + "_angle" + file_name_extension
    
    #     plot_2d(I2_x, I2_y, size_PerPixel, 0, 
    #             U2_Z0_Superposition_relative_error_phase, U2_Z0_Superposition_relative_error_phase_address, "U2_" + str(float('%.2g' % Z0)) + "mm" + "_Superposition_relative_error" + "_phase", 
    #             is_save, dpi, size_fig,  
    #             cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #             fontsize, font,
    #             1, is_colorbar_on, 0, vmax, vmin)

    # #%%
    # # 储存 U2_Z0_Superposition_relative_error 到 txt 文件

    # # if is_save == 1:
    # np.savetxt(U2_Z0_Superposition_relative_error_full_name, U2_Z0_Superposition_relative_error) if is_save_txt else savemat(U2_Z0_Superposition_relative_error_full_name, {"U":U2_Z0_Superposition_relative_error})
    
    #%%
    
Consistency_NLA_AST(U1_txt_name = "", 
                    file_full_name = "lena1.png", 
                    border_percentage = 0.1, 
                    phase_only = 0, 
                    #%%
                    is_LG = 0, is_Gauss = 0, is_OAM = 0, 
                    l = 0, p = 0, 
                    theta_x = 0, theta_y = 0, 
                    is_H_l = 0, is_H_theta = 0, 
                    #%%
                    U1_0_NonZero_size = 1, w0 = 0, 
                    z0 = 3, z0_new = 5, 
                    #%%
                    lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, 
                    deff = 30, is_linear_convolution = 0, 
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
                    is_self_colorbar = 1, is_colorbar_on = 1, 
                    is_energy = 1, vmax = 1, vmin = 0)

# 注意 colorbar 上的数量级