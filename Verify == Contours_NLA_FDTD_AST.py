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
from B_3_NLA_FDTD import NLA_FDTD
from scipy.io import loadmat, savemat
from fun_plot import plot_1d, plot_2d, plot_3d_XYZ, plot_3d_XYz

def Contours_NLA_FDTD_AST(U1_txt_name = "", 
                          file_full_name = "lena.png", 
                          border_percentage = 0.3, 
                          phase_only = 0, 
                          #%%
                          is_LG = 0, is_Gauss = 0, is_OAM = 0, 
                          l = 0, p = 0, 
                          theta_x = 0, theta_y = 0, 
                          is_H_l = 0, is_H_theta = 0, 
                          #%%
                          U1_0_NonZero_size = 1, w0 = 0, 
                          z0_AST = 0.03, z0_NLA = 0.01, deff_structure_sheet_expect = 1, is_energy_evolution_on = 1, 
                          #%%
                          lam1 = 0.8, is_air_pump = 0, is_air = 0, T = 25, 
                          deff = 30, 
                          Tx = 10, Ty = 10, Tz = "2*lc", 
                          mx = 0, my = 0, mz = 0, 
                          #%%
                          is_save = 0, is_save_txt = 0, dpi = 100, 
                          #%%
                          color_1d = 'b', cmap_2d = 'viridis', 
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
    # 非线性 惠更斯 菲涅尔 原理
    
    Image_Add_Black_border(file_full_name, border_percentage)
    
    #%%
    # 路径设定
    
    file_name = os.path.splitext(file_full_name)[0]
    file_name_extension = os.path.splitext(file_full_name)[1]
    
    location = os.path.dirname(os.path.abspath(__file__))
    file_squared_address = location + "\\" + "1." + file_name + "_squared" + file_name_extension
    file_squared_bordered_address = location + "\\" + "2." + file_name + "_squared" + "_bordered" + file_name_extension
    
    img_squared = cv2.imdecode(np.fromfile(file_squared_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
    img_squared_bordered = cv2.imdecode(np.fromfile(file_squared_bordered_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
    
    size_fig = img_squared_bordered.shape[0] / dpi
    size_PerPixel = U1_0_NonZero_size / img_squared.shape[0] # Unit: mm / 个 每个 像素点 的 尺寸，相当于 △x = △y = △z
    I2_x, I2_y = img_squared_bordered.shape[0], img_squared_bordered.shape[1]
    
    #%%
    # 定义 调制区域切片厚度 的 纵向实际像素、调制区域切片厚度 的 实际纵向尺寸

    if mz != 0: # 如过你想 让结构 提供 z 向倒格矢
        if deff_structure_sheet_expect >= 0.1 * Tz or deff_structure_sheet_expect <= 0 or (type(deff_structure_sheet_expect) != float and type(deff_structure_sheet_expect) != int): # 则 deff_structure_sheet_expect 不能超过 0.1 * Tz（以保持 良好的 占空比）
            deff_structure_sheet_expect = 0.1 * Tz # Unit: μm
    else:
        if deff_structure_sheet_expect >= 0.01 * 1 * 1000 or deff_structure_sheet_expect <= 0 or (type(deff_structure_sheet_expect) != float and type(deff_structure_sheet_expect) != int): # 则 deff_structure_sheet_expect 不能超过 0.01 * deff_structure_length_expect（以保持 良好的 精度）
            deff_structure_sheet_expect = 0.01 * 1 * 1000 # Unit: μm
            
    diz = deff_structure_sheet_expect / 1000 / size_PerPixel # Unit: mm
    # diz = int( deff_structure_sheet_expect / 1000 / size_PerPixel )
    deff_structure_sheet = diz * size_PerPixel # Unit: mm 调制区域切片厚度 的 实际纵向尺寸
    # print("deff_structure_sheet = {} mm".format(deff_structure_sheet))
    
    #%%
    # 定义 晶体 的 纵向实际像素、晶体 的 实际纵向尺寸
    
    Iz = z0_AST / size_PerPixel
    sheets_num = int(Iz // diz)
    Iz = sheets_num * diz
    z0_AST_real = Iz * size_PerPixel
    print("z0_AST_real = {} mm".format(z0_AST_real))
    
    #%%
    # 定义 晶体 的 纵向实际像素、晶体 的 实际纵向尺寸
    
    Iz = z0_NLA / size_PerPixel
    sheets_num = int(Iz // diz)
    Iz = sheets_num * diz
    z0_NLA_real = Iz * size_PerPixel
    print("z0_NLA_real = {} mm".format(z0_NLA_real))
    
    #%%
    # 先衍射 z0_AST(z0_AST_real) 后倍频 z0_NLA
    
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
        z0_AST_real, 
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
        is_energy, vmax, vmin)
    
    U1_txt_name = "6. AST - U1_" + str(float('%.2g' % z0_AST_real)) + "mm"
    # U1_txt_full_name = U1_txt_name + (is_save_txt and ".txt" or ".mat")
    # U1_txt_short_name = U1_txt_name.replace('6. AST - ', '')
    
    NLA_FDTD(U1_txt_name, 
             file_full_name, 
             phase_only, 
             #%%
             is_LG, is_Gauss, is_OAM, 
             l, p, 
             theta_x, theta_y, 
             is_H_l, is_H_theta, 
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
             Tx, Ty, Tz, 
             mx, my, mz, 
             #%%
             is_save, is_save_txt, dpi, 
             #%%
             color_1d, cmap_2d, 'rainbow', 
             10, -65, 2, 
             #%%
             ticks_num, is_contourf, 
             is_title_on, is_axes_on, 
             is_mm, is_propagation, 
             #%%
             fontsize, font, 
             #%%
             is_self_colorbar, is_colorbar_on, 
             is_energy, vmax, vmin)
    
    U1_NLA_txt_name = "6. NLA - U2_" + str(float('%.2g' % z0_NLA_real)) + "mm" + "_FDTD"
    U1_NLA_txt_full_name = U1_NLA_txt_name + (is_save_txt and ".txt" or ".mat")
    # U1_NLA_txt_short_name = U1_NLA_txt_name.replace('6. NLA - ', '')
    U1_NLA_txt_short_name = U1_NLA_txt_name.replace('6. NLA - ', 'NLA - ')
    
    #%%
    # 先倍频 z0_NLA 后衍射 z0_AST(z0_AST_real)
    
    NLA_FDTD('', 
             file_full_name, 
             phase_only, 
             #%%
             is_LG, is_Gauss, is_OAM, 
             l, p, 
             theta_x, theta_y, 
             is_H_l, is_H_theta, 
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
             Tx, Ty, Tz, 
             mx, my, mz, 
             #%%
             is_save, is_save_txt, dpi, 
             #%%
             color_1d, cmap_2d, 'rainbow', 
             10, -65, 2, 
             #%%
             ticks_num, is_contourf, 
             is_title_on, is_axes_on, 
             is_mm, is_propagation, 
             #%%
             fontsize, font, 
             #%%
             is_self_colorbar, is_colorbar_on, 
             is_energy, vmax, vmin)
    
    U2_txt_name = "6. NLA - U2_" + str(float('%.2g' % z0_NLA_real)) + "mm" + "_FDTD"
    # U2_txt_full_name = U2_txt_name + (is_save_txt and ".txt" or ".mat")
    # U2_txt_short_name = U2_txt_name.replace('6. NLA - ', '')
    
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
        z0_AST_real, 
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
        is_energy, vmax, vmin)
    
    U2_AST_txt_name = "6. AST - U2_" + str(float('%.2g' % z0_AST_real)) + "mm"
    U2_AST_txt_full_name = U2_AST_txt_name + (is_save_txt and ".txt" or ".mat")
    # U2_AST_txt_short_name = U2_AST_txt_name.replace('6. AST - ', '')
    U2_AST_txt_short_name = U2_AST_txt_name.replace('6. AST - ', 'AST - ')
    
    #%%
    # 加和 U1_NLA 与 U2_AST = U2_Z0_Superposition
    
    Z0 = z0_AST_real + z0_NLA_real + diz / 2 * size_PerPixel
    
    #%%
    # 定义 晶体 的 纵向实际像素、晶体 的 实际纵向尺寸
    
    Iz = Z0 / size_PerPixel
    sheets_num = int(Iz // diz)
    Iz = sheets_num * diz
    Z0_real = Iz * size_PerPixel
    print("Z0_real = {} mm".format(Z0_real))
    
    U1_NLA = np.loadtxt(U1_NLA_txt_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U1_NLA_txt_full_name)['U'] # 加载 复振幅场
    U2_AST = np.loadtxt(U2_AST_txt_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U2_AST_txt_full_name)['U'] # 加载 复振幅场
    
    U2_Z0_Superposition = U1_NLA + U2_AST
    
    U2_Z0_Superposition_amp = np.abs(U2_Z0_Superposition)
    # print(np.max(U2_Z0_Superposition_amp))
    U2_Z0_Superposition_phase = np.angle(U2_Z0_Superposition)

    print("NLAST - U2_{}mm.total_energy = {}".format(Z0_real, np.sum(U2_Z0_Superposition_amp**2)))

    if is_save == 1:
        if not os.path.isdir("6. U2_" + str(float('%.2g' % Z0_real)) + "mm" + "_FDTD"):
            os.makedirs("6. U2_" + str(float('%.2g' % Z0_real)) + "mm" + "_FDTD")

    #%%
    #绘图：U2_Z0_Superposition_amp

    U2_Z0_Superposition_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0_real)) + "mm" + "_FDTD" + "\\" + "6.1. NLAST - " + "U2_" + str(float('%.2g' % Z0_real)) + "mm" + "_FDTD" + "_Superposition_amp" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_abs" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, 0, 
            U2_Z0_Superposition_amp, U2_Z0_Superposition_amp_address, "U2_" + str(float('%.2g' % Z0_real)) + "mm" + "_FDTD" + "_Superposition_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    #%%
    #绘图：U2_Z0_Superposition_phase

    U2_Z0_Superposition_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0_real)) + "mm" + "_FDTD" + "\\" + "6.2. NLAST - " + "U2_" + str(float('%.2g' % Z0_real)) + "mm" + "_FDTD" + "_Superposition_phase" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_angle" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, 0, 
            U2_Z0_Superposition_phase, U2_Z0_Superposition_phase_address, "U2_" + str(float('%.2g' % Z0_real)) + "mm" + "_FDTD" + "_Superposition_phase", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)
    
    #%%
    # 储存 U2_Z0_Superposition 到 txt 文件

    U2_Z0_Superposition_full_name = "6. NLAST - U2_" + str(float('%.2g' % Z0_real)) + "mm" + "_FDTD" + (is_save_txt and ".txt" or ".mat")
    if is_save == 1:
        U2_Z0_Superposition_txt_address = location + "\\" + "6. U2_" + str(float('%.2g' % Z0_real)) + "mm" + "_FDTD" + "\\" + U2_Z0_Superposition_full_name
        np.savetxt(U2_Z0_Superposition_txt_address, U2_Z0_Superposition) if is_save_txt else savemat(U2_Z0_Superposition_txt_address, {"U":U2_Z0_Superposition})

        #%%
        #再次绘图：U2_Z0_Superposition_amp
    
        U2_Z0_Superposition_amp_address = location + "\\" + "6.1. NLAST - " + "U2_" + str(float('%.2g' % Z0_real)) + "mm" + "_FDTD" + "_Superposition_amp" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_abs" + file_name_extension
    
        plot_2d(I2_x, I2_y, size_PerPixel, 0, 
                U2_Z0_Superposition_amp, U2_Z0_Superposition_amp_address, "U2_" + str(float('%.2g' % Z0_real)) + "mm" + "_FDTD" + "_Superposition_amp", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, is_energy, vmax, vmin)
        
        #再次绘图：U2_Z0_Superposition_phase
    
        U2_Z0_Superposition_phase_address = location + "\\" + "6.2. NLAST - " + "U2_" + str(float('%.2g' % Z0_real)) + "mm" + "_FDTD" + "_Superposition_phase" + " = " + U1_NLA_txt_short_name + "_Plus" + "_" + U2_AST_txt_short_name + "_angle" + file_name_extension
    
        plot_2d(I2_x, I2_y, size_PerPixel, 0, 
                U2_Z0_Superposition_phase, U2_Z0_Superposition_phase_address, "U2_" + str(float('%.2g' % Z0_real)) + "mm" + "_FDTD" + "_Superposition_phase", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, 0, vmax, vmin)

    #%%
    # 储存 U2_Z0_Superposition 到 txt 文件

    # if is_save == 1:
    np.savetxt(U2_Z0_Superposition_full_name, U2_Z0_Superposition) if is_save_txt else savemat(U2_Z0_Superposition_full_name, {"U":U2_Z0_Superposition})
    
#%%
    
Contours_NLA_FDTD_AST(U1_txt_name = "", 
                      file_full_name = "grating.png", 
                      border_percentage = 0.3, 
                      phase_only = 0, 
                      #%%
                      is_LG = 0, is_Gauss = 0, is_OAM = 0, 
                      l = 0, p = 0, 
                      theta_x = 0, theta_y = 0, 
                      is_H_l = 0, is_H_theta = 0, 
                      #%%
                      U1_0_NonZero_size = 1, w0 = 0, 
                      z0_AST = 0.21, z0_NLA = 0.1, deff_structure_sheet_expect = 0.2, is_energy_evolution_on = 1,
                      #%%
                      lam1 = 0.8, is_air_pump = 0, is_air = 0, T = 25, 
                      deff = 30, 
                      Tx = 10, Ty = 10, Tz = "2*lc", 
                      mx = 0, my = 0, mz = 0, 
                      #%%
                      is_save = 0, is_save_txt = 0, dpi = 100, 
                      #%%
                      color_1d = 'b', cmap_2d = 'viridis', 
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
                      is_energy = 0, vmax = 1, vmin = 0)

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
