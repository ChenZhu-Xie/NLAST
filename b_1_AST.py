# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

#%%

import os
import cv2
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import math
# import copy
from PIL import Image
Image.MAX_IMAGE_PIXELS = 10E10 #Image 的 默认参数 无法处理那么大的图片
from n_dispersion import LN_n, KTP_n
# import scipy
from scipy.io import loadmat, savemat
from fun_plot import plot_1d, plot_2d, plot_3d_XYZ, plot_3d_XYz
from fun_pump import pump_LG

#%%

def AST(U1_txt_name = "", 
        file_full_name = "Grating.png", 
        phase_only = 0, 
        #%%
        is_LG = 0, is_Gauss = 0, is_OAM = 0, 
        l = 0, p = 0, 
        theta_x = 0, theta_y = 0, 
        is_H_l = 0, is_H_theta = 0, 
        #%%
        U1_0_NonZero_size = 1, w0 = 0.3,
        z0 = 1, 
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
        vmax = 1, vmin = 0):
    
    # #%%
    # U1_txt_name = ""
    # file_full_name = "lena.png"
    # #%%
    # phase_only = 0
    # is_LG, is_Gauss, is_OAM = 0, 1, 1
    # l, p = 1, 0
    # theta_x, theta_y = 1, 0
    # is_H_l, is_H_theta = 0, 0
    # # 正空间：右，下 = +, +
    # # 倒空间：左, 上 = +, +
    # # 朝着 x, y 轴 分别偏离 θ_1_x, θ_1_y 度
    # #%%
    # U1_0_NonZero_size = 1 # Unit: mm 不包含边框，图片 的 实际尺寸
    # w0 = 0.5 # Unit: mm 束腰（z = 0 处）
    # z0 = 10 # Unit: mm 传播距离
    # # size_modulate = 1e-3 # Unit: mm χ2 调制区域 的 横向尺寸，即 公式中的 d
    # #%%
    # lam1 = 1.064 # Unit: um 基波 或 倍频波长
    # is_air, T = 0, 25 # is_air = 0, 1, 2 分别表示 LN, 空气, KTP；T 表示 温度
    # #%%
    # is_save = 0
    # is_save_txt = 0
    # dpi = 100
    # #%%
    # cmap_2d='viridis'
    # # cmap_2d.set_under('black')
    # # cmap_2d.set_over('red')
    # #%%
    # ticks_num = 6 # 不包含 原点的 刻度数，也就是 区间数（植数问题）
    # is_contourf = 0
    # is_title_on, is_axes_on = 1, 1
    # is_mm, is_propagation = 1, 0
    # #%%
    # fontsize = 9
    # font = {'family': 'serif',
    #         'style': 'normal', # 'normal', 'italic', 'oblique'
    #         'weight': 'normal',
    #         'color': 'black', # 'black','gray','darkred'
    #         }
    # #%%
    # is_self_colorbar, is_colorbar_on = 0, 1 # vmax 与 vmin 是否以 自己的 U 的 最大值 最小值 为 相应的值；是，则覆盖设定；否的话，需要自己设定。
    # vmax, vmin = 1, 0
    
    if (type(U1_txt_name) != str) or U1_txt_name == "":
        #%%
        # 导入 方形，以及 加边框 的 图片
    
        file_name = os.path.splitext(file_full_name)[0]
        file_name_extension = os.path.splitext(file_full_name)[1]
    
        location = os.path.dirname(os.path.abspath(__file__))
        file_squared_address = location + "\\" + "1." + file_name + "_squared" + file_name_extension
        file_squared_bordered_address = location + "\\" + "2." + file_name + "_squared" + "_bordered" + file_name_extension
    
        img_squared = cv2.imdecode(np.fromfile(file_squared_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
        img_squared_bordered = cv2.imdecode(np.fromfile(file_squared_bordered_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
    
        size_fig = img_squared_bordered.shape[0] / dpi
    
        #%%
        # 线性 角谱理论 - 基波 begin
    
        size_PerPixel = U1_0_NonZero_size / img_squared.shape[0] # Unit: mm / 个 每个 像素点 的 尺寸，相当于 △x = △y = △z
        I1_x, I1_y = img_squared_bordered.shape[0], img_squared_bordered.shape[1]
        U1_size = I1_x * size_PerPixel # Unit: mm 包含 边框 后，图片 的 实际尺寸
        print("U1_size = U2_size = {} mm".format(U1_size))
        # print("U1_size = {} mm".format(U1_size))
        # print("%f mm" %(U1_size))
    
        #%%
        # U1_0 = U(x, y, 0) = img_squared_bordered
    
        # I_img_squared_bordered = np.empty([I1_x, I1_y], dtype=np.uint64)
        # I_img_squared_bordered = copy.deepcopy(img_squared_bordered) # 但这 深拷贝 也不行，因为把 最底层的 数据类型 uint8 也拷贝了，我 tm 无语，不如直接 astype 算了
        if phase_only == 1:
            U1_0 = np.power(math.e, (img_squared_bordered.astype(np.complex128()) / 255 * 2 * math.pi - math.pi) * 1j) # 变成相位图
        else:
            U1_0 = img_squared_bordered.astype(np.complex128)
        
        #%%
        # 预处理 输入场
        
        if is_air_pump == 1:
            n1 = 1
        elif is_air_pump == 0:
            n1 = LN_n(lam1, T, "e")
        else:
            n1 = KTP_n(lam1, T, "e")

        k1 = 2 * math.pi * size_PerPixel / (lam1 / 1000 / n1) # lam / 1000 即以 mm 为单位
        
        U1_0 = pump_LG(file_full_name, 
                       I1_x, I1_y, size_PerPixel, 
                       U1_0, w0, k1, 0, 
                       is_LG, is_Gauss, is_OAM, 
                       l, p, 
                       theta_x, theta_y, 
                       is_H_l, is_H_theta, 
                       is_save, is_save_txt, dpi, 
                       cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                       fontsize, font, 
                       1, is_colorbar_on, vmax, vmin) 
        
    else:
    
        #%%
        # 导入 方形，以及 加边框 的 图片
        
        U1_txt_full_name = U1_txt_name + (is_save_txt and ".txt" or ".mat")
        U1_txt_short_name = U1_txt_name.replace('6. AST - ', '')
        
        file_name = os.path.splitext(file_full_name)[0]
        file_name_extension = os.path.splitext(file_full_name)[1]

        location = os.path.dirname(os.path.abspath(__file__))
        file_squared_address = location + "\\" + "1." + file_name + "_squared" + file_name_extension

        img_squared = cv2.imdecode(np.fromfile(file_squared_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
        U1_0 = np.loadtxt(U1_txt_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U1_txt_full_name)['U'] # 加载 复振幅场

        size_fig = U1_0.shape[0] / dpi

        #%%
        # 线性 角谱理论 - 基波 begin

        size_PerPixel = U1_0_NonZero_size / img_squared.shape[0] # Unit: mm / 个 每个 像素点 的 尺寸，相当于 △x = △y = △z
        I1_x, I1_y = U1_0.shape[0], U1_0.shape[1]
        # U1_size = I1_x * size_PerPixel # Unit: mm 包含 边框 后，图片 的 实际尺寸
        # print("U1_size = U2_size = {} mm".format(U1_size))
        # print("U1_size = {} mm".format(U1_size))
        # print("%f mm" %(U1_size))
        
    #%%
    
    if U1_txt_name.find("U2") != -1: # 如果找到了 U2 字样
        lam1 = lam1 / 2

    if is_air == 1:
        n1 = 1
    elif is_air == 0:
        n1 = LN_n(lam1, T, "e")
    else:
        n1 = KTP_n(lam1, T, "e")

    k1 = 2 * math.pi * size_PerPixel / (lam1 / 1000 / n1) # lam / 1000 即以 mm 为单位

    #%%
    # U1_0 = U(x, y, 0) → FFT2 → g1_shift(k1_x, k1_y) = g1_shift

    g1 = np.fft.fft2(U1_0)

    g1_shift = np.fft.fftshift(g1)

    g1_shift_amp = np.abs(g1_shift)
    # print(np.max(g1_shift_amp))
    g1_shift_phase = np.angle(g1_shift)

    if is_save == 1:
        if not os.path.isdir("3. g" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_shift"):
            os.makedirs("3. g" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_shift")
     
    # #%%
    # #绘图：g1_shift_amp

    # g1_shift_amp_address = location + "\\" + "3. g" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_shift" + "\\" + "3.1. AST - " + "g" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_shift" + "_amp" + file_name_extension

    # plot_2d(I1_x, I1_y, size_PerPixel, 0, 
    #         g1_shift_amp, g1_shift_amp_address, "g" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_shift" + "_amp", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         1, is_colorbar_on, vmax, vmin)

    # #%%
    # #绘图：g1_shift_phase

    # g1_shift_phase_address = location + "\\" + "3. g" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_shift" + "\\" + "3.1. AST - " + "g" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_shift" + "_phase" + file_name_extension

    # plot_2d(I1_x, I1_y, size_PerPixel, 0, 
    #         g1_shift_phase, g1_shift_phase_address, "g" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_shift" + "_phase", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         1, is_colorbar_on, vmax, vmin)
    
    #%%
    # 储存 g1_shift 到 txt 文件

    if is_save == 1:
        g1_shift_full_name = "3. AST - g" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_shift" + (is_save_txt and ".txt" or ".mat")
        g1_shift_txt_address = location + "\\" + "3. g" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_shift" + "\\" + g1_shift_full_name
        np.savetxt(g1_shift_txt_address, g1_shift) if is_save_txt else savemat(g1_shift_txt_address, {"g":g1_shift})
        
    #%%
    # g1_shift = { g1_shift(k1_x, k1_y) } → 每个元素，乘以，频域 传递函数 e^{i*k1_z*z0} → G1_z0(k1_x, k1_y) = G1_z0

    z1_0 = z0
    i1_z0 = z1_0 / size_PerPixel

    n1_x, n1_y = np.meshgrid([i for i in range(I1_x)], [j for j in range(I1_y)])
    Mesh_n1_x_n1_y = np.dstack((n1_x, n1_y))
    Mesh_n1_x_n1_y_shift = Mesh_n1_x_n1_y - (I1_x // 2, I1_y // 2)
    Mesh_k1_x_k1_y_shift = np.dstack((2 * math.pi * Mesh_n1_x_n1_y_shift[:, :, 0] / I1_x, 2 * math.pi * Mesh_n1_x_n1_y_shift[:, :, 1] / I1_y))

    # k = 2 * math.pi * n / lam # Unit: 1 / um 并不在 计算 中 使用
    # lam_pixels = lam / 1000 / size_PerPixel # Unit: pixels 一个波长内，占有几个像素点；本身可在 直接计算中使用，且用起来更方便，但为了形式对称，弃用了之

    k1_z_shift = (k1**2 - np.square(Mesh_k1_x_k1_y_shift[:, :, 0]) - np.square(Mesh_k1_x_k1_y_shift[:, :, 1]) + 0j )**0.5
    H1_z0_shift = np.power(math.e, k1_z_shift * i1_z0 * 1j)

    H1_z0_shift_amp = np.abs(H1_z0_shift)
    H1_z0_shift_phase = np.angle(H1_z0_shift)

    if is_save == 1:
        if not os.path.isdir("4. H" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift"):
            os.makedirs("4. H" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift")

    # #%%
    # #绘图：H1_z0_shift_amp

    # H1_z0_shift_amp_address = location + "\\" + "4. H" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + "\\" + "4.1. AST - " + "H" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_amp" + file_name_extension

    # plot_2d(I1_x, I1_y, size_PerPixel, 0, 
    #         H1_z0_shift_amp, H1_z0_shift_amp_address, "H" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_amp", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         1, is_colorbar_on, vmax, vmin)

    # #%%
    # #绘图：H1_z0_shift_phase

    # H1_z0_shift_phase_address = location + "\\" + "4. H" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + "\\" + "4.2. AST - " + "H" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_phase" + file_name_extension

    # plot_2d(I1_x, I1_y, size_PerPixel, 0, 
    #         H1_z0_shift_phase, H1_z0_shift_phase_address, "H" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_phase", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         1, is_colorbar_on, vmax, vmin)
    
    #%%
    # 储存 H1_z0_shift 到 txt 文件

    if is_save == 1:
        H1_z0_shift_full_name = "4. AST - H" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + (is_save_txt and ".txt" or ".mat")
        H1_z0_shift_txt_address = location + "\\" + "4. H" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + "\\" + H1_z0_shift_full_name
        np.savetxt(H1_z0_shift_txt_address, H1_z0_shift) if is_save_txt else savemat(H1_z0_shift_txt_address, {'H':H1_z0_shift})
        
    #%%

    G1_z0_shift = g1_shift * H1_z0_shift
    G1_z0_shift_amp = np.abs(G1_z0_shift)
    # print(np.max(G1_z0_shift_amp))
    G1_z0_shift_phase = np.angle(G1_z0_shift)

    if is_save == 1:
        if not os.path.isdir("5. G" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift"):
            os.makedirs("5. G" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift")

    #%%
    #绘图：G1_z0_shift_amp

    G1_z0_shift_amp_address = location + "\\" + "5. G" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + "\\" + "5.1. AST - " + "G" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_amp" + file_name_extension

    plot_2d(I1_x, I1_y, size_PerPixel, 0, 
            G1_z0_shift_amp, G1_z0_shift_amp_address, "G" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, vmax, vmin)

    #%%
    #绘图：G1_z0_shift_phase

    G1_z0_shift_phase_address = location + "\\" + "5. G" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + "\\" + "5.2. AST - " + "G" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_phase" + file_name_extension

    plot_2d(I1_x, I1_y, size_PerPixel, 0, 
            G1_z0_shift_phase, G1_z0_shift_phase_address, "G" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_phase", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, vmax, vmin)
    
    #%%
    # 储存 G1_z0_shift 到 txt 文件

    if is_save == 1:
        G1_z0_shift_full_name = "5. AST - G" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + (is_save_txt and ".txt" or ".mat")
        G1_z0_shift_txt_address = location + "\\" + "5. G" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + "\\" + G1_z0_shift_full_name
        np.savetxt(G1_z0_shift_txt_address, G1_z0_shift) if is_save_txt else savemat(G1_z0_shift_txt_address, {'G':G1_z0_shift})
        
    #%%
    # G1_z0 = G1_z0(k1_x, k1_y) → IFFT2 → U1(x0, y0, z0) = U1_z0 ，毕竟 标量场 整体，是个 数组，就不写成 U1_x0_y0_z0 了

    G1_z0 = np.fft.ifftshift(G1_z0_shift)
    U1_z0 = np.fft.ifft2(G1_z0)
    # U1_z0_shift = np.fft.fftshift(U1_z0)

    U1_z0_amp = np.abs(U1_z0)
    # print(np.max(U1_z0_amp))
    U1_z0_phase = np.angle(U1_z0)

    # print("AST - U1_{}mm.total_amp = {}".format(z0, np.sum(U1_z0_amp)))
    print("AST - U1_{}mm.total_energy = {}".format(z0, np.sum(U1_z0_amp**2)))

    if is_save == 1:
        if not os.path.isdir("6. U" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm"):
            os.makedirs("6. U" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm")

    #%%
    #绘图：U1_z0_amp

    U1_z0_amp_address = location + "\\" + "6. U" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "\\" + "6.1. AST - " + "U" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_amp" + file_name_extension

    plot_2d(I1_x, I1_y, size_PerPixel, 0, 
            U1_z0_amp, U1_z0_amp_address, "U" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, vmax, vmin)

    #%%
    #绘图：U1_z0_phase

    U1_z0_phase_address = location + "\\" + "6. U" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "\\" + "6.2. AST - " + "U" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_phase" + file_name_extension

    plot_2d(I1_x, I1_y, size_PerPixel, 0, 
            U1_z0_phase, U1_z0_phase_address, "U" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_phase", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, vmax, vmin)
    
    #%%
    # 储存 U1_z0 到 txt 文件

    U1_z0_full_name = "6. AST - U" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + (is_save_txt and ".txt" or ".mat")
    if is_save == 1:
        U1_z0_txt_address = location + "\\" + "6. U" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "\\" + U1_z0_full_name
        np.savetxt(U1_z0_txt_address, U1_z0) if is_save_txt else savemat(U1_z0_txt_address, {'U':U1_z0})

        #%%
        #再次绘图：U1_z0_amp
    
        U1_z0_amp_address = location + "\\" + "6.1. AST - " + "U" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_amp" + file_name_extension
    
        plot_2d(I1_x, I1_y, size_PerPixel, 0, 
                U1_z0_amp, U1_z0_amp_address, "U" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_amp", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, vmax, vmin)
    
        #再次绘图：U1_z0_phase
    
        U1_z0_phase_address = location + "\\" + "6.2. AST - " + "U" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_phase" + file_name_extension
    
        plot_2d(I1_x, I1_y, size_PerPixel, 0, 
                U1_z0_phase, U1_z0_phase_address, "U" + ((U1_txt_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_phase", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, vmax, vmin)

    #%%
    # 储存 U1_z0 到 txt 文件

    # if is_save == 1:
    np.savetxt(U1_z0_full_name, U1_z0) if is_save_txt else savemat(U1_z0_full_name, {'U':U1_z0})
    