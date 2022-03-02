# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

#%%

import os
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import math
# import copy
# import scipy
from scipy.io import savemat
from fun_os import img_squared_bordered_Read, U_Read
from fun_img_Resize import image_Add_black_border
from fun_plot import plot_1d, plot_2d, plot_3d_XYZ, plot_3d_XYz
from fun_pump import pump_LG
from fun_linear import Cal_n, Cal_kz

#%%
U1_name = ""
img_full_name = "lena.png"
border_percentage = 0.3 # 边框 占图片的 百分比，也即 图片 放大系数
#%%
is_phase_only = 0
is_LG, is_Gauss, is_OAM = 0, 1, 1
l, p = 1, 0
theta_x, theta_y = 1, 0
is_H_l, is_H_theta = 0, 0
# 正空间：右，下 = +, +
# 倒空间：左, 上 = +, +
# 朝着 x, y 轴 分别偏离 θ_1_x, θ_1_y 度
#%%
U1_0_NonZero_size = 1 # Unit: mm 不包含边框，图片 的 实际尺寸
w0 = 0.5 # Unit: mm 束腰（z = 0 处）
z0 = 10 # Unit: mm 传播距离
# size_modulate = 1e-3 # Unit: mm χ2 调制区域 的 横向尺寸，即 公式中的 d
#%%
lam1 = 0.4 # Unit: um 基波 或 倍频波长
is_air_pump, is_air, T = 0, 0, 25 # is_air = 0, 1, 2 分别表示 LN, 空气, KTP；T 表示 温度
#%%
is_save = 0
is_save_txt = 0
dpi = 100
#%%
cmap_2d='viridis'
# cmap_2d.set_under('black')
# cmap_2d.set_over('red')
#%%
ticks_num = 6 # 不包含 原点的 刻度数，也就是 区间数（植数问题）
is_contourf = 0
is_title_on, is_axes_on = 1, 1
is_mm, is_propagation = 1, 0
#%%
fontsize = 9
font = {'family': 'serif',
        'style': 'normal', # 'normal', 'italic', 'oblique'
        'weight': 'normal',
        'color': 'black', # 'black','gray','darkred'
        }
#%%
is_self_colorbar, is_colorbar_on = 0, 1 # vmax 与 vmin 是否以 自己的 U 的 最大值 最小值 为 相应的值；是，则覆盖设定；否的话，需要自己设定。
is_energy = 0
vmax, vmin = 1, 0
#%%
is_print = 1

#%%

location = os.path.dirname(os.path.abspath(__file__)) # 其实不需要，默认就是在 相对路径下 读，只需要 文件名 即可

if (type(U1_name) != str) or U1_name == "":
    #%%
    # 预处理 导入图片 为方形，并加边框
    
    image_Add_black_border(img_full_name, 
                           border_percentage, 
                           is_print, )
    
    #%%
    # 导入 方形，以及 加边框 的 图片
    
    img_name, img_name_extension, img_squared, size_PerPixel, size_fig, I1_x, I1_y, U1_0 = img_squared_bordered_Read(img_full_name, 
                                                                                                                     U1_0_NonZero_size, dpi, 
                                                                                                                     is_phase_only)
    
    #%%
    # 预处理 输入场
    
    n1, k1 = Cal_n(size_PerPixel, 
                   is_air, 
                   lam1, T, p = "e")
    
    U1_0 = pump_LG(img_full_name, 
                   I1_x, I1_y, size_PerPixel, 
                   U1_0, w0, k1, 0, 
                   is_LG, is_Gauss, is_OAM, 
                   l, p, 
                   theta_x, theta_y, 
                   is_H_l, is_H_theta, 
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

if U1_name.find("U2") != -1: # 如果找到了 U2 字样
    lam1 = lam1 / 2

n1, k1 = Cal_n(size_PerPixel, 
               is_air, 
               lam1, T, p = "e")

#%%
# U1_0 = U(x, y, 0) → FFT2 → g1_shift(k1_x, k1_y) = g1_shift

g1 = np.fft.fft2(U1_0)

g1_shift = np.fft.fftshift(g1)

g1_shift_amp = np.abs(g1_shift)
# print(np.max(g1_shift_amp))
g1_shift_phase = np.angle(g1_shift)

if is_save == 1:
    if not os.path.isdir("3. g" + ((U1_name.find("U2") + 1) and "2" or "1") + "_shift"):
        os.makedirs("3. g" + ((U1_name.find("U2") + 1) and "2" or "1") + "_shift")

# #%%
# #绘图：g1_shift_amp

# g1_shift_amp_address = location + "\\" + "3. g" + ((U1_name.find("U2") + 1) and "2" or "1") + "_shift" + "\\" + "3.1. AST - " + "g" + ((U1_name.find("U2") + 1) and "2" or "1") + "_shift" + "_amp" + img_name_extension

# plot_2d(I1_x, I1_y, size_PerPixel, 0, 
#         g1_shift_amp, g1_shift_amp_address, "g" + ((U1_name.find("U2") + 1) and "2" or "1") + "_shift" + "_amp", 
#         is_save, dpi, size_fig,  
#         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
#         fontsize, font,
#         1, is_colorbar_on, is_energy, vmax, vmin)

# #%%
# #绘图：g1_shift_phase

# g1_shift_phase_address = location + "\\" + "3. g" + ((U1_name.find("U2") + 1) and "2" or "1") + "_shift" + "\\" + "3.1. AST - " + "g" + ((U1_name.find("U2") + 1) and "2" or "1") + "_shift" + "_phase" + img_name_extension

# plot_2d(I1_x, I1_y, size_PerPixel, 0, 
#         g1_shift_phase, g1_shift_phase_address, "g" + ((U1_name.find("U2") + 1) and "2" or "1") + "_shift" + "_phase", 
#         is_save, dpi, size_fig,  
#         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
#         fontsize, font,
#         1, is_colorbar_on, 0, vmax, vmin)

#%%
# 储存 g1_shift 到 txt 文件

if is_save == 1:
    g1_shift_full_name = "3. AST - g" + ((U1_name.find("U2") + 1) and "2" or "1") + "_shift" + (is_save_txt and ".txt" or ".mat")
    g1_shift_txt_address = location + "\\" + "3. g" + ((U1_name.find("U2") + 1) and "2" or "1") + "_shift" + "\\" + g1_shift_full_name
    np.savetxt(g1_shift_txt_address, g1_shift) if is_save_txt else savemat(g1_shift_txt_address, {"g":g1_shift})
    
#%%
# g1_shift = { g1_shift(k1_x, k1_y) } → 每个元素，乘以，频域 传递函数 e^{i*k1_z*z0} → G1_z0(k1_x, k1_y) = G1_z0

z1_0 = z0
i1_z0 = z1_0 / size_PerPixel

k1_z_shift, mesh_k1_x_k1_y_shift = Cal_kz(I1_x, I1_y, k1)
H1_z0_shift = np.power(math.e, k1_z_shift * i1_z0 * 1j)

H1_z0_shift_amp = np.abs(H1_z0_shift)
H1_z0_shift_phase = np.angle(H1_z0_shift)

if is_save == 1:
    if not os.path.isdir("4. H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift"):
        os.makedirs("4. H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift")

# #%%
# #绘图：H1_z0_shift_amp

# H1_z0_shift_amp_address = location + "\\" + "4. H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + "\\" + "4.1. AST - " + "H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_amp" + img_name_extension

# plot_2d(I1_x, I1_y, size_PerPixel, 0, 
#         H1_z0_shift_amp, H1_z0_shift_amp_address, "H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_amp", 
#         is_save, dpi, size_fig,  
#         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
#         fontsize, font,
#         1, is_colorbar_on, is_energy, vmax, vmin)

# #%%
# #绘图：H1_z0_shift_phase

# H1_z0_shift_phase_address = location + "\\" + "4. H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + "\\" + "4.2. AST - " + "H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_phase" + img_name_extension

# plot_2d(I1_x, I1_y, size_PerPixel, 0, 
#         H1_z0_shift_phase, H1_z0_shift_phase_address, "H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_phase", 
#         is_save, dpi, size_fig,  
#         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
#         fontsize, font,
#         1, is_colorbar_on, 0, vmax, vmin)

#%%
# 储存 H1_z0_shift 到 txt 文件

if is_save == 1:
    H1_z0_shift_full_name = "4. AST - H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + (is_save_txt and ".txt" or ".mat")
    H1_z0_shift_txt_address = location + "\\" + "4. H" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + "\\" + H1_z0_shift_full_name
    np.savetxt(H1_z0_shift_txt_address, H1_z0_shift) if is_save_txt else savemat(H1_z0_shift_txt_address, {'H':H1_z0_shift})
    
#%%

G1_z0_shift = g1_shift * H1_z0_shift
G1_z0_shift_amp = np.abs(G1_z0_shift)
# print(np.max(G1_z0_shift_amp))
G1_z0_shift_phase = np.angle(G1_z0_shift)

if is_save == 1:
    if not os.path.isdir("5. G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift"):
        os.makedirs("5. G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift")

#%%
#绘图：G1_z0_shift_amp

G1_z0_shift_amp_address = location + "\\" + "5. G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + "\\" + "5.1. AST - " + "G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_amp" + img_name_extension

plot_2d(I1_x, I1_y, size_PerPixel, 0, 
        G1_z0_shift_amp, G1_z0_shift_amp_address, "G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_amp", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        1, is_colorbar_on, is_energy, vmax, vmin)

#%%
#绘图：G1_z0_shift_phase

G1_z0_shift_phase_address = location + "\\" + "5. G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + "\\" + "5.2. AST - " + "G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_phase" + img_name_extension

plot_2d(I1_x, I1_y, size_PerPixel, 0, 
        G1_z0_shift_phase, G1_z0_shift_phase_address, "G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift_phase", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        1, is_colorbar_on, 0, vmax, vmin)

#%%
# 储存 G1_z0_shift 到 txt 文件

if is_save == 1:
    G1_z0_shift_full_name = "5. AST - G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + (is_save_txt and ".txt" or ".mat")
    G1_z0_shift_txt_address = location + "\\" + "5. G" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_shift" + "\\" + G1_z0_shift_full_name
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
print("AST - U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_{}mm.total_energy = {}".format(z0, np.sum(U1_z0_amp**2)))

if is_save == 1:
    if not os.path.isdir("6. U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm"):
        os.makedirs("6. U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm")

#%%
#绘图：U1_z0_amp

U1_z0_amp_address = location + "\\" + "6. U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "\\" + "6.1. AST - " + "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_amp" + img_name_extension

plot_2d(I1_x, I1_y, size_PerPixel, 0, 
        U1_z0_amp, U1_z0_amp_address, "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_amp", 
        is_save, dpi, size_fig, 
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        1, is_colorbar_on, is_energy, vmax, vmin)

#%%
#绘图：U1_z0_phase

U1_z0_phase_address = location + "\\" + "6. U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "\\" + "6.2. AST - " + "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_phase" + img_name_extension

plot_2d(I1_x, I1_y, size_PerPixel, 0, 
        U1_z0_phase, U1_z0_phase_address, "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_phase", 
        is_save, dpi, size_fig, 
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        1, is_colorbar_on, 0, vmax, vmin)

#%%
# 储存 U1_z0 到 txt 文件

U1_z0_full_name = "6. AST - U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + (is_save_txt and ".txt" or ".mat")
if is_save == 1:
    U1_z0_txt_address = location + "\\" + "6. U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "\\" + U1_z0_full_name
    np.savetxt(U1_z0_txt_address, U1_z0) if is_save_txt else savemat(U1_z0_txt_address, {'U':U1_z0})

    #%%
    #再次绘图：U1_z0_amp

    U1_z0_amp_address = location + "\\" + "6.1. AST - " + "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_amp" + img_name_extension

    plot_2d(I1_x, I1_y, size_PerPixel, 0, 
            U1_z0_amp, U1_z0_amp_address, "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    #再次绘图：U1_z0_phase

    U1_z0_phase_address = location + "\\" + "6.2. AST - " + "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_phase" + img_name_extension

    plot_2d(I1_x, I1_y, size_PerPixel, 0, 
            U1_z0_phase, U1_z0_phase_address, "U" + ((U1_name.find("U2") + 1) and "2" or "1") + "_" + str(float('%.2g' % z0)) + "mm" + "_phase", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)

#%%
# 储存 U1_z0 到 txt 文件

if is_save == 1:
    np.savetxt(U1_z0_full_name, U1_z0) if is_save_txt else savemat(U1_z0_full_name, {'U':U1_z0})
