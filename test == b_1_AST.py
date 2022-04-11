# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
import math
from fun_os import U_dir, U_energy_print, img_squared_bordered_Read, U_Read, U_plot, U_save
from fun_img_Resize import image_Add_black_border
from fun_pump import pump_LG
from fun_linear import Cal_n, Cal_kz
np.seterr(divide='ignore', invalid='ignore')
# %%
U1_name = ""
img_full_name = "lena1.png"
border_percentage = 0.1  # 边框 占图片的 百分比，也即 图片 放大系数
is_phase_only = 0
# %%
z_pump = 0
is_LG, is_Gauss, is_OAM = 1, 1, 1
l, p = 10, 3
theta_x, theta_y = -0.5, 0
# 正空间：右，下 = +, +
# 倒空间：左, 上 = +, +
# 朝着 x, y 轴 分别偏离 θ_1_x, θ_1_y 度
is_random_phase = 0
is_H_l, is_H_theta, is_H_random_phase = 0, 0, 0
# %%
U1_0_NonZero_size = 0.9  # Unit: mm 不包含边框，图片 的 实际尺寸
w0 = 0.05  # Unit: mm 束腰（z = 0 处）
z0 = 5  # Unit: mm 传播距离
# size_modulate = 1e-3 # Unit: mm χ2 调制区域 的 横向尺寸，即 公式中的 d
# %%
lam1 = 1.064  # Unit: um 基波 或 倍频波长
is_air_pump, is_air, T = 0, 0, 25  # is_air = 0, 1, 2 分别表示 LN, 空气, KTP；T 表示 温度
# %%
is_save = 1
is_save_txt = 0
dpi = 100
# %%
cmap_2d = 'viridis'
# cmap_2d.set_under('black')
# cmap_2d.set_over('red')
# %%
ticks_num = 6  # 不包含 原点的 刻度数，也就是 区间数（植数问题）
is_contourf = 0
is_title_on, is_axes_on = 1, 1
is_mm, is_propagation = 1, 0
# %%
fontsize = 9
font = {'family': 'serif',
        'style': 'normal',  # 'normal', 'italic', 'oblique'
        'weight': 'normal',
        'color': 'black',  # 'black','gray','darkred'
        }
# %%
is_self_colorbar, is_colorbar_on = 0, 1  # vmax 与 vmin 是否以 自己的 U 的 最大值 最小值 为 相应的值；是，则覆盖设定；否的话，需要自己设定。
is_energy = 0
vmax, vmin = 1, 0
# %%
is_print = 1

# %%

if (type(U1_name) != str) or U1_name == "":
    # %%
    # 预处理 导入图片 为方形，并加边框

    image_Add_black_border(img_full_name,
                           border_percentage,
                           is_print, )

    # %%
    # 导入 方形，以及 加边框 的 图片

    img_name, img_name_extension, img_squared, size_PerPixel, size_fig, I1_x, I1_y, U1_0 = img_squared_bordered_Read(
                        img_full_name,
                        U1_0_NonZero_size, dpi,
                        is_phase_only)

    # %%
    # 预处理 输入场

    n1, k1 = Cal_n(size_PerPixel,
                   is_air_pump,
                   lam1, T, p="e")

    U1_0, g1_shift = pump_LG(img_full_name,
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

    # %%
    # 导入 方形 的 图片，以及 U

    img_name, img_name_extension, img_squared, size_PerPixel, size_fig, I1_x, I1_y, U1_0 = U_Read(U1_name,
                                                                                                  img_full_name,
                                                                                                  U1_0_NonZero_size,
                                                                                                  dpi,
                                                                                                  is_save_txt, )

# %%

if U1_name.find("U2") != -1:  # 如果找到了 U2 字样
    lam1 = lam1 / 2

n1, k1 = Cal_n(size_PerPixel,
               is_air,
               lam1, T, p="e")

# %%
# U1_0 = U(x, y, 0) → FFT2 → g1_shift(k1_x, k1_y) = g1_shift

if is_save == 1:
    folder_address = U_dir(U1_name, "g1_shift", 1, )

#%%
#绘图：g1_shift

# U_amp_plot_address, U_phase_plot_address = U_plot(U1_name, folder_address, 1, 
#                                                   g1_shift, "g1_shift", "AST", 
#                                                   img_name_extension, 
#                                                   #%%
#                                                   [], 1, size_PerPixel,
#                                                   is_save, dpi, size_fig,
#                                                   cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
#                                                   fontsize, font,
#                                                   1, is_colorbar_on, is_energy, vmax, vmin, 
#                                                   #%%
#                                                   z0, )

# %%
# 储存 g1_shift 到 txt 文件

if is_save == 1:
    U_address = U_save(g1_shift, "g1_shift", folder_address,
                       is_save_txt, )

# %%
# g1_shift = { g1_shift(k1_x, k1_y) } → 每个元素，乘以，频域 传递函数 e^{i*k1_z*z0} → G1_z0(k1_x, k1_y) = G1_z0

z1_0 = z0
i1_z0 = z1_0 / size_PerPixel

k1_z_shift, mesh_k1_x_k1_y_shift = Cal_kz(I1_x, I1_y, k1)
H1_z0_shift = np.power(math.e, k1_z_shift * i1_z0 * 1j)

if is_save == 1:
    folder_address = U_dir(U1_name, "H1_z0_shift", 1, z0, )

#%%
#绘图：H1_z0_shift

# U_amp_plot_address, U_phase_plot_address = U_plot(U1_name, folder_address, 1, 
#                                                   H1_z0_shift, "H1_z0_shift", "AST", 
#                                                   img_name_extension, 
#                                                   #%%
#                                                   [], 1, size_PerPixel,
#                                                   is_save, dpi, size_fig,
#                                                   cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
#                                                   fontsize, font,
#                                                   1, is_colorbar_on, is_energy, vmax, vmin, 
#                                                   #%%
#                                                   z0, )

# %%
# 储存 H1_z0_shift 到 txt 文件

if is_save == 1:
    U_address = U_save(H1_z0_shift, "H1_z0_shift", folder_address,
                       is_save_txt, z=z0, )

# %%

G1_z0_shift = g1_shift * H1_z0_shift

if is_save == 1:
    folder_address = U_dir(U1_name, "G1_z0_shift", 1, z0)

# %%
# 绘图：G1_z0_shift

U_amp_plot_address, U_phase_plot_address = U_plot(U1_name, folder_address, 1, 
                                                  G1_z0_shift, "G1_z0_shift", "AST", 
                                                  img_name_extension, 
                                                  #%%
                                                  [], 1, size_PerPixel,
                                                  is_save, dpi, size_fig,
                                                  cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                                                  fontsize, font,
                                                  1, is_colorbar_on, is_energy, vmax, vmin, 
                                                  #%%
                                                  z0, )

# %%
# 储存 G1_z0_shift 到 txt 文件

if is_save == 1:
    U_address = U_save(G1_z0_shift, "G1_z0_shift", folder_address,
                       is_save_txt, z=z0, )

# %%
# G1_z0 = G1_z0(k1_x, k1_y) → IFFT2 → U1(x0, y0, z0) = U1_z0 ，毕竟 标量场 整体，是个 数组，就不写成 U1_x0_y0_z0 了

G1_z0 = np.fft.ifftshift(G1_z0_shift)
U1_z0 = np.fft.ifft2(G1_z0)

U_energy_print(U1_name, 1, 1, 
               U1_z0, "U1_z0", "AST", 
               z0, )

if is_save == 1:
    folder_address = U_dir(U1_name, "U1_z0", 1, z0)

# %%
# 绘图：U1_z0

U_amp_plot_address, U_phase_plot_address = U_plot(U1_name, folder_address, 1, 
                                                  U1_z0, "U1_z0", "AST", 
                                                  img_name_extension, 
                                                  #%%
                                                  [], 1, size_PerPixel,
                                                  is_save, dpi, size_fig,
                                                  cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                                                  fontsize, font,
                                                  1, is_colorbar_on, is_energy, vmax, vmin, 
                                                  #%%
                                                  z0, )

# %%
# 储存 U1_z0 到 txt 文件

if is_save == 1:
    U_address = U_save(U1_name, folder_address, 1, 
                       U1_z0, "U1_z0", "AST", 
                       is_save_txt, z0, )