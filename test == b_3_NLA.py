# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import os
import numpy as np
import math
from scipy.io import savemat
from fun_os import img_squared_bordered_Read, U_Read
from fun_img_Resize import image_Add_black_border
from fun_array_Transform import Rotate_180, Roll_xy
from fun_plot import plot_2d
from fun_pump import pump_LG
from fun_linear import Cal_n, Cal_kz
from fun_nonlinear import Eikz, C_m, Cal_lc_SHG, Cal_GxGyGz, Cal_dk_z_Q_shift_SHG, Cal_roll_xy, G2_z_modulation_NLAST, \
    G2_z_NLAST, G2_z_NLAST_false, Info_find_contours_SHG
from fun_thread import noop, my_thread
from fun_CGH import structure_chi2_Generate_2D

np.seterr(divide='ignore', invalid='ignore')
# %%
U1_name = ""
img_full_name = "lena.png"
border_percentage = 0.1  # 边框 占图片的 百分比，也即 图片 放大系数
is_phase_only = 0
# %%
z_pump = 0
is_LG, is_Gauss, is_OAM = 1, 1, 1
l, p = 1, 3
theta_x, theta_y = -0.5, 0
# 正空间：右，下 = +, +
# 倒空间：左, 上 = +, +
# 朝着 x, y 轴 分别偏离 θ_1_x, θ_1_y 度
is_random_phase = 0
is_H_l, is_H_theta, is_H_random_phase = 0, 0, 0
# %%
U1_0_NonZero_size = 0.9  # Unit: mm 不包含边框，图片 的 实际尺寸
w0 = 0.1  # Unit: mm 束腰（z = 0 处）
z0 = 3  # Unit: mm 传播距离
# size_modulate = 1e-3 # Unit: mm χ2 调制区域 的 横向尺寸，即 公式中的 d
# %%
lam1 = 1.064  # Unit: um 基波波长
is_air_pump, is_air, T = 0, 0, 25  # is_air = 0, 1, 2 分别表示 LN, 空气, KTP；T 表示 温度
# %%
deff = 30  # pm / V
Tx, Ty, Tz = 35, 50, 7.004  # Unit: um
mx, my, mz = 1, 0, 1
# 倒空间：右, 下 = +, +
is_fft = 1
fft_mode = 0
is_linear_convolution = 1  # 0 代表 循环卷积，1 代表 线性卷积
# %%
is_save = 0
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
is_contours = 66  # —— 我草，描边 需要 线性卷积，而不是 循环卷积，这样才 描得清楚
n_TzQ = 1
Gz_max_Enhance = 1
match_mode = 1

# %%

location = os.path.dirname(os.path.abspath(__file__))  # 其实不需要，默认就是在 相对路径下 读，只需要 文件名 即可

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

    # %%
    # 导入 方形 的 图片，以及 U

    img_name, img_name_extension, img_squared, size_PerPixel, size_fig, I1_x, I1_y, U1_0 = U_Read(U1_name,
                                                                                                  img_full_name,
                                                                                                  U1_0_NonZero_size,
                                                                                                  dpi,
                                                                                                  is_save_txt, )

# %%

n1, k1 = Cal_n(size_PerPixel,
               is_air,
               lam1, T, p="e")

# %%
# 线性 角谱理论 - 基波 begin

g1 = np.fft.fft2(U1_0)
g1_shift = np.fft.fftshift(g1)

k1_z_shift, mesh_k1_x_k1_y_shift = Cal_kz(I1_x, I1_y, k1)

# %%
# 非线性 角谱理论 - 无近似 begin

I2_x, I2_y = U1_0.shape[0], U1_0.shape[1]

# %%
# const

lam2 = lam1 / 2

n2, k2 = Cal_n(size_PerPixel,
               is_air,
               lam2, T, p="e")

k2_z_shift, mesh_k2_x_k2_y_shift = Cal_kz(I2_x, I2_y, k2)

# %%
# 提供描边信息，并覆盖值

z0, Tz, deff_structure_length_expect = Info_find_contours_SHG(g1_shift, k1_z_shift, k2_z_shift, Tz, mz,
                                                              z0, size_PerPixel, z0,
                                                              is_print, is_contours, n_TzQ, Gz_max_Enhance,
                                                              match_mode, )

# %%
# 引入 倒格矢，对 k2 的 方向 进行调整，其实就是对 k2 的 k2x, k2y, k2z 网格的 中心频率 从 (0, 0, k2z) 移到 (Gx, Gy, k2z + Gz)

dk, lc, Tz = Cal_lc_SHG(k1, k2, Tz, size_PerPixel,
                        is_print=0)

Gx, Gy, Gz = Cal_GxGyGz(mx, my, mz,
                        Tx, Ty, Tz, size_PerPixel,
                        is_print)

# %%
# const

deff = C_m(mx) * C_m(my) * C_m(mz) * deff * 1e-12  # pm / V 转换成 m / V
const = (k2 / size_PerPixel / n2) ** 2 * deff

# %%

z2_0 = z0
i2_z0 = z2_0 / size_PerPixel

if is_fft == 0:

    integrate_z0_shift = np.zeros((I2_x, I2_y), dtype=np.complex128())

    g1_shift_rotate_180 = Rotate_180(g1_shift)


    # g1_shift_rotate_180_shift = g1_shift_rotate_180
    # # 往下（行） 循环平移 I2_x 像素
    # g1_shift_rotate_180_shift = np.roll(g1_shift_rotate_180_shift, I2_x, axis=0)
    # # 往右（列） 循环平移 I2_y 像素
    # g1_shift_rotate_180_shift = np.roll(g1_shift_rotate_180_shift, I2_y, axis=1)

    def Cal_integrate_z0_shift(for_th, fors_num, *arg, ):
        for n2_y in range(I2_y):
            dk_z_Q_shift = Cal_dk_z_Q_shift_SHG(k1,
                                                k1_z_shift, k2_z_shift,
                                                mesh_k1_x_k1_y_shift, mesh_k2_x_k2_y_shift,
                                                for_th, n2_y,
                                                Gx, Gy, Gz, )

            roll_x, roll_y = Cal_roll_xy(Gx, Gy,
                                         I2_x, I2_y,
                                         for_th, n2_y, )

            g1_shift_dk_x_dk_y = Roll_xy(g1_shift_rotate_180,
                                         roll_x, roll_y,
                                         is_linear_convolution, )

            integrate_z0_shift[for_th, n2_y] = np.sum(
                g1_shift * g1_shift_dk_x_dk_y * Eikz(dk_z_Q_shift * i2_z0) * i2_z0 * size_PerPixel \
                * (2 / (dk_z_Q_shift / k2_z_shift[for_th, n2_y] + 2)))


    my_thread(10, I2_x,
              Cal_integrate_z0_shift, noop, noop,
              is_ordered=1, is_print=is_print, )

    # integrate_z0_shift = integrate_z0_shift * (2 * math.pi / I2_x / size_PerPixel) * (2 * math.pi / I2_y / size_PerPixel)
    g2_z0_shift = const * integrate_z0_shift / k2_z_shift * size_PerPixel

    G2_z0_shift = g2_z0_shift * np.power(math.e, k2_z_shift * i2_z0 * 1j)

else:

    if fft_mode == 0:
        # %% generate structure

        U1_name_Structure = ''
        is_phase_only_Structure = 0

        w0_Structure = 0
        z_pump_Structure = 0

        is_LG_Structure, is_Gauss_Structure, is_OAM_Structure = 0,1,0
        l_Structure, p_Structure = 0,0
        theta_x_Structure, theta_y_Structure = 0,0

        is_random_phase_Structure = 0
        is_H_l_Structure, is_H_theta_Structure, is_H_random_phase_Structure = 0,0,0

        structure_size_Enlarge = border_percentage
        Duty_Cycle_x = 0.5
        Duty_Cycle_y = 0.5

        Depth = 2
        structure_xy_mode = 'x'

        is_continuous = 0
        is_target_far_field = 1
        is_transverse_xy = 0
        is_reverse_xy = 0
        is_positive_xy = 1
        is_no_backgroud = 0
        
        n1, k1, k1_z_shift, lam2, n2, k2, k2_z_shift, \
        dk, lc, Tz, Gx, Gy, Gz, \
        size_PerPixel, U1_0_structure, g1_shift_structure, \
        structure, structure_opposite, modulation, modulation_opposite, modulation_squared, modulation_opposite_squared \
            = structure_chi2_Generate_2D(U1_name_Structure,
                                         img_full_name,
                                         is_phase_only_Structure,
                                         # %%
                                         z_pump_Structure,
                                         is_LG_Structure, is_Gauss_Structure, is_OAM_Structure,
                                         l_Structure, p_Structure,
                                         theta_x_Structure, theta_y_Structure,
                                         # %%
                                         is_random_phase_Structure,
                                         is_H_l_Structure, is_H_theta_Structure, is_H_random_phase_Structure,
                                         # %%
                                         U1_0_NonZero_size, w0_Structure,
                                         structure_size_Enlarge,
                                         Duty_Cycle_x, Duty_Cycle_y,
                                         structure_xy_mode, Depth,
                                         # %%
                                         is_continuous, is_target_far_field,
                                         is_transverse_xy, is_reverse_xy,
                                         is_positive_xy, is_no_backgroud,
                                         # %%
                                         lam1, is_air_pump, is_air, T,
                                         Tx, Ty, Tz,
                                         mx, my, mz,
                                         # %%
                                         is_save, is_save_txt, dpi,
                                         # %%
                                         cmap_2d,
                                         # %%
                                         ticks_num, is_contourf,
                                         is_title_on, is_axes_on,
                                         is_mm, is_propagation,
                                         # %%
                                         fontsize, font,
                                         # %%
                                         is_self_colorbar, is_colorbar_on,
                                         is_energy, vmax, vmin,
                                         # %%
                                         is_print, )

        G2_z0_shift = G2_z_modulation_NLAST(k1, k2, Gz,
                                            modulation_squared, U1_0, i2_z0, const, )

    elif fft_mode == 1:

        G2_z0_shift = G2_z_NLAST(k1, k2, Gx, Gy, Gz,
                                 U1_0, i2_z0, const,
                                 is_linear_convolution, )

    elif fft_mode == 2:

        G2_z0_shift = G2_z_NLAST_false(k1, k2, Gx, Gy, Gz,
                                       U1_0, i2_z0, const,
                                       is_linear_convolution, )

G2_z0_shift_amp = np.abs(G2_z0_shift)
# print(np.max(G2_z0_shift_amp))
G2_z0_shift_phase = np.angle(G2_z0_shift)
if is_save == 1:
    if not os.path.isdir("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_shift"):
        os.makedirs("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_shift")

# %%
# 绘图：G2_z0_shift_amp

G2_z0_shift_amp_address = location + "\\" + "5. G2_" + str(
    float('%.2g' % z0)) + "mm" + "_shift" + "\\" + "5.1. NLA - " + "G2_" + str(
    float('%.2g' % z0)) + "mm" + "_shift_amp" + img_name_extension

plot_2d([], 1, size_PerPixel,
        G2_z0_shift_amp, G2_z0_shift_amp_address, "G2_" + str(float('%.2g' % z0)) + "mm" + "_shift_amp",
        is_save, dpi, size_fig,
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
        fontsize, font,
        1, is_colorbar_on, is_energy, vmax, vmin)

# %%
# 绘图：G2_z0_shift_phase

G2_z0_shift_phase_address = location + "\\" + "5. G2_" + str(
    float('%.2g' % z0)) + "mm" + "_shift" + "\\" + "5.2. NLA - " + "G2_" + str(
    float('%.2g' % z0)) + "mm" + "_shift_phase" + img_name_extension

plot_2d([], 1, size_PerPixel,
        G2_z0_shift_phase, G2_z0_shift_phase_address, "G2_" + str(float('%.2g' % z0)) + "mm" + "_shift_phase",
        is_save, dpi, size_fig,
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
        fontsize, font,
        1, is_colorbar_on, 0, vmax, vmin)

# %%
# 储存 G2_z0_shift 到 txt 文件

if is_save == 1:
    G2_z0_shift_full_name = "5. NLA - G2_" + str(float('%.2g' % z0)) + "mm" + "_shift" + (
            is_save_txt and ".txt" or ".mat")
    G2_z0_shift_txt_address = location + "\\" + "5. G2_" + str(
        float('%.2g' % z0)) + "mm" + "_shift" + "\\" + G2_z0_shift_full_name
    np.savetxt(G2_z0_shift_txt_address, G2_z0_shift) if is_save_txt else savemat(G2_z0_shift_txt_address,
                                                                                 {"G": G2_z0_shift})

# %%
# % H2_z0

H2_z0_shift = G2_z0_shift / np.max(np.abs(G2_z0_shift)) / (g1_shift / np.max(np.abs(g1_shift)))
# 扔掉 amp 偏离 amp 均值 3 倍于 总体 标准差 以外 的 数据，保留 剩下的 3 倍 以内的 数据。
H2_z0_shift_amp_mean = np.mean(np.abs(H2_z0_shift))
H2_z0_shift_amp_std = np.std(np.abs(H2_z0_shift))
H2_z0_shift_amp_trust = np.abs(np.abs(H2_z0_shift) - H2_z0_shift_amp_mean) <= 3 * H2_z0_shift_amp_std
H2_z0_shift = H2_z0_shift * H2_z0_shift_amp_trust.astype(np.int8)

if is_save == 1:
    if not os.path.isdir("4. H2_" + str(float('%.2g' % z0)) + "mm" + "_shift"):
        os.makedirs("4. H2_" + str(float('%.2g' % z0)) + "mm" + "_shift")

# %%
# % H2_z0_shift_amp

H2_z0_shift_amp_address = location + "\\" + "4. H2_" + str(
    float('%.2g' % z0)) + "mm" + "_shift" + "\\" + "4.1. NLA - " + "H2_" + str(
    float('%.2g' % z0)) + "mm" + "_shift_amp" + img_name_extension

plot_2d([], 1, size_PerPixel,
        np.abs(H2_z0_shift), H2_z0_shift_amp_address, "H2_" + str(float('%.2g' % z0)) + "mm" + "_shift_amp",
        is_save, dpi, size_fig,
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
        fontsize, font,
        1, is_colorbar_on, is_energy, vmax, vmin)

# %%
# 绘图：H2_z0_shift_phase

H2_z0_shift_phase_address = location + "\\" + "4. H2_" + str(
    float('%.2g' % z0)) + "mm" + "_shift" + "\\" + "4.2. NLA - " + "H2_" + str(
    float('%.2g' % z0)) + "mm" + "_shift_phase" + img_name_extension

plot_2d([], 1, size_PerPixel,
        np.angle(H2_z0_shift), H2_z0_shift_phase_address, "H2_" + str(float('%.2g' % z0)) + "mm" + "_shift_phase",
        is_save, dpi, size_fig,
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
        fontsize, font,
        1, is_colorbar_on, 0, vmax, vmin)

# %%
# 储存 H2_z0_shift 到 txt 文件

if is_save == 1:
    H2_z0_shift_full_name = "4. NLA - H2_" + str(float('%.2g' % z0)) + "mm" + "_shift" + (
            is_save_txt and ".txt" or ".mat")
    H2_z0_shift_txt_address = location + "\\" + "4. H2_" + str(
        float('%.2g' % z0)) + "mm" + "_shift" + "\\" + H2_z0_shift_full_name
    np.savetxt(H2_z0_shift_txt_address, H2_z0_shift) if is_save_txt else savemat(H2_z0_shift_txt_address,
                                                                                 {"H": H2_z0_shift})

# %%
# G2_z0 = G2_z0(k1_x, k1_y) → IFFT2 → U2(x0, y0, z0) = U2_z0

G2_z0 = np.fft.ifftshift(G2_z0_shift)
U2_z0 = np.fft.ifft2(G2_z0)
# 2 维 坐标空间 中的 复标量场，是 i2_x0, i2_y0 的函数
# U2_z0 = U2_z0 * scale_down_factor # 归一化
U2_z0_amp = np.abs(U2_z0)
# print(np.max(U2_z0_amp))
U2_z0_phase = np.angle(U2_z0)

print("NLA - U2_{}mm.total_energy = {}".format(z0, np.sum(U2_z0_amp ** 2)))

if is_save == 1:
    if not os.path.isdir("6. U2_" + str(float('%.2g' % z0)) + "mm"):
        os.makedirs("6. U2_" + str(float('%.2g' % z0)) + "mm")

# %%
# 绘图：U2_z0_amp

U2_z0_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "\\" + "6.1. NLA - " + "U2_" + str(
    float('%.2g' % z0)) + "mm" + "_amp" + img_name_extension

plot_2d([], 1, size_PerPixel,
        U2_z0_amp, U2_z0_amp_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_amp",
        is_save, dpi, size_fig,
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
        fontsize, font,
        1, is_colorbar_on, is_energy, vmax, vmin)

# %%
# 绘图：U2_z0_phase

U2_z0_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "\\" + "6.2. NLA - " + "U2_" + str(
    float('%.2g' % z0)) + "mm" + "_phase" + img_name_extension

plot_2d([], 1, size_PerPixel,
        U2_z0_phase, U2_z0_phase_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_phase",
        is_save, dpi, size_fig,
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
        fontsize, font,
        1, is_colorbar_on, 0, vmax, vmin)

# %%
# 储存 U2_z0 到 txt 文件

U2_z0_full_name = "6. NLA - U2_" + str(float('%.2g' % z0)) + "mm" + (is_save_txt and ".txt" or ".mat")
if is_save == 1:
    U2_z0_txt_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "\\" + U2_z0_full_name
    np.savetxt(U2_z0_txt_address, U2_z0) if is_save_txt else savemat(U2_z0_txt_address, {"U": U2_z0})

    # %%
    # 再次绘图：U2_z0_amp

    U2_z0_amp_address = location + "\\" + "6.1. NLA - " + "U2_" + str(
        float('%.2g' % z0)) + "mm" + "_amp" + img_name_extension

    plot_2d([], 1, size_PerPixel,
            U2_z0_amp, U2_z0_amp_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_amp",
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    # 再次绘图：U2_z0_phase

    U2_z0_phase_address = location + "\\" + "6.2. NLA - " + "U2_" + str(
        float('%.2g' % z0)) + "mm" + "_phase" + img_name_extension

    plot_2d([], 1, size_PerPixel,
            U2_z0_phase, U2_z0_phase_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_phase",
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)

# %%
# 储存 U2_z0 到 txt 文件

if is_save == 1:
    np.savetxt(U2_z0_full_name, U2_z0) if is_save_txt else savemat(U2_z0_full_name, {"U": U2_z0})
