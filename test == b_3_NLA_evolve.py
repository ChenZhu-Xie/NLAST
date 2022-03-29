# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

#%%

import os
import numpy as np
import math
from scipy.io import savemat
from fun_os import img_squared_bordered_Read, U_Read
from fun_img_Resize import image_Add_black_border
from fun_array_Transform import Rotate_180, Roll_xy
from fun_plot import plot_1d, plot_2d, plot_3d_XYz, plot_3d_XYZ
from fun_pump import pump_LG
from fun_linear import Cal_n, Cal_kz
from fun_nonlinear import Eikz, C_m, Cal_lc_SHG, Cal_GxGyGz, Cal_dk_z_Q_shift_SHG, Cal_roll_xy, G2_z_modulation_NLAST, \
    G2_z_NLAST, G2_z_NLAST_false, Info_find_contours_SHG
from fun_thread import noop, my_thread
from fun_CGH import structure_chi2_Generate_2D
np.seterr(divide='ignore', invalid='ignore')
# %%
U1_name = ""
img_full_name = "lena1.png"
border_percentage = 0.1  # 边框 占图片的 百分比，也即 图片 放大系数
is_phase_only = 0
# %%
z_pump = 0
is_LG, is_Gauss, is_OAM = 1, 1, 1
l, p = 1, 0
theta_x, theta_y = 0, 0
# 正空间：右，下 = +, +
# 倒空间：左, 上 = +, +
# 朝着 x, y 轴 分别偏离 θ_1_x, θ_1_y 度
is_random_phase = 0
is_H_l, is_H_theta, is_H_random_phase = 0, 0, 0
# %%
U1_0_NonZero_size = 0.9  # Unit: mm 不包含边框，图片 的 实际尺寸
w0 = 0.1  # Unit: mm 束腰（z = 0 处）
z0 = 10  # Unit: mm 传播距离
# size_modulate = 1e-3 # Unit: mm χ2 调制区域 的 横向尺寸，即 公式中的 d
# %%
lam1 = 1.064  # Unit: um 基波波长
is_air_pump, is_air, T = 0, 0, 25  # is_air = 0, 1, 2 分别表示 LN, 空气, KTP；T 表示 温度
# %%
deff = 30  # pm / V
Tx, Ty, Tz = 10, 50, 7.004  # Unit: um
mx, my, mz = 1, 0, 1
# 倒空间：右, 下 = +, +
is_fft = 1
fft_mode = 0
is_sum_Gm = 0
mG = 0
is_linear_convolution = 1  # 0 代表 循环卷积，1 代表 线性卷积
# %%
is_save = 0
is_save_txt = 0
dpi = 100
# %%
color_1d = 'b'
cmap_2d = 'viridis'
# cmap_2d.set_under('black')
# cmap_2d.set_over('red')
cmap_3d = 'rainbow'
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
#%%
is_energy_evolution_on = 1
is_stored = 1
sheets_stored_num = 10
sample = 2
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

#%%
# G2_z0_shift

global G2_z_shift
G2_z_shift = np.zeros( (I2_x, I2_y), dtype=np.complex128() )
U2_z = np.zeros( (I2_x, I2_y), dtype=np.complex128() )

sheets_num = sheets_stored_num
if is_energy_evolution_on == 1:
    G2_z_shift_energy = np.zeros( (sheets_num + 1), dtype=np.float64() )
    U2_z_energy = np.zeros( (sheets_num + 1), dtype=np.float64() )
G2_z_shift_energy[0] = np.sum(np.abs(G2_z_shift)**2)
U2_z_energy[0] = np.sum(np.abs(U2_z)**2)

if is_stored == 1:
    # sheet_stored_th = np.zeros( (sheets_stored_num + 1), dtype=np.int64() ) # 这个其实 就是 0123...
    sheet_th_stored = np.zeros( int(sheets_stored_num + 1), dtype=np.int64() )
    iz_stored = np.zeros( int(sheets_stored_num + 1), dtype=np.float64() )
    z_stored = np.zeros( int(sheets_stored_num + 1), dtype=np.float64() )
    G2_z_shift_stored = np.zeros( (I2_x, I2_y, int(sheets_stored_num + 1)), dtype=np.complex128() )
    U2_z_stored = np.zeros( (I2_x, I2_y, int(sheets_stored_num + 1)), dtype=np.complex128() )

# %%
# const

const = (k2 / size_PerPixel / n2) ** 2 * C_m(mx) * C_m(my) * C_m(mz) * deff * 1e-12  # pm / V 转换成 m / V

# %%

zj = np.linspace(0, z0, sheets_stored_num + 1)
izj = zj / size_PerPixel

def Cal_G2_zm_shift(for_th2, fors_num2, *arg, ):
    
    names = globals() # 获取 全局变量名 列表
    names['G2_z' + str(for_th2) + '_shift'] = np.zeros((I2_x, I2_y), dtype=np.complex128()) # 在其中 新增一个 初始化的 全局变量

    if for_th2 == 0:
        
        pass
        
    else:
        
        if is_fft == 0:
        
            integrate_z0_shift = np.zeros((I2_x, I2_y), dtype=np.complex128())
        
            g1_shift_rotate_180 = Rotate_180(g1_shift)
        
        
            # g1_shift_rotate_180_shift = g1_shift_rotate_180
            # # 往下（行） 循环平移 I2_x 像素
            # g1_shift_rotate_180_shift = np.roll(g1_shift_rotate_180_shift, I2_x, axis=0)
            # # 往右（列） 循环平移 I2_y 像素
            # g1_shift_rotate_180_shift = np.roll(g1_shift_rotate_180_shift, I2_y, axis=1)
        
            def Cal_integrate_z0_shift(for_th, fors_num, *arg, ):
                print("forth = {}".format(for_th))
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
                        g1_shift * g1_shift_dk_x_dk_y * Eikz(dk_z_Q_shift * izj[for_th2]) * izj[for_th2] * size_PerPixel \
                        * (2 / (dk_z_Q_shift / k2_z_shift[for_th, n2_y] + 2)))
        
        
            my_thread(10, I2_x,
                      Cal_integrate_z0_shift, noop, noop,
                      is_ordered=1, is_print=is_print, )
        
            # integrate_z0_shift = integrate_z0_shift * (2 * math.pi / I2_x / size_PerPixel) * (2 * math.pi / I2_y / size_PerPixel)
            g2_z0_shift = const * integrate_z0_shift / k2_z_shift * size_PerPixel
        
            names['G2_z' + str(for_th2) + '_shift'] = g2_z0_shift * np.power(math.e, k2_z_shift * izj[for_th2] * 1j)
        
        else:
        
            if fft_mode == 0:
        
                if is_sum_Gm == 0:
                    names['G2_z' + str(for_th2) + '_shift'] = G2_z_modulation_NLAST(k1, k2, Gz,
                                                                                    modulation_squared, U1_0, izj[for_th2], const, )
                else:
                    # G2_z0_shift = np.zeros((I2_x, I2_y), dtype=np.complex128())
                    
                    def Cal_G2_z0_shift_Gm(for_th, fors_num, *arg, ):
                        m_z = for_th - mG
                        Gz_m = 2 * math.pi * m_z * size_PerPixel / (Tz / 1000)
                        # print(m_z, C_m(m_z), "\n")
                        
                        # 注意这个系数 C_m(m_z) 只对应 Duty_Cycle_z = 50% 占空比...
                        Const = (k2 / size_PerPixel / n2) ** 2 * C_m(mx) * C_m(my) * C_m(m_z) * deff * 1e-12
                        G2_z0_shift_Gm = G2_z_modulation_NLAST(k1, k2, Gz_m,
                                                               modulation_squared, U1_0, izj[for_th2], Const, ) if m_z != 0 else 0
                        return G2_z0_shift_Gm
                        
                    def Cal_G2_z0_shift(for_th, fors_num, G2_z0_shift_Gm, *arg, ):
                        print("forth = {}".format(for_th))
                        # global G2_z0_shift
                        # G2_z0_shift = G2_z0_shift + G2_z0_shift_Gm
                        
                        # names = globals()
                        names['G2_z' + str(for_th2) + '_shift'] = names['G2_z' + str(for_th2) + '_shift'] + G2_z0_shift_Gm
                        
                        return names['G2_z' + str(for_th2) + '_shift']
                    
                    my_thread(10, 2 * mG + 1,
                              Cal_G2_z0_shift_Gm, Cal_G2_z0_shift, noop, 
                              is_ordered=1, is_print=is_print, )
        
            elif fft_mode == 1:
                
                if is_sum_Gm == 0:
                    names['G2_z' + str(for_th2) + '_shift'] = G2_z_NLAST(k1, k2, Gx, Gy, Gz,
                                                                         U1_0, izj[for_th2], const,
                                                                         is_linear_convolution, )
                else:
                    
                    def Cal_G2_z0_shift_Gm(for_th, fors_num, *arg, ):
                        m_x = for_th - mG
                        Gx_m = 2 * math.pi * m_x * size_PerPixel / (Tx / 1000)
                        # print(m_x, C_m(m_x), "\n")
                        
                        # 注意这个系数 C_m(m_x) 只对应 Duty_Cycle_x = 50% 占空比...
                        Const = (k2 / size_PerPixel / n2) ** 2 * C_m(m_x) * C_m(my) * C_m(mz) * deff * 1e-12
                        G2_z0_shift_Gm = G2_z_NLAST(k1, k2, Gx_m, Gy, Gz,
                                                 U1_0, izj[for_th2], Const,
                                                 is_linear_convolution, ) if m_x != 0 else 0
                        return G2_z0_shift_Gm
                        
                    def Cal_G2_z0_shift(for_th, fors_num, G2_z0_shift_Gm, *arg, ):
                        
                        # global G2_z0_shift
                        # G2_z0_shift = G2_z0_shift + G2_z0_shift_Gm
                        
                        # names = globals()
                        names['G2_z' + str(for_th2) + '_shift'] = names['G2_z' + str(for_th2) + '_shift'] + G2_z0_shift_Gm
                        
                        return names['G2_z' + str(for_th2) + '_shift']
                    
                    my_thread(10, 2 * mG + 1,
                              Cal_G2_z0_shift_Gm, Cal_G2_z0_shift, noop, 
                              is_ordered=1, is_print=is_print, )
        
            elif fft_mode == 2:
        
                names['G2_z' + str(for_th2) + '_shift'] = G2_z_NLAST_false(k1, k2, Gx, Gy, Gz,
                                                                           U1_0, izj[for_th2], const,
                                                                            is_linear_convolution, )
    # print("forth2 = {}".format(for_th2))
    return names['G2_z' + str(for_th2) + '_shift']

def pass_G2_zm_shift(for_th2, fors_num2, G2_zm_shift, *arg, ):
    
    global G2_z_shift
    G2_z_shift = G2_zm_shift
    
    return G2_zm_shift

def After_G2_zm_shift(for_th2, fors_num2, G2_zm_shift_temp, *arg, ):
    
    G2_z = np.fft.ifftshift(G2_zm_shift_temp)
    U2_z = np.fft.ifft2(G2_z)
    
    if is_energy_evolution_on == 1:
        G2_z_shift_energy[for_th2] = np.sum(np.abs(G2_zm_shift_temp)**2)
        U2_z_energy[for_th2] = np.sum(np.abs(U2_z)**2)
    
    if is_stored == 1:
        sheet_th_stored[for_th2] = for_th2
        iz_stored[for_th2] = izj[for_th2]
        z_stored[for_th2] = zj[for_th2]
        G2_z_shift_stored[:, :, for_th2] = G2_zm_shift_temp #　储存的 第一层，实际上不是 G2_0，而是 G2_dz
        U2_z_stored[:, :, for_th2] = U2_z #　储存的 第一层，实际上不是 U2_0，而是 U2_dz

my_thread(10, sheets_num + 1, 
          Cal_G2_zm_shift, pass_G2_zm_shift, After_G2_zm_shift,  
          is_ordered = 1, is_print = is_print, )
    
#%%

G2_z0_SSI_shift = G2_z_shift
# G2_z0_SSI_shift = G2_z_shift_stored[:, :, -1]

G2_z0_SSI_shift_amp = np.abs(G2_z0_SSI_shift)
# print(np.max(G2_z0_SSI_shift_amp))
G2_z0_SSI_shift_phase = np.angle(G2_z0_SSI_shift)
if is_save == 1:
    if not os.path.isdir("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift"):
        os.makedirs("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift")

#%%
#绘图：G2_z0_SSI_shift_amp

G2_z0_SSI_shift_amp_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + "5.1. NLA - " + "G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_amp" + img_name_extension

plot_2d(zj, sample, size_PerPixel, 
        G2_z0_SSI_shift_amp, G2_z0_SSI_shift_amp_address, "G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_amp", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        1, is_colorbar_on, is_energy, vmax, vmin)

#%%
#绘图：G2_z0_SSI_shift_phase

G2_z0_SSI_shift_phase_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + "5.2. NLA - " + "G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_phase" + img_name_extension

plot_2d(zj, sample, size_PerPixel, 
        G2_z0_SSI_shift_phase, G2_z0_SSI_shift_phase_address, "G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_phase", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        1, is_colorbar_on, 0, vmax, vmin)

#%%
# 储存 G2_z0_SSI_shift 到 txt 文件

if is_save == 1:
    G2_z0_SSI_shift_full_name = "5. NLA - G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + (is_save_txt and ".txt" or ".mat")
    G2_z0_SSI_shift_txt_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + G2_z0_SSI_shift_full_name
    np.savetxt(G2_z0_SSI_shift_txt_address, G2_z0_SSI_shift) if is_save_txt else savemat(G2_z0_SSI_shift_txt_address, {'G':G2_z0_SSI_shift})

#%%    
# 绘制 G2_z_shift_energy 随 z 演化的 曲线

if is_energy_evolution_on == 1:
    
    vmax_G2_z_shift_energy = np.max(G2_z_shift_energy)
    vmin_G2_z_shift_energy = np.min(G2_z_shift_energy)
    
    G2_z_shift_energy_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + "5.1. NLA - " + "G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_energy_evolution" + img_name_extension
    
    plot_1d(zj, sample, size_PerPixel, 
            G2_z_shift_energy, G2_z_shift_energy_address, "G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_energy_evolution", 
            is_save, dpi, size_fig * 10, size_fig, 
            color_1d, ticks_num, is_title_on, is_axes_on, is_mm, 1, 
            fontsize, font, 
            0, vmax_G2_z_shift_energy, vmin_G2_z_shift_energy)
    
#%%
#% H2_z0

H2_z0_SSI_shift = G2_z0_SSI_shift/np.max(np.abs(G2_z0_SSI_shift)) / (g1_shift/np.max(np.abs(g1_shift)))
# 扔掉 amp 偏离 amp 均值 3 倍于 总体 标准差 以外 的 数据，保留 剩下的 3 倍 以内的 数据。
H2_z0_SSI_shift_amp_mean = np.mean(np.abs(H2_z0_SSI_shift))
H2_z0_SSI_shift_amp_std = np.std(np.abs(H2_z0_SSI_shift))
H2_z0_SSI_shift_amp_trust = np.abs(np.abs(H2_z0_SSI_shift) - H2_z0_SSI_shift_amp_mean) <= 3*H2_z0_SSI_shift_amp_std
H2_z0_SSI_shift = H2_z0_SSI_shift * H2_z0_SSI_shift_amp_trust.astype(np.int8)

if is_save == 1:
    if not os.path.isdir("4. H2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift"):
        os.makedirs("4. H2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift")

#%%
#% H2_z0_SSI_shift_amp

H2_z0_SSI_shift_amp_address = location + "\\" + "4. H2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + "4.1. NLA - " + "H2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift_amp" + img_name_extension

plot_2d(zj, sample, size_PerPixel, 
        np.abs(H2_z0_SSI_shift), H2_z0_SSI_shift_amp_address, "H2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift_amp", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        1, is_colorbar_on, is_energy, vmax, vmin)

#%%
#绘图：H2_z0_SSI_shift_phase

H2_z0_SSI_shift_phase_address = location + "\\" + "4. H2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + "4.2. NLA - " + "H2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift_phase" + img_name_extension

plot_2d(zj, sample, size_PerPixel, 
        np.angle(H2_z0_SSI_shift), H2_z0_SSI_shift_phase_address, "H2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift_phase", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        1, is_colorbar_on, 0, vmax, vmin)

#%%
# 储存 H2_z0_SSI_shift 到 txt 文件

if is_save == 1:
    H2_z0_SSI_shift_full_name = "4. NLA - H2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + (is_save_txt and ".txt" or ".mat")
    H2_z0_SSI_shift_txt_address = location + "\\" + "4. H2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + H2_z0_SSI_shift_full_name
    np.savetxt(H2_z0_SSI_shift_txt_address, H2_z0_SSI_shift) if is_save_txt else savemat(H2_z0_SSI_shift_txt_address, {"H":H2_z0_SSI_shift})

#%%
# G2_z0_SSI = G2_z0_SSI(k1_x, k1_y) → IFFT2 → U2(x0, y0, z0) = U2_z0_SSI

G2_z0_SSI = np.fft.ifftshift(G2_z0_SSI_shift)
U2_z0_SSI = np.fft.ifft2(G2_z0_SSI)
# 2 维 坐标空间 中的 复标量场，是 i2_x0, i2_y0 的函数
# U2_z0_SSI = U2_z0_SSI * scale_down_factor # 归一化

#%%

if is_stored == 1:
    
    if is_save == 1:
        if not os.path.isdir("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_stored"):
            os.makedirs("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_stored")
        if not os.path.isdir("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_stored"):
            os.makedirs("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_stored")
    
    #-------------------------
    
    vmax_G2_z_shift_stored_amp = np.max(np.abs(G2_z_shift_stored))
    vmin_G2_z_shift_stored_amp = np.min(np.abs(G2_z_shift_stored))
    
    for sheet_stored_th in range(sheets_stored_num + 1):
        
        G2_z_shift_sheet_stored_th_amp_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_stored" + "\\" + "5.1. NLA - " + "G2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI_shift" + "_amp" + img_name_extension
        
        plot_2d(zj, sample, size_PerPixel, 
                np.abs(G2_z_shift_stored[:, :, sheet_stored_th]), G2_z_shift_sheet_stored_th_amp_address, "G2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI_shift" + "_amp", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                is_self_colorbar, is_colorbar_on, is_energy, vmax_G2_z_shift_stored_amp, vmin_G2_z_shift_stored_amp)
        
    vmax_G2_z_shift_stored_phase = np.max(np.angle(G2_z_shift_stored))
    vmin_G2_z_shift_stored_phase = np.min(np.angle(G2_z_shift_stored))
        
    for sheet_stored_th in range(sheets_stored_num + 1):
        
        G2_z_shift_sheet_stored_th_phase_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_stored" + "\\" + "5.2. NLA - " + "G2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI_shift" + "_phase" + img_name_extension
    
        plot_2d(zj, sample, size_PerPixel, 
                np.angle(G2_z_shift_stored[:, :, sheet_stored_th]), G2_z_shift_sheet_stored_th_phase_address, "G2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI_shift" + "_phase", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                is_self_colorbar, is_colorbar_on, 0, vmax_G2_z_shift_stored_phase, vmin_G2_z_shift_stored_phase)
    
    #-------------------------    
    
    vmax_U2_z_stored_amp = np.max(np.abs(U2_z_stored))
    vmin_U2_z_stored_amp = np.min(np.abs(U2_z_stored))
    
    for sheet_stored_th in range(sheets_stored_num + 1):
        
        U2_z_sheet_stored_th_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_stored" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI" + "_amp" + img_name_extension
    
        plot_2d(zj, sample, size_PerPixel, 
                np.abs(U2_z_stored[:, :, sheet_stored_th]), U2_z_sheet_stored_th_amp_address, "U2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI" + "_amp", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                is_self_colorbar, is_colorbar_on, is_energy, vmax_U2_z_stored_amp, vmin_U2_z_stored_amp)
        
    vmax_U2_z_stored_phase = np.max(np.angle(U2_z_stored))
    vmin_U2_z_stored_phase = np.min(np.angle(U2_z_stored))
        
    for sheet_stored_th in range(sheets_stored_num + 1):
        
        U2_z_sheet_stored_th_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_stored" + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI" + "_phase" + img_name_extension
    
        plot_2d(zj, sample, size_PerPixel, 
                np.angle(U2_z_stored[:, :, sheet_stored_th]), U2_z_sheet_stored_th_phase_address, "U2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI" + "_phase", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                is_self_colorbar, is_colorbar_on, 0, vmax_U2_z_stored_phase, vmin_U2_z_stored_phase)
    
    #%%
    # 这 sheets_stored_num 层 也可以 画成 3D，就是太丑了，所以只 整个 U2_amp 示意一下即可
    
    # U2_z_sheets_stored_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_stored" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_sheets_stored" + "_amp" + img_name_extension
    
    # plot_3d_XYz(zj, sample, size_PerPixel, 
    #             sheets_stored_num, U2_z_stored, z_stored, 
    #             U2_z_sheets_stored_amp_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_sheets_stored" + "_amp", 
    #             is_save, dpi, size_fig, 
    #             cmap_3d, elev, azim, alpha, 
    #             ticks_num, is_title_on, is_axes_on, is_mm,  
    #             fontsize, font,
    #             is_self_colorbar, is_colorbar_on, is_energy, vmax_U2_z_stored_amp, vmin_U2_z_stored_amp)

#%%

U2_z0_SSI_amp = np.abs(U2_z0_SSI)
# print(np.max(U2_z0_SSI_amp))
U2_z0_SSI_phase = np.angle(U2_z0_SSI)

print("NLA - U2_{}mm_SSI.total_energy = {}".format(z0, np.sum(U2_z0_SSI_amp**2)))

if is_save == 1:
    if not os.path.isdir("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI"):
        os.makedirs("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI")

#%%
#绘图：U2_z0_SSI_amp

U2_z0_SSI_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_amp" + img_name_extension

plot_2d(zj, sample, size_PerPixel, 
        U2_z0_SSI_amp, U2_z0_SSI_amp_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_amp", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        1, is_colorbar_on, is_energy, vmax, vmin)

#%%
#绘图：U2_z0_SSI_phase

U2_z0_SSI_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_phase" + img_name_extension

plot_2d(zj, sample, size_PerPixel, 
        U2_z0_SSI_phase, U2_z0_SSI_phase_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_phase", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        1, is_colorbar_on, 0, vmax, vmin)

#%%
# 储存 U2_z0_SSI 到 txt 文件

U2_z0_SSI_full_name = "6. NLA - U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + (is_save_txt and ".txt" or ".mat")
if is_save == 1:
    U2_z0_SSI_txt_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "\\" + U2_z0_SSI_full_name
    np.savetxt(U2_z0_SSI_txt_address, U2_z0_SSI) if is_save_txt else savemat(U2_z0_SSI_txt_address, {'U':U2_z0_SSI})
    
    #%%
    #再次绘图：U2_z0_SSI_amp

    U2_z0_SSI_amp_address = location + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_amp" + img_name_extension

    plot_2d(zj, sample, size_PerPixel, 
            U2_z0_SSI_amp, U2_z0_SSI_amp_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    #再次绘图：U2_z0_SSI_phase

    U2_z0_SSI_phase_address = location + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_phase" + img_name_extension

    plot_2d(zj, sample, size_PerPixel, 
            U2_z0_SSI_phase, U2_z0_SSI_phase_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_phase", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)

#%%
# 储存 U2_z0_SSI 到 txt 文件

if is_save == 1:
    np.savetxt(U2_z0_SSI_full_name, U2_z0_SSI) if is_save_txt else savemat(U2_z0_SSI_full_name, {'U':U2_z0_SSI})
 
#%%
# 绘制 U2_z_energy 随 z 演化的 曲线
    
if is_energy_evolution_on == 1:
    
    vmax_U2_z_energy = np.max(U2_z_energy)
    vmin_U2_z_energy = np.min(U2_z_energy)
    
    U2_z_energy_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_energy_evolution" + img_name_extension
    
    plot_1d(zj, sample, size_PerPixel, 
            U2_z_energy, U2_z_energy_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_energy_evolution", 
            is_save, dpi, size_fig * 10, size_fig, 
            color_1d, ticks_num, is_title_on, is_axes_on, is_mm, 1, 
            fontsize, font, 
            0, vmax_U2_z_energy, vmin_U2_z_energy)
