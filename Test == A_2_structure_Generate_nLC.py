# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

#%%

import os
import cv2
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import math
from scipy.io import loadmat, savemat
from fun_os import img_squared_bordered_Read
from fun_img_Resize import image_Add_black_border, img_squared_Resize
from fun_plot import plot_2d
from fun_pump import pump_LG
from fun_SSI import Cal_diz, Cal_Iz_structure, Cal_IxIy
from fun_linear import Cal_n
from fun_nonlinear import Cal_lc_SHG, Cal_GxGyGz
from fun_CGH import structure_Generate
from fun_thread import noop, my_thread

#%%
U1_name = ""
img_full_name = "l=1.png"
border_percentage = 0.3 # 边框 占图片的 百分比，也即 图片 放大系数
#%%
is_phase_only = 0
is_LG, is_Gauss, is_OAM = 0, 1, 1
l, p = 0, 0
theta_x, theta_y = 0, 0
is_H_l, is_H_theta = 0, 0
# 正空间：右，下 = +, +
# 倒空间：左, 上 = +, +
# 朝着 x, y 轴 分别偏离 θ_1_x, θ_1_y 度
#%%
U1_0_NonZero_size = 0.5 # Unit: mm 不包含边框，图片 的 实际尺寸 5e-1
w0 = 5 # Unit: mm 束腰（z = 0 处），一般 设定地 比 U1_0_NonZero_size 小，但 CGH 生成结构的时候 得大
Enlarge_percentage = 0.1
# deff_structure_size_expect = 0.4 # Unit: mm 不包含边框，chi_2 的 实际尺寸 4e-1，一般 设定地 比 U1_0_NonZero_size 小，这样 从非线性过程 一开始，基波 就覆盖了 结构，而不是之后 衍射般 地 覆盖结构
deff_structure_length_expect = 1 # Unit: mm 调制区域 z 向长度（类似 z）
deff_structure_sheet_expect = 1.8 # Unit: μm z 向 切片厚度
# 一般得比 size_PerPixel 大？ 不用，z 不需要 离散化，因为已经定义 i_z0 = z0 / size_PerPixel，而不是 z0 // size_PerPixel
# 但一般得比 min(Tz * Duty_Cycle_z, Tz * (1-Duty_Cycle_z)) 小；
# 下面会令其：当 mz 不为零 时（你想 匹配了），若超过 0.1 * Tz 则直接等于 0.1 * Tz，这样在 大多数 情况下，小于 min(Tz * Duty_Cycle_z, Tz * (1-Duty_Cycle_z))
# 当 mz 为零时（你不想 匹配），则 保留 你的 原设定 不变。
Duty_Cycle_x = 0.5 # Unit: 1 x 向 占空比
Duty_Cycle_y = 0.5 # Unit: 1 y 向 占空比
Duty_Cycle_z = 0.5 # Unit: 1 z 向 占空比，一个周期内 （有）结构（的 长度） / 一个周期（的 长度）
structure_xy_mode = 'x'
Depth = 1 # 调制深度
# size_modulate = 1e-3 # Unit: mm n1 调制区域 的 横向尺寸，即 公式中的 d
is_continuous = 0 # 值为 1 表示 连续调制，否则 二值化调制
is_target_far_field = 1 # 值为 1 表示 想要的 U1_0 是远场分布
is_transverse_xy = 0 # 值为 1 表示 对生成的 structure 做转置
is_reverse_xy = 0 # 值为 1 表示 对生成的 structure 做 1 - structure （01 反转）
is_positive_xy = 1 # 值为 1 表示 正常的 占空比 逻辑；若为 0 则表示： 占空比 := 一个周期内，无结构长度 / 一个周期长度
#%%
lam1 = 1.5 # Unit: um 基波波长
is_air_pump, is_air, T = 0, 0, 25 # is_air = 0, 1, 2 分别表示 LN, 空气, KTP；T 表示 温度
#%%
Tx, Ty, Tz = 19.769, 20, 18.139 # Unit: um "2*lc"，测试： 0 度 - 20.155, 20, 17.885 、 -2 度 ： 6.633, 20, 18.437 、-3 度 ： 4.968, 20, 19.219
mx, my, mz = -1, 1, 1
# 倒空间：右, 下 = +, +
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
# 定义 调制区域 的 横向实际像素、调制区域 的 实际横向尺寸

deff_structure_size_expect = U1_0_NonZero_size * ( 1 + Enlarge_percentage )
is_print and print("deff_structure_size_expect = {} mm".format(deff_structure_size_expect))

Ix, Iy, deff_structure_size = Cal_IxIy(I1_x, I1_y, 
                                       deff_structure_size_expect, size_PerPixel, 
                                       is_print)

#%%
# 需要先将 目标 U1_0_NonZero = img_squared 给 放大 或 缩小 到 与 全息图（结构） 横向尺寸 Ix, Iy 相同，才能开始 之后的工作

border_width, img_squared_resize_full_name, img_squared_resize = img_squared_Resize(img_name, img_name_extension, img_squared, 
                                                                                    Ix, Iy, I1_x, 
                                                                                    is_print, )

if (type(U1_name) != str) or U1_name == "":
    #%%
    # U1_0 = U(x, y, 0) = img_squared_resize
    
    if is_phase_only == 1:
        U1_0 = np.power(math.e, (img_squared_resize.astype(np.complex128()) / 255 * 2 * math.pi - math.pi) * 1j) # 变成相位图
    else:
        U1_0 = img_squared_resize.astype(np.complex128)
    
    #%%
    # 预处理 输入场
    
    n1, k1 = Cal_n(size_PerPixel, 
                   is_air, 
                   lam1, T, p = "e")
    
    U1_0 = pump_LG(img_squared_resize_full_name, 
                   Ix, Iy, size_PerPixel, 
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
    # 导入 方形，以及 加边框 的 图片
    
    U1_full_name = U1_name + (is_save_txt and ".txt" or ".mat")
    U1_0 = np.loadtxt(U1_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U1_full_name)['U'] # 加载 复振幅场
    
    U1_0 = cv2.resize(np.real(U1_0), (Ix, Iy), interpolation=cv2.INTER_AREA) + cv2.resize(np.imag(U1_0), (Ix, Iy), interpolation=cv2.INTER_AREA) * 1j
    # U1_0 必须 resize 为 Ix,Iy 大小； 
    # 但 cv2 、 skimage.transform 中 resize 都能处理 图片 和 float64，
    # 但似乎 没有东西 能直接 处理 complex128，但可 分别处理 实部和虚部，再合并为 complex128

#%%

n1, k1 = Cal_n(size_PerPixel, 
               is_air, 
               lam1, T, p = "e")

#%%

lam2 = lam1 / 2

n2, k2 = Cal_n(size_PerPixel, 
               is_air, 
               lam2, T, p = "e")

#%%

dk, lc, Tz = Cal_lc_SHG(k1, k2, Tz, size_PerPixel, 
                        is_print = 0)

Gx, Gy, Gz = Cal_GxGyGz(mx, my, mz,
                        Tx, Ty, Tz, size_PerPixel, 
                        is_print)

#%%
# 定义 调制区域切片厚度 的 纵向实际像素、调制区域切片厚度 的 实际纵向尺寸

diz, deff_structure_sheet = Cal_diz(deff_structure_sheet_expect, deff_structure_length_expect, size_PerPixel, 
                                    Tz, mz,
                                    is_print)

#%%
# 定义 调制区域 的 纵向实际像素、调制区域 的 实际纵向尺寸

sheets_num, Iz, deff_structure_length = Cal_Iz_structure(diz, 
                                                         deff_structure_length_expect, size_PerPixel, 
                                                         is_print)

#%%

Tz_unit = (Tz / 1000) / size_PerPixel

#%%
# 开始生成 调制函数 structure 和 modulation = n1 - Depth * structure，以及 structure_opposite = 1 - structure 及其 modulation

structure = structure_Generate(U1_0, structure_xy_mode, 
                               Duty_Cycle_x, Duty_Cycle_y, 
                               is_positive_xy, 
                               #%%
                               Gx, Gy, 
                               is_Gauss, l, 
                               is_continuous, 
                               #%%
                               is_target_far_field, is_transverse_xy, is_reverse_xy, )

vmax_structure, vmin_structure = 1, 0
vmax_modulation, vmin_modulation = n1, n1 - Depth

plot_2d(I1_x, I1_y, size_PerPixel, diz, 
        structure, location + "\\" + "n1_structure" + img_name_extension, "n1_structure", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        0, is_colorbar_on, 0, vmax_structure, vmin_structure)

modulation = n1 - Depth * structure
modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (n1, n1))

plot_2d(I1_x, I1_y, size_PerPixel, diz, 
        modulation_squared, location + "\\" + "n1_modulation_squared" + img_name_extension, "n1_modulation_squared", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        0, is_colorbar_on, 0, vmax_modulation, vmin_modulation)

#%%

if mz != 0:

    structure_opposite = 1 - structure

    plot_2d(I1_x, I1_y, size_PerPixel, diz, 
            structure_opposite, location + "\\" + "n1_structure_opposite" + img_name_extension, "n1_structure_opposite", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            0, is_colorbar_on, 0, vmax_structure, vmin_structure)

    modulation_opposite = n1 - Depth * structure_opposite
    modulation_opposite_squared = np.pad(modulation_opposite, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (n1, n1))

    plot_2d(I1_x, I1_y, size_PerPixel, diz, 
            modulation_opposite_squared, location + "\\" + "n1_modulation_opposite_squared" + img_name_extension, "n1_modulation_opposite_squared", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            0, is_colorbar_on, 0, vmax_modulation, vmin_modulation)

#%%
# 逐层 绘制 并 输出 structure
if is_save == 1:
    if not os.path.isdir("0.n1_modulation_squared"):
        os.makedirs("0.n1_modulation_squared")

def structure_Generate_z(for_th, fors_num, *arg, ):
    
    iz = for_th * diz
    
    if mz != 0: # 如果 要用 Tz，则如下 分层；
    
        if np.mod(iz, Tz_unit) < Tz_unit * Duty_Cycle_z: # 如果 左端面 小于 占空比 【减去一个微小量（比如 diz / 10）】，则以 正向畴结构 输出为 该端面结构
            modulation = n1 - Depth * structure
            
        else: # 如果 左端面 大于等于 占空比，则以 反向畴结构 输出为 该端面结构
            modulation = n1 - Depth * structure_opposite
        modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (n1, n1))    
        
        modulation_squared_full_name = str(for_th) + (is_save_txt and ".txt" or ".mat")
        modulation_squared_address = location + "\\" + "0.n1_modulation_squared" + "\\" + modulation_squared_full_name
        
        if is_save == 1:
            
            np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') if is_save_txt else savemat(modulation_squared_address, {'n1_modulation_squared':modulation_squared})
    
    else: # 如果不用 Tz，则 z 向 无结构，则一直输出 正向畴
    
        modulation = n1 - Depth * structure
        modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (n1, n1))
        
        modulation_squared_full_name = str(for_th) + (is_save_txt and ".txt" or ".mat")
        modulation_squared_address = location + "\\" + "0.n1_modulation_squared" + "\\" + modulation_squared_full_name
        
        if is_save == 1:
            
            np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') if is_save_txt else savemat(modulation_squared_address, {'n1_modulation_squared':modulation_squared})
            
my_thread(10, sheets_num, 
          structure_Generate_z, noop, noop, 
          is_ordered = 1, is_print = is_print, )