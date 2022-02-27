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
# import copy
from PIL import Image
Image.MAX_IMAGE_PIXELS = 10E10 #Image 的 默认参数 无法处理那么大的图片
import threading
# import scipy
from scipy.io import loadmat, savemat
import time
from fun_plot import plot_1d, plot_2d, plot_3d_XYZ, plot_3d_XYz
from fun_pump import pump_LG
from fun_SSI import Cal_diz, Cal_Iz_structure
from fun_linear import Cal_n
from fun_nonlinear import Cal_lc_SHG, Cal_GxGyGz
from a_Image_Add_Black_border import Image_Add_Black_border

#%%
U1_txt_name = ""
file_full_name = "l=1.png"
border_percentage = 0.3 # 边框 占图片的 百分比，也即 图片 放大系数
#%%
phase_only = 0
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
deff_structure_size_expect = 0.4 # Unit: mm 不包含边框，chi_2 的 实际尺寸 4e-1，一般 设定地 比 U1_0_NonZero_size 小，这样 从非线性过程 一开始，基波 就覆盖了 结构，而不是之后 衍射般 地 覆盖结构
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
Depth = 2 # 调制深度
# size_modulate = 1e-3 # Unit: mm χ2 调制区域 的 横向尺寸，即 公式中的 d
is_continuous = 0 # 值为 1 表示 连续调制，否则 二值化调制
is_target_far_field = 1 # 值为 1 表示 想要的 U1_0 是远场分布
is_transverse_xy = 0 # 值为 1 表示 对生成的 structure 做转置
is_reverse_xy = 0 # 值为 1 表示 对生成的 structure 做 1 - structure （01 反转）
is_positive_xy = 1 # 值为 1 表示 正常的 占空比 逻辑；若为 0 则表示： 占空比 := 一个周期内，无结构长度 / 一个周期长度
is_no_backgroud = 0 # 若为 1.则 1,-1 调制，将变为 0,-2 调制
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
is_energy = 1
vmax, vmin = 1, 0

def image_border(src, dst, loc='a', width=3, color=(0, 0, 0, 255)):
    '''
    src: (str) 需要加边框的图片路径
    dst: (str) 加边框的图片保存路径
    loc: (str) 边框添加的位置, 默认是'a'(
        四周: 'a' or 'all'
        上: 't' or 'top'
        右: 'r' or 'rigth'
        下: 'b' or 'bottom'
        左: 'l' or 'left'
    )
    width: (int) 边框宽度 (默认是3)
    color: (int or 3-tuple) 边框颜色 (默认是0, 表示黑色; 也可以设置为三元组表示RGB颜色)
    '''
    # 读取图片
    img_ori = Image.open(src)
    w = img_ori.size[0]
    h = img_ori.size[1]

    # 添加边框
    if loc in ['a', 'all']:
        w += 2*width
        h += 2*width
        img_new = Image.new('RGBA', (w, h), color)
        img_new.paste(img_ori, (width, width))
    elif loc in ['t', 'top']:
        h += width
        img_new = Image.new('RGBA', (w, h), color)
        img_new.paste(img_ori, (0, width, w, h))
    elif loc in ['r', 'right']:
        w += width
        img_new = Image.new('RGBA', (w, h), color)
        img_new.paste(img_ori, (0, 0, w-width, h))
    elif loc in ['b', 'bottom']:
        h += width
        img_new = Image.new('RGBA', (w, h), color)
        img_new.paste(img_ori, (0, 0, w, h-width))
    elif loc in ['l', 'left']:
        w += width
        img_new = Image.new('RGBA', (w, h), color)
        img_new.paste(img_ori, (width, 0, w, h))
    else:
        pass

    # 保存图片
    img_new.save(dst)
        
#%%
# 预处理 导入图片 为方形，并加边框

Image_Add_Black_border(file_full_name, border_percentage)
    
#%%
# 导入 方形，以及 加边框 的 图片

file_name = os.path.splitext(file_full_name)[0]
file_name_extension = os.path.splitext(file_full_name)[1]

location = os.path.dirname(os.path.abspath(__file__))
file_squared_address = location + "\\" + "1." + file_name + "_squared" + file_name_extension
file_squared_bordered_address = location + "\\" + "2." + file_name + "_squared" + "_bordered" + file_name_extension

img_squared = cv2.imdecode(np.fromfile(file_squared_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
img_squared_bordered = cv2.imdecode(np.fromfile(file_squared_bordered_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片

# img_squared_bordered = np.roll(img_squared_bordered, 50, axis=0) # 往下（行） 循环平移 50 像素
# img_squared_bordered = np.pad(img_squared_bordered, ((50, 50), (50, 50)), 'constant', constant_values = (0, 0)) # 看看 加 黑边框（填充零）后 是啥样子

size_fig = img_squared_bordered.shape[0] / dpi

#%%
# 定义 像素尺寸、加边框后的 U1 像素、加边框后的 U1 尺寸

size_PerPixel = U1_0_NonZero_size / img_squared.shape[0] # Unit: mm / 个 每个 像素点 的 尺寸，相当于 △x = △y = △z
I1_x, I1_y = img_squared_bordered.shape[0], img_squared_bordered.shape[1]
U1_size = I1_x * size_PerPixel # Unit: mm 包含 边框 后，图片 的 实际尺寸
print("U1_size = U2_size = {} mm".format(U1_size))

#%%
# 定义 调制区域 的 横向实际像素、调制区域 的 实际横向尺寸

Ix, Iy = int( deff_structure_size_expect / size_PerPixel ), int( deff_structure_size_expect / size_PerPixel )
# Ix, Iy 需要与 I1_x, I1_y 同奇偶性，这样 加边框 才好加（对称地加 而不用考虑 左右两边加的量 可能不一样）
Ix, Iy = Ix + np.mod(I1_x - Ix,2), Iy + np.mod(I1_y - Iy,2)
deff_structure_size = Ix * size_PerPixel # Unit: mm 不包含 边框，调制区域 的 实际横向尺寸
print("deff_structure_size = {} mm".format(deff_structure_size))

#%%
# 需要先将 目标 U1_0_NonZero = img_squared 给 放大 或 缩小 到 与 全息图（结构） 横向尺寸 Ix, Iy 相同，才能开始 之后的工作

img_squared_resize = cv2.resize(img_squared, (Ix, Iy), interpolation=cv2.INTER_AREA)
img_squared_resize_full_name = "1." + file_name + "_squared" + "_resize" + file_name_extension
img_squared_resize_address = location + "\\" + "1." + file_name + "_squared" + "_resize" + file_name_extension
cv2.imwrite(img_squared_resize_full_name, img_squared_resize) # 保存 img_squared_resize
print("img_squared_resize.shape = {}".format(img_squared_resize.shape))

img_squared_resize_bordered_address = location + "\\" + "2." + file_name + "_squared" + "_resize" + "_bordered" + file_name_extension
border_width = (I1_x - Ix) // 2
image_border(img_squared_resize_address, img_squared_resize_bordered_address, loc='a', width=border_width, color=(0, 0, 0, 255))
img_squared_resize_bordered = cv2.imdecode(np.fromfile(img_squared_resize_bordered_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
print("structure_squared.shape = img_squared_resize_bordered.shape = {}".format(img_squared_resize_bordered.shape))

if (type(U1_txt_name) != str) or U1_txt_name == "":
    #%%
    # U1_0 = U(x, y, 0) = img_squared_resize
    
    if phase_only == 1:
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
                   1, is_colorbar_on, is_energy, vmax, vmin) 

else:
    
    #%%
    # 导入 方形，以及 加边框 的 图片
    
    U1_txt_full_name = U1_txt_name + (is_save_txt and ".txt" or ".mat")
    U1_txt_short_name = U1_txt_name.replace('6. AST - ', '')
    U1_0 = np.loadtxt(U1_txt_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U1_txt_full_name)['U'] # 加载 复振幅场
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

dk, lc, Tz = Cal_lc_SHG(k1, k2, Tz, size_PerPixel, 
                        is_print = 0)

Gx, Gy, Gz = Cal_GxGyGz(mx, my, mz,
                        Tx, Ty, Tz, size_PerPixel, 
                        is_print = 1)

#%%
# 定义 调制区域切片厚度 的 纵向实际像素、调制区域切片厚度 的 实际纵向尺寸

diz, deff_structure_sheet = Cal_diz(deff_structure_sheet_expect, deff_structure_length_expect, size_PerPixel, 
                                    Tz, mz,
                                    is_print = 1)

#%%
# 定义 调制区域 的 纵向实际像素、调制区域 的 实际纵向尺寸

sheets_num, Iz, deff_structure_length = Cal_Iz_structure(diz, 
                                                         deff_structure_length_expect, size_PerPixel, 
                                                         is_print = 1)

#%%

Tz_unit = (Tz / 1000) / size_PerPixel

#%%

def step(U,mode='x'):
    if mode == 'x':
        return ( U > (2 * is_positive_xy - 1) * np.cos(Duty_Cycle_x * math.pi) ).astype(np.int8()) # uint8 会导致 之后 structure 和 modulation 也变成 无符号 整形，以致于 在 0 - 1 时 变成 255 而不是 -1...
    elif mode == 'y':
        return ( U > (2 * is_positive_xy - 1) * np.cos(Duty_Cycle_y * math.pi) ).astype(np.int8()) # uint8 会导致 之后 structure 和 modulation 也变成 无符号 整形，以致于 在 0 - 1 时 变成 255 而不是 -1...

def CGH(U,mode='x'):
    i1_x0, i1_y0 = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
    i1_x0_shift, i1_y0_shift = i1_x0 - Ix // 2, i1_y0 - Iy // 2
    if is_Gauss == 1 and l == 0:
        if mode == 'x*y':
            cgh = np.cos(Gx * i1_x0_shift)
            cgh_x = step(cgh,'x') if is_continuous == 0 else 0.5 + 0.5 * cgh
            cgh = np.cos(Gy * i1_y0_shift)
            cgh_y = step(cgh,'y') if is_continuous == 0 else 0.5 + 0.5 * cgh
            cgh = cgh_x * cgh_y
        elif mode == 'x':
            cgh = np.cos(Gx * i1_x0_shift)
            cgh = step(cgh,'x') if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'y':
            cgh = np.cos(Gy * i1_y0_shift)
            cgh = step(cgh,'y') if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'x+y':
            cgh = np.cos(Gx * i1_x0_shift + Gy * i1_y0_shift)
            cgh = step(cgh,'x') if is_continuous == 0 else 0.5 + 0.5 * cgh # 在所有方向的占空比都认为是 Duty_Cycle_x
        return cgh
    else:
        if mode == 'x*y':
            cgh = np.cos(Gx * i1_x0_shift - ( np.angle(U) + math.pi )) - np.cos( np.arcsin( np.abs(U) / np.max( np.abs(U) ) ) )
            cgh_x = step(cgh,'x') if is_continuous == 0 else 0.5 + 0.5 * cgh
            cgh = np.cos(Gy * i1_y0_shift - ( np.angle(U) + math.pi )) - np.cos( np.arcsin( np.abs(U) / np.max( np.abs(U) ) ) )
            cgh_y = step(cgh,'y') if is_continuous == 0 else 0.5 + 0.5 * cgh
            cgh = cgh_x * cgh_y
        elif mode == 'x':
            cgh = np.cos(Gx * i1_x0_shift - ( np.angle(U) + math.pi )) - np.cos( np.arcsin( np.abs(U) / np.max( np.abs(U) ) ) )
            cgh = step(cgh,'x') if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'y':
            cgh = np.cos(Gy * i1_y0_shift - ( np.angle(U) + math.pi )) - np.cos( np.arcsin( np.abs(U) / np.max( np.abs(U) ) ) )
            cgh = step(cgh,'y') if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'x+y':
            cgh = np.cos(Gx * i1_x0_shift + Gy * i1_y0_shift - ( np.angle(U) + math.pi )) - np.cos( np.arcsin( np.abs(U) / np.max( np.abs(U) ) ) )
            cgh = step(cgh,'x') if is_continuous == 0 else 0.5 + 0.5 * cgh
        return cgh

#%%
# 开始生成 调制函数 structure 和 modulation = 1 - is_no_backgroud - Depth * structure，以及 structure_opposite = 1 - structure 及其 modulation

# structure = np.zeros( (Ix, Iy) ,dtype=np.int8() )

def structure_generate(U1_0, mode='x*y'):
    
    if is_target_far_field == 0: # 如果 想要的 U1_0 是近场（晶体后端面）分布
        
        g1_0 = np.fft.fft2(U1_0)
        g1_0_shift = np.fft.fftshift(g1_0)
        
        if is_transverse_xy == 1:
            structure = CGH(g1_0_shift,mode).T # 转置（沿 右下 对角线 翻转）
        else:
            structure = CGH(g1_0_shift,mode)[::-1] # 上下翻转
    else: # 如果 想要的 U1_0 是远场分布
        if is_transverse_xy == 1:
            structure = CGH(U1_0,mode).T # 转置（沿 右下 对角线 翻转）
        else:
            structure = CGH(U1_0,mode)[::-1] # 上下翻转
        
    if is_reverse_xy == 1:
        structure = 1 - structure

    return structure

vmax_structure, vmin_structure = 1, 0
vmax_modulation, vmin_modulation = 1 - is_no_backgroud, -1 - is_no_backgroud

structure = structure_generate(U1_0, structure_xy_mode)

plot_2d(I1_x, I1_y, size_PerPixel, diz, 
        structure, location + "\\" + "χ2_structure" + file_name_extension, "χ2_structure", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        0, is_colorbar_on, 0, vmax_structure, vmin_structure)

modulation = 1 - is_no_backgroud - Depth * structure
modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (1 - is_no_backgroud, 1 - is_no_backgroud))

plot_2d(I1_x, I1_y, size_PerPixel, diz, 
        modulation_squared, location + "\\" + "χ2_modulation_squared" + file_name_extension, "χ2_modulation_squared", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        0, is_colorbar_on, 0, vmax_modulation, vmin_modulation)

#%%

if mz != 0:

    structure_opposite = 1 - structure

    plot_2d(I1_x, I1_y, size_PerPixel, diz, 
            structure_opposite, location + "\\" + "χ2_structure_opposite" + file_name_extension, "χ2_structure_opposite", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            0, is_colorbar_on, 0, vmax_structure, vmin_structure)

    modulation_opposite = 1 - is_no_backgroud - Depth * structure_opposite
    modulation_opposite_squared = np.pad(modulation_opposite, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (1 - is_no_backgroud, 1 - is_no_backgroud))

    plot_2d(I1_x, I1_y, size_PerPixel, diz, 
            modulation_opposite_squared, location + "\\" + "χ2_modulation_opposite_squared" + file_name_extension, "χ2_modulation_opposite_squared", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            0, is_colorbar_on, 0, vmax_modulation, vmin_modulation)

#%%
# 逐层 绘制 并 输出 structure
if is_save == 1:
    if not os.path.isdir("0.χ2_modulation_squared"):
        os.makedirs("0.χ2_modulation_squared")

iz = 0

#%%
# 单线程

# tick_start = time.time()
# for sheet_th in range(sheets_num):
    
#     iz = sheet_th * diz
    
#     if mz != 0: # 如果 要用 Tz，则如下 分层；
    
#         if np.mod(iz, Tz_unit) < Tz_unit * Duty_Cycle_z: # 如果 左端面 小于 占空比 【减去一个微小量（比如 diz / 10）】，则以 正向畴结构 输出为 该端面结构
#             modulation = 1 - is_no_backgroud - Depth * structure
#         else: # 如果 左端面 大于等于 占空比，则以 反向畴结构 输出为 该端面结构
#             modulation = 1 - is_no_backgroud - Depth * structure_opposite
#         modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (1 - is_no_backgroud, 1 - is_no_backgroud))
            
#         modulation_squared_full_name = str(sheet_th) + (is_save_txt and ".txt" or ".mat")
#         # modulation_squared_full_name = str(sheet_th) + ".txt" # 52 kb
#         # modulation_squared_full_name = str(sheet_th) + ".mat" # 21 kb
#         # modulation_squared_full_name = str(sheet_th) # 21 kb
#         modulation_squared_address = location + "\\" + "0.χ2_modulation_squared" + "\\" + modulation_squared_full_name
        
#         if is_save == 1:
            
#             np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') if is_save_txt else savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared})
#             # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') # 输出 iz ~ iz + diz 间的 结构分布 # 52 kb
#             # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i', delimiter=",") # 52 kb
#             # savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared}) # 21 kb
#             # np.save(modulation_squared_address, modulation_squared) # 21 kb
    
#     else: # 如果不用 Tz，则 z 向 无结构，则一直输出 正向畴
    
#         modulation = 1 - is_no_backgroud - Depth * structure
#         modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (1 - is_no_backgroud, 1 - is_no_backgroud))
        
#         modulation_squared_full_name = str(sheet_th) + (is_save_txt and ".txt" or ".mat")
#         # modulation_squared_full_name = str(sheet_th) + ".txt" # 52 kb
#         # modulation_squared_full_name = str(sheet_th) + ".mat" # 21 kb
#         # modulation_squared_full_name = str(sheet_th) # 21 kb
#         modulation_squared_address = location + "\\" + "0.χ2_modulation_squared" + "\\" + modulation_squared_full_name
        
#         if is_save == 1:
            
#             np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') if is_save_txt else savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared})
#             # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') # 输出 iz ~ iz + diz 间的 结构分布 # 52 kb
#             # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i', delimiter=",") # 52 kb
#             # savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared}) # 21 kb
#             # np.save(modulation_squared_address, modulation_squared) # 21 kb
# print("----- consume time: {} s -----".format(time.time() - tick_start))

#%%
# 多线程

global thread_th, for_th # 好怪，作为 被封装 和 被引用的 函数，还得在 这一层 声明 全局变量，光是在 内层 子线程 里声明 的话，没用。
thread_th = 0 # 生产出的 第几个 / 一共几个 线程，全局
for_th = 0 # 正在计算到的 第几个 for 循环的序数，全局（非顺序的情况下，这个的含义只是计数，即一共计算了几个 序数 了）
con = threading.Condition() # 锁不必定义为全局变量

class Producer(threading.Thread):
    """线程 生产者"""
    def __init__(self, threads_num, fors_num):
        self.threads_num = threads_num
        self.fors_num = fors_num
        self.for_th = 0 # 生产出的 第几个 for_th，这个不必定义为全局变量
        super().__init__()
    
    def run(self):
        global thread_th
        
        con.acquire()
        while True:
            
            if self.for_th >= self.fors_num and for_th == self.fors_num: # 退出生产线程 的 条件：p 线程 完成 其母线程功能，且 最后一个 t 线程 完成其子线程功能
                break
            else:
                if thread_th >= self.threads_num or self.for_th == self.fors_num : # 暂停生产线程 的 条件： 运行线程数 达到 设定，或 p 线程 完成 其母线程功能
                    con.notify()
                    con.wait()
                else:
                    # print(self.for_th)
                    t = Customer('thread:%s' % thread_th, self.for_th)
                    t.setDaemon(True)
                    t.start()
                    
                    thread_th += 1
                    # print('已生产了 共 {} 个 线程'.format(thread_th))
                    self.for_th += 1
                    # print('已算到了 第 {} 个 for_th'.format(self.for_th))
                    # time.sleep(1)
        con.release()

class Customer(threading.Thread):
    """线程 消费者"""
    def __init__(self, name, for_th):
        self.thread_name = name
        self.for_th = for_th
        super().__init__()
        
    def run(self):
        global thread_th, for_th
        """----- 你 需累积的 全局变量，替换 最末一个 g2_z_plus_dz_shift -----"""
            
        """----- your code begin 1 -----"""
        if self.for_th == fors_num - 1:
            iz = Iz
        else:
            iz = self.for_th * diz
        
        if mz != 0: # 如果 要用 Tz，则如下 分层；
        
            if np.mod(iz, Tz_unit) < Tz_unit * Duty_Cycle_z: # 如果 左端面 小于 占空比 【减去一个微小量（比如 diz / 10）】，则以 正向畴结构 输出为 该端面结构
                modulation = 1 - is_no_backgroud - Depth * structure
                
            else: # 如果 左端面 大于等于 占空比，则以 反向畴结构 输出为 该端面结构
                modulation = 1 - is_no_backgroud - Depth * structure_opposite
            modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (1 - is_no_backgroud, 1 - is_no_backgroud))    
            
            modulation_squared_full_name = str(self.for_th) + (is_save_txt and ".txt" or ".mat")
            # modulation_squared_full_name = str(sheet_th) + ".txt" # 52 kb
            # modulation_squared_full_name = str(sheet_th) + ".mat" # 21 kb
            # modulation_squared_full_name = str(sheet_th) # 21 kb
            modulation_squared_address = location + "\\" + "0.χ2_modulation_squared" + "\\" + modulation_squared_full_name
            
            if is_save == 1:
                
                np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') if is_save_txt else savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared})
                # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') # 输出 iz ~ iz + diz 间的 结构分布 # 52 kb
                # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i', delimiter=",") # 52 kb
                # savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared}) # 21 kb
                # np.save(modulation_squared_address, modulation_squared) # 21 kb
        
        else: # 如果不用 Tz，则 z 向 无结构，则一直输出 正向畴
        
            modulation = 1 - is_no_backgroud - Depth * structure
            modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (1 - is_no_backgroud, 1 - is_no_backgroud))
            
            modulation_squared_full_name = str(self.for_th) + (is_save_txt and ".txt" or ".mat")
            # modulation_squared_full_name = str(sheet_th) + ".txt" # 52 kb
            # modulation_squared_full_name = str(sheet_th) + ".mat" # 21 kb
            # modulation_squared_full_name = str(sheet_th) # 21 kb
            modulation_squared_address = location + "\\" + "0.χ2_modulation_squared" + "\\" + modulation_squared_full_name
            
            if is_save == 1:
                
                np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') if is_save_txt else savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared})
                # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') # 输出 iz ~ iz + diz 间的 结构分布 # 52 kb
                # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i', delimiter=",") # 52 kb
                # savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared}) # 21 kb
                # np.save(modulation_squared_address, modulation_squared) # 21 kb
        """----- your code end 1 -----"""
        
        con.acquire() # 上锁
        
        if is_ordered == 1:
            
            while True:
                if for_th == self.for_th:
                    # print(self.for_th)
                    """----- your code begin 2 -----"""
                    
                    """----- your code end 2 -----"""
                    for_th += 1
                    break
                else:
                    con.notify()
                    con.wait() # 但只有当 for_th 不等于 self.for_th， 才等待
        else:
            
            # print(self.for_th)
            """----- your code begin 2 -----"""
            
            """----- your code end 2 -----"""
            for_th += 1
        
        thread_th -= 1 # 在解锁之前 减少 1 个线程数量，以便 p 线程 收到消息后，生产 1 个 线程出来
        con.notify() # 无论如何 都得通知一下 其他线程，让其别 wait() 了
        con.release() # 解锁

"""----- your code begin 0 -----"""
is_ordered = 0 # for_th 是否 按顺序执行
threads_num = 10 # 需要开启 多少个线程 持续计算
fors_num = sheets_num # 需要计算 for 循环中 多少个 序数
"""----- your code end 0 -----"""

tick_start = time.time()

p = Producer(threads_num, fors_num)
p.setDaemon(True)
p.start()
p.join() # 添加join使 p 线程执行完

print("----- consume time: {} s -----".format(time.time() - tick_start))

#%%
# sheet_th = 0 # 层序号
# is_in_DutyCycle = 1

# tick_start = time.time()
# for sheet_th in range(sheets_num): # range(sheets_num + 1) 也行，这样 后端面为 Iz // diz + 1，始终大于 Iz 于 1 个 diz 之内（这里注释里 说的 Iz 在定义上 是 连续的）

#     iz = sheet_th * diz
    
#     if mz != 0: # 如果 要用 Tz，则如下 分层；
        
#         if np.mod(iz, Tz_unit) < Tz_unit * Duty_Cycle_z: # 如果 左端面 小于 占空比 【减去一个微小量（比如 diz / 10）】，则以 正向畴结构 输出为 该端面结构
#             if is_in_DutyCycle == 0:
#                 structure = 1 - structure
#                 modulation = 1 - is_no_backgroud - Depth * structure
#                 modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (1 - is_no_backgroud, 1 - is_no_backgroud))
#             is_in_DutyCycle = 1
            
#             modulation_squared_full_name = str(sheet_th) + (is_save_txt and ".txt" or ".mat")
#             # modulation_squared_full_name = str(sheet_th) + ".txt" # 52 kb
#             # modulation_squared_full_name = str(sheet_th) + ".mat" # 21 kb
#             # modulation_squared_full_name = str(sheet_th) # 21 kb
#             modulation_squared_address = location + "\\" + "0.χ2_modulation_squared" + "\\" + modulation_squared_full_name
            
#             if is_save == 1:
                
#                 np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') if is_save_txt else savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared})
#                 # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') # 输出 iz ~ iz + diz 间的 结构分布 # 52 kb
#                 # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i', delimiter=",") # 52 kb
#                 # savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared}) # 21 kb
#                 # np.save(modulation_squared_address, modulation_squared) # 21 kb
            
#         else: # 如果 左端面 大于等于 占空比，则以 反向畴结构 输出为 该端面结构
#             if is_in_DutyCycle == 1:
#                 structure = 1 - structure
#                 modulation = 1 - is_no_backgroud - Depth * structure
#                 modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (1 - is_no_backgroud, 1 - is_no_backgroud))
#             is_in_DutyCycle = 0
            
#             modulation_squared_full_name = str(sheet_th) + (is_save_txt and ".txt" or ".mat")
#             # modulation_squared_full_name = str(sheet_th) + ".txt" # 52 kb
#             # modulation_squared_full_name = str(sheet_th) + ".mat" # 21 kb
#             # modulation_squared_full_name = str(sheet_th) # 21 kb
#             modulation_squared_address = location + "\\" + "0.χ2_modulation_squared" + "\\" + modulation_squared_full_name
            
#             if is_save == 1:
                
#                 np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') if is_save_txt else savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared})
#                 # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') # 输出 iz ~ iz + diz 间的 结构分布 # 52 kb
#                 # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i', delimiter=",") # 52 kb
#                 # savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared}) # 21 kb
#                 # np.save(modulation_squared_address, modulation_squared) # 21 kb
#     else: # 如果不用 Tz，则 z 向 无结构
        
#         modulation_squared_full_name = str(sheet_th) + (is_save_txt and ".txt" or ".mat")
#         # modulation_squared_full_name = str(sheet_th) + ".txt" # 52 kb
#         # modulation_squared_full_name = str(sheet_th) + ".mat" # 21 kb
#         # modulation_squared_full_name = str(sheet_th) # 21 kb
#         modulation_squared_address = location + "\\" + "0.χ2_modulation_squared" + "\\" + modulation_squared_full_name
        
#         if is_save == 1:
            
#             np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') if is_save_txt else savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared})
#             # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') # 输出 iz ~ iz + diz 间的 结构分布 # 52 kb
#             # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i', delimiter=",") # 52 kb
#             # savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared}) # 21 kb
#             # np.save(modulation_squared_address, modulation_squared) # 21 kb
            
# print("----- consume time: {} s -----".format(time.time() - tick_start))

#%%

# while iz < Iz: # 这样 后端面为 Iz // diz + 1，始终大于 Iz 于 1 个 diz 之内 【注意，这里的 Iz 在定义上 是 连续的】
    
#     if np.mod(iz, Tz_unit) < Tz_unit * Duty_Cycle_z: # 如果 左端面 小于 占空比 【减去一个微小量（比如 diz / 10）】，则以 正向畴结构 输出为 该端面结构
#         if is_in_DutyCycle == 0:
#             structure = 1 - structure
#             modulation = 1 - is_no_backgroud - Depth * structure
#             modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (1 - is_no_backgroud, 1 - is_no_backgroud))
#         is_in_DutyCycle = 1
        
#         modulation_squared_full_name = str(sheet_th) + (is_save_txt and ".txt" or ".mat")
#         # modulation_squared_full_name = str(sheet_th) + ".txt" # 52 kb
#         # modulation_squared_full_name = str(sheet_th) + ".mat" # 21 kb
#         # modulation_squared_full_name = str(sheet_th) # 21 kb
#         modulation_squared_address = location + "\\" + "0.χ2_modulation_squared" + "\\" + modulation_squared_full_name
        
#         if is_save == 1:
            
#             np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') if is_save_txt else savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared})
#             # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') # 输出 iz ~ iz + diz 间的 结构分布 # 52 kb
#             # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i', delimiter=",") # 52 kb
#             # savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared}) # 21 kb
#             # np.save(modulation_squared_address, modulation_squared) # 21 kb
        
#     else: # 如果 左端面 大于等于 占空比，则以 反向畴结构 输出为 该端面结构
#         if is_in_DutyCycle == 1:
#             structure = 1 - structure
#             modulation = 1 - is_no_backgroud - Depth * structure
#             modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant', constant_values = (1 - is_no_backgroud, 1 - is_no_backgroud))
#         is_in_DutyCycle = 0
        
#         modulation_squared_full_name = str(sheet_th) + (is_save_txt and ".txt" or ".mat")
#         # modulation_squared_full_name = str(sheet_th) + ".txt" # 52 kb
#         # modulation_squared_full_name = str(sheet_th) + ".mat" # 21 kb
#         # modulation_squared_full_name = str(sheet_th) # 21 kb
#         modulation_squared_address = location + "\\" + "0.χ2_modulation_squared" + "\\" + modulation_squared_full_name
        
#         if is_save == 1:
            
#             np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') if is_save_txt else savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared})
#             # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i') # 输出 iz ~ iz + diz 间的 结构分布 # 52 kb
#             # np.savetxt(modulation_squared_address, modulation_squared, fmt='%i', delimiter=",") # 52 kb
#             # savemat(modulation_squared_address, {'chi2_modulation_squared':modulation_squared}) # 21 kb
#             # np.save(modulation_squared_address, modulation_squared) # 21 kb
    
#     sheet_th += 1
#     iz = sheet_th * diz

#%%

