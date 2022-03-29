# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

#%%

import os
import numpy as np
from scipy.io import savemat
from fun_SSI import Cal_diz, Cal_Iz_structure
from fun_thread import noop, my_thread
from fun_CGH import structure_n1_Generate_2D
np.seterr(divide='ignore',invalid='ignore')

#%%

def structure_n1_3D(U1_name = "", 
                    img_full_name = "Grating.png", 
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
                    U1_0_NonZero_size = 1, w0 = 0.3, structure_size_Enlarge = 0.1, 
                    deff_structure_length_expect = 2, deff_structure_sheet_expect = 1.8, 
                    #%%
                    Duty_Cycle_x = 0.5, Duty_Cycle_y = 0.5, Duty_Cycle_z = 0.5, structure_xy_mode = 'x', Depth = 1, 
                    #%%
                    is_continuous = 1, is_target_far_field = 1, is_transverse_xy = 0, is_reverse_xy = 0, is_positive_xy = 1, 
                    #%%
                    lam1 = 0.8, is_air_pump = 0, is_air = 0, T = 25, 
                    #%%
                    Tx = 10, Ty = 10, Tz = "2*lc", 
                    mx = 0, my = 0, mz = 0, 
                    is_stripe = 0, 
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
                    is_energy = 0, vmax = 1, vmin = 0, 
                    #%%
                    is_print = 1, ):
    # #%%
    # U1_name = ""
    # img_full_name = "l=1.png"
    # #%%
    # is_phase_only = 0
    # is_LG, is_Gauss, is_OAM = 0, 1, 1
    # l, p = 1, 0
    # theta_x, theta_y = 1, 0
    # is_H_l, is_H_theta = 0, 0
    # #%%
    # U1_0_NonZero_size = 0.5 # Unit: mm 不包含边框，图片 的 实际尺寸 5e-1
    # w0 = 5 # Unit: mm 束腰（z = 0 处），一般 设定地 比 U1_0_NonZero_size 小，但 CGH 生成结构的时候 得大
    # structure_size_Enlarge = 0.1
    # # deff_structure_size_expect = 0.4 # Unit: mm 不包含边框，chi_2 的 实际尺寸 4e-1，一般 设定地 比 U1_0_NonZero_size 小，这样 从非线性过程 一开始，基波 就覆盖了 结构，而不是之后 衍射般 地 覆盖结构
    # deff_structure_length_expect = 1 # Unit: mm 调制区域 z 向长度（类似 z）
    # deff_structure_sheet_expect = 1.8 # Unit: μm z 向 切片厚度
    # # 一般得比 size_PerPixel 大？ 不用，z 不需要 离散化，因为已经定义 i_z0 = z0 / size_PerPixel，而不是 z0 // size_PerPixel
    # # 但一般得比 min(Tz * Duty_Cycle_z, Tz * (1-Duty_Cycle_z)) 小；
    # # 下面会令其：当 mz 不为零 时（你想 匹配了），若超过 0.1 * Tz 则直接等于 0.1 * Tz，这样在 大多数 情况下，小于 min(Tz * Duty_Cycle_z, Tz * (1-Duty_Cycle_z))
    # # 当 mz 为零时（你不想 匹配），则 保留 你的 原设定 不变。
    # Duty_Cycle_x = 0.5 # Unit: 1 x 向 占空比
    # Duty_Cycle_y = 0.5 # Unit: 1 y 向 占空比
    # Duty_Cycle_z = 0.5 # Unit: 1 z 向 占空比，一个周期内 （有）结构（的 长度） / 一个周期（的 长度）
    # structure_xy_mode = 'x'
    # Depth = 2 # 调制深度
    # # size_modulate = 1e-3 # Unit: mm n1 调制区域 的 横向尺寸，即 公式中的 d
    # is_continuous = 1 # 值为 1 表示 连续调制，否则 二值化调制
    # is_target_far_field = 1 # 值为 1 表示 想要的 U1_0 是远场分布
    # is_transverse_xy = 0 # 值为 1 表示 对生成的 structure 做转置
    # is_reverse_xy = 0 # 值为 1 表示 对生成的 structure 做 1 - structure （01 反转）
    # is_positive_xy = 1 # 值为 1 表示 正常的 占空比 逻辑；若为 0 则表示： 占空比 := 一个周期内，无结构长度 / 一个周期长度
    # #%%
    # lam1 = 1.5 # Unit: um 基波波长
    # is_air, T = 0, 25 # is_air = 0, 1, 2 分别表示 LN, 空气, KTP；T 表示 温度
    # #%%
    # Tx, Ty, Tz = 6.633, 20, 18.437 # Unit: um "2*lc"，测试： 0 度 - 20.155, 20, 17.885 、 -2 度 ： 6.633, 20, 18.437 、-3 度 ： 4.968, 20, 19.219
    # mx, my, mz = -1, 0, 1
    # # 倒空间：右, 下 = +, +
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
    # is_energy = 0
    # vmax, vmin = 1, 0

    #%%

    n1, k1, k1_z_shift, lam2, n2, k2, k2_z_shift, \
    dk, lc, Tz, Gx, Gy, Gz, \
    size_PerPixel, U1_0, g1_shift, \
    structure, structure_opposite, modulation, modulation_opposite, modulation_squared, modulation_opposite_squared \
        = structure_n1_Generate_2D(U1_name,
                                   img_full_name,
                                   is_phase_only,
                                   # %%
                                   z_pump,
                                   is_LG, is_Gauss, is_OAM,
                                   l, p,
                                   theta_x, theta_y,
                                   # %%
                                   is_random_phase,
                                   is_H_l, is_H_theta, is_H_random_phase,
                                   # %%
                                   U1_0_NonZero_size, w0, structure_size_Enlarge,
                                   Duty_Cycle_x, Duty_Cycle_y, structure_xy_mode, Depth,
                                   # %%
                                   is_continuous, is_target_far_field, is_transverse_xy, is_reverse_xy, is_positive_xy,
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
    # 逐层 绘制 并 输出 structure
    if is_save == 1:
        if not os.path.isdir("0.n1_modulation_squared"):
            os.makedirs("0.n1_modulation_squared")

    def structure_n1_Generate_z(for_th, fors_num, *arg, ):
        iz = for_th * diz

        if mz != 0:  # 如果 要用 Tz，则如下 分层；

            if is_stripe == 0:
                if iz - iz // Tz_unit * Tz_unit < Tz_unit * Duty_Cycle_z:  # 如果 左端面 小于 占空比 【减去一个微小量（比如 diz / 10）】，则以 正向畴结构 输出为 该端面结构
                    m = modulation_squared
        
                else:  # 如果 左端面 大于等于 占空比，则以 反向畴结构 输出为 该端面结构
                    m = modulation_opposite_squared
            else:
                if structure_xy_mode == 'x': # 往右（列） 线性平移 mj[for_th] 像素
                    m = np.roll(modulation_squared, int(mx * Tx / Tz * iz), axis=1)
                elif structure_xy_mode == 'y': # 往下（行） 线性平移 mj[for_th] 像素
                    m = np.roll(modulation_squared, int(my * Ty / Tz * iz), axis=0)
                elif structure_xy_mode == 'xy': # 往右（列） 线性平移 mj[for_th] 像素
                    m = np.roll(modulation_squared, int(mx * Tx / Tz * iz), axis=1)
                    m = np.roll(modulation_squared, int(my * Ty / Tz * iz), axis=0)

            modulation_squared_full_name = str(for_th) + (is_save_txt and ".txt" or ".mat")
            modulation_squared_address = "0.n1_modulation_squared" + "\\" + modulation_squared_full_name

            if is_save == 1:
                np.savetxt(modulation_squared_address, m, fmt='%i') if is_save_txt else savemat(
                    modulation_squared_address, {'n1_modulation_squared': m})

        else:  # 如果不用 Tz，则 z 向 无结构，则一直输出 正向畴

            m = modulation_squared

            modulation_squared_full_name = str(for_th) + (is_save_txt and ".txt" or ".mat")
            modulation_squared_address = "0.n1_modulation_squared" + "\\" + modulation_squared_full_name

            if is_save == 1:
                np.savetxt(modulation_squared_address, m, fmt='%i') if is_save_txt else savemat(
                    modulation_squared_address, {'n1_modulation_squared': m})
                
    my_thread(10, sheets_num, 
              structure_n1_Generate_z, noop, noop, 
              is_ordered = 1, is_print = is_print, )
    
# structure_n1_3D(U1_name = "", 
#              img_full_name = "l=1.png", 
#              is_phase_only = 0, 
#              #%%
#              is_LG = 0, is_Gauss = 1, is_OAM = 1, 
#              l = 0, p = 0, 
#              theta_x = 0, theta_y = 0, 
#              is_H_l = 0, is_H_theta = 0, 
#              #%%
#              U1_0_NonZero_size = 0.5, w0 = 5, deff_structure_size_expect = 0.4, 
#              deff_structure_length_expect = 2, deff_structure_sheet_expect = 1.8, 
#              Duty_Cycle_x = 0.5, Duty_Cycle_y = 0.5, Duty_Cycle_z = 0.5, structure_xy_mode = 'x*y', Depth = 1, 
#              #%%
#              is_continuous = 1, is_target_far_field = 1, is_transverse_xy = 0, is_reverse_xy = 0, is_positive_xy = 1, 
#              #%%
#              lam1 = 0.8, is_air_pump = 0, is_air = 0, T = 25, 
#              Tx = 10, Ty = 10, Tz = 18, 
#              mx = 1, my = 1, mz = 1, 
#              #%%
#              is_save = 0, is_save_txt = 0, dpi = 100, 
#              #%%
#              cmap_2d = 'viridis', 
#              #%%
#              ticks_num = 6, is_contourf = 0, 
#              is_title_on = 1, is_axes_on = 1, 
#              is_mm = 1, is_propagation = 0, 
#              #%%
#              fontsize = 9, 
#              font = {'family': 'serif',
#                      'style': 'normal', # 'normal', 'italic', 'oblique'
#                      'weight': 'normal',
#                      'color': 'black', # 'black','gray','darkred'
#                      }, 
#              #%%
#              is_self_colorbar = 0, is_colorbar_on = 1, 
#              vmax = 1, vmin = 0)