# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
from scipy.io import savemat
from fun_os import U_dir
from fun_img_Resize import image_Add_black_border
from fun_SSI import Cal_diz, Cal_Iz_structure
from fun_nonlinear import Info_find_contours_SHG
from fun_thread import noop, my_thread
from fun_CGH import structure_chi2_Generate_2D
np.seterr(divide='ignore', invalid='ignore')

# %%

def structure_chi2_3D(U1_name="",
                      img_full_name="Grating.png",
                      is_phase_only=0,
                      # %%
                      z_pump=0,
                      is_LG=0, is_Gauss=0, is_OAM=0,
                      l=0, p=0,
                      theta_x=0, theta_y=0,
                      #%%
                      is_random_phase=0,
                      is_H_l=0, is_H_theta=0, is_H_random_phase=0,
                      # %%
                      U1_0_NonZero_size=1, w0=0.3, structure_size_Enlarge=0.1,
                      deff_structure_length_expect=2, deff_structure_sheet_expect=1.8,
                      #%%
                      Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5, 
                      structure_xy_mode='x', Depth=2,
                      # %%
                      is_continuous=1, is_target_far_field=1, is_transverse_xy=0, 
                      is_reverse_xy=0, is_positive_xy=1, is_no_backgroud=1,
                      # %%
                      lam1=0.8, is_air_pump=0, is_air=0, T=25,
                      #%%
                      Tx=10, Ty=10, Tz="2*lc",
                      mx=0, my=0, mz=0,
                      is_stripe = 0,
                      # %%
                      is_save=0, is_save_txt=0, dpi=100,
                      # %%
                      cmap_2d='viridis',
                      # %%
                      ticks_num=6, is_contourf=0,
                      is_title_on=1, is_axes_on=1,
                      is_mm=1,
                      # %%
                      fontsize=9,
                      font={'family': 'serif',
                            'style': 'normal',  # 'normal', 'italic', 'oblique'
                            'weight': 'normal',
                            'color': 'black',  # 'black','gray','darkred'
                            },
                      # %%
                      is_colorbar_on=1, is_energy=0,
                      # %%
                      is_print=1, is_contours=1, n_TzQ=1, 
                      Gz_max_Enhance=1, match_mode=1,
                      # %%
                      *args, **kwargs, ):
    
    # %%
    # 预处理 导入图片 为方形，并加边框
    if __name__ == "__main__" :
        
        border_percentage = kwargs["border_percentage"] if len(kwargs) != 0 else 0.1
        
        image_Add_black_border(img_full_name, # 预处理 导入图片 为方形，并加边框
                               border_percentage,
                               is_print, )

    # %%

    n1, k1, k1_z_shift, lam2, n2, k2, k2_z_shift, \
    dk, lc, Tz, Gx, Gy, Gz, \
    size_PerPixel, U1_0, g1_shift, \
    structure, structure_opposite, modulation, modulation_opposite, modulation_squared, modulation_opposite_squared \
        = structure_chi2_Generate_2D(U1_name,
                                     img_full_name,
                                     is_phase_only,
                                     # %%
                                     z_pump,
                                     is_LG, is_Gauss, is_OAM,
                                     l, p,
                                     theta_x, theta_y,
                                     # %%s
                                     is_random_phase,
                                     is_H_l,
                                     is_H_theta,
                                     is_H_random_phase,
                                     # %%
                                     U1_0_NonZero_size, w0,
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
                                     is_mm,
                                     # %%
                                     fontsize, font,
                                     # %%
                                     is_colorbar_on, is_energy,
                                     # %%
                                     is_print, )

    # %%
    # 提供描边信息，并覆盖值

    # 这里 传 deff_structure_length_expect 进去 而不是 z0，是有问题的，导致只有 周期 Tz 能与 NLA_SSI 保持一致，长度并不能，
    # 这样若 deff_structure_length_expect < NLA_SSI 中的 z0 则 无法读取到 > deff_structure_length_expect 的 结构，只能手动在 A_to_B_3_NLA_SSI 中设置 deff_structure_length_expect 比 z0 大
    # 并不打算改这一点，因为否则的话，需要向这个函数传入一个参数，而这个参数却是之后要引用的函数 NLA_SSI 才能给出的，违反了 因果律

    if len(args) != 0:
        g1_shift = args[0]

    z0_recommend, Tz, deff_structure_length_expect = Info_find_contours_SHG(g1_shift, k1_z_shift, k2_z_shift, Tz, mz,
                                                                            deff_structure_length_expect, size_PerPixel,
                                                                            deff_structure_length_expect,
                                                                            0, is_contours, n_TzQ, Gz_max_Enhance,
                                                                            match_mode, )

    # %%
    # 定义 调制区域切片厚度 的 纵向实际像素、调制区域切片厚度 的 实际纵向尺寸

    diz, deff_structure_sheet = Cal_diz(deff_structure_sheet_expect, deff_structure_length_expect, size_PerPixel,
                                        Tz, mz,
                                        is_print)

    # %%
    # 定义 调制区域 的 纵向实际像素、调制区域 的 实际纵向尺寸

    sheets_num, Iz, deff_structure_length = Cal_Iz_structure(diz,
                                                             deff_structure_length_expect, size_PerPixel,
                                                             is_print)

    # %%

    Tz_unit = (Tz / 1000) / size_PerPixel

    # %%
    # 逐层 绘制 并 输出 structure
    
    folder_address = ''
    
    if is_save == 1:
        folder_address = U_dir("", "0.χ2_modulation_squared", 0, )

    def structure_chi2_Generate_z(for_th, fors_num, *args, ):
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
            modulation_squared_address = folder_address + "\\" + modulation_squared_full_name

            if is_save == 1:
                np.savetxt(modulation_squared_address, m, fmt='%i') if is_save_txt else savemat(
                    modulation_squared_address, {'chi2_modulation_squared': m})

        else:  # 如果不用 Tz，则 z 向 无结构，则一直输出 正向畴

            m = modulation_squared

            modulation_squared_full_name = str(for_th) + (is_save_txt and ".txt" or ".mat")
            modulation_squared_address = folder_address + "\\" + modulation_squared_full_name

            if is_save == 1:
                np.savetxt(modulation_squared_address, m, fmt='%i') if is_save_txt else savemat(
                    modulation_squared_address, {'chi2_modulation_squared': m})

    my_thread(10, sheets_num,
              structure_chi2_Generate_z, noop, noop,
              is_ordered=1, is_print=is_print, )

if __name__ == '__main__':
    
    structure_chi2_3D(U1_name="",
                    img_full_name="Grating.png",
                    is_phase_only=0,
                    # %%
                    z_pump=0,
                    is_LG=0, is_Gauss=0, is_OAM=0,
                    l=0, p=0,
                    theta_x=0, theta_y=0,
                    #%%
                    is_random_phase=0,
                    is_H_l=0, is_H_theta=0, is_H_random_phase=0,
                    # %%
                    U1_0_NonZero_size=1, w0=0.3, structure_size_Enlarge=0.1,
                    deff_structure_length_expect=2, deff_structure_sheet_expect=1.8,
                    #%%
                    Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5, 
                    structure_xy_mode='x', Depth=2,
                    # %%
                    is_continuous=1, is_target_far_field=1, is_transverse_xy=0, 
                    is_reverse_xy=0, is_positive_xy=1, is_no_backgroud=1,
                    # %%
                    lam1=0.8, is_air_pump=0, is_air=0, T=25,
                    #%%
                    Tx=10, Ty=10, Tz="2*lc",
                    mx=0, my=0, mz=0,
                    is_stripe = 0,
                    # %%
                    is_save=0, is_save_txt=0, dpi=100,
                    # %%
                    cmap_2d='viridis',
                    # %%
                    ticks_num=6, is_contourf=0,
                    is_title_on=1, is_axes_on=1,
                    is_mm=1,
                    # %%
                    fontsize=9,
                    font={'family': 'serif',
                          'style': 'normal',  # 'normal', 'italic', 'oblique'
                          'weight': 'normal',
                          'color': 'black',  # 'black','gray','darkred'
                          },
                    # %%
                    is_colorbar_on=1, is_energy=0,
                    # %%
                    is_print=1, is_contours=1, n_TzQ=1, 
                    Gz_max_Enhance=1, match_mode=1,
                    # %%
                    border_percentage=0.1, )
