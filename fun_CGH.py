# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

# %%

import math
import numpy as np
from fun_plot import plot_2d
from fun_pump import pump_pic_or_U_structure
from fun_linear import init_AST, init_SHG
from fun_nonlinear import args_SHG


# %%

def Step_U(U, mode,
           Duty_Cycle_x, Duty_Cycle_y,
           is_positive_xy):
    if mode == 'x':
        return (U > (2 * is_positive_xy - 1) * np.cos(Duty_Cycle_x * math.pi)).astype(
            np.int8())  # uint8 会导致 之后 structure 和 modulation 也变成 无符号 整形，以致于 在 0 - 1 时 变成 255 而不是 -1...
    elif mode == 'y':
        return (U > (2 * is_positive_xy - 1) * np.cos(Duty_Cycle_y * math.pi)).astype(
            np.int8())  # uint8 会导致 之后 structure 和 modulation 也变成 无符号 整形，以致于 在 0 - 1 时 变成 255 而不是 -1...


# %%

def CGH(U, mode,
        Duty_Cycle_x, Duty_Cycle_y,
        is_positive_xy,
        # %%
        Gx, Gy,
        is_Gauss, l,
        is_continuous, ):
    i1_x0, i1_y0 = np.meshgrid([i for i in range(U.shape[0])], [j for j in range(U.shape[1])])
    i1_x0_shift, i1_y0_shift = i1_x0 - U.shape[0] // 2, i1_y0 - U.shape[1] // 2
    if is_Gauss == 1 and l == 0:
        if mode == 'x*y' or mode == 'x+y':
            cgh = np.cos(Gx * i1_x0_shift)
            cgh_x = Step_U(cgh, 'x',
                           Duty_Cycle_x, Duty_Cycle_y,
                           is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
            cgh = np.cos(Gy * i1_y0_shift)
            cgh_y = Step_U(cgh, 'y',
                           Duty_Cycle_x, Duty_Cycle_y,
                           is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
            if mode == 'x*y':
                cgh = cgh_x * cgh_y
            else:
                cgh = np.mod(cgh_x + cgh_y, 2)
        elif mode == 'x':
            cgh = np.cos(Gx * i1_x0_shift)
            cgh = Step_U(cgh, 'x',
                         Duty_Cycle_x, Duty_Cycle_y,
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'y':
            cgh = np.cos(Gy * i1_y0_shift)
            cgh = Step_U(cgh, 'y',
                         Duty_Cycle_x, Duty_Cycle_y,
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'xy':
            cgh = np.cos(Gx * i1_x0_shift + Gy * i1_y0_shift)
            cgh = Step_U(cgh, 'x',
                         Duty_Cycle_x, Duty_Cycle_y,
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh  # 在所有方向的占空比都认为是 Duty_Cycle_x
        return cgh
    else:
        if mode == 'x*y' or mode == 'x+y':
            cgh = np.cos(Gx * i1_x0_shift - (np.angle(U) + math.pi)) - np.cos(np.arcsin(np.abs(U) / np.max(np.abs(U))))
            cgh_x = Step_U(cgh, 'x',
                           Duty_Cycle_x, Duty_Cycle_y,
                           is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
            cgh = np.cos(Gy * i1_y0_shift - (np.angle(U) + math.pi)) - np.cos(np.arcsin(np.abs(U) / np.max(np.abs(U))))
            cgh_y = Step_U(cgh, 'y',
                           Duty_Cycle_x, Duty_Cycle_y,
                           is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
            if mode == 'x*y':
                cgh = cgh_x * cgh_y
            else:
                cgh = np.mod(cgh_x + cgh_y, 2)
        elif mode == 'x':
            cgh = np.cos(Gx * i1_x0_shift - (np.angle(U) + math.pi)) - np.cos(np.arcsin(np.abs(U) / np.max(np.abs(U))))
            cgh = Step_U(cgh, 'x',
                         Duty_Cycle_x, Duty_Cycle_y,
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'y':
            cgh = np.cos(Gy * i1_y0_shift - (np.angle(U) + math.pi)) - np.cos(np.arcsin(np.abs(U) / np.max(np.abs(U))))
            cgh = Step_U(cgh, 'y',
                         Duty_Cycle_x, Duty_Cycle_y,
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'xy':
            cgh = np.cos(Gx * i1_x0_shift + Gy * i1_y0_shift - (np.angle(U) + math.pi)) - np.cos(
                np.arcsin(np.abs(U) / np.max(np.abs(U))))
            cgh = Step_U(cgh, 'x',
                         Duty_Cycle_x, Duty_Cycle_y,
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
        return cgh


# %%

def structure_Generate_2D_CGH(U, mode,
                              Duty_Cycle_x, Duty_Cycle_y,
                              is_positive_xy,
                              # %%
                              Gx, Gy,
                              is_Gauss, l,
                              is_continuous,
                              # %%
                              is_target_far_field, is_transverse_xy, is_reverse_xy, ):
    if is_target_far_field == 0:  # 如果 想要的 U1_0 是近场（晶体后端面）分布

        g = np.fft.fft2(U)
        g_shift = np.fft.fftshift(g)

        if is_transverse_xy == 1:
            structure = CGH(g_shift, mode,
                            Duty_Cycle_x, Duty_Cycle_y,
                            is_positive_xy,
                            # %%
                            Gx, Gy,
                            is_Gauss, l,
                            is_continuous, ).T  # 转置（沿 右下 对角线 翻转）
        else:
            structure = CGH(g_shift, mode,
                            Duty_Cycle_x, Duty_Cycle_y,
                            is_positive_xy,
                            # %%
                            Gx, Gy,
                            is_Gauss, l,
                            is_continuous, )[::-1]  # 上下翻转

    else:  # 如果 想要的 U1_0 是远场分布
        if is_transverse_xy == 1:
            structure = CGH(U, mode,
                            Duty_Cycle_x, Duty_Cycle_y,
                            is_positive_xy,
                            # %%
                            Gx, Gy,
                            is_Gauss, l,
                            is_continuous, ).T  # 转置（沿 右下 对角线 翻转）
        else:
            structure = CGH(U, mode,
                            Duty_Cycle_x, Duty_Cycle_y,
                            is_positive_xy,
                            # %%
                            Gx, Gy,
                            is_Gauss, l,
                            is_continuous, )[::-1]  # 上下翻转

    if is_reverse_xy == 1:
        structure = 1 - structure

    return structure


def structure_Generate_2D_radial_G(Ix, Iy,
                                   G, Duty_Cycle,
                                   is_positive_xy, is_continuous, is_reverse_xy, ):
    ix, iy = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
    ix_shift, iy_shift = ix - Ix // 2, iy - Iy // 2

    cgh = np.cos(G * (ix_shift ** 2 + iy_shift ** 2) ** 0.5)
    structure = Step_U(cgh, 'x',
                       Duty_Cycle, Duty_Cycle,
                       is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh  # 在所有方向的占空比都认为是 Duty_Cycle_x

    if is_reverse_xy == 1:
        structure = 1 - structure

    return structure


def structure_chi2_Generate_2D(U1_name="",
                               img_full_name="Grating.png",
                               is_phase_only=0,
                               # %%
                               z_pump=0,
                               is_LG=0, is_Gauss=0, is_OAM=0,
                               l=0, p=0,
                               theta_x=0, theta_y=0,
                               is_random_phase=0,
                               is_H_l=0, is_H_theta=0, is_H_random_phase=0,
                               # %%
                               U1_0_NonZero_size=1, w0=0.3, structure_size_Enlarge=0.1,
                               Duty_Cycle_x=0.5, Duty_Cycle_y=0.5,
                               structure_xy_mode='x', Depth=2,
                               # %%
                               is_continuous=1, is_target_far_field=1, is_transverse_xy=0,
                               is_reverse_xy=0, is_positive_xy=1, is_no_backgroud=1,
                               # %%
                               lam1=0.8, is_air_pump=0, is_air=0, T=25,
                               Tx=10, Ty=10, Tz="2*lc",
                               mx=0, my=0, mz=0,
                               # %%
                               is_save=0, is_save_txt=0, dpi=100,
                               # %%
                               cmap_2d='viridis',
                               # %%
                               ticks_num=6, is_contourf=0,
                               is_title_on=1, is_axes_on=1, is_mm=1,
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
                               is_print=1,
                               # %%
                               **kwargs, ):
    
    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, I1_x, I1_y, \
    Ix, Iy, deff_structure_size, \
    border_width, img_squared_resize_full_name, img_squared_resize, \
    U1_0, g1_shift = pump_pic_or_U_structure(U1_name,
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
                                             # %%
                                             lam1, is_air_pump, T,
                                             # %%
                                             is_save, is_save_txt, dpi,
                                             cmap_2d,
                                             # %%
                                             ticks_num, is_contourf,
                                             is_title_on, is_axes_on, is_mm,
                                             # %%
                                             fontsize, font,
                                             # %%
                                             is_colorbar_on, is_energy,
                                             # %%
                                             is_print,
                                             # %%
                                             **kwargs, )

    # %%

    n1, k1, k1_z, k1_xy = init_AST(Ix, Iy, size_PerPixel,
                                   lam1, is_air, T, )

    lam2, n2, k2, k2_z, k2_xy = init_SHG(Ix, Iy, size_PerPixel,
                                         lam1, is_air, T, )

    dk, lc, Tz, \
    Gx, Gy, Gz = args_SHG(k1, k2, size_PerPixel,
                          mx, my, mz,
                          Tx, Ty, Tz,
                          is_print, )

    # %%
    # 开始生成 调制函数 structure 和 modulation = 1 - is_no_backgroud - Depth * structure，以及 structure_opposite = 1 - structure 及其 modulation

    if structure_xy_mode == "r":

        structure = structure_Generate_2D_radial_G(Ix, Iy,
                                                   Gx, Duty_Cycle_x,
                                                   is_positive_xy, is_continuous, is_reverse_xy, )
    else:

        structure = structure_Generate_2D_CGH(U1_0, structure_xy_mode,
                                              Duty_Cycle_x, Duty_Cycle_y,
                                              is_positive_xy,
                                              # %%
                                              Gx, Gy,
                                              is_Gauss, l,
                                              is_continuous,
                                              # %%
                                              is_target_far_field, is_transverse_xy, is_reverse_xy, )

    # vmax_structure, vmin_structure = 1, 0
    vmax_modulation, vmin_modulation = 1 - is_no_backgroud, -1 - is_no_backgroud

    # plot_2d([], 1, size_PerPixel,  
    #         structure, "χ2_structure" + img_name_extension, "χ2_structure", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf,
    #         is_title_on, is_axes_on, is_mm, 0,
    #         fontsize, font,
    #         0, is_colorbar_on, 0, vmax_structure, vmin_structure)

    modulation = 1 - is_no_backgroud - Depth * structure
    modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant',
                                constant_values=(1 - is_no_backgroud, 1 - is_no_backgroud))

    plot_2d([], 1, size_PerPixel,
            modulation_squared, "χ2_modulation_squared" + img_name_extension, "χ2_modulation_squared",
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            0, is_colorbar_on, 0, vmax_modulation, vmin_modulation)

    # %%

    if mz != 0:
        structure_opposite = 1 - structure
    else:
        structure_opposite = structure

    # plot_2d([], 1, size_PerPixel,  
    #         structure_opposite, "χ2_structure_opposite" + img_name_extension, "χ2_structure_opposite", 
    #         is_save, dpi, size_fig,
    #         cmap_2d, ticks_num, is_contourf,
    #         is_title_on, is_axes_on, is_mm, 0,
    #         fontsize, font,
    #         0, is_colorbar_on, 0, vmax_structure, vmin_structure)

    modulation_opposite = 1 - is_no_backgroud - Depth * structure_opposite
    modulation_opposite_squared = np.pad(modulation_opposite,
                                         ((border_width, border_width), (border_width, border_width)), 'constant',
                                         constant_values=(1 - is_no_backgroud, 1 - is_no_backgroud))

    plot_2d([], 1, size_PerPixel,
            modulation_opposite_squared, "χ2_modulation_opposite_squared" + img_name_extension,
            "χ2_modulation_opposite_squared",
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            0, is_colorbar_on, 0, vmax_modulation, vmin_modulation)

    return n1, k1, k1_z, lam2, n2, k2, k2_z, \
           dk, lc, Tz, Gx, Gy, Gz, \
           size_PerPixel, U1_0, g1_shift, \
           structure, structure_opposite, modulation, modulation_opposite, modulation_squared, modulation_opposite_squared


def structure_n1_Generate_2D(U1_name="",
                             img_full_name="Grating.png",
                             is_phase_only=0,
                             # %%
                             z_pump=0,
                             is_LG=0, is_Gauss=0, is_OAM=0,
                             l=0, p=0,
                             theta_x=0, theta_y=0,
                             is_random_phase=0,
                             is_H_l=0, is_H_theta=0, is_H_random_phase=0,
                             # %%
                             U1_0_NonZero_size=1, w0=0.3, structure_size_Enlarge=0.1,
                             Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, structure_xy_mode='x', Depth=2,
                             # %%
                             is_continuous=1, is_target_far_field=1, is_transverse_xy=0,
                             is_reverse_xy=0, is_positive_xy=1,
                             # %%
                             lam1=0.8, is_air_pump=0, is_air=0, T=25,
                             Tx=10, Ty=10, Tz="2*lc",
                             mx=0, my=0, mz=0,
                             # %%
                             is_save=0, is_save_txt=0, dpi=100,
                             # %%
                             cmap_2d='viridis',
                             # %%
                             ticks_num=6, is_contourf=0,
                             is_title_on=1, is_axes_on=1, is_mm=1,
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
                             is_print=1,
                             # %%
                             **kwargs, ):
    
    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, I1_x, I1_y, \
    Ix, Iy, deff_structure_size, \
    border_width, img_squared_resize_full_name, img_squared_resize, \
    U1_0, g1_shift = pump_pic_or_U_structure(U1_name,
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
                                             # %%
                                             lam1, is_air_pump, T,
                                             # %%
                                             is_save, is_save_txt, dpi,
                                             cmap_2d,
                                             # %%
                                             ticks_num, is_contourf,
                                             is_title_on, is_axes_on, is_mm,
                                             # %%
                                             fontsize, font,
                                             # %%
                                             is_colorbar_on, is_energy,
                                             # %%
                                             is_print,
                                             # %%
                                             **kwargs, )

    # %%

    n1, k1, k1_z, k1_xy = init_AST(Ix, Iy, size_PerPixel,
                                   lam1, is_air, T, )

    lam2, n2, k2, k2_z, k2_xy = init_SHG(Ix, Iy, size_PerPixel,
                                   lam1, is_air, T, )

    dk, lc, Tz, \
    Gx, Gy, Gz = args_SHG(k1, k2, size_PerPixel,
                          mx, my, mz,
                          Tx, Ty, Tz,
                          is_print, )

    # %%
    # 开始生成 调制函数 structure 和 modulation = n1 - Depth * structure，以及 structure_opposite = 1 - structure 及其 modulation

    if structure_xy_mode == "r":

        structure = structure_Generate_2D_radial_G(Ix, Iy,
                                                   Gx, Duty_Cycle_x,
                                                   is_positive_xy, is_continuous, is_reverse_xy, )
    else:

        structure = structure_Generate_2D_CGH(U1_0, structure_xy_mode,
                                              Duty_Cycle_x, Duty_Cycle_y,
                                              is_positive_xy,
                                              # %%
                                              Gx, Gy,
                                              is_Gauss, l,
                                              is_continuous,
                                              # %%
                                              is_target_far_field, is_transverse_xy, is_reverse_xy, )

    # vmax_structure, vmin_structure = 1, 0
    vmax_modulation, vmin_modulation = n1, n1 - Depth

    # plot_2d([], 1, size_PerPixel,  
    #         structure, "n1_structure" + img_name_extension, "n1_structure", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf,
    #         is_title_on, is_axes_on, is_mm, 0,
    #         fontsize, font,
    #         0, is_colorbar_on,
    #         0, vmax_structure, vmin_structure)

    modulation = n1 - Depth * structure
    modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant',
                                constant_values=(n1, n1))

    plot_2d([], 1, size_PerPixel,
            modulation_squared, "n1_modulation_squared" + img_name_extension, "n1_modulation_squared",
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            0, is_colorbar_on,
            0, vmax_modulation, vmin_modulation)

    # %%

    if mz != 0:
        structure_opposite = 1 - structure
    else:
        structure_opposite = structure

    # plot_2d([], 1, size_PerPixel,  
    #         structure_opposite, "n1_structure_opposite" + img_name_extension, "n1_structure_opposite", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf,
    #         is_title_on, is_axes_on, is_mm, 0,
    #         fontsize, font,
    #         0, is_colorbar_on,
    #         0, vmax_structure, vmin_structure)

    modulation_opposite = n1 - Depth * structure_opposite
    modulation_opposite_squared = np.pad(modulation_opposite,
                                         ((border_width, border_width), (border_width, border_width)), 'constant',
                                         constant_values=(n1, n1))

    plot_2d([], 1, size_PerPixel,
            modulation_opposite_squared, "n1_modulation_opposite_squared" + img_name_extension,
            "n1_modulation_opposite_squared",
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            0, is_colorbar_on,
            0, vmax_modulation, vmin_modulation)

    return n1, k1, k1_z, lam2, n2, k2, k2_z, \
           dk, lc, Tz, Gx, Gy, Gz, \
           size_PerPixel, U1_0, g1_shift, \
           structure, structure_opposite, modulation, modulation_opposite, modulation_squared, modulation_opposite_squared
