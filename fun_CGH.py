# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

# %%

import math
import numpy as np
from fun_os import U_dir
from fun_global_var import Get, tree_print
from fun_plot import plot_2d
from fun_pump import pump_pic_or_U_structure
from fun_linear import init_AST, init_SHG, fft2
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
    def args_CGH(mode, ):
        return [cgh, mode,
                Duty_Cycle_x, Duty_Cycle_y,
                is_positive_xy, ]
    if is_Gauss == 1 and l == 0:
        if mode == 'x*y' or mode == 'x+y':
            cgh = np.cos(Gx * i1_x0_shift)
            cgh_x = Step_U(*args_CGH('x')) if is_continuous == 0 else 0.5 + 0.5 * cgh
            cgh = np.cos(Gy * i1_y0_shift)
            cgh_y = Step_U(*args_CGH('y')) if is_continuous == 0 else 0.5 + 0.5 * cgh
            if mode == 'x*y':
                cgh = cgh_x * cgh_y
            else:
                cgh = np.mod(cgh_x + cgh_y, 2)
        elif mode == 'x':
            cgh = np.cos(Gx * i1_x0_shift)
            cgh = Step_U(*args_CGH('x')) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'y':
            cgh = np.cos(Gy * i1_y0_shift)
            cgh = Step_U(*args_CGH('y')) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'xy':
            cgh = np.cos(Gx * i1_x0_shift + Gy * i1_y0_shift)
            cgh = Step_U(*args_CGH('x')) if is_continuous == 0 else 0.5 + 0.5 * cgh  # 在所有方向的占空比都认为是 Duty_Cycle_x
        return cgh
    else:
        if mode == 'x*y' or mode == 'x+y':
            cgh = np.cos(Gx * i1_x0_shift - (np.angle(U) + math.pi)) - np.cos(np.arcsin(np.abs(U) / np.max(np.abs(U))))
            cgh_x = Step_U(*args_CGH('x')) if is_continuous == 0 else 0.5 + 0.5 * cgh
            cgh = np.cos(Gy * i1_y0_shift - (np.angle(U) + math.pi)) - np.cos(np.arcsin(np.abs(U) / np.max(np.abs(U))))
            cgh_y = Step_U(*args_CGH('y')) if is_continuous == 0 else 0.5 + 0.5 * cgh
            if mode == 'x*y':
                cgh = cgh_x * cgh_y
            else:
                cgh = np.mod(cgh_x + cgh_y, 2)
        elif mode == 'x':
            cgh = np.cos(Gx * i1_x0_shift - (np.angle(U) + math.pi)) - np.cos(np.arcsin(np.abs(U) / np.max(np.abs(U))))
            cgh = Step_U(*args_CGH('x')) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'y':
            cgh = np.cos(Gy * i1_y0_shift - (np.angle(U) + math.pi)) - np.cos(np.arcsin(np.abs(U) / np.max(np.abs(U))))
            cgh = Step_U(*args_CGH('y')) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'xy':
            cgh = np.cos(Gx * i1_x0_shift + Gy * i1_y0_shift - (np.angle(U) + math.pi)) - np.cos(
                np.arcsin(np.abs(U) / np.max(np.abs(U))))
            cgh = Step_U(*args_CGH('x')) if is_continuous == 0 else 0.5 + 0.5 * cgh
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
    # print(U)
    def args_CGH(U, ):
        return [U, mode,
                Duty_Cycle_x, Duty_Cycle_y,
                is_positive_xy,
                # %%
                Gx, Gy,
                is_Gauss, l,
                is_continuous, ]
    if is_target_far_field == 0:  # 如果 想要的 U_0 是近场（晶体后端面）分布
        g_shift = fft2(U)
        if is_transverse_xy == 1:
            structure = CGH(*args_CGH(g_shift)).T  # 转置（沿 右下 对角线 翻转）
        else:
            structure = CGH(*args_CGH(g_shift))[::-1]  # 上下翻转

    else:  # 如果 想要的 U_0 是远场分布
        if is_transverse_xy == 1:
            structure = CGH(*args_CGH(U)).T  # 转置（沿 右下 对角线 翻转）
        else:
            structure = CGH(*args_CGH(U))[::-1]  # 上下翻转

    if is_reverse_xy == 1:
        structure = 1 - structure
    # print(structure)
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


def structure_chi2_Generate_2D(U_structure_name="",
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
                               U_0_NonZero_size=1, w0=0.3, structure_size_Enlarge=0.1,
                               Duty_Cycle_x=0.5, Duty_Cycle_y=0.5,
                               structure_xy_mode='x', Depth=2,
                               # %%
                               is_continuous=1, is_target_far_field=1, is_transverse_xy=0,
                               is_reverse_xy=0, is_positive_xy=1,
                               is_bulk=0, is_no_backgroud=1,
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
    info = "χ2_2D_横向绘制"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs["is_end"], kwargs["add_level"] = 0, 0  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    #%%
    kwargs["is_end"] = 1 if Get("args_SHG") else 0
    # 如果 Get("args_SHG") 非 False（则为 1,2,...），说明之前 用过 args_SHG，那么之后这里的 args_SHG 就不会被用，则下面 这个就应是 is_end=1
    # 否则 Get("args_SHG") == False.

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, \
    Ix_structure, Iy_structure, deff_structure_size, \
    border_width, img_squared_resize_full_name, img_squared_resize, \
    U_0_structure, g_shift_structure = pump_pic_or_U_structure(U_structure_name,
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
                                                                 U_0_NonZero_size, w0, structure_size_Enlarge,
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

    kwargs["is_end"], kwargs["add_level"] = 0, 0  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # %%

    n1, k1, k1_z, k1_xy = init_AST(Ix, Iy, size_PerPixel,
                                   lam1, is_air, T, )

    lam2, n2, k2, k2_z, k2_xy = init_SHG(Ix, Iy, size_PerPixel,
                                         lam1, is_air, T, )

    dk, lc, Tz, \
    Gx, Gy, Gz = args_SHG(k1, k2, size_PerPixel,
                          mx, my, mz,
                          Tx, Ty, Tz,
                          is_print, is_end=1)

    # %%
    # 开始生成 调制函数 structure 和 modulation = 1 - is_no_backgroud - Depth * structure，以及 structure_opposite = 1 - structure 及其 modulation

    if is_bulk == 0:
        if structure_xy_mode == "r":

            structure = structure_Generate_2D_radial_G(Ix_structure, Iy_structure,
                                                       Gx, Duty_Cycle_x,
                                                       is_positive_xy, is_continuous, is_reverse_xy, )
        else:

            structure = structure_Generate_2D_CGH(U_0_structure, structure_xy_mode,
                                                  Duty_Cycle_x, Duty_Cycle_y,
                                                  is_positive_xy,
                                                  # %%
                                                  Gx, Gy,
                                                  is_Gauss, l,
                                                  is_continuous,
                                                  # %%
                                                  is_target_far_field, is_transverse_xy, is_reverse_xy, )
    else:
        structure = np.ones((Ix_structure, Iy_structure), dtype=np.int64()) - is_no_backgroud

    #%%

    method = "MOD"
    folder_name = method + " - " + "χ2_modulation_squared"
    folder_address = U_dir(folder_name, is_save - 0.5 * is_bulk, )

    #%%

    # vmax_structure, vmin_structure = 1, 0
    vmax_modulation, vmin_modulation = 1 - is_no_backgroud, -1 - is_no_backgroud

    # name = "χ2_structure"
    # full_name = method + " - " + name
    # title = full_name.replace("χ2", "$\chi_2$")
    # address = folder_address + "\\" + full_name + img_name_extension
    # plot_2d([], 1, size_PerPixel,
    #         structure, address, title,
    #         is_save - 0.5 * is_bulk, dpi, size_fig,
    #         cmap_2d, ticks_num, is_contourf,
    #         is_title_on, is_axes_on, is_mm, 0,
    #         fontsize, font,
    #         0, is_colorbar_on, 0, vmax_structure, vmin_structure)

    modulation = 1 - is_no_backgroud - Depth * structure
    modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant',
                                constant_values=(1 - is_no_backgroud, 1 - is_no_backgroud))

    name = "χ2_modulation_squared"
    full_name = method + " - " + name
    title = full_name.replace("χ2", "$\chi_2$")
    address = folder_address + "\\" + full_name + img_name_extension
    plot_2d([], 1, size_PerPixel,
            modulation_squared, address, title,
            is_save - 0.5 * is_bulk, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            0, is_colorbar_on, 0,
            # %%
            vmax = vmax_modulation, vmin = vmin_modulation)

    # %%

    if mz != 0:
        structure_opposite = 1 - structure
    else:
        structure_opposite = structure

    # name = "χ2_structure_opposite"
    # full_name = method + " - " + name
    # title = full_name.replace("χ2", "$\chi_2$")
    # address = folder_address + "\\" + full_name + img_name_extension
    # plot_2d([], 1, size_PerPixel,  
    #         structure_opposite, address, title,
    #         is_save - 0.5 * is_bulk, dpi, size_fig,
    #         cmap_2d, ticks_num, is_contourf,
    #         is_title_on, is_axes_on, is_mm, 0,
    #         fontsize, font,
    #         0, is_colorbar_on, 0, vmax_structure, vmin_structure)

    modulation_opposite = 1 - is_no_backgroud - Depth * structure_opposite
    modulation_opposite_squared = np.pad(modulation_opposite,
                                         ((border_width, border_width), (border_width, border_width)), 'constant',
                                         constant_values=(1 - is_no_backgroud, 1 - is_no_backgroud))

    name = "χ2_modulation_opposite_squared"
    full_name = method + " - " + name
    title = full_name.replace("χ2", "$\chi_2$")
    address = folder_address + "\\" + full_name + img_name_extension
    plot_2d([], 1, size_PerPixel,
            modulation_opposite_squared, address, title,
            is_save - 0.5 * is_bulk, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            0, is_colorbar_on, 0,
            # %%
            vmax = vmax_modulation, vmin = vmin_modulation, )

    return n1, k1, k1_z, lam2, n2, k2, k2_z, \
           dk, lc, Tz, Gx, Gy, Gz, \
           size_PerPixel, U_0_structure, g_shift_structure, \
           structure, structure_opposite, modulation, modulation_opposite, modulation_squared, modulation_opposite_squared


def structure_n1_Generate_2D(U_structure_name="",
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
                             U_0_NonZero_size=1, w0=0.3, structure_size_Enlarge=0.1,
                             Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, structure_xy_mode='x', Depth=2,
                             # %%
                             is_continuous=1, is_target_far_field=1, is_transverse_xy=0,
                             is_reverse_xy=0, is_positive_xy=1,
                             is_bulk=0,
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
    info = "n_2D_横向绘制"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs["is_end"], kwargs["add_level"] = 0, 0  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    #%%

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, \
    Ix_structure, Iy_structure, deff_structure_size, \
    border_width, img_squared_resize_full_name, img_squared_resize, \
    U_0_structure, g_shift_structure = pump_pic_or_U_structure(U_structure_name,
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
                                                                 U_0_NonZero_size, w0, structure_size_Enlarge,
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
                          is_print, is_end=1)

    # %%
    # 开始生成 调制函数 structure 和 modulation = n1 - Depth * structure，以及 structure_opposite = 1 - structure 及其 modulation

    if is_bulk == 0:
        if structure_xy_mode == "r":

            structure = structure_Generate_2D_radial_G(Ix_structure, Iy_structure,
                                                       Gx, Duty_Cycle_x,
                                                       is_positive_xy, is_continuous, is_reverse_xy, )
        else:

            structure = structure_Generate_2D_CGH(U_0_structure, structure_xy_mode,
                                                  Duty_Cycle_x, Duty_Cycle_y,
                                                  is_positive_xy,
                                                  # %%
                                                  Gx, Gy,
                                                  is_Gauss, l,
                                                  is_continuous,
                                                  # %%
                                                  is_target_far_field, is_transverse_xy, is_reverse_xy, )
    else:
        structure = np.ones((Ix_structure, Iy_structure), dtype=np.int64()) * n1

    #%%

    method = "MOD"
    folder_name = method + " - " + "n1_modulation_squared"
    folder_address = U_dir(folder_name, is_save - 0.5 * is_bulk, )

    #%%

    # vmax_structure, vmin_structure = 1, 0
    vmax_modulation, vmin_modulation = n1, n1 - Depth

    # name = "n1_structure"
    # title = method + " - " + name.replace("1", "$_1$")
    # address = folder_address + "\\" + name + img_name_extension
    # plot_2d([], 1, size_PerPixel,  
    #         structure, address, title,
    #         is_save - 0.5 * is_bulk, dpi, size_fig,
    #         cmap_2d, ticks_num, is_contourf,
    #         is_title_on, is_axes_on, is_mm, 0,
    #         fontsize, font,
    #         0, is_colorbar_on,
    #         0, vmax_structure, vmin_structure)

    modulation = n1 - Depth * structure
    modulation_squared = np.pad(modulation, ((border_width, border_width), (border_width, border_width)), 'constant',
                                constant_values=(n1, n1))

    name = "n1_modulation_squared"
    title = method + " - " + name.replace("1", "$_1$")
    address = folder_address + "\\" + name + img_name_extension
    plot_2d([], 1, size_PerPixel,
            modulation_squared, address, title,
            is_save - 0.5 * is_bulk, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            0, is_colorbar_on, 0,
            # %%
            vmax=vmax_modulation, vmin=vmin_modulation)

    # %%

    if mz != 0:
        structure_opposite = 1 - structure
    else:
        structure_opposite = structure

    # name = "n1_structure_opposite"
    # title = method + " - " + name.replace("1", "$_1$")
    # address = folder_address + "\\" + name + img_name_extension
    # plot_2d([], 1, size_PerPixel,  
    #         structure_opposite, address, title,
    #         is_save - 0.5 * is_bulk, dpi, size_fig,
    #         cmap_2d, ticks_num, is_contourf,
    #         is_title_on, is_axes_on, is_mm, 0,
    #         fontsize, font,
    #         0, is_colorbar_on,
    #         0, vmax_structure, vmin_structure)

    modulation_opposite = n1 - Depth * structure_opposite
    modulation_opposite_squared = np.pad(modulation_opposite,
                                         ((border_width, border_width), (border_width, border_width)), 'constant',
                                         constant_values=(n1, n1))

    name = "n1_modulation_opposite_squared"
    title = method + " - " + name.replace("1", "$_1$")
    address = folder_address + "\\" + name + img_name_extension
    plot_2d([], 1, size_PerPixel,
            modulation_opposite_squared, address, title,
            is_save - 0.5 * is_bulk, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            0, is_colorbar_on, 0,
            # %%
            vmax=vmax_modulation, vmin=vmin_modulation)

    return n1, k1, k1_z, lam2, n2, k2, k2_z, \
           dk, lc, Tz, Gx, Gy, Gz, \
           size_PerPixel, U_0_structure, g_shift_structure, \
           structure, structure_opposite, modulation, modulation_opposite, modulation_squared, modulation_opposite_squared
