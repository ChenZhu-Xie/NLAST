# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

# %%

import math
import numpy as np
from fun_os import U_dir, U_amp_plot_save
from fun_global_var import Get, tree_print
from fun_pump import pump_pic_or_U_structure
from fun_linear import init_AST_12oe, fft2, ifft2
from fun_nonlinear import args_SFG


# %%

def Step_U(U, mode,
           Duty_Cycle_x, Duty_Cycle_y,
           is_positive_xy):
    if mode == 'x':
        return (U >= (2 * is_positive_xy - 1) * np.cos(Duty_Cycle_x * math.pi)).astype(
            np.int8())  # uint8 会导致 之后 structure 和 modulation 也变成 无符号 整形，以致于 在 0 - 1 时 变成 255 而不是 -1...
    elif mode == 'y':
        return (U >= (2 * is_positive_xy - 1) * np.cos(Duty_Cycle_y * math.pi)).astype(
            np.int8())  # uint8 会导致 之后 structure 和 modulation 也变成 无符号 整形，以致于 在 0 - 1 时 变成 255 而不是 -1...


# %%

def CGH(U, mode,
        Duty_Cycle_x, Duty_Cycle_y,
        is_positive_xy,
        # %%
        Gx, Gy,
        is_Gauss, l,
        is_continuous, ):
    i1_x0, i1_y0 = np.meshgrid(range(U.shape[1]), range(U.shape[0]))
    i1_x0_shift, i1_y0_shift = i1_x0 - U.shape[1] // 2, i1_y0 - U.shape[0] // 2

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
    ix, iy = np.meshgrid(range(Iy), range(Ix))
    ix_shift, iy_shift = ix - Iy // 2, iy - Ix // 2

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
                               # %%
                               is_random_phase=0,
                               is_H_l=0, is_H_theta=0, is_H_random_phase=0,
                               # %%
                               U_0_size=1, w0=0.3, structure_size_Shrink=0.1,
                               Duty_Cycle_x=0.5, Duty_Cycle_y=0.5,
                               structure_xy_mode='x', Depth=2,
                               # %%
                               is_continuous=1, is_target_far_field=1, is_transverse_xy=0,
                               is_reverse_xy=0, is_positive_xy=1,
                               is_bulk=0, is_no_backgroud=1,
                               # %%
                               lam1=0.8, is_air_pump_structure=0, is_air=0, T=25,
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
                               # %% --------------------- for Info_find_contours_SHG
                               deff_structure_length_expect=1,
                               is_contours=1, n_TzQ=1,
                               Gz_max_Enhance=1, match_mode=1,
                               # %%
                               is_plot_n=0, is_print2=0, **kwargs, ):
    # print(kwargs)
    # %%
    is_HOPS = kwargs.get("is_HOPS_SHG", 0)
    is_twin_pump_degenerate = int(is_HOPS >= 1)  # is_HOPS == 0.x 的情况 仍是单泵浦
    is_single_pump_birefringence = int(is_HOPS > 0 and is_HOPS < 1)
    is_birefringence_deduced = int(is_twin_pump_degenerate == 1 or is_single_pump_birefringence == 1)
    kwargs['ray'] = "2" if is_birefringence_deduced == 1 else kwargs.get('ray', "2")
    ray_tag = "f" if kwargs['ray'] == "3" else "h"
    is_twin_pump = int(ray_tag == "f" or is_twin_pump_degenerate == 1)
    is_add_polarizer = int(is_HOPS > 0 and type(is_HOPS) != int)
    is_add_analyzer = int(type(kwargs.get("phi_a", 0)) != str)
    # %%
    lam_structure = kwargs.get("lam_structure", lam1)
    T_structure = kwargs.get("T_structure", T)
    kwargs.pop("lam_structure", None)
    kwargs.pop("T_structure", None)
    lam_structure = lam1  # 懒得搞 去管 lam_structure 的赋值了
    T_structure = T  # 懒得搞 去管 T 的赋值了
    # %%
    theta2_x = kwargs.get("theta2_x", theta_x)
    theta2_y = kwargs.get("theta2_y", theta_y)
    lam2 = kwargs.get("lam2", lam1)
    polar2 = kwargs.get("polar2", 'e')
    # %%
    if is_twin_pump == 1:
        [kwargs.pop(key) for key in kwargs["pump2_keys"]]  # 及时清理 kwargs ，尽量 保持 其干净
        kwargs.pop("pump2_keys")  # 这个有点意思， "pump2_keys" 这个键本身 也会被删除。

    is_air_pump = kwargs.get("is_air_pump", 1)
    if "is_air_pump" in kwargs:
        kwargs.pop("is_air_pump")

    # %%
    info = "χ2_2D_横向绘制"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    # %%
    kwargs["is_end"] = 1 if Get("args_SFG") else 0
    # 如果 Get("args_SFG") 非 False（则为 1,2,...），说明之前 用过 args_SFG，那么之后这里的 args_SFG 就不会被用，则下面 这个就应是 is_end=1
    # 否则 Get("args_SFG") == False.

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, \
    Ix_structure, Iy_structure, deff_structure_size_x, deff_structure_size_y, \
    border_width_x, border_width_y, img_squared_resize_full_name, img_squared_resize, \
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
                                                               U_0_size, w0, structure_size_Shrink,
                                                               # %%
                                                               lam_structure, is_air_pump_structure, T_structure,
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

    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话

    # %% 确定 公有参数

    z0 = kwargs["L0_Crystal"] if "L0_Crystal" in kwargs else deff_structure_length_expect
    kwargs.pop("L0_Crystal", None)

    g_shift = kwargs.get("g1", 0)
    g2 = kwargs.get("g2", 0)
    U_0 = ifft2(g_shift) if type(g_shift) == np.ndarray else 0
    U2_0 = ifft2(g2) if type(g2) == np.ndarray else 0

    args_init_AST = \
        [Ix, Iy, size_PerPixel,
         lam1, is_air, T,
         theta_x, theta_y, ]
    kwargs_init_AST = {"is_air_pump": is_air_pump, "gp": g_shift, }

    args_gan_args_SFG = \
        [z0, z0,
         mx, my, mz,
         Tx, Ty, Tz,
         is_contours, n_TzQ,
         Gz_max_Enhance, match_mode, ]

    def args_U_amp_plot_save(folder_address, U, U_name):
        return [folder_address,
                # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
                U, U_name,
                Get("img_name_extension"),
                is_save_txt,
                # %%
                [], 1, size_PerPixel,
                0, dpi, Get("size_fig"),  # is_save = 1 - is_bulk 改为 不储存，因为 反正 都储存了
                # %%
                cmap_2d, ticks_num, is_contourf,
                is_title_on, is_axes_on, is_mm, 0,  # 1, 1 或 0, 0
                fontsize, font,
                # %%
                1, is_colorbar_on, 0, ]  # 折射率分布差别很小，而 is_self_colorbar = 0 只看前 3 位小数的差异，因此用自动 colorbar。

    kwargs_U_amp_plot_save = {"suffix": ""}

    # %%  只提供 Gx, Gy 给自己，也提供 dk 给 A_3_structure_chi2_Generate_3D 的 Info_find_contours_SHG
    # 也提供 dk 来矫正 Tz...
    from b_3_SFG_NLA import gan_gpnkE_123VHoe_xyzinc_SFG

    g_p, p_p, g_V, g_H, p_V, p_H, \
    n1_inc, n1, k1_inc, k1, k1_z, k1_xy, E1_u, \
    n2_inc, n2, k2_inc, k2, k2_z, k2_xy, E2_u, \
    lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
    n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
    n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue, \
    n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
    n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
    n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
    n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He, \
    dk_z, lc, Gx, Gy, Gz, \
    z0_recommend, Tz, deff_structure_length_expect \
        = gan_gpnkE_123VHoe_xyzinc_SFG(is_birefringence_deduced, is_air,
                                       is_add_polarizer, is_HOPS,
                                       is_save, is_print2,
                                       ray_tag, is_twin_pump, is_air_pump,
                                       lam2, theta2_x, theta2_y,
                                       g_shift, g2, U_0, U2_0, polar2,
                                       args_init_AST, args_gan_args_SFG,
                                       args_U_amp_plot_save,
                                       kwargs_init_AST, kwargs_U_amp_plot_save,
                                       is_plot_n=is_plot_n, is_end=1, **kwargs)

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

    # %%

    method = "MOD"
    folder_name = method + " - " + "χ2_modulation_squared"
    folder_address = U_dir(folder_name, 1 - is_bulk, )

    kwargs.pop('U', None)  # 要想把 kwargs 传入 U_amp_plot_save，kwargs 里不能含 'U'

    # %%

    # vmax_structure, vmin_structure = 1, 0
    vmax_modulation, vmin_modulation = 1 - is_no_backgroud, -1 - is_no_backgroud

    # name = "χ2_structure"
    # full_name = method + " - " + name
    # U_amp_plot_save(folder_address,
    #                 # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
    #                 structure, full_name,
    #                 img_name_extension,
    #                 is_save_txt,
    #                 # %%
    #                 [], 1, size_PerPixel,
    #                 1 - is_bulk, dpi, size_fig,
    #                 # %%
    #                 cmap_2d, ticks_num, is_contourf,
    #                 is_title_on, is_axes_on, is_mm, 0,
    #                 fontsize, font,
    #                 # %%
    #                 0, is_colorbar_on, 0,
    #                 vmax=vmax_structure, vmin=vmin_structure,
    #                 # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
    #                 # %%
    #                 suffix="", **kwargs, )

    modulation = 1 - is_no_backgroud - Depth * structure
    # print(modulation.shape, border_width_x, border_width_y)  # ((行前, 行后) 填充行, (列前, 列后) 填充列)
    modulation_squared = np.pad(modulation, ((border_width_x, border_width_x), (border_width_y, border_width_y)),
                                'constant', constant_values=(1 - is_no_backgroud, 1 - is_no_backgroud))
    # print(modulation_squared.shape)

    name = "χ2_modulation_squared"
    full_name = method + " - " + name
    U_amp_plot_save(folder_address,
                    # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
                    modulation_squared, full_name,
                    img_name_extension,
                    is_save_txt,
                    # %%
                    [], 1, size_PerPixel,
                    1 - is_bulk, dpi, size_fig,
                    # %%
                    cmap_2d, ticks_num, is_contourf,
                    is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    # %%
                    0, is_colorbar_on, 0,
                    vmax=vmax_modulation, vmin=vmin_modulation,
                    # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
                    # %%
                    suffix="", **kwargs, )

    # %%

    if mz != 0:
        structure_opposite = 1 - structure
    else:
        structure_opposite = structure

    # name = "χ2_structure_opposite"
    # full_name = method + " - " + name
    # U_amp_plot_save(folder_address,
    #                 # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
    #                 structure_opposite, full_name,
    #                 img_name_extension,
    #                 is_save_txt,
    #                 # %%
    #                 [], 1, size_PerPixel,
    #                 1 - is_bulk, dpi, size_fig,
    #                 # %%
    #                 cmap_2d, ticks_num, is_contourf,
    #                 is_title_on, is_axes_on, is_mm, 0,
    #                 fontsize, font,
    #                 # %%
    #                 0, is_colorbar_on, 0,
    #                 vmax=vmax_structure, vmin=vmin_structure,
    #                 # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
    #                 # %%
    #                 suffix="", **kwargs, )

    modulation_opposite = 1 - is_no_backgroud - Depth * structure_opposite
    modulation_opposite_squared = np.pad(modulation_opposite,
                                         ((border_width_x, border_width_x), (border_width_y, border_width_y)),
                                         'constant', constant_values=(1 - is_no_backgroud, 1 - is_no_backgroud))

    name = "χ2_modulation_opposite_squared"
    full_name = method + " - " + name
    U_amp_plot_save(folder_address,
                    # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
                    modulation_opposite_squared, full_name,
                    img_name_extension,
                    is_save_txt,
                    # %%
                    [], 1, size_PerPixel,
                    1 - is_bulk, dpi, size_fig,
                    # %%
                    cmap_2d, ticks_num, is_contourf,
                    is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    # %%
                    0, is_colorbar_on, 0,
                    vmax=vmax_modulation, vmin=vmin_modulation,
                    # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
                    # %%
                    suffix="", **kwargs, )

    return folder_address, size_PerPixel, U_0_structure, g_shift_structure, \
           g_p, p_p, g_V, g_H, p_V, p_H, \
           n1_inc, n1, k1_inc, k1, k1_z, k1_xy, E1_u, \
           n2_inc, n2, k2_inc, k2, k2_z, k2_xy, E2_u, \
           lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
           n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
           n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue, \
           n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
           n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
           n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
           n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He, \
           dk_z, lc, Gx, Gy, Gz, z0_recommend, Tz, deff_structure_length_expect, \
           structure, structure_opposite, \
           modulation, modulation_opposite, \
           modulation_squared, modulation_opposite_squared


# %%

def structure_nonrect_chi2_Generate_2D(z_pump=0,
                                       is_LG=0, is_Gauss=0, is_OAM=0,
                                       l=0, p=0,
                                       theta_x=0, theta_y=0,
                                       # %%
                                       is_random_phase=0,
                                       is_H_l=0, is_H_theta=0, is_H_random_phase=0,
                                       # %%
                                       Ix_structure=1, Iy_structure=1, w0=0.3,
                                       Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, structure_xy_mode='x', Depth=2,
                                       # %%
                                       is_continuous=1, is_target_far_field=1, is_transverse_xy=0,
                                       is_reverse_xy=0, is_positive_xy=1,
                                       is_bulk=0, is_no_backgroud=1,
                                       # %%
                                       lam1=0.8, is_air_pump_structure=0, T=25,
                                       # %%
                                       is_save=0, is_save_txt=0, dpi=100,
                                       # %%
                                       cmap_2d='viridis',
                                       # %%
                                       ticks_num=6, is_contourf=0,
                                       is_title_on=1, is_axes_on=1, is_mm=1, zj_structure=[],
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
                                       **kwargs, ):
    lam_structure = kwargs.get("lam_structure", lam1)
    T_structure = kwargs.get("T_structure", T)
    kwargs.pop("lam_structure", None)
    kwargs.pop("T_structure", None)
    lam_structure = lam1  # 懒得搞 去管 lam_structure 的赋值了
    T_structure = T  # 懒得搞 去管 T 的赋值了
    # %%
    from fun_pump import pump
    from fun_linear import Cal_n
    # %%  只提供 Gx, Gy 给自己

    Gx, Gy = Get("Gx"), Get("Gy")

    # %%

    n_inc, n, k_inc, k = Cal_n(Get("size_PerPixel"),
                               is_air_pump_structure,
                               lam_structure, T_structure, p=kwargs.get("polar_structure", 'e'),
                               theta_x=theta_x,
                               theta_y=theta_y,
                               Ix_structure=Ix_structure,
                               Iy_structure=Iy_structure, **kwargs)

    # %%
    U_structure = np.ones((Ix_structure, Iy_structure))
    # print(Ix_structure, Iy_structure)
    U_structure, g_shift_structure = pump(Ix_structure, Iy_structure, Get("size_PerPixel"),
                                          U_structure, w0, k_inc, k, z_pump,
                                          is_LG, is_Gauss, is_OAM,
                                          l, p,
                                          theta_x, theta_y,
                                          is_random_phase,
                                          is_H_l, is_H_theta, is_H_random_phase,
                                          is_save, is_save_txt, dpi,
                                          ticks_num, is_contourf,
                                          is_title_on, is_axes_on, is_mm,
                                          fontsize, font,
                                          is_colorbar_on, is_energy,
                                          **kwargs, )

    # %%
    if is_bulk == 0:
        if structure_xy_mode == "r":

            structure = structure_Generate_2D_radial_G(Ix_structure, Iy_structure,
                                                       Gx, Duty_Cycle_x,
                                                       is_positive_xy, is_continuous, is_reverse_xy, )
        else:

            structure = structure_Generate_2D_CGH(U_structure, structure_xy_mode,
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

    # %%

    method = "MOD"
    folder_name = method + " - " + "χ2_modulation_squared"
    folder_address = U_dir(folder_name, 1 - is_bulk, )

    kwargs.pop('U', None)  # 要想把 kwargs 传入 U_amp_plot_save，kwargs 里不能含 'U'
    # %%
    vmax_modulation, vmin_modulation = 1 - is_no_backgroud, -1 - is_no_backgroud
    modulation_lie_down = 1 - is_no_backgroud - Depth * structure
    # %%
    name = "χ2_modulation_lie_down"
    full_name = method + " - " + name
    # is_propa_ax_reverse = 1 if Iy_structure == Get("Iy") else 0  # 以前是 Get("Iy")
    is_propa_ax_reverse = 1  # 反正在这里 恒有 Iy_structure == modulation.shape[1]
    # 所以没必要把 modulation.shape[1] 传进来，以及 与之比较了
    U_amp_plot_save(folder_address,
                    # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
                    modulation_lie_down, full_name,
                    Get("img_name_extension"),
                    is_save_txt,
                    # %%
                    zj_structure, 1, Get("size_PerPixel"),
                    1 - is_bulk, dpi, Get("size_fig"),
                    # %%
                    cmap_2d, ticks_num, is_contourf,
                    is_title_on, is_axes_on, 1, 1,  # 1, 1 或 0, 0
                    fontsize, font,
                    # %%
                    0, is_colorbar_on, 0, is_propa_ax_reverse=is_propa_ax_reverse,
                    vmax=vmax_modulation, vmin=vmin_modulation,
                    # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
                    # %%
                    suffix="", **kwargs, )

    return modulation_lie_down, folder_address


# %%

def structure_nonrect_chi2_interp2d_2D(folder_address=1, modulation=1,
                                       sheets_num=1,
                                       # %%
                                       is_save_txt=0, dpi=100,
                                       # %%
                                       cmap_2d='viridis',
                                       # %%
                                       ticks_num=6, is_contourf=0,
                                       is_title_on=1, is_axes_on=1, is_mm=1, zj_structure=[],
                                       # %%
                                       fontsize=9,
                                       font={'family': 'serif',
                                             'style': 'normal',  # 'normal', 'italic', 'oblique'
                                             'weight': 'normal',
                                             'color': 'black',  # 'black','gray','darkred'
                                             },
                                       # %%
                                       is_colorbar_on=1,
                                       # %%
                                       **kwargs, ):
    # %%
    from scipy.interpolate import interp2d
    ix, iy = range(modulation.shape[1]), range(modulation.shape[0])
    # iz = [k for k in range(sheets_num)]
    iz = np.linspace(0, modulation.shape[0] - 1, sheets_num)
    kind = 'linear'  # 'linear', 'cubic'
    f = interp2d(ix, iy, modulation, kind=kind)
    # if structure_xy_mode == 'x':
    #     modulation_lie_down = f(ix, iz)  # 行数重排
    # elif structure_xy_mode == 'y':
    #     modulation_lie_down = f(iz, iy)  # 列数重排
    modulation_lie_down = f(ix, iz)  # 行数重排
    # print(modulation_lie_down.shape)

    # 插值后 会变得连续，得 重新 二值化
    mod_max, mod_min = np.max(modulation), np.min(modulation)
    mod_middle = (mod_max + mod_min) / 2
    modulation_lie_down = (modulation_lie_down > mod_middle).astype(np.int8()) * mod_max + \
                          (modulation_lie_down <= mod_middle).astype(np.int8()) * mod_min

    # print(Ix, sheets_num)
    # print(zj_structure)
    mod_name = "χ2_modulation_lie_down"
    # is_propa_ax_reverse = 1 if structure_xy_mode == 'x' else 0
    is_propa_ax_reverse = 1
    U_amp_plot_save(folder_address,
                    # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
                    modulation_lie_down, mod_name,
                    Get("img_name_extension"),
                    is_save_txt,
                    # %%
                    zj_structure, 1, Get("size_PerPixel"),
                    0, dpi, Get("size_fig"),  # is_save = 1 - is_bulk 改为 不储存，因为 反正 都储存了
                    # %%
                    cmap_2d, ticks_num, is_contourf,
                    is_title_on, is_axes_on, 1, 1,  # 1, 1 或 0, 0
                    fontsize, font,
                    # %%
                    0, is_colorbar_on, 0, is_propa_ax_reverse=is_propa_ax_reverse,
                    # %%
                    suffix="", **kwargs, )

    return modulation_lie_down


# %%

def structure_n1_Generate_2D(U_structure_name="",
                             img_full_name="Grating.png",
                             is_phase_only=0,
                             # %%
                             z_pump=0,
                             is_LG=0, is_Gauss=0, is_OAM=0,
                             l=0, p=0,
                             theta_x=0, theta_y=0,
                             # %%
                             is_random_phase=0,
                             is_H_l=0, is_H_theta=0, is_H_random_phase=0,
                             # %%
                             U_0_size=1, w0=0.3, structure_size_Shrink=0.1,
                             Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, structure_xy_mode='x', Depth=2,
                             # %%
                             is_continuous=1, is_target_far_field=1, is_transverse_xy=0,
                             is_reverse_xy=0, is_positive_xy=1,
                             is_bulk=0,
                             # %%
                             lam1=0.8, is_air_pump_structure=0, is_air=0, T=25,
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
    lam_structure = kwargs.get("lam_structure", lam1)
    T_structure = kwargs.get("T_structure", T)
    kwargs.pop("lam_structure", None)
    kwargs.pop("T_structure", None)
    lam_structure = lam1  # 懒得搞 去管 lam_structure 的赋值了
    T_structure = T  # 懒得搞 去管 T 的赋值了

    is_air_pump = kwargs.get("is_air_pump", 1)
    if "is_air_pump" in kwargs:
        kwargs.pop("is_air_pump")
    # %%
    info = "n_2D_横向绘制"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    # %%

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, \
    Ix_structure, Iy_structure, deff_structure_size_x, deff_structure_size_y, \
    border_width_x, border_width_y, img_squared_resize_full_name, img_squared_resize, \
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
                                                               U_0_size, w0, structure_size_Shrink,
                                                               # %%
                                                               lam_structure, is_air_pump_structure, T_structure,
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

    # %%  只提供 Gx, Gy 给自己

    n1_inc, n1, k1_inc, k1, k1_z, k1_xy, g_shift, E1_u = \
        init_AST_12oe(Ix, Iy, size_PerPixel,
                      lam1, is_air, T,
                      theta_x, theta_y, is_print,
                      is_air_pump=is_air_pump, **kwargs, )

    dk_z, lc, Tz, \
    Gx, Gy, Gz, \
    lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u = \
        args_SFG(Ix, Iy, size_PerPixel,
                 is_air, T, lam1, lam1,
                 k1, k1_inc, k1, k1_inc,
                 theta_x, theta_y, theta_x, theta_y,
                 mx, my, mz,
                 Tx, Ty, Tz,
                 is_print, is_end=1,
                 is_air_pump=is_air_pump, **kwargs)

    # %%
    # 开始生成 调制函数 structure 和 modulation = n1_inc - Depth * structure，以及 structure_opposite = 1 - structure 及其 modulation

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
        structure = np.ones((Ix_structure, Iy_structure), dtype=np.int64()) * n1_inc

    # %%

    method = "MOD"
    folder_name = method + " - " + "n1_modulation_squared"
    folder_address = U_dir(folder_name, 1 - is_bulk, )

    kwargs.pop('U', None)  # 要想把 kwargs 传入 U_amp_plot_save，kwargs 里不能含 'U'

    # %%

    # vmax_structure, vmin_structure = 1, 0
    vmax_modulation, vmin_modulation = n1_inc, n1_inc - Depth

    # name = "n1_structure"
    # full_name = method + " - " + name
    # U_amp_plot_save(folder_address,
    #                 # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
    #                 structure, full_name,
    #                 img_name_extension,
    #                 is_save_txt,
    #                 # %%
    #                 [], 1, size_PerPixel,
    #                 1 - is_bulk, dpi, size_fig,
    #                 # %%
    #                 cmap_2d, ticks_num, is_contourf,
    #                 is_title_on, is_axes_on, is_mm, 0,
    #                 fontsize, font,
    #                 # %%
    #                 0, is_colorbar_on, 0,
    #                 vmax=vmax_structure, vmin=vmin_structure,
    #                 # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
    #                 # %%
    #                 suffix="", **kwargs, )

    modulation = n1_inc - Depth * structure
    modulation_squared = np.pad(modulation, ((border_width_x, border_width_x), (border_width_y, border_width_y)),
                                'constant', constant_values=(n1_inc, n1_inc))

    name = "n1_modulation_squared"
    full_name = method + " - " + name
    U_amp_plot_save(folder_address,
                    # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
                    modulation_squared, full_name,
                    img_name_extension,
                    is_save_txt,
                    # %%
                    [], 1, size_PerPixel,
                    1 - is_bulk, dpi, size_fig,
                    # %%
                    cmap_2d, ticks_num, is_contourf,
                    is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    # %%
                    0, is_colorbar_on, 0,
                    vmax=vmax_modulation, vmin=vmin_modulation,
                    # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
                    # %%
                    suffix="", **kwargs, )

    # %%

    if mz != 0:
        structure_opposite = 1 - structure
    else:
        structure_opposite = structure

    # name = "n1_structure_opposite"
    # full_name = method + " - " + name
    # U_amp_plot_save(folder_address,
    #                 # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
    #                 structure_opposite, full_name,
    #                 img_name_extension,
    #                 is_save_txt,
    #                 # %%
    #                 [], 1, size_PerPixel,
    #                 1 - is_bulk, dpi, size_fig,
    #                 # %%
    #                 cmap_2d, ticks_num, is_contourf,
    #                 is_title_on, is_axes_on, is_mm, 0,
    #                 fontsize, font,
    #                 # %%
    #                 0, is_colorbar_on, 0,
    #                 vmax=vmax_structure, vmin=vmin_structure,
    #                 # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
    #                 # %%
    #                 suffix="", **kwargs, )

    modulation_opposite = n1_inc - Depth * structure_opposite
    modulation_opposite_squared = np.pad(modulation_opposite,
                                         ((border_width_x, border_width_x), (border_width_y, border_width_y)),
                                         'constant', constant_values=(n1_inc, n1_inc))

    name = "n1_modulation_opposite_squared"
    full_name = method + " - " + name
    U_amp_plot_save(folder_address,
                    # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
                    modulation_opposite_squared, full_name,
                    img_name_extension,
                    is_save_txt,
                    # %%
                    [], 1, size_PerPixel,
                    1 - is_bulk, dpi, size_fig,
                    # %%
                    cmap_2d, ticks_num, is_contourf,
                    is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    # %%
                    0, is_colorbar_on, 0,
                    vmax=vmax_modulation, vmin=vmin_modulation,
                    # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
                    # %%
                    suffix="", **kwargs, )

    return n1_inc, n1, k1_inc, k1, k1_z, lam3, n3_inc, n3, k3_inc, k3, k3_z, \
           dk_z, lc, Tz, Gx, Gy, Gz, folder_address, \
           size_PerPixel, U_0_structure, g_shift_structure, \
           structure, structure_opposite, modulation, modulation_opposite, modulation_squared, modulation_opposite_squared


# %%

def structure_nonrect_n1_Generate_2D(z_pump=0,
                                     is_LG=0, is_Gauss=0, is_OAM=0,
                                     l=0, p=0,
                                     theta_x=0, theta_y=0,
                                     # %%
                                     is_random_phase=0,
                                     is_H_l=0, is_H_theta=0, is_H_random_phase=0,
                                     # %%
                                     Ix_structure=1, Iy_structure=1, w0=0.3,
                                     Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, structure_xy_mode='x', Depth=2,
                                     # %%
                                     is_continuous=1, is_target_far_field=1, is_transverse_xy=0,
                                     is_reverse_xy=0, is_positive_xy=1,
                                     is_bulk=0,
                                     # %%
                                     lam1=0.8, is_air_pump_structure=0, n1_inc=1, T=25,
                                     # %%
                                     is_save=0, is_save_txt=0, dpi=100,
                                     # %%
                                     cmap_2d='viridis',
                                     # %%
                                     ticks_num=6, is_contourf=0,
                                     is_title_on=1, is_axes_on=1, is_mm=1, zj_structure=[],
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
                                     **kwargs, ):
    lam_structure = kwargs.get("lam_structure", lam1)
    T_structure = kwargs.get("T_structure", T)
    kwargs.pop("lam_structure", None)
    kwargs.pop("T_structure", None)
    lam_structure = lam1  # 懒得搞 去管 lam_structure 的赋值了
    T_structure = T  # 懒得搞 去管 T 的赋值了
    # %%
    from fun_pump import pump
    from fun_linear import Cal_n
    # %%  只提供 Gx, Gy 给自己

    Gx, Gy = Get("Gx"), Get("Gy")

    # %%

    n_inc, n, k_inc, k = Cal_n(Get("size_PerPixel"),
                               is_air_pump_structure,
                               lam_structure, T_structure, p=kwargs.get("polar_structure", "e"),
                               theta_x=theta_x,
                               theta_y=theta_y,
                               Ix_structure=Ix_structure,
                               Iy_structure=Iy_structure, **kwargs)

    # %%
    U_structure = np.ones((Ix_structure, Iy_structure))
    # print(Ix_structure, Iy_structure)
    U_structure, g_shift_structure = pump(Ix_structure, Iy_structure, Get("size_PerPixel"),
                                          U_structure, w0, k_inc, k, z_pump,
                                          is_LG, is_Gauss, is_OAM,
                                          l, p,
                                          theta_x, theta_y,
                                          is_random_phase,
                                          is_H_l, is_H_theta, is_H_random_phase,
                                          is_save, is_save_txt, dpi,
                                          ticks_num, is_contourf,
                                          is_title_on, is_axes_on, is_mm,
                                          fontsize, font,
                                          is_colorbar_on, is_energy,
                                          **kwargs, )

    # %%
    if is_bulk == 0:
        if structure_xy_mode == "r":

            structure = structure_Generate_2D_radial_G(Ix_structure, Iy_structure,
                                                       Gx, Duty_Cycle_x,
                                                       is_positive_xy, is_continuous, is_reverse_xy, )
        else:

            structure = structure_Generate_2D_CGH(U_structure, structure_xy_mode,
                                                  Duty_Cycle_x, Duty_Cycle_y,
                                                  is_positive_xy,
                                                  # %%
                                                  Gx, Gy,
                                                  is_Gauss, l,
                                                  is_continuous,
                                                  # %%
                                                  is_target_far_field, is_transverse_xy, is_reverse_xy, )
    else:
        structure = np.ones((Ix_structure, Iy_structure), dtype=np.int64()) * n1_inc

    # %%

    method = "MOD"
    folder_name = method + " - " + "n1_modulation_squared"
    folder_address = U_dir(folder_name, 1 - is_bulk, )

    kwargs.pop('U', None)  # 要想把 kwargs 传入 U_amp_plot_save，kwargs 里不能含 'U'
    # %%
    vmax_modulation, vmin_modulation = n1_inc, n1_inc - Depth
    modulation_lie_down = n1_inc - Depth * structure
    # %%
    name = "n1_modulation_lie_down"
    full_name = method + " - " + name
    # is_propa_ax_reverse = 1 if Iy_structure == Get("Iy") else 0
    is_propa_ax_reverse = 1
    U_amp_plot_save(folder_address,
                    # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
                    modulation_lie_down, full_name,
                    Get("img_name_extension"),
                    is_save_txt,
                    # %%
                    zj_structure, 1, Get("size_PerPixel"),
                    1 - is_bulk, dpi, Get("size_fig"),
                    # %%
                    cmap_2d, ticks_num, is_contourf,
                    is_title_on, is_axes_on, 1, 1,  # 1, 1 或 0, 0
                    fontsize, font,
                    # %%
                    0, is_colorbar_on, 0, is_propa_ax_reverse=is_propa_ax_reverse,
                    vmax=vmax_modulation, vmin=vmin_modulation,
                    # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
                    # %%
                    suffix="", **kwargs, )

    return modulation_lie_down, folder_address


def structure_nonrect_n1_interp2d_2D(folder_address=1, modulation=1,
                                     sheets_num=1,
                                     # %%
                                     is_save_txt=0, dpi=100,
                                     # %%
                                     cmap_2d='viridis',
                                     # %%
                                     ticks_num=6, is_contourf=0,
                                     is_title_on=1, is_axes_on=1, is_mm=1, zj_structure=[],
                                     # %%
                                     fontsize=9,
                                     font={'family': 'serif',
                                           'style': 'normal',  # 'normal', 'italic', 'oblique'
                                           'weight': 'normal',
                                           'color': 'black',  # 'black','gray','darkred'
                                           },
                                     # %%
                                     is_colorbar_on=1,
                                     # %%
                                     **kwargs, ):
    # %%
    from scipy.interpolate import interp2d
    ix, iy = range(modulation.shape[1]), range(modulation.shape[0])
    # iz = [k for k in range(sheets_num)]
    iz = np.linspace(0, modulation.shape[0] - 1, sheets_num)
    kind = 'linear'  # 'linear', 'cubic'
    f = interp2d(ix, iy, modulation, kind=kind)
    # if structure_xy_mode == 'x':
    #     modulation_lie_down = f(ix, iz)  # 行数重排
    # elif structure_xy_mode == 'y':
    #     modulation_lie_down = f(iz, iy)  # 列数重排
    modulation_lie_down = f(ix, iz)  # 行数重排

    # 插值后 会变得连续，得 重新 二值化
    mod_max, mod_min = np.max(modulation), np.min(modulation)
    mod_middle = (mod_max + mod_min) / 2
    modulation_lie_down = (modulation_lie_down > mod_middle).astype(np.int8()) * mod_max + \
                          (modulation_lie_down <= mod_middle).astype(np.int8()) * mod_min

    # print(Ix, sheets_num)
    mod_name = "n1_modulation_lie_down"
    # is_propa_ax_reverse = 1 if structure_xy_mode == 'x' else 0
    is_propa_ax_reverse = 1
    U_amp_plot_save(folder_address,
                    # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
                    modulation_lie_down, mod_name,
                    Get("img_name_extension"),
                    is_save_txt,
                    # %%
                    zj_structure, 1, Get("size_PerPixel"),
                    0, dpi, Get("size_fig"),  # is_save = 1 - is_bulk 改为 不储存，因为 反正 都储存了
                    # %%
                    cmap_2d, ticks_num, is_contourf,
                    is_title_on, is_axes_on, 1, 1,  # 1, 1 或 0, 0
                    fontsize, font,
                    # %%
                    0, is_colorbar_on, 0, is_propa_ax_reverse=is_propa_ax_reverse,
                    # %%
                    suffix="", **kwargs, )

    return modulation_lie_down
