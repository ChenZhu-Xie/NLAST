# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:57 2021

@author: Xcz
"""

# %%

import copy
import numpy as np
from fun_os import img_squared_bordered_Read, U_twin_energy_error_plot_save, U_twin_error_energy_plot_save
from fun_global_var import init_GLV_DICT, tree_print, Get, eget, sget, skey
from fun_img_Resize import if_image_Add_black_border
from fun_linear import fft2
from fun_compare import U_compare
from b_3_SFG_NLA_EVV import SFG_NLA_EVV
from B_3_SFG_NLA_SSI_chi2 import SFG_NLA_SSI
from B_3_SFG_SSF_SSI_chi2 import SFG_SSF_SSI

np.seterr(divide='ignore', invalid='ignore')


def compare_SFG_NLA_EVV__SSI(U_name_Structure="",
                             is_phase_only_Structure=0,
                             # %%
                             z_pump_Structure=0,
                             is_LG_Structure=0, is_Gauss_Structure=1, is_OAM_Structure=1,
                             l_Structure=0, p_Structure=0,
                             theta_x_Structure=0, theta_y_Structure=0,
                             # %%
                             is_random_phase_Structure=0,
                             is_H_l_Structure=0, is_H_theta_Structure=0, is_H_random_phase_Structure=0,
                             # %%
                             U_name="",
                             img_full_name="l=1.png",
                             is_phase_only=0,
                             # %%
                             z_pump=0,
                             is_LG=0, is_Gauss=1, is_OAM=1,
                             l=1, p=0,
                             theta_x=1, theta_y=0,
                             # %%
                             is_random_phase=0,
                             is_H_l=0, is_H_theta=0, is_H_random_phase=0,
                             # %%---------------------------------------------------------------------
                             # %%
                             U_size=0.5, w0=0.1, w0_Structure=5, structure_size_Shrink=0.1,
                             L0_Crystal=2, z0_structure_frontface_expect=0.5, deff_structure_length_expect=1,
                             SSI_zoomout_times=1, sheets_stored_num=10,
                             z0_section_1_expect=1, z0_section_2_expect=1,
                             X=0, Y=0,
                             # %%
                             Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
                             structure_xy_mode='x', Depth=2,
                             # %%
                             is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
                             is_reverse_xy=0, is_positive_xy=1,
                             # %%
                             is_bulk=0, is_no_backgroud=1,
                             is_stored=0, is_show_structure_face=1, is_energy_evolution_on=1,
                             # %%
                             lam1=1.5, is_air_pump=0, is_air=0, T=25,
                             is_air_pump_structure=0,
                             deff=30, is_fft=1, fft_mode=0,
                             is_sum_Gm=0, mG=0,
                             is_linear_convolution=0,
                             # %%
                             Tx=19.769, Ty=20, Tz=8.139,
                             mx=-1, my=0, mz=1,
                             is_stripe=0, is_NLAST=0,
                             # %%
                             is_save=0, is_save_txt=0, dpi=100,
                             # %%
                             color_1d='b', color_1d2='r', cmap_2d='viridis', cmap_3d='rainbow',
                             elev=10, azim=-65, alpha=2,
                             # %%
                             sample=1, ticks_num=6, is_contourf=0,
                             is_title_on=1, is_axes_on=1, is_mm=1,
                             # %%
                             fontsize=9,
                             font={'family': 'serif',
                                   'style': 'normal',  # 'normal', 'italic', 'oblique'
                                   'weight': 'normal',
                                   'color': 'black',  # 'black','gray','darkred'
                                   },
                             # %%
                             is_colorbar_on=1, is_colorbar_log=0,
                             is_energy=1,
                             # %%
                             plot_group="UGa", is_animated=1,
                             loop=0, duration=0.033, fps=5,
                             # %%
                             is_plot_EVV=1, is_plot_3d_XYz=0, is_plot_selective=0,
                             is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
                             # %%
                             is_print=1, is_contours=1, n_TzQ=1,
                             Gz_max_Enhance=1, match_mode=1,
                             # %%
                             is_EVV_SSI=1,
                             # %%
                             is_NLA=1, is_amp_relative=1,
                             is_energy_normalized=2, is_output_error_EVV=0,
                             # %%
                             **kwargs, ):
    # %%
    info = "利用 SHG 对比：EVV 与 SSI"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    # %%

    if_image_Add_black_border("", img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%

    args_SSI = \
        [U_name,
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
         # 生成横向结构
         U_name_Structure,
         structure_size_Shrink,
         is_phase_only_Structure,
         # %%
         w0_Structure, z_pump_Structure,
         is_LG_Structure, is_Gauss_Structure, is_OAM_Structure,
         l_Structure, p_Structure,
         theta_x_Structure, theta_y_Structure,
         # %%
         is_random_phase_Structure,
         is_H_l_Structure, is_H_theta_Structure, is_H_random_phase_Structure,
         # %%
         U_size, w0,
         L0_Crystal, z0_structure_frontface_expect, deff_structure_length_expect,
         SSI_zoomout_times, sheets_stored_num,
         z0_section_1_expect, z0_section_2_expect,
         X, Y,
         # %%
         is_bulk, is_no_backgroud,
         is_stored, is_show_structure_face, is_energy_evolution_on,
         # %%
         lam1, is_air_pump, is_air, T,
         is_air_pump_structure,
         deff,
         # %%
         Tx, Ty, Tz,
         mx, my, mz,
         is_stripe, is_NLAST,
         # %%
         # 生成横向结构
         Duty_Cycle_x, Duty_Cycle_y, Duty_Cycle_z,
         Depth, structure_xy_mode,
         is_continuous, is_target_far_field, is_transverse_xy,
         is_reverse_xy, is_positive_xy,
         # %%
         is_save, is_save_txt, dpi,
         # %%
         color_1d, cmap_2d, cmap_3d,
         elev, azim, alpha,
         # %%
         sample, ticks_num, is_contourf,
         is_title_on, is_axes_on, is_mm,
         # %%
         fontsize, font,
         # %%
         is_colorbar_on, is_colorbar_log,
         is_energy,
         # %%
         plot_group, is_animated,
         loop, duration, fps,
         # %%
         is_plot_EVV, is_plot_3d_XYz, is_plot_selective,
         is_plot_YZ_XZ, is_plot_3d_XYZ,
         # %%
         is_print, is_contours, n_TzQ,
         Gz_max_Enhance, match_mode, ]

    kwargs_SSI = copy.deepcopy(kwargs)
    kwargs_SSI.update({"ray": kwargs.get("ray", "2"), })
    U2_SSI, G2_SSI, ray2_SSI, method_and_way2_SSI, U_key2_SSI = \
        SFG_NLA_SSI(*args_SSI, **kwargs_SSI, ) if is_NLA == 1 else \
            SFG_SSF_SSI(*args_SSI, **kwargs_SSI, )

    if is_energy_evolution_on == 1:  # 截获一下 SSI 的 能量曲线
        zj_SSI = Get("zj")
        # print(len(zj_SSI))
        U2_energy_SSI = eget("U")
    if abs(is_stored) == 1:
        U2_stored_SSI, G2_stored_SSI, U2_stored_key_SSI = sget("U"), sget("G"), skey("U")

    args_EVV = \
        [U_name,
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
         # 生成横向结构
         U_name_Structure,
         structure_size_Shrink,
         is_phase_only_Structure,
         # %%
         w0_Structure, z_pump_Structure,
         is_LG_Structure, is_Gauss_Structure, is_OAM_Structure,
         l_Structure, p_Structure,
         theta_x_Structure, theta_y_Structure,
         # %%
         is_random_phase_Structure,
         is_H_l_Structure, is_H_theta_Structure, is_H_random_phase_Structure,
         # %%
         U_size, w0,
         L0_Crystal, sheets_stored_num,
         # %%
         lam1, is_air_pump, is_air, T,
         is_air_pump_structure,
         deff, is_fft, fft_mode,
         is_sum_Gm, mG,
         is_linear_convolution,
         # %%
         Tx, Ty, Tz,
         mx, my, mz,
         # %%
         # 生成横向结构
         Duty_Cycle_x, Duty_Cycle_y, Duty_Cycle_z,
         Depth, structure_xy_mode,
         # %%
         is_continuous, is_target_far_field, is_transverse_xy,
         is_reverse_xy, is_positive_xy, is_no_backgroud,
         is_stored, is_energy_evolution_on,
         # %%
         is_save, is_save_txt, dpi,
         # %%
         color_1d, cmap_2d, cmap_3d,
         elev, azim, alpha,
         # %%
         sample, ticks_num, is_contourf,
         is_title_on, is_axes_on,
         is_mm,
         # %%
         fontsize, font,
         # %%
         is_colorbar_on, is_colorbar_log,
         is_energy,
         # %%
         is_plot_EVV, is_plot_3d_XYz, is_plot_selective,
         X, Y, is_plot_YZ_XZ, is_plot_3d_XYZ,
         # %%
         plot_group, is_animated,
         loop, duration, fps,
         # %%
         is_print, is_contours, n_TzQ,
         Gz_max_Enhance, match_mode,
         # %%
         is_EVV_SSI, ]

    # print(Get("z_stored"))
    kwargs_EVV = copy.deepcopy(kwargs)
    # print(kwargs)
    if abs(is_stored) == 1:
        kwargs_EVV.update({"ray": kwargs.get("ray", "2"), "zj_EVV": Get("z_stored"), })
    else:
        kwargs_EVV.update({"ray": kwargs.get("ray", "2"), })
    U2_NLA, G2_NLA, ray2_NLA, method_and_way2_NLA, U_key2_NLA = \
        SFG_NLA_EVV(*args_EVV, **kwargs_EVV, )
    # 如果 is_stored == 1 或 -1，则把 SSI 或 ssi 生成的 z_stored 传进 SFG_NLA_EVV 作为 他的 zj，方便 比较。不画图 则传 -1 进去。

    if is_energy_evolution_on == 1:  # 截获一下 EVV 的 能量曲线
        zj_EVV = Get("zj")
        # print(zj_EVV)
        U2_energy_EVV = eget("U")
    if abs(is_stored) == 1:
        U2_stored_EVV, G2_stored_EVV, U2_stored_key_EVV = sget("U"), sget("G"), skey("U")

    # %%

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, U = \
        img_squared_bordered_Read(img_full_name,
                                  U_size, dpi,
                                  is_phase_only, **kwargs, )

    # %%
    if kwargs.get('ray', "2") == "3":  #  防止 l2 关键字 进 U_twin_energy_error_plot_save 等， 与 line2 冲突
        [kwargs.pop(key) for key in kwargs["pump2_keys"]]  # 及时清理 kwargs ，尽量 保持 其干净
        kwargs.pop("pump2_keys")

    if is_output_error_EVV != 1:
        # %%
        # 对比 G2_NLA 与 G2_SSI 的 （绝对）误差

        U_compare(fft2(U2_NLA), fft2(U2_SSI), U_key2_SSI.replace("U", "G"), L0_Crystal,
                  # %%
                  img_name_extension, size_PerPixel, size_fig,
                  # %%
                  is_save, is_save_txt, dpi,
                  # %%
                  cmap_2d,
                  # %%
                  ticks_num, is_contourf,
                  is_title_on, is_axes_on, is_mm,
                  # %%
                  fontsize, font,
                  # %%S
                  is_colorbar_on, is_energy,
                  # %%
                  is_amp_relative, is_print, )

        # %%
        # 对比 U2_NLA 与 U2_SSI 的 （绝对）误差

        U_compare(U2_NLA, U2_SSI, U_key2_SSI, L0_Crystal,
                  # %%
                  img_name_extension, size_PerPixel, size_fig,
                  # %%
                  is_save, is_save_txt, dpi,
                  # %%
                  cmap_2d,
                  # %%
                  ticks_num, is_contourf,
                  is_title_on, is_axes_on, is_mm,
                  # %%
                  fontsize, font,
                  # %%S
                  is_colorbar_on, is_energy,
                  # %%
                  is_amp_relative, is_print,
                  # %%
                  is_end=1, )

        if is_energy_evolution_on == 1:
            U_twin_energy_error_plot_save(U2_energy_SSI, U2_energy_EVV, U_key2_SSI.replace("_SSI", ""),
                                          img_name_extension, is_save_txt,
                                          # %%
                                          zj_SSI, zj_EVV, sample, size_PerPixel,
                                          is_save, dpi, Get("size_fig_x"), Get("size_fig_y"),
                                          # %%
                                          color_1d, color_1d2,
                                          ticks_num, is_title_on, is_axes_on, is_mm,
                                          fontsize, font,
                                          # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                                          # %%
                                          L0_Crystal,
                                          # %%
                                          is_energy_normalized=is_energy_normalized, **kwargs, )

    else:
        G0_energy = []
        G_energy = []
        G_error_energy = []
        U_energy = []
        U0_energy = []
        U_error_energy = []

        # %%
        # 对比 G2_NLA 与 G2_SSI 的 （绝对）误差

        is_end = [0] * (len(zj_EVV) - 1)
        is_end.append(1)

        is_print and print(tree_print(add_level=2) + "G_z_对比：G2_EVV 与 G2_SSI 的 （绝对）误差，随 z 的演化")
        for i in range(len(zj_EVV)):
            G_and_G_error_energy = U_compare(G2_stored_EVV[:, :, i], G2_stored_SSI[:, :, i],
                                             U2_stored_key_SSI.replace("U", "G"), zj_EVV[i],
                                             # %%
                                             img_name_extension, size_PerPixel, size_fig,
                                             # %%
                                             is_save, is_save_txt, dpi,
                                             # %%
                                             cmap_2d,
                                             # %%
                                             ticks_num, is_contourf,
                                             is_title_on, is_axes_on, is_mm,
                                             # %%
                                             fontsize, font,
                                             # %%S
                                             is_colorbar_on, is_energy,
                                             # %%
                                             is_amp_relative, is_print,
                                             # %%
                                             is_end=is_end[i], )

            G_energy.append(G_and_G_error_energy[0])
            G0_energy.append(G_and_G_error_energy[1])
            G_error_energy.append(G_and_G_error_energy[2])

        # %%
        # 对比 U2_EVV 与 U2_SSI 的 （绝对）误差

        is_print and print(tree_print(add_level=2) + "U_z_对比：U2_EVV 与 U2_SSI 的 （绝对）误差，随 z 的演化")
        for i in range(len(zj_EVV)):
            U_and_U_error_energy = U_compare(U2_stored_EVV[:, :, i], U2_stored_SSI[:, :, i], U2_stored_key_SSI,
                                             zj_EVV[i],
                                             # %%
                                             img_name_extension, size_PerPixel, size_fig,
                                             # %%
                                             is_save, is_save_txt, dpi,
                                             # %%
                                             cmap_2d,
                                             # %%
                                             ticks_num, is_contourf,
                                             is_title_on, is_axes_on, is_mm,
                                             # %%
                                             fontsize, font,
                                             # %%S
                                             is_colorbar_on, is_energy,
                                             # %%
                                             is_amp_relative, is_print,
                                             # %%
                                             is_end=is_end[i], )

            U_energy.append(U_and_U_error_energy[0])
            U0_energy.append(U_and_U_error_energy[1])
            U_error_energy.append(U_and_U_error_energy[2])

        G0_energy = np.array(G0_energy, dtype='float64')  # 需要把 list 转换为 array
        G_error_energy = np.array(G_error_energy, dtype='float64')
        U0_energy = np.array(U0_energy, dtype='float64')
        U_error_energy = np.array(U_error_energy, dtype='float64')

        is_end = [0] * (len(zj_EVV) - 1)
        is_end.append(-1)

        is_print and print(tree_print(add_level=1) + "G_energy 和 G_error")
        for i in range(len(zj_EVV)):
            is_print and print(tree_print(is_end[i]) + "zj, G_error, G0_energy, G_energy = {}, {}, {}, {}"
                               .format(format(zj_EVV[i], Get("F_f")), format(G_error_energy[i], Get("F_E")),
                                       format(G0_energy[i], Get("F_E")), format(G_energy[i], Get("F_E")), ))

        is_print and print(tree_print(is_end=1, add_level=1) + "U_energy 和 U_error")
        for i in range(len(zj_EVV)):
            is_print and print(tree_print(is_end[i]) + "zj, U_error, U0_energy, U_energy = {}, {}, {}, {}"
                               .format(format(zj_EVV[i], Get("F_f")), format(U_error_energy[i], Get("F_E")),
                                       format(U0_energy[i], Get("F_E")), format(U_energy[i], Get("F_E")), ))

        if is_energy_evolution_on == 1:
            U_twin_error_energy_plot_save(U2_energy_SSI, U2_energy_EVV, G_error_energy,
                                          U_key2_SSI.replace("_SSI", "").replace("U", "G"),
                                          img_name_extension, is_save_txt,
                                          # %%
                                          zj_SSI, zj_EVV, sample, size_PerPixel,
                                          is_save, dpi, Get("size_fig_x"), Get("size_fig_y"),
                                          # %%
                                          color_1d, color_1d2,
                                          ticks_num, is_title_on, is_axes_on, is_mm,
                                          fontsize, font,
                                          # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                                          # %%
                                          L0_Crystal,
                                          # %%
                                          is_energy_normalized=is_energy_normalized, **kwargs, )

            U_twin_error_energy_plot_save(U2_energy_SSI, U2_energy_EVV, U_error_energy, U_key2_SSI.replace("_SSI", ""),
                                          img_name_extension, is_save_txt,
                                          # %%
                                          zj_SSI, zj_EVV, sample, size_PerPixel,
                                          is_save, dpi, Get("size_fig_x"), Get("size_fig_y"),
                                          # %%
                                          color_1d, color_1d2,
                                          ticks_num, is_title_on, is_axes_on, is_mm,
                                          fontsize, font,
                                          # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                                          # %%
                                          L0_Crystal,
                                          # %%
                                          is_energy_normalized=is_energy_normalized, **kwargs, )

    # %%


if __name__ == '__main__':
    kwargs = \
        {"U_name_Structure": "",
         "is_phase_only_Structure": 0,
         # %%
         "z_pump_Structure": 0,
         "is_LG_Structure": 0, "is_Gauss_Structure": 1, "is_OAM_Structure": 1,
         "l_Structure": 2, "p_Structure": 0,
         "theta_x_Structure": 0, "theta_y_Structure": 0,
         # %%
         "is_random_phase_Structure": 0,
         "is_H_l_Structure": 0, "is_H_theta_Structure": 0, "is_H_random_phase_Structure": 0,
         # %%
         "U_name": "",
         "img_full_name": "lena1.png",
         "U_pixels_x": 0, "U_pixels_y": 0,
         "is_phase_only": 0,
         # %%
         "z_pump": 0,
         "is_LG": 0, "is_Gauss": 0, "is_OAM": 0,
         "l": 0, "p": 0,
         "theta_x": 0, "theta_y": 0,
         # %%
         "is_random_phase": 0,
         "is_H_l": 0, "is_H_theta": 0, "is_H_random_phase": 0,
         # %%---------------------------------------------------------------------
         # %%
         "U_size": 1, "w0": 0, "w0_Structure": 0, 
         "structure_size_Shrink": 0.1, "structure_size_Shrinker": 0,
         "is_U_size_x_structure_side_y": 1,
         "L0_Crystal": 5, "z0_structure_frontface_expect": 0, "deff_structure_length_expect": 1,
         # %%
         "SSI_zoomout_times": 1, "sheets_stored_num": 10,
         "z0_section_1_expect": 0, "z0_section_2_expect": 0,
         "X": 0, "Y": 0,
         # %%
         "Duty_Cycle_x": 0.5, "Duty_Cycle_y": 0.5, "Duty_Cycle_z": 0.5,
         "structure_xy_mode": 'x', "Depth": 2,
         # %%
         "is_continuous": 0, "is_target_far_field": 1, "is_transverse_xy": 0,
         "is_reverse_xy": 0, "is_positive_xy": 1,
         # %%
         "is_bulk": 0, "is_no_backgroud": 0,
         "is_stored": -1, "is_show_structure_face": 0, "is_energy_evolution_on": 1,
         # %%
         "lam1": 1.064, "is_air_pump": 1, "is_air": 0, "T": 25,
         "lam_structure": 1.064, "is_air_pump_structure": 1, "T_structure": 25,
         "deff": 30, "is_fft": 1, "fft_mode": 0,
         "is_sum_Gm": 0, "mG": 0, 'is_NLAST_sum': 0,
         "is_linear_convolution": 0,
         # %%
         "Tx": 18.769, "Ty": 20, "Tz": 0, # 11.873
         "mx": 1, "my": 0, "mz": 1,
         "is_stripe": 0, "is_NLAST": 1,
         # %%
         "is_save": 2, "is_no_data_save": 0,
         "is_save_txt": 0, "dpi": 100,
         # %%
         "color_1d": 'b', "color_1d2": 'r', "cmap_2d": 'viridis', "cmap_3d": 'rainbow',
         "elev": 10, "azim": -65, "alpha": 2,
         # %%
         "sample": 1, "ticks_num": 7, "is_contourf": 0,
         "is_title_on": 1, "is_axes_on": 1, "is_mm": 1,
         # %%
         "fontsize": 10,
         "font": {'family': 'serif',
                  'style': 'normal',  # 'normal', 'italic', 'oblique'
                  'weight': 'normal',
                  'color': 'black',  # 'black','gray','darkred'
                  },
         # %%
         "is_colorbar_on": 1, "is_colorbar_log": 0, 
         "is_energy": 1,
         # %%
         "plot_group": "UGa", "is_animated": 1,
         "loop": 0, "duration": 0.033, "fps": 5,
         # %%
         "is_plot_EVV": 1, "is_plot_3d_XYz": 0, "is_plot_selective": 0,
         "is_plot_YZ_XZ": 1, "is_plot_3d_XYZ": 0,
         # %%
         "is_print": 1, "is_contours": 0, "n_TzQ": 1,
         "Gz_max_Enhance": 1, "match_mode": 1,
         # %%
         "is_EVV_SSI": 0,
         # %% 该程序 独有 -------------------------------
         "is_NLA": 1, "is_amp_relative": 1,
         "is_energy_normalized": 2, "is_output_error_EVV": 1,
         # %% 该程序 作为 主入口时 -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "border_percentage": 0.1, "is_end": -1,
         # %%
         "size_fig_x_scale": 10, "size_fig_y_scale": 2,
         "ax_yscale": 'linear',
         # %%
         "theta_z": 90, "phi_z": 0, "phi_c": 24.3,
         # KTP 50 度 ：deff 最高： 90, ~, 24.3，（24.3 - 2002, 25.3 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.9 - 2000）
         # KTP 25 度 ：deff 最高： 90, ~, 23.7，（23.7 - 2002, 24.8 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.7 - 2000）
         # LN 25 度 ：90, ~, ~
         "polar": "e",
         "ray": "3", "polar3": "e",
         }

    if kwargs.get("ray", "2") == "3":  # 如果 ray == 3，则 默认 双泵浦 is_twin_pumps == 1
        pump2_kwargs = {
            "U2_name": "",
            "img2_full_name": "lena1.png",
            "is_phase_only_2": 0,
            # %%
            "z_pump2": 0,
            "is_LG_2": 0, "is_Gauss_2": 0, "is_OAM_2": 0,
            "l2": 0, "p2": 0,
            "theta2_x": 0, "theta2_y": 0,
            # %%
            "is_random_phase_2": 0,
            "is_H_l2": 0, "is_H_theta2": 0, "is_H_random_phase_2": 0,
            # %%
            "w0_2": 0,
            # %%
            "lam2": 1.064, "is_air_pump2": 1, "T2": 25,
            "polar2": 'e',
        }
        pump2_kwargs.update({"pump2_keys": list(pump2_kwargs.keys())})
        # Object of type dict_keys is not JSON serializable，所以 得转为 list
        kwargs.update(pump2_kwargs)

    kwargs = init_GLV_DICT(**kwargs)
    compare_SFG_NLA_EVV__SSI(**kwargs)

    # compare_SFG_NLA_EVV__SSI(U_name_Structure="",
    #                          is_phase_only_Structure=0,
    #                          # %%
    #                          z_pump_Structure=0,
    #                          is_LG_Structure=0, is_Gauss_Structure=1, is_OAM_Structure=0,
    #                          l_Structure=0, p_Structure=0,
    #                          theta_x_Structure=0, theta_y_Structure=0,
    #                          # %%
    #                          is_random_phase_Structure=0,
    #                          is_H_l_Structure=0, is_H_theta_Structure=0, is_H_random_phase_Structure=0,
    #                          # %%
    #                          U_name="",
    #                          img_full_name="lena1.png",
    #                          is_phase_only=0,
    #                          # %%
    #                          z_pump=0,
    #                          is_LG=0, is_Gauss=0, is_OAM=0,
    #                          l=0, p=0,
    #                          theta_x=0, theta_y=0,
    #                          # %%
    #                          is_random_phase=0,
    #                          is_H_l=0, is_H_theta=0, is_H_random_phase=0,
    #                          # %%---------------------------------------------------------------------
    #                          # %%
    #                          U_size=0.9, w0=0.3, w0_Structure=0, structure_size_Shrink=0.1,
    #                          L0_Crystal=2.66, z0_structure_frontface_expect=0, deff_structure_length_expect=1,
    #                          sheets_stored_num=10,
    #                          z0_section_1_expect=0, z0_section_2_expect=0,
    #                          X=0, Y=0,
    #                          # %%
    #                          Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
    #                          structure_xy_mode='x', Depth=2,
    #                          # %%
    #                          is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
    #                          is_reverse_xy=0, is_positive_xy=1,
    #                          # %%
    #                          is_bulk=0, is_no_backgroud=0,
    #                          is_stored=-1, is_show_structure_face=0, is_energy_evolution_on=1,
    #                          # %%
    #                          lam1=1.064, is_air_pump=0, is_air=0, T=25,
    #                          deff=30, is_fft=1, fft_mode=0,
    #                          is_sum_Gm=0, mG=0,
    #                          is_linear_convolution=0,
    #                          # %%
    #                          Tx=18.769, Ty=20, Tz=5.9,
    #                          mx=1, my=0, mz=0,
    #                          is_stripe=0, is_NLAST=1,
    #                          # %%
    #                          is_save=2, is_save_txt=0, dpi=100,
    #                          # %%
    #                          color_1d='b', color_1d2='r', cmap_2d='viridis', cmap_3d='rainbow',
    #                          elev=10, azim=-65, alpha=2,
    #                          # %%
    #                          sample=1, ticks_num=7, is_contourf=0,
    #                          is_title_on=1, is_axes_on=1, is_mm=1,
    #                          # %%
    #                          fontsize=9,
    #                          font={'family': 'serif',
    #                                'style': 'normal',  # 'normal', 'italic', 'oblique'
    #                                'weight': 'normal',
    #                                'color': 'black',  # 'black','gray','darkred'
    #                                },
    #                          # %%
    #                          is_colorbar_on=1, is_energy=0,
    #                          # %%
    #                          plot_group="UGa", is_animated=1,
    #                          loop=0, duration=0.033, fps=5,
    #                          # %%
    #                          is_plot_3d_XYz=0, is_plot_selective=0,
    #                          is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
    #                          # %%
    #                          is_print=1, is_contours=0, n_TzQ=1,
    #                          Gz_max_Enhance=1, match_mode=1,
    #                          # %%
    #                          is_NLA=1, is_amp_relative=1,
    #                          is_energy_normalized=2, is_output_error_EVV=1,
    #                          # %%
    #                          root_dir=r'',
    #                          border_percentage=0.1, is_end=-1,
    #                          size_fig_x_scale=10, size_fig_y_scale=2,
    #                          ax_yscale='linear', )

# 注意 colorbar 上的数量级
