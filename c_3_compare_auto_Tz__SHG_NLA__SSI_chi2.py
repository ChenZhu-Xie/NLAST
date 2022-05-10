# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:57 2021

@author: Xcz
"""

# %%

import math
import numpy as np
from fun_global_var import init_GLV_DICT, Get, tree_print, GU_error_energy_plot_save
from fun_img_Resize import if_image_Add_black_border
from fun_pump import pump_pic_or_U
from fun_linear import init_AST, init_SHG
from fun_nonlinear import Cal_lc_SHG
from c_2_compare_SHG_NLA__SSI_chi2 import compare_SHG_NLA__SSI

np.seterr(divide='ignore', invalid='ignore')


def auto_compare_SHG_NLA__SSI(U_name_Structure="",
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
                              U_NonZero_size=0.5, w0=0.1, w0_Structure=5, structure_size_Enlarge=0.1,
                              L0_Crystal=2, z0_structure_frontface_expect=0.5, deff_structure_length_expect=1,
                              sheets_stored_num=10,
                              z0_section_1_expect=1, z0_section_2_expect=1,
                              X=0, Y=0,
                              # %%
                              Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
                              structure_xy_mode='x', Depth=2,
                              # %%
                              is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
                              is_reverse_xy=0, is_positive_xy=1,
                              # %%
                              is_bulk=1, is_no_backgroud=1,
                              is_stored=0, is_show_structure_face=1, is_energy_evolution_on=1,
                              # %%
                              lam1=1.5, is_air_pump=0, is_air=0, T=25,
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
                              sample=2, ticks_num=6, is_contourf=0,
                              is_title_on=1, is_axes_on=1, is_mm=1,
                              # %%
                              fontsize=9,
                              font={'family': 'serif',
                                    'style': 'normal',  # 'normal', 'italic', 'oblique'
                                    'weight': 'normal',
                                    'color': 'black',  # 'black','gray','darkred'
                                    },
                              # %%
                              is_colorbar_on=1, is_energy=1,
                              # %%
                              plot_group="UGa", is_animated=1,
                              loop=0, duration=0.033, fps=5,
                              # %%
                              is_plot_3d_XYz=0, is_plot_selective=0,
                              is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
                              # %%
                              is_print=1, is_contours=1, n_TzQ=1,
                              Gz_max_Enhance=1, match_mode=1,
                              # %% 该程序 独有
                              is_NLA=1, is_amp_relative=1,
                              num_data_points=3, center_times=40, shift_right=3,
                              # %%
                              **kwargs, ):

    # %%
    info = "扫描 Tz，自动对比：NLA 与 SSI"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    # %%

    if_image_Add_black_border("", img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, \
    U_0, g_shift = pump_pic_or_U(U_name,
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
                                 U_NonZero_size, w0,
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

    dk, lc, Tz = Cal_lc_SHG(k1, k2, Tz, size_PerPixel,
                            0, )

    Tc = 2 * lc

    ticks_Num = num_data_points
    ticks_Num += 1

    # center_times >= 1，以防止 Tz 出现负数
    zoomout_times = (ticks_Num + 1) // 2 * center_times  # 步长缩得更小，这样 步长 * 步数 更小一些，防止 Tz 出现负数
    if Tz != Tc:
        Gz = 2 * math.pi * mz / (Tz / 1000)  # Tz / 1000 即以 mm 为单位
        delta_k = abs(dk / size_PerPixel + Gz)  # Unit: 1 / mm
    else:
        delta_k = abs(dk / size_PerPixel) / zoomout_times  # delta_k 是 恒正的

    # 这里的 shift_right “往右移” 是指 图 往右，左加右减，则 对应 自变量 x 是做差，所以下面是 - shift_right
    # x 不要 减 太多了，也就是 图不要 往右 移太多，否则 Tz 可能 出现负数
    array_1d = np.arange(0, ticks_Num, 1) - (ticks_Num - 1) // 2 - shift_right  # 尺子整体 更偏右一点，这样负的不多，防止 Tz 出现负数

    array_dkQ = array_1d * delta_k
    array_Gz = array_dkQ - dk / size_PerPixel  # Unit: 1 / mm
    array_Tz = 2 * math.pi * mz / array_Gz  # Unit: mm

    array_dkQ /= 1000  # Unit: 1 / μm
    array_Tz *= 1000  # Unit: μm
    # print(array_dkQ)
    # print(array_Tz)

    G_energy = []
    G_error_energy = []
    U_energy = []
    U_error_energy = []

    for i in range(ticks_Num):
        tuple_temp = \
            compare_SHG_NLA__SSI(U_name_Structure,
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
                                 U_name,
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
                                 # %%---------------------------------------------------------------------
                                 # %%
                                 U_NonZero_size, w0, w0_Structure, structure_size_Enlarge,
                                 L0_Crystal, z0_structure_frontface_expect, deff_structure_length_expect,
                                 sheets_stored_num,
                                 z0_section_1_expect, z0_section_2_expect,
                                 X, Y,
                                 # %%
                                 Duty_Cycle_x, Duty_Cycle_y, Duty_Cycle_z,
                                 structure_xy_mode, Depth,
                                 # %%
                                 is_continuous, is_target_far_field, is_transverse_xy,
                                 is_reverse_xy, is_positive_xy,
                                 # %%
                                 is_bulk, is_no_backgroud,
                                 is_stored, is_show_structure_face, is_energy_evolution_on,
                                 # %%
                                 lam1, is_air_pump, is_air, T,
                                 deff, is_fft, fft_mode,
                                 is_sum_Gm, mG,
                                 is_linear_convolution,
                                 # %%
                                 Tx, Ty, array_Tz[i],
                                 mx, my, mz,
                                 is_stripe, is_NLAST,
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
                                 is_colorbar_on, is_energy,
                                 # %%
                                 plot_group, is_animated,
                                 loop, duration, fps,
                                 # %%
                                 is_plot_3d_XYz, is_plot_selective,
                                 is_plot_YZ_XZ, is_plot_3d_XYZ,
                                 # %%
                                 is_print, is_contours, n_TzQ,
                                 Gz_max_Enhance, match_mode,
                                 # %%
                                 is_NLA, is_amp_relative, is_end=0, )

        G_energy.append(tuple_temp[0][0])
        G_error_energy.append(tuple_temp[0][1])
        U_energy.append(tuple_temp[1][0])
        U_error_energy.append(tuple_temp[1][1])

    G_energy = np.array(G_energy)  # 需要把 list 转换为 array
    G_error_energy = np.array(G_error_energy)
    U_energy = np.array(U_energy)
    U_error_energy = np.array(U_error_energy)

    is_end = [0] * (ticks_Num - 1)
    is_end.append(-1)

    is_print and print(tree_print(add_level=1) + "G_energy 和 G_error")
    for i in range(ticks_Num):
        is_print and print(tree_print(is_end[i]) + "Tz, G_error, dkQ, G_energy = {}, {}, {}, {}"
                           .format(format(array_Tz[i], Get("F_f")), format(G_error_energy[i], Get("F_E")),
                                   format(array_dkQ[i], Get("F_E")), format(G_energy[i], Get("F_E")), ))

    is_print and print(tree_print(is_end=1, add_level=1) + "U_energy 和 U_error")
    for i in range(ticks_Num):
        is_print and print(tree_print(is_end[i]) + "Tz, U_error, dkQ, U_energy = {}, {}, {}, {}"
                           .format(format(array_Tz[i], Get("F_f")), format(U_error_energy[i], Get("F_E")),
                                   format(array_dkQ[i], Get("F_E")), format(U_energy[i], Get("F_E")), ))

    GU_error_energy_plot_save(G_energy, G_error_energy, U_energy, U_error_energy,
                              img_name_extension, is_save_txt,
                              # %%
                              array_dkQ, array_Tz, sample, size_PerPixel,
                              is_save, dpi, Get("size_fig_x"), Get("size_fig_y"),
                              # %%
                              color_1d, color_1d2,
                              ticks_num, is_title_on, is_axes_on, is_mm,
                              fontsize, font,  # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                              # %%
                              L0_Crystal, **kwargs, )

    # %%


if __name__ == '__main__':
    kwargs = \
        {"U_name_Structure": "",
          "is_phase_only_Structure": 0,
          # %%
          "z_pump_Structure": 0,
          "is_LG_Structure": 0, "is_Gauss_Structure": 1, "is_OAM_Structure": 1,
          "l_Structure": 1, "p_Structure": 0,
          "theta_x_Structure": 0, "theta_y_Structure": 0,
          # %%
          "is_random_phase_Structure": 0,
          "is_H_l_Structure": 0, "is_H_theta_Structure": 0, "is_H_random_phase_Structure": 0,
          # %%
          "U_name": "",
          "img_full_name": "lena1.png",
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
          "U_NonZero_size": 0.9, "w0": 0.3, "w0_Structure": 0, "structure_size_Enlarge": 0.1,
          "L0_Crystal": 1, "z0_structure_frontface_expect": 0, "deff_structure_length_expect": 1,
          "sheets_stored_num": 10,
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
          "is_stored": 0, "is_show_structure_face": 0, "is_energy_evolution_on": 1,
          # %%
          "lam1": 1.064, "is_air_pump": 0, "is_air": 0, "T": 25,
          "deff": 30, "is_fft": 1, "fft_mode": 0,
          "is_sum_Gm": 0, "mG": 0, 'is_NLAST_sum': 1, 
          "is_linear_convolution": 0,
          # %%
          "Tx": 14.769, "Ty": 20, "Tz": 0,
          "mx": 1, "my": 0, "mz": 1,
          "is_stripe": 0, "is_NLAST": 1,
          # %%
          "is_save": 2, "is_save_txt": 0, "dpi": 100,
          # %%
          "color_1d": 'b', "color_1d2": 'r', "cmap_2d": 'viridis', "cmap_3d": 'rainbow',
          "elev": 10, "azim": -65, "alpha": 2,
          # %%
          "sample": 1, "ticks_num": 6, "is_contourf": 0,
          "is_title_on": 1, "is_axes_on": 1, "is_mm": 1,
          # %%
          "fontsize": 9,
          "font": {'family': 'serif',
                'style': 'normal',  # 'normal', 'italic', 'oblique'
                'weight': 'normal',
                'color': 'black',  # 'black','gray','darkred'
                },
          # %%
          "is_colorbar_on": 1, "is_energy": 1,
          # %%
          "plot_group": "UGa", "is_animated": 1,
          "loop": 0, "duration": 0.033, "fps": 5,
          # %%
          "is_plot_3d_XYz": 0, "is_plot_selective": 0,
          "is_plot_YZ_XZ": 1, "is_plot_3d_XYZ": 0,
          # %%
          "is_print": 1, "is_contours": 0, "n_TzQ": 1,
          "Gz_max_Enhance": 1, "match_mode": 1,
          # %% 该程序 独有
          "is_NLA": 1, "is_amp_relative": 1,
          "num_data_points": 40, "center_times": 1.5, "shift_right": 3,
          # %% 该程序 作为 主入口时
          "kwargs_seq": 0, "root_dir": r'1',
          "border_percentage": 0.1, "is_end": -1,
          "size_fig_x_scale": 10, "size_fig_y_scale": 1,
          "ax_yscale": 'linear', }

    kwargs = init_GLV_DICT(**kwargs)
    auto_compare_SHG_NLA__SSI(**kwargs)

    # auto_compare_SHG_NLA__SSI(U_name_Structure="",
    #                           is_phase_only_Structure=0,
    #                           # %%
    #                           z_pump_Structure=0,
    #                           is_LG_Structure=0, is_Gauss_Structure=1, is_OAM_Structure=1,
    #                           l_Structure=1, p_Structure=0,
    #                           theta_x_Structure=0, theta_y_Structure=0,
    #                           # %%
    #                           is_random_phase_Structure=0,
    #                           is_H_l_Structure=0, is_H_theta_Structure=0, is_H_random_phase_Structure=0,
    #                           # %%
    #                           U_name="",
    #                           img_full_name="lena1.png",
    #                           is_phase_only=0,
    #                           # %%
    #                           z_pump=0,
    #                           is_LG=0, is_Gauss=0, is_OAM=0,
    #                           l=0, p=0,
    #                           theta_x=0, theta_y=0,
    #                           # %%
    #                           is_random_phase=0,
    #                           is_H_l=0, is_H_theta=0, is_H_random_phase=0,
    #                           # %%---------------------------------------------------------------------
    #                           # %%
    #                           U_NonZero_size=0.9, w0=0.3, w0_Structure=0, structure_size_Enlarge=0.1,
    #                           L0_Crystal=1, z0_structure_frontface_expect=0, deff_structure_length_expect=1,
    #                           sheets_stored_num=10,
    #                           z0_section_1_expect=0, z0_section_2_expect=0,
    #                           X=0, Y=0,
    #                           # %%
    #                           Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
    #                           structure_xy_mode='x', Depth=2,
    #                           # %%
    #                           is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
    #                           is_reverse_xy=0, is_positive_xy=1,
    #                           # %%
    #                           is_bulk=0, is_no_backgroud=0,
    #                           is_stored=0, is_show_structure_face=0, is_energy_evolution_on=1,
    #                           # %%
    #                           lam1=1.064, is_air_pump=0, is_air=0, T=25,
    #                           deff=30, is_fft=1, fft_mode=0,
    #                           is_sum_Gm=0, mG=0,
    #                           is_linear_convolution=0,
    #                           # %%
    #                           Tx=14.769, Ty=20, Tz=0,
    #                           mx=1, my=0, mz=1,
    #                           is_stripe=0, is_NLAST=1,
    #                           # %%
    #                           is_save=2, is_save_txt=0, dpi=100,
    #                           # %%
    #                           color_1d='b', color_1d2='r', cmap_2d='viridis', cmap_3d='rainbow',
    #                           elev=10, azim=-65, alpha=2,
    #                           # %%
    #                           sample=1, ticks_num=6, is_contourf=0,
    #                           is_title_on=1, is_axes_on=1, is_mm=1,
    #                           # %%
    #                           fontsize=9,
    #                           font={'family': 'serif',
    #                                 'style': 'normal',  # 'normal', 'italic', 'oblique'
    #                                 'weight': 'normal',
    #                                 'color': 'black',  # 'black','gray','darkred'
    #                                 },
    #                           # %%
    #                           is_colorbar_on=1, is_energy=1,
    #                           # %%
    #                           plot_group="UGa", is_animated=1,
    #                           loop=0, duration=0.033, fps=5,
    #                           # %%
    #                           is_plot_3d_XYz=0, is_plot_selective=0,
    #                           is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
    #                           # %%
    #                           is_print=1, is_contours=0, n_TzQ=1,
    #                           Gz_max_Enhance=1, match_mode=1,
    #                           # %% 该程序 独有
    #                           is_NLA=1, is_amp_relative=1,
    #                           num_data_points=40, center_times=1.5, shift_right=3,
    #                           # %% 该程序 作为 主入口时
    #                           root_dir=r'',
    #                           border_percentage=0.1, is_end=-1,
    #                           size_fig_x_scale=10, size_fig_y_scale=1,
    #                           ax_yscale='linear', )

# 注意 colorbar 上的数量级
