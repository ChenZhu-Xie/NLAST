# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:57 2021

@author: Xcz
"""

# %%

import copy
import numpy as np
from fun_os import img_squared_bordered_Read
from fun_global_var import init_GLV_DICT, tree_print
from fun_img_Resize import if_image_Add_black_border
from fun_linear import fft2
from fun_compare import U_compare
from b_3_SFG_NLA import SFG_NLA
from B_3_SFG_NLA_SSI_chi2 import SFG_NLA_SSI
from B_3_SFG_SSF_SSI_chi2 import SFG_SSF_SSI

np.seterr(divide='ignore', invalid='ignore')


def compare_SFG_NLA__SSI(U_name_Structure="",
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
                         is_bulk=1, is_no_backgroud=1,
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
                         color_1d='b', cmap_2d='viridis', cmap_3d='rainbow',
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
                         # %%
                         is_NLA=1, is_amp_relative=1,
                         # %%
                         **kwargs, ):
    # %%
    info = "利用 SHG 对比：NLA 与 SSI"
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
         structure_size_Enlarge,
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
         U_NonZero_size, w0,
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
         is_colorbar_on, is_energy,
         # %%
         plot_group, is_animated,
         loop, duration, fps,
         # %%
         is_plot_3d_XYz, is_plot_selective,
         is_plot_YZ_XZ, is_plot_3d_XYZ,
         # %%
         is_print, is_contours, n_TzQ,
         Gz_max_Enhance, match_mode, ]

    kwargs_SSI = copy.deepcopy(kwargs)
    kwargs_SSI.update({"ray": kwargs.get("ray", "2"), })
    U2_SSI, G2_SSI, ray2_SSI, method_and_way2_SSI, U_key2_SSI = \
        SFG_NLA_SSI(*args_SSI, **kwargs_SSI, ) if is_NLA == 1 else \
            SFG_SSF_SSI(*args_SSI, **kwargs_SSI, )

    kwargs_NLA = copy.deepcopy(kwargs)
    kwargs_NLA.update({"ray": kwargs.get("ray", "2"), })
    U2_NLA, G2_NLA, ray2_NLA, method_and_way2_NLA, U_key2_NLA = \
        SFG_NLA(U_name,
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
                structure_size_Enlarge,
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
                U_NonZero_size, w0,
                L0_Crystal,
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
                is_continuous, is_target_far_field, is_transverse_xy,
                is_reverse_xy, is_positive_xy, is_no_backgroud,
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
                is_print, is_contours, n_TzQ,
                Gz_max_Enhance, match_mode,
                **kwargs_NLA, )

    # %%

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, U = \
        img_squared_bordered_Read(img_full_name,
                                  U_NonZero_size, dpi,
                                  is_phase_only)

    # %%
    # 对比 G2_NLA 与 G2_SSI 的 （绝对）误差

    G_and_G_error_energy = U_compare(fft2(U2_NLA), fft2(U2_SSI), U_key2_SSI.replace("U", "G"), L0_Crystal,
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
    # 对比 U2_NLA 与 U2_ssi 的 （绝对）误差

    U_and_U_error_energy = U_compare(U2_NLA, U2_SSI, U_key2_SSI, L0_Crystal,
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

    # %%

    return (G_and_G_error_energy, U_and_U_error_energy)


if __name__ == '__main__':
    kwargs = \
        {"U_name_Structure": "",
         "is_phase_only_Structure": 0,
         # %%
         "z_pump_Structure": 0,
         "is_LG_Structure": 0, "is_Gauss_Structure": 1, "is_OAM_Structure": 0,
         "l_Structure": 0, "p_Structure": 0,
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
         "is_stored": 0, "is_show_structure_face": 0, "is_energy_evolution_on": 1,
         # %%
         "lam1": 1.064, "is_air_pump": 1, "is_air": 0, "T": 25,
         "lam_structure": 1.064, "is_air_pump_structure": 1, "T_structure": 25,
         "deff": 30, "is_fft": 1, "fft_mode": 0,
         "is_sum_Gm": 0, "mG": 0, 'is_NLAST_sum': 0,
         "is_linear_convolution": 0,
         # %%
         "Tx": 10, "Ty": 20, "Tz": 0,
         "mx": 1, "my": 0, "mz": 1,
         "is_stripe": 0, "is_NLAST": 1,
         # %%
         "is_save": 0, "is_save_txt": 0, "dpi": 100,
         # %%
         "color_1d": 'b', "cmap_2d": 'viridis', "cmap_3d": 'rainbow',
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
         "is_colorbar_on": 1, "is_energy": 0,
         # %%
         "plot_group": "UGa", "is_animated": 1,
         "loop": 0, "duration": 0.033, "fps": 5,
         # %%
         "is_plot_3d_XYz": 0, "is_plot_selective": 0,
         "is_plot_YZ_XZ": 1, "is_plot_3d_XYZ": 0,
         # %%
         "is_print": 1, "is_contours": 0, "n_TzQ": 1,
         "Gz_max_Enhance": 1, "match_mode": 1,
         # %% 该程序 独有 -------------------------------
         "is_NLA": 1, "is_amp_relative": 1,
         # %% 该程序 作为 主入口时 -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "border_percentage": 0.1, "is_end": -1,
         # %%
         "size_fig_x_scale": 10, "size_fig_y_scale": 1,
         # %%
         "theta_z": 90, "phi_z": 0, "phi_c": 24.3,
         # KTP 25 度 ：deff 最高： 90, ~, 24.3，（24.3 - 2002, 24.8 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.7 - 2000）
         # LN 25 度 ：90, ~, ~
         "polar": "e",
         "ray": "3", "polar3": "e",
         }

    if kwargs.get("ray", "2") == "3":  # 如果 ray == 3，则 默认 双泵浦 is_twin_pumps == 1
        pump2_kwargs = {
            "U2_name": "",
            "img2_full_name": "spaceship.png",
            "is_phase_only_2": 0,
            # %%
            "z_pump2": 0,
            "is_LG_2": 0, "is_Gauss_2": 1, "is_OAM_2": 0,
            "l2": 0, "p2": 0,
            "theta2_x": 0, "theta2_y": 0,
            # %%
            "is_random_phase_2": 0,
            "is_H_l2": 0, "is_H_theta2": 0, "is_H_random_phase_2": 0,
            # %%
            "w0_2": 0.3,
            # %%
            "lam2": 1, "is_air_pump2": 1, "T2": 25,
            "polar2": 'e',
        }
        pump2_kwargs.update({"pump2_keys": list(pump2_kwargs.keys())})
        # Object of type dict_keys is not JSON serializable，所以 得转为 list
        kwargs.update(pump2_kwargs)

    kwargs = init_GLV_DICT(**kwargs)
    compare_SFG_NLA__SSI(**kwargs)

    # compare_SFG_NLA__SSI(U_name_Structure="",
    #                      is_phase_only_Structure=0,
    #                      # %%
    #                      z_pump_Structure=0,
    #                      is_LG_Structure=0, is_Gauss_Structure=1, is_OAM_Structure=0,
    #                      l_Structure=0, p_Structure=0,
    #                      theta_x_Structure=0, theta_y_Structure=0,
    #                      # %%
    #                      is_random_phase_Structure=0,
    #                      is_H_l_Structure=0, is_H_theta_Structure=0, is_H_random_phase_Structure=0,
    #                      # %%
    #                      U_name="",
    #                      img_full_name="lena1.png",
    #                      is_phase_only=0,
    #                      # %%
    #                      z_pump=0,
    #                      is_LG=0, is_Gauss=0, is_OAM=0,
    #                      l=0, p=0,
    #                      theta_x=0, theta_y=0,
    #                      # %%
    #                      is_random_phase=0,
    #                      is_H_l=0, is_H_theta=0, is_H_random_phase=0,
    #                      # %%---------------------------------------------------------------------
    #                      # %%
    #                      U_NonZero_size=0.9, w0=0.3, w0_Structure=0, structure_size_Enlarge=0.1,
    #                      L0_Crystal=2.66, z0_structure_frontface_expect=0, deff_structure_length_expect=1,
    #                      sheets_stored_num=10,
    #                      z0_section_1_expect=0, z0_section_2_expect=0,
    #                      X=0, Y=0,
    #                      # %%
    #                      Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
    #                      structure_xy_mode='x', Depth=2,
    #                      # %%
    #                      is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
    #                      is_reverse_xy=0, is_positive_xy=1,
    #                      # %%
    #                      is_bulk=0, is_no_backgroud=0,
    #                      is_stored=0, is_show_structure_face=0, is_energy_evolution_on=1,
    #                      # %%
    #                      lam1=1.064, is_air_pump=0, is_air=0, T=25,
    #                      deff=30, is_fft=1, fft_mode=0,
    #                      is_sum_Gm=0, mG=0,
    #                      is_linear_convolution=0,
    #                      # %%
    #                      Tx=10, Ty=20, Tz=10,
    #                      mx=1, my=0, mz=1,
    #                      is_stripe=0, is_NLAST=1,
    #                      # %%
    #                      is_save=0, is_save_txt=0, dpi=100,
    #                      # %%
    #                      color_1d='b', cmap_2d='viridis', cmap_3d='rainbow',
    #                      elev=10, azim=-65, alpha=2,
    #                      # %%
    #                      sample=2, ticks_num=6, is_contourf=0,
    #                      is_title_on=1, is_axes_on=1, is_mm=1,
    #                      # %%
    #                      fontsize=9,
    #                      font={'family': 'serif',
    #                            'style': 'normal',  # 'normal', 'italic', 'oblique'
    #                            'weight': 'normal',
    #                            'color': 'black',  # 'black','gray','darkred'
    #                            },
    #                      # %%
    #                      is_colorbar_on=1, is_energy=1,
    #                      # %%
    #                      plot_group="UGa", is_animated=1,
    #                      loop=0, duration=0.033, fps=5,
    #                      # %%
    #                      is_plot_3d_XYz=0, is_plot_selective=0,
    #                      is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
    #                      # %%
    #                      is_print=1, is_contours=0, n_TzQ=1,
    #                      Gz_max_Enhance=1, match_mode=1,
    #                      # %%
    #                      is_NLA=1, is_amp_relative=1,
    #                      # %%
    #                      root_dir=r'',
    #                      border_percentage=0.1, is_end=-1,
    #                      size_fig_x_scale=10, size_fig_y_scale=1, )

# 注意 colorbar 上的数量级
