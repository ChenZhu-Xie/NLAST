# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import copy
import numpy as np
from fun_img_Resize import if_image_Add_black_border
from fun_global_var import init_GLV_DICT, init_GLV_rmw, Get, fkey, tree_print
from fun_nonlinear import gan_G3_z_sinc_reverse
from fun_compare import U_compare
from b_3_SFG_NLA import SFG_NLA
from B_3_SFG_NLA_SSI_chi2 import SFG_NLA_SSI

np.seterr(divide='ignore', invalid='ignore')


# %%

def SFG_NLA_reverse(U_name="",
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
                    # 生成横向结构
                    U_name_Structure='',
                    structure_size_Enlarge=0.1,
                    is_phase_only_Structure=0,
                    # %%
                    w0_Structure=0, z_pump_Structure=0,
                    is_LG_Structure=0, is_Gauss_Structure=0, is_OAM_Structure=0,
                    l_Structure=0, p_Structure=0,
                    theta_x_Structure=0, theta_y_Structure=0,
                    # %%
                    is_random_phase_Structure=0,
                    is_H_l_Structure=0, is_H_theta_Structure=0, is_H_random_phase_Structure=0,
                    # %%
                    U_NonZero_size=1, w0=0.3,
                    z0=1,
                    # %%
                    lam1=0.8, is_air_pump=0, is_air=0, T=25,
                    is_air_pump_structure=0,
                    deff=30, is_fft=1, fft_mode=0,
                    is_sum_Gm=0, mG=0,
                    is_linear_convolution=0,
                    # %%
                    Tx=10, Ty=10, Tz="2*lc",
                    mx=0, my=0, mz=0,
                    # %%
                    # 生成横向结构
                    Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
                    Depth=2, structure_xy_mode='x',
                    # %%
                    is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
                    is_reverse_xy=0, is_positive_xy=1, is_no_backgroud=0,
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
                    is_print=1, is_contours=1, n_TzQ=1,
                    Gz_max_Enhance=1, match_mode=1,
                    # %%  SSI 参数 -------------------------------
                    z0_structure_frontface_expect=0.5, deff_structure_length_expect=2,
                    SSI_zoomout_times=1, sheets_stored_num=10,
                    z0_section_1_expect=1, z0_section_2_expect=1,
                    X=0, Y=0,
                    # %%
                    is_bulk=1,
                    is_stored=0, is_show_structure_face=1, is_energy_evolution_on=1,
                    # %%
                    is_stripe=0, is_NLAST=0,
                    # %%
                    color_1d='b', cmap_3d='rainbow',
                    elev=10, azim=-65, alpha=2,
                    # %%
                    sample=1,
                    # %%
                    plot_group="UGa", is_animated=1,
                    loop=0, duration=0.033, fps=5,
                    # %%
                    is_plot_EVV=1, is_plot_3d_XYz=0, is_plot_selective=0,
                    is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
                    # %%  该程序 独有 -------------------------------
                    Cal_target=1, is_amp_relative=1,
                    is_SSI=1,
                    # %%
                    **kwargs, ):
    # %%
    info = "反向 NLA：已知 E3、χ2、E1、E2 中的 三者，可求 剩下的 一者"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    # %%
    # 非线性 惠更斯 菲涅尔 原理

    if_image_Add_black_border("", img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%

    args_NLA = [U_name,
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
                z0,
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
                Gz_max_Enhance, match_mode, ]

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
         z0, z0_structure_frontface_expect, deff_structure_length_expect,
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
         is_plot_EVV, is_plot_3d_XYz, is_plot_selective,
         is_plot_YZ_XZ, is_plot_3d_XYZ,
         # %%
         is_print, is_contours, n_TzQ,
         Gz_max_Enhance, match_mode, ]

    # %%
    kwargs_forward = copy.deepcopy(kwargs)

    U3_z, U1_0, U2_0, modulation_squared, k1_inc, k2_inc, \
    theta_x, theta_y, theta2_x, theta2_y, kiizQ, \
    k1, k2, k3, const, iz, Gz = \
        SFG_NLA_SSI(*args_SSI, **kwargs_forward, ) if is_SSI == 1 else \
            SFG_NLA(*args_NLA, **kwargs_forward, )

    # Cal_targets = ["U_0", "U1_0", "U2_0", "mod_0", "mod_2", "U_half_z_Squared_modulated"]
    # Cal_targets = ["U", "U1", "U2", "m", "m2"]

    if Cal_target == "U":  # "U_0"
        target = U1_0

        method = "PUMP"
        way = ""
        ray = ""
        ray_tag = "p"
        init_GLV_rmw("", ray_tag, method, way, ray=ray)
        U0_name = fkey("U").replace("_z", "")

        kwargs_Cal_reverse = {
            "mod": np.where(modulation_squared == 0, 1, modulation_squared),
            "k1_inc": k1_inc,
            "theta_x": theta_x,
            "theta_y": theta_y,
            "kiizQ": kiizQ,
        }
    elif Cal_target == "U1":  # "U1_0"
        target = U1_0

        method = "PUMP"
        way = ""
        ray = "1"
        ray_tag = "p"
        init_GLV_rmw("", ray_tag, method, way, ray=ray)
        U0_name = fkey("U").replace("_z", "")

        # print(modulation_squared)
        kwargs_Cal_reverse = {
            "mod": np.where(modulation_squared == 0, 1, modulation_squared),
            "U2_0": U2_0,
            "k1_inc": k1_inc,
            "theta_x": theta_x,
            "theta_y": theta_y,
            "kiizQ": kiizQ,
        }
    elif Cal_target == "U2":  # "U2_0"
        target = U2_0

        method = "PUMP"
        way = ""
        ray = "2"
        ray_tag = "p"
        init_GLV_rmw("", ray_tag, method, way, ray=ray)
        U0_name = fkey("U").replace("_z", "")

        kwargs_Cal_reverse = {
            "mod": np.where(modulation_squared == 0, 1, modulation_squared),
            "U1_0": U1_0,
            "k2_inc": k2_inc,
            "theta2_x": theta2_x,
            "theta2_x": theta2_x,
            "kiizQ": kiizQ,
        }
    elif Cal_target == "m":  # "mod_0"
        target = modulation_squared

        method = "MOD"
        name = "χ2_modulation_squared"
        suffix = "_squared"
        U0_name = method + " - " + name + suffix

        kwargs_Cal_reverse = {
            "U_0": U1_0,
        }
    elif Cal_target == "m2":  # "mod_2"
        target = modulation_squared

        method = "MOD"
        name = "χ2_modulation"
        suffix = "_squared"
        U0_name = method + " - " + name + suffix

        kwargs_Cal_reverse = {
            "U1_0": U1_0,
            "U2_0": U2_0,
            "kiizQ": kiizQ,
        }
    else:  # "U_half_z_Squared_modulated"
        target = None
        U0_name = ""
        kwargs_Cal_reverse = {
            "k1_inc": k1_inc,
            "k2_inc": k2_inc,
            "theta_x": theta_x,
            "theta_y": theta_y,
            "theta2_x": theta2_x,
            "theta2_x": theta2_x,
            "kiizQ": kiizQ,
        }

    if method == "MOD":
        suffix = "_guessed"
        U_name = method + " - " + name + suffix
    else:
        method = "NLA"
        way = "REV"
        init_GLV_rmw("", ray_tag, method, way, ray=ray)
        U_name = fkey("U").replace("_z", "")

        cmap_2d = "inferno"

    Cal_result = gan_G3_z_sinc_reverse(k1, k2, k3, U3_z, is_no_backgroud,
                                       const, iz, Gz, **kwargs_Cal_reverse)

    if method != "MOD":
        from fun_linear import fft2
        U_compare(fft2(Cal_result), fft2(target), U0_name.replace("- U", "- G"), -z0,
                  # %%
                  Get("img_name_extension"), Get("size_PerPixel"), Get("size_fig"),
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
                  U_title=U_name.replace("- U", "- G"), )

    U_compare(Cal_result, target, U0_name, -z0,
              # %%
              Get("img_name_extension"), Get("size_PerPixel"), Get("size_fig"),
              # %%
              is_save, is_save_txt, dpi,
              # %%
              cmap_2d,
              # %%
              ticks_num, is_contourf,
              is_title_on, is_axes_on, is_mm,
              # %%
              fontsize, font,
              # %%
              is_colorbar_on, is_energy,
              # %%
              is_amp_relative, is_print,
              # %%
              is_end=1, U_title=U_name, )


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",  # 要么从 U_name 里传 ray 和 U 进来，要么 单独传个 U 和 ray
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
         # %%
         # 生成横向结构
         "U_name_Structure": '',
         "structure_size_Enlarge": 0.19,
         "is_phase_only_Structure": 0,
         # %%
         "w0_Structure": 0, "z_pump_Structure": 0,
         "is_LG_Structure": 0, "is_Gauss_Structure": 1, "is_OAM_Structure": 0,
         "l_Structure": 0, "p_Structure": 0,
         "theta_x_Structure": 0, "theta_y_Structure": 0,
         # %%
         "is_random_phase_Structure": 0,
         "is_H_l_Structure": 0, "is_H_theta_Structure": 0, "is_H_random_phase_Structure": 0,
         # %%
         "U_NonZero_size": 1, "w0": 0.3,
         "z0": 2,
         # %%
         "lam1": 1.064, "is_air_pump": 1, "is_air": 0, "T": 50,
         "lam_structure": 1, "is_air_pump_structure": 1, "T_structure": 25,
         "deff": 30, "is_fft": 1, "fft_mode": 0,
         "is_sum_Gm": 0, "mG": 0, 'is_NLAST_sum': 0,
         "is_linear_convolution": 0,
         # %%
         "Tx": 30, "Ty": 30, "Tz": 6,
         "mx": 0, "my": 0, "mz": 1,
         # %%
         # 生成横向结构
         "Duty_Cycle_x": 0.5, "Duty_Cycle_y": 0.5, "Duty_Cycle_z": 0.5,
         "Depth": 2, "structure_xy_mode": 'x',
         # %%
         "is_continuous": 0, "is_target_far_field": 1, "is_transverse_xy": 0,
         "is_reverse_xy": 0, "is_positive_xy": 1, "is_no_backgroud": 0,
         # %%
         "is_save": 0, "is_save_txt": 0, "dpi": 100,
         # %%
         "cmap_2d": 'viridis',
         # %%
         "ticks_num": 6, "is_contourf": 0,
         "is_title_on": 1, "is_axes_on": 1, "is_mm": 1,
         # %%
         "fontsize": 10,
         "font": {'family': 'serif',
                  'style': 'normal',  # 'normal', 'italic', 'oblique'
                  'weight': 'normal',
                  'color': 'black',  # 'black','gray','darkred'
                  },
         # %%
         "is_colorbar_on": 1, "is_energy": 0,
         # %%
         "is_print": 1, "is_contours": 0, "n_TzQ": 1,
         "Gz_max_Enhance": 1, "match_mode": 1,
         # %%  SSI 参数 -------------------------------
         # %%
         "z0_structure_frontface_expect": 0, "deff_structure_length_expect": 2,
         "SSI_zoomout_times": 1, "sheets_stored_num": 10,
         "z0_section_1_expect": 1, "z0_section_2_expect": 1,
         "X": 0, "Y": 0,
         # %%
         "is_bulk": 0,
         "is_stored": 0, "is_show_structure_face": 1, "is_energy_evolution_on": 1,
         # %%
         "is_stripe": 0, "is_NLAST": 1,
         # %%
         "color_1d": 'b', "cmap_3d": 'rainbow',
         "elev": 10, "azim": -65, "alpha": 2,
         # %%
         "sample": 1,
         # %%
         "plot_group": "UGa", "is_animated": 1,
         "loop": 0, "duration": 0.033, "fps": 5,
         # %%
         "is_plot_EVV": 1, "is_plot_3d_XYz": 0, "is_plot_selective": 0,
         "is_plot_YZ_XZ": 0, "is_plot_3d_XYZ": 0,
         # %% 该程序 独有 -------------------------------
         "Cal_target": "U1", "is_amp_relative": 1,
         "is_SSI": 0,
         # %% 该程序 作为 主入口时 -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "border_percentage": 0.1, "is_end": -1,
         # %%
         "theta_z": 90, "phi_z": 0, "phi_c": 0,
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
            "w0_2": 0.3,
            # %%
            "lam2": 1.064, "is_air_pump2": 1, "T2": 25,
            "polar2": 'e',
        }
        pump2_kwargs.update({"pump2_keys": list(pump2_kwargs.keys())})
        # Object of type dict_keys is not JSON serializable，所以 得转为 list
        kwargs.update(pump2_kwargs)

    # 要不要把 “要不要（是否）使用 上一次使用 的 参数”
    # or “这次 使用了 哪一次 使用的 参数” 传进去记录呢？—— 也不是不行。

    # kwargs.update(init_GLV_DICT(**kwargs))
    kwargs = init_GLV_DICT(**kwargs)
    SFG_NLA_reverse(**kwargs)

    # SFG_NLA(U_name="", # 要么从 U_name 里传 ray 和 U 进来，要么 单独传个 U 和 ray
    #         img_full_name="lena1.png",
    #         is_phase_only=0,
    #         # %%
    #         z_pump=0,
    #         is_LG=0, is_Gauss=1, is_OAM=0,
    #         l=0, p=0,
    #         theta_x=0, theta_y=0,
    #         # %%
    #         is_random_phase=0,
    #         is_H_l=0, is_H_theta=0, is_H_random_phase=0,
    #         # %%
    #         # 生成横向结构
    #         U_name_Structure='',
    #         structure_size_Enlarge=0.1,
    #         is_phase_only_Structure=0,
    #         # %%
    #         w0_Structure=0, z_pump_Structure=0,
    #         is_LG_Structure=0, is_Gauss_Structure=1, is_OAM_Structure=0,
    #         l_Structure=0, p_Structure=0,
    #         theta_x_Structure=0, theta_y_Structure=0,
    #         # %%
    #         is_random_phase_Structure=0,
    #         is_H_l_Structure=0, is_H_theta_Structure=0, is_H_random_phase_Structure=0,
    #         # %%
    #         U_NonZero_size=0.9, w0=0.1,
    #         z0=10,
    #         # %%
    #         lam1=1.064, is_air_pump=0, is_air=0, T=25,
    #         deff=30, is_fft=1, fft_mode=0,
    #         is_sum_Gm=0, mG=0,
    #         is_linear_convolution=0,
    #         # %%
    #         Tx=10, Ty=10, Tz=3,
    #         mx=1, my=0, mz=0,
    #         # %%
    #         # 生成横向结构
    #         Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
    #         Depth=2, structure_xy_mode='x',
    #         # %%
    #         is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
    #         is_reverse_xy=0, is_positive_xy=1, is_no_backgroud=0,
    #         # %%
    #         is_save=1, is_save_txt=0, dpi=100,
    #         # %%
    #         cmap_2d='viridis',
    #         # %%
    #         ticks_num=6, is_contourf=0,
    #         is_title_on=1, is_axes_on=1, is_mm=1,
    #         # %%
    #         fontsize=9,
    #         font={'family': 'serif',
    #               'style': 'normal',  # 'normal', 'italic', 'oblique'
    #               'weight': 'normal',
    #               'color': 'black',  # 'black','gray','darkred'
    #               },
    #         # %%
    #         is_colorbar_on=1, is_energy=0,
    #         # %%
    #         is_print=1, is_contours=0, n_TzQ=1,
    #         Gz_max_Enhance=1, match_mode=1,
    #         # %%
    #         root_dir=r'af', ray="2", 
    #         border_percentage=0.1, is_end=-1, )
