# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import copy
import numpy as np
from fun_img_Resize import if_image_Add_black_border
from fun_global_var import init_GLV_DICT, tree_print, Get
from b_1_AST_EVV import AST_EVV
from b_3_SFG_NLA_EVV import SFG_NLA_EVV

np.seterr(divide='ignore', invalid='ignore')


# %%

def SFG_NLA_EVV__AST_EVV(U_name="",
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
                         L0_Crystal=1, z_AST=1, sheets_stored_num=10,
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
                         is_stored=1, is_energy_evolution_on=1,
                         # %%
                         is_save=0, is_save_txt=0, dpi=100,
                         # %%
                         color_1d='b', cmap_2d='viridis', cmap_3d='rainbow',
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
                         is_energy=0,
                         # %%
                         is_plot_EVV=1, is_plot_3d_XYz=0, is_plot_selective=0,
                         X=0, Y=0, is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
                         # %%
                         plot_group="UGa", is_animated=1,
                         loop=0, duration=0.033, fps=5,
                         # %%
                         is_print=1, is_contours=1, n_TzQ=1,
                         Gz_max_Enhance=1, match_mode=1,
                         # %% 该程序 独有 -------------------------------
                         is_EVV_SSI=0, is_add_lens=0,
                         # %%
                         **kwargs, ):
    # %%
    info = "先 NLA_EVV，后 AST_EVV"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    # %%

    if_image_Add_black_border("", img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%

    def args_EVV(z_SFG):
        return [U_name,
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
                z_SFG, sheets_stored_num,
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

    def args_AST(z_AST):
        return ["",
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
                z_AST,
                # %%
                lam1, is_air_pump, 1, T,  # 后续 线性衍射 过程中， is_air = 1
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
                is_colorbar_on, is_colorbar_log,
                is_energy,
                # %%
                is_print,
                # %% 该程序 独有 -------------------------------
                is_EVV_SSI, is_stored, sheets_stored_num,
                # %%
                sample, cmap_3d,
                elev, azim, alpha,
                # %%
                is_plot_EVV, is_plot_3d_XYz, is_plot_selective,
                X, Y, is_plot_YZ_XZ, is_plot_3d_XYZ,
                # %%
                plot_group, is_animated,
                loop, duration, fps, ]

    kwargs_EVV = copy.deepcopy(kwargs)
    U2_NLA, G2_NLA, ray2_NLA, method_and_way2_NLA, U_key2_NLA = \
        SFG_NLA_EVV(*args_EVV(L0_Crystal), **kwargs_EVV, )

    kwargs_AST = copy.deepcopy(kwargs)
    kwargs_AST.update({"U": U2_NLA, "ray": ray2_NLA,
                       "lam3": Get("lam3"), "polar": kwargs_AST["polar3"], })

    if is_add_lens != 1:
        kwargs_AST.update({"is_end": 1, })
        U1_AST, G1_AST, ray1_AST, method_and_way1_AST, U_key1_AST = \
            AST_EVV(*args_AST(z_AST), **kwargs_AST, )
    else:
        f = z_AST / 2

        U1_AST, G1_AST, ray1_AST, method_and_way1_AST, U_key1_AST = \
            AST_EVV(*args_AST(f), **kwargs_AST, )

        from fun_nonlinear import init_SFG
        lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy = init_SFG(Get("Ix"), Get("Iy"), Get("size_PerPixel"),
                                                             lam1, 0, T,
                                                             0, 0, **kwargs)
        from fun_linear import Cal_H_lens
        # H_lens = Cal_H_lens(Get("Ix"), Get("Iy"), Get("size_PerPixel"), Get("k3"), z_AST / 2, Cal_mode=1)
        H_lens = Cal_H_lens(Get("Ix"), Get("Iy"), Get("size_PerPixel"), k3, f, Cal_mode=1)
        U1_AST *= H_lens

        kwargs_AST.update({"U": U1_AST, "ray": ray1_AST, "is_end": 1, })
        U1_AST, G1_AST, ray1_AST, method_and_way1_AST, U_key1_AST = \
            AST_EVV(*args_AST(f), **kwargs_AST, )


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",
         "img_full_name": "spaceship.png",
         "is_phase_only": 0,
         # %%
         "z_pump": 0,
         "is_LG": 1, "is_Gauss": 1, "is_OAM": 1,
         "l": 3, "p": 0,
         "theta_x": 0, "theta_y": 0,
         # %%
         "is_random_phase": 0,
         "is_H_l": 0, "is_H_theta": 0, "is_H_random_phase": 0,
         # %%
         # 生成横向结构
         "U_name_Structure": '',
         "structure_size_Enlarge": 0.1, "structure_side_Enlarger": -1.05,
         "is_U_NonZero_size_x_structure_side_y": 1,
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
         "U_NonZero_size": 1, "w0": 0.05,
         "L0_Crystal": 15, "z_AST": 20, "sheets_stored_num": 10,
         # %%
         "lam1": 1.064, "is_air_pump": 1, "is_air": 0, "T": 25,
         "lam_structure": 1, "is_air_pump_structure": 1, "T_structure": 25,
         "deff": 30, "is_fft": 1, "fft_mode": 0,
         "is_sum_Gm": 0, "mG": 0, 'is_NLAST_sum': 0,
         "is_linear_convolution": 0,
         # %%
         "Tx": 8, "Ty": 25, "Tz": 0,
         "mx": 1, "my": 0, "mz": 0,
         # %%
         # 生成横向结构
         "Duty_Cycle_x": 0.5, "Duty_Cycle_y": 0.5, "Duty_Cycle_z": 0.5,
         "Depth": 2, "structure_xy_mode": 'x',
         # %%
         "is_continuous": 0, "is_target_far_field": 1, "is_transverse_xy": 0,
         "is_reverse_xy": 0, "is_positive_xy": 1, "is_no_backgroud": 0,
         "is_stored": 1, "is_energy_evolution_on": 1,
         # %%
         "is_save": 0, "is_no_data_save": 0,
         "is_save_txt": 0, "dpi": 100,
         # %%
         "color_1d": 'b', "cmap_2d": 'viridis', "cmap_3d": 'rainbow',
         "elev": 10, "azim": -65, "alpha": 2,
         # %%
         "sample": 1, "ticks_num": 6, "is_contourf": 0,
         "is_title_on": 1, "is_axes_on": 1, "is_mm": 1,
         # %%
         "fontsize": 10,
         "font": {'family': 'serif',
                  'style': 'normal',  # 'normal', 'italic', 'oblique'
                  'weight': 'normal',
                  'color': 'black',  # 'black','gray','darkred'
                  },
         # %%
         "is_colorbar_on": 1, "is_colorbar_log": -1,
         "is_energy": 1,
         # %%
         "is_plot_EVV": 1, "is_plot_3d_XYz": 0, "is_plot_selective": 0,
         "X": 0, "Y": 0, "is_plot_YZ_XZ": 0, "is_plot_3d_XYZ": 0,
         # %%
         "plot_group": "Ua", "is_animated": 1,
         "loop": 0, "duration": 0.033, "fps": 5,
         # %%
         "is_print": 1, "is_contours": 0, "n_TzQ": 1,
         "Gz_max_Enhance": 1, "match_mode": 1,
         # %% 该程序 独有 -------------------------------
         "is_EVV_SSI": 0, "is_add_lens": 0,
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
            "img2_full_name": "lena1.png",
            "is_phase_only_2": 0,
            # %%
            "z_pump2": 0,
            "is_LG_2": 1, "is_Gauss_2": 1, "is_OAM_2": 1,
            "l2": 3, "p2": 0,
            "theta2_x": 0, "theta2_y": 0,
            # %%
            "is_random_phase_2": 0,
            "is_H_l2": 0, "is_H_theta2": 0, "is_H_random_phase_2": 0,
            # %%
            "w0_2": 0.05,
            # %%
            "lam2": 1.064, "is_air_pump2": 1, "T2": 25,
            "polar2": 'e',
        }
        pump2_kwargs.update({"pump2_keys": list(pump2_kwargs.keys())})
        # Object of type dict_keys is not JSON serializable，所以 得转为 list
        kwargs.update(pump2_kwargs)

    kwargs = init_GLV_DICT(**kwargs)
    SFG_NLA_EVV__AST_EVV(**kwargs)

    # SFG_NLA_EVV(U_name="",
    #             img_full_name="lena1.png",
    #             is_phase_only=0,
    #             # %%
    #             z_pump=0,
    #             is_LG=0, is_Gauss=0, is_OAM=0,
    #             l=0, p=0,
    #             theta_x=0, theta_y=0,
    #             # %%
    #             is_random_phase=0,
    #             is_H_l=0, is_H_theta=0, is_H_random_phase=0,
    #             # %%
    #             # 生成横向结构
    #             U_name_Structure='',
    #             structure_size_Enlarge=0.1,
    #             is_phase_only_Structure=0,
    #             # %%
    #             w0_Structure=0, z_pump_Structure=0,
    #             is_LG_Structure=0, is_Gauss_Structure=1, is_OAM_Structure=0,
    #             l_Structure=0, p_Structure=0,
    #             theta_x_Structure=0, theta_y_Structure=0,
    #             # %%
    #             is_random_phase_Structure=0,
    #             is_H_l_Structure=0, is_H_theta_Structure=0, is_H_random_phase_Structure=0,
    #             # %%
    #             U_NonZero_size=0.9, w0=0.3,
    #             z0=10, sheets_stored_num=10,
    #             # %%
    #             lam1=1.064, is_air_pump=0, is_air=0, T=25,
    #             deff=30, is_fft=1, fft_mode=0,
    #             is_sum_Gm=0, mG=0,
    #             is_linear_convolution=0,
    #             # %%
    #             Tx=10, Ty=10, Tz=0,
    #             mx=1, my=0, mz=0,
    #             # %%
    #             # 生成横向结构
    #             Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
    #             Depth=2, structure_xy_mode='x',
    #             # %%
    #             is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
    #             is_reverse_xy=0, is_positive_xy=1, is_no_backgroud=0,
    #             is_stored=0, is_energy_evolution_on=1,
    #             # %%
    #             is_save=0, is_save_txt=0, dpi=100,
    #             # %%
    #             color_1d='b', cmap_2d='viridis', cmap_3d='rainbow',
    #             elev=10, azim=-65, alpha=2,
    #             # %%
    #             sample=1, ticks_num=6, is_contourf=0,
    #             is_title_on=1, is_axes_on=1, is_mm=1,
    #             # %%
    #             fontsize=9,
    #             font={'family': 'serif',
    #                   'style': 'normal',  # 'normal', 'italic', 'oblique'
    #                   'weight': 'normal',
    #                   'color': 'black',  # 'black','gray','darkred'
    #                   },
    #             # %%
    #             is_colorbar_on=1, is_energy=0, is_plot_3d_XYz = 0,
    #             #%%
    #             plot_group = "UGa", is_animated = 1,
    #             loop = 0, duration = 0.033, fps = 5,
    #             # %%
    #             is_print=1, is_contours=66, n_TzQ=1,
    #             Gz_max_Enhance=1, match_mode=1,
    #             # %%
    #             root_dir=r'',
    #             border_percentage=0.1, ray="2", is_end=-1,
    #             size_fig_x_scale=10, size_fig_y_scale=1, )
