# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:57 2021

@author: Xcz
"""

# %%

import copy
import numpy as np
from fun_img_Resize import if_image_Add_black_border
from fun_global_var import init_GLV_DICT
from fun_pump import pump_pic_or_U
from A_3_structure_chi2_Generate_3D import structure_chi2_3D
from B_3_SFG_NLA_ssi import SFG_NLA_ssi
from B_3_SFG_SSF_ssi import SFG_SSF_ssi

np.seterr(divide='ignore', invalid='ignore')


# %%

def A_3_to_B_3_SFG_NLA_ssi(U_name_Structure="",
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
                           structure_xy_mode='x', Depth=2, ssi_zoomout_times=5,
                           # %%
                           is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
                           is_reverse_xy=0, is_positive_xy=1,
                           # %%
                           is_bulk=1, is_no_backgroud=1,
                           is_stored=0, is_show_structure_face=1, is_energy_evolution_on=1,
                           # %%
                           lam1=1.5, is_air_pump=0, is_air=0, T=25,
                           is_air_pump_structure=0,
                           deff=30,
                           Tx=19.769, Ty=20, Tz=18.139,
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
                           is_NLA=1,
                           # %%
                           **kwargs, ):
    is_end, add_level = kwargs.get("is_end", 0), kwargs.get("add_level", 0)  # 将 is_end 拦截 下来，传给最末尾 的 含 print 函数
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    # %%
    # Image_Add_Black_border

    if_image_Add_black_border(U_name, img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%
    # 为了生成 U_0 和 g_shift

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
                                 ray_pump='1', **kwargs, )

    # %%

    structure_chi2_3D(U_name_Structure,
                      img_full_name,
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
                      U_NonZero_size, w0_Structure, structure_size_Enlarge,
                      deff_structure_length_expect,
                      # %%
                      Duty_Cycle_x, Duty_Cycle_y, Duty_Cycle_z,
                      structure_xy_mode, Depth, ssi_zoomout_times,
                      # %%
                      is_continuous, is_target_far_field, is_transverse_xy,
                      is_reverse_xy, is_positive_xy, is_no_backgroud,
                      # %%
                      lam1, is_air_pump_structure, is_air, T,
                      Tx, Ty, Tz,
                      mx, my, mz,
                      is_stripe,
                      # %%
                      is_save, is_save_txt, dpi,
                      is_bulk,
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
                      is_print, is_contours, n_TzQ,
                      Gz_max_Enhance, match_mode,
                      # %%
                      g_shift=g_shift, L0_Crystal=L0_Crystal, **kwargs, )

    # %%
    # B_3_NLA_SSI

    args_SFG_ssi = \
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
         U_NonZero_size, w0,
         L0_Crystal, z0_structure_frontface_expect, deff_structure_length_expect,
         Duty_Cycle_z, ssi_zoomout_times, sheets_stored_num,
         z0_section_1_expect, z0_section_2_expect,
         X, Y,
         # %%
         is_bulk, is_no_backgroud,
         is_stored, is_show_structure_face, is_energy_evolution_on,
         # %%
         lam1, is_air_pump, is_air, T,
         deff,
         Tx, Ty, Tz,
         mx, my, mz,
         is_NLAST,
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

    kwargs_ssi = copy.deepcopy(kwargs)
    kwargs_ssi.update({"is_end": is_end})
    if is_NLA == 1:
        return SFG_NLA_ssi(*args_SFG_ssi, **kwargs_ssi, )
    else:
        return SFG_SSF_ssi(*args_SFG_ssi, **kwargs_ssi, )


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
         "is_LG": 0, "is_Gauss": 1, "is_OAM": 0,
         "l": 0, "p": 0,
         "theta_x": 0, "theta_y": 0,
         # %%
         "is_random_phase": 0,
         "is_H_l": 0, "is_H_theta": 0, "is_H_random_phase": 0,
         # %%---------------------------------------------------------------------
         # %%
         "U_NonZero_size": 0.9, "w0": 0.1, "w0_Structure": 0, "structure_size_Enlarge": 0.1,
         "L0_Crystal": 2.25, "z0_structure_frontface_expect": 0, "deff_structure_length_expect": 0.5,
         "sheets_stored_num": 10,
         "z0_section_1_expect": 0, "z0_section_2_expect": 0,
         "X": 0, "Y": 0,
         # %%
         "Duty_Cycle_x": 0.5, "Duty_Cycle_y": 0.5, "Duty_Cycle_z": 0.5,
         "structure_xy_mode": 'x', "Depth": 2, "ssi_zoomout_times": 5,
         # %%
         "is_continuous": 0, "is_target_far_field": 1, "is_transverse_xy": 0,
         "is_reverse_xy": 0, "is_positive_xy": 1,
         # %%
         "is_bulk": 0, "is_no_backgroud": 0,
         "is_stored": 1, "is_show_structure_face": 0, "is_energy_evolution_on": 1,
         # %%
         "lam1": 1.064, "is_air_pump": 1, "is_air": 0, "T": 25,
         "lam_structure": 1.064, "is_air_pump_structure": 1, "T_structure": 25,
         "deff": 30,
         "Tx": 10, "Ty": 20, "Tz": 12.319,
         "mx": 1, "my": 0, "mz": 0,
         "is_stripe": 0, "is_NLAST": 1,
         # %%
         "is_save": 1, "is_save_txt": 0, "dpi": 100,
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
         "is_print": 1, "is_contours": 66, "n_TzQ": 1,
         "Gz_max_Enhance": 1, "match_mode": 0,
         # %% 该程序 独有 -------------------------------
         "is_NLA": 1,
         # %% 该程序 作为 主入口时 -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "border_percentage": 0.1, "is_end": -1,
         # %%
         "size_fig_x_scale": 10, "size_fig_y_scale": 1,
         # %%
         "theta_z": 90, "phi_z": 22.5, "phi_c": 22.5,
         "polar": "e",
         "ray": "2", "polar3": "e",
         }

    if kwargs.get("ray", "2") == "3":  # 如果 ray == 3，则 默认 双泵浦 is_twin_pumps == 1
        pump2_kwargs = {
            "U2_name": "",
            "img2_full_name": "lena.png",
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
    A_3_to_B_3_SFG_NLA_ssi(**kwargs)

    # A_3_to_B_3_SFG_NLA_ssi(U_name_Structure="",
    #                        is_phase_only_Structure=0,
    #                        # %%
    #                        z_pump_Structure=0,
    #                        is_LG_Structure=0, is_Gauss_Structure=1, is_OAM_Structure=1,
    #                        l_Structure=1, p_Structure=0,
    #                        theta_x_Structure=0, theta_y_Structure=0,
    #                        # %%
    #                        is_random_phase_Structure=0,
    #                        is_H_l_Structure=0, is_H_theta_Structure=0, is_H_random_phase_Structure=0,
    #                        # %%
    #                        U_name="",
    #                        img_full_name="lena1.png",
    #                        is_phase_only=0,
    #                        # %%
    #                        z_pump=0,
    #                        is_LG=0, is_Gauss=1, is_OAM=0,
    #                        l=0, p=0,
    #                        theta_x=0, theta_y=0,
    #                        # %%
    #                        is_random_phase=0,
    #                        is_H_l=0, is_H_theta=0, is_H_random_phase=0,
    #                        # %%---------------------------------------------------------------------
    #                        # %%
    #                        U_NonZero_size=0.9, w0=0.1, w0_Structure=0, structure_size_Enlarge=0.1,
    #                        L0_Crystal=2.25, z0_structure_frontface_expect=0, deff_structure_length_expect=0.5,
    #                        sheets_stored_num=10,
    #                        z0_section_1_expect=0, z0_section_2_expect=0,
    #                        X=0, Y=0,
    #                        # %%
    #                        Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
    #                        structure_xy_mode='x', Depth=2, ssi_zoomout_times=5,
    #                        # %%
    #                        is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
    #                        is_reverse_xy=0, is_positive_xy=1,
    #                        # %%
    #                        is_bulk=0, is_no_backgroud=0,
    #                        is_stored=1, is_show_structure_face=0, is_energy_evolution_on=1,
    #                        # %%
    #                        lam1=1.064, is_air_pump=0, is_air=0, T=25,
    #                        deff=30,
    #                        Tx=10, Ty=20, Tz=12.319,
    #                        mx=1, my=0, mz=0,
    #                        is_stripe=0, is_NLAST=1,
    #                        # %%
    #                        is_save=1, is_save_txt=0, dpi=100,
    #                        # %%
    #                        color_1d='b', cmap_2d='viridis', cmap_3d='rainbow',
    #                        elev=10, azim=-65, alpha=2,
    #                        # %%
    #                        sample=2, ticks_num=6, is_contourf=0,
    #                        is_title_on=1, is_axes_on=1, is_mm=1,
    #                        # %%
    #                        fontsize=9,
    #                        font={'family': 'serif',
    #                              'style': 'normal',  # 'normal', 'italic', 'oblique'
    #                              'weight': 'normal',
    #                              'color': 'black',  # 'black','gray','darkred'
    #                              },
    #                        # %%
    #                        is_colorbar_on=1, is_energy=0,
    #                        # %%
    #                        plot_group="UGa", is_animated=1,
    #                        loop=0, duration=0.033, fps=5,
    #                        # %%
    #                        is_plot_3d_XYz=0, is_plot_selective=0,
    #                        is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
    #                        # %%
    #                        is_print=1, is_contours=66, n_TzQ=1,
    #                        Gz_max_Enhance=1, match_mode=0,
    #                        # %%
    #                        is_NLA=1,
    #                        # %%
    #                        root_dir=r'',
    #                        border_percentage=0.1, is_end=-1,
    #                        size_fig_x_scale=10, size_fig_y_scale=1, )
