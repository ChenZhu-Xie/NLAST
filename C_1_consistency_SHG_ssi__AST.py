# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:57 2021

@author: Xcz
"""

# %%

import copy
import numpy as np
from fun_os import img_squared_bordered_Read, U_plot_save
from fun_img_Resize import if_image_Add_black_border
from fun_global_var import init_GLV_DICT, tree_print, init_GLV_rmw, fset, fget, fkey
from fun_linear import fft2
from fun_pump import pump_pic_or_U
from fun_compare import U_compare
from A_3_structure_chi2_Generate_3D import structure_chi2_3D
from b_1_AST import AST
from B_3_SFG_NLA_ssi import SFG_NLA_ssi
from B_3_SFG_SSF_ssi import SFG_SSF_ssi

np.seterr(divide='ignore', invalid='ignore')


def consistency_SHG_ssi__AST(img_full_name="Grating.png",
                             is_phase_only=0,
                             # %%
                             z_pump=0,
                             is_LG=0, is_Gauss=0, is_OAM=0,
                             l=0, p=0,
                             theta_x=0, theta_y=0,
                             # %%
                             is_random_phase=0,
                             is_H_l=0, is_H_theta=0, is_H_random_phase=0,
                             # %% 生成横向结构
                             U_name_Structure='',
                             structure_size_Enlarge=0.1,
                             is_phase_only_Structure=0,
                             # %%
                             w0_Structure=0, z_pump_Structure=0,
                             is_LG_Structure=0, is_Gauss_Structure=1, is_OAM_Structure=0,
                             l_Structure=0, p_Structure=0,
                             theta_x_Structure=0, theta_y_Structure=0,
                             # %%
                             is_random_phase_Structure=0,
                             is_H_l_Structure=0, is_H_theta_Structure=0, is_H_random_phase_Structure=0,
                             # %%
                             U_NonZero_size=1, w0=0.3,
                             z1=1, z2=2,
                             # %% 不关心
                             z0_structure_frontface_expect=0, deff_structure_length_expect=10,
                             sheets_stored_num=10, z0_section_1_expect=0, z0_section_2_expect=0,
                             X=0, Y=0,
                             # %% 不关心
                             is_bulk=1, is_no_backgroud=0,
                             is_stored=0, is_show_structure_face=0, is_energy_evolution_on=1,
                             # %%
                             lam1=0.8, is_air_pump=0, is_air=0, T=25,
                             is_air_pump_structure=0,
                             deff=30,
                             # %%
                             Tx=10, Ty=10, Tz="2*lc",
                             mx=0, my=0, mz=0,
                             is_stripe=0, is_NLAST=1,  # 不关心 is_stripe
                             # %% 生成横向结构
                             Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
                             Depth=2, ssi_zoomout_times=5, structure_xy_mode='x',
                             # %%
                             is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
                             is_reverse_xy=0, is_positive_xy=1,
                             # %%
                             is_save=0, is_save_txt=0, dpi=100,
                             # %%
                             color_1d='b', cmap_2d='viridis',
                             # %% 不关心
                             cmap_3d='rainbow', elev=10, azim=-65, alpha=2,
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
                             is_colorbar_on=1, is_energy=1,
                             # %% 不关心
                             plot_group="UGa", is_animated=1,
                             loop=0, duration=0.033, fps=5,
                             # %% 不关心
                             is_plot_3d_XYz=0, is_plot_selective=0,
                             is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
                             # %%
                             is_print=1, is_contours=66, n_TzQ=1,
                             Gz_max_Enhance=1, match_mode=1,
                             # %% 该程序 独有 -------------------------------
                             is_NLA=1, is_amp_relative=1,
                             # %%
                             **kwargs, ):
    # %%
    info = "利用 SHG 检验：ssi 自洽性"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    # %%
    # 非线性 惠更斯 菲涅尔 原理

    if_image_Add_black_border("", img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%
    # 为了生成 U_0 和 g_shift

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, \
    U_0, g_shift = pump_pic_or_U("",
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
                                 kwargs.get("polar", "e"),
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
                      g_shift=g_shift, L0_Crystal=max(z1, z2), **kwargs, )

    # %%
    # 先衍射 z1 后倍频 z2

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
                lam1, is_air_pump, is_air, T,
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
                is_print, ]

    def args_ssi(z_ssi):
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
                z_ssi, z0_structure_frontface_expect, deff_structure_length_expect,
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

    kwargs_AST = copy.deepcopy(kwargs)
    kwargs_AST.update({"ray": "1", })
    U1_z1, G1_z1, ray1_z1, method_and_way1_z1, U_key1_z1 = \
        AST(*args_AST(z1), **kwargs_AST, )

    kwargs_ssi = copy.deepcopy(kwargs)
    kwargs_ssi.update({"U": U1_z1, "ray": ray1_z1, })
    U2_z2, G2_z2, ray2_z2, method_and_way2_z2, U_key2_z2 = \
        SFG_NLA_ssi(*args_ssi(z2), **kwargs_ssi, ) if is_NLA == 1 else \
            SFG_SSF_ssi(*args_ssi(z2), **kwargs_ssi, )

    # %%
    # 先倍频 z1 后衍射 z2

    kwargs_ssi = copy.deepcopy(kwargs)
    kwargs_ssi.update({"ray": kwargs.get("ray", "2"), })
    U2_z1, G2_z1, ray2_z1, method_and_way2_z1, U_key2_z1 = \
        SFG_NLA_ssi(*args_ssi(z1), **kwargs_ssi, ) if is_NLA == 1 else \
            SFG_SSF_ssi(*args_ssi(z1), **kwargs_ssi, )

    kwargs_AST = copy.deepcopy(kwargs)
    kwargs_AST.update({"U": U2_z1, "ray": ray2_z1, "polar": kwargs_AST["polar3"], })
    U1_z2, G1_z2, ray1_z2, method_and_way1_z2, U_key1_z2 = \
        AST(*args_AST(z2), **kwargs_AST, )

    # %%
    # 直接倍频 Z = z1 + z2

    Z = z1 + z2

    kwargs_ssi = copy.deepcopy(kwargs)
    kwargs_ssi.update({"ray": kwargs.get("ray", "2"), })
    U2_Z, G2_Z, ray2_Z, method_and_way2_Z, U_key2_Z = \
        SFG_NLA_ssi(*args_ssi(Z), ) if is_NLA == 1 else \
            SFG_SSF_ssi(*args_ssi(Z), **kwargs_ssi, )

    # %%
    # 加和 U1_NLA 与 U2_AST = U2_Z_Superposition

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, U = \
        img_squared_bordered_Read(img_full_name,
                                  U_NonZero_size, dpi,
                                  is_phase_only)

    U2_Z_ADD = U1_z2 + U2_z2
    kwargs.update({"ray": kwargs.get("ray", "2"), })
    init_GLV_rmw("", "a", "ADD", "ssi", **kwargs)  # 主要是用于 之后 compare 里 fkey 获取 U_title 或 U_name 用
    fset("U", U2_Z_ADD)
    fset("G", fft2(U2_Z_ADD))

    folder_address = U_plot_save(fget("U"), fkey("U"), 1,
                                 img_name_extension,
                                 # %%
                                 size_PerPixel,
                                 is_save, is_save_txt, dpi, size_fig,
                                 # %%
                                 cmap_2d, ticks_num, is_contourf,
                                 is_title_on, is_axes_on, is_mm,
                                 fontsize, font,
                                 # %%
                                 is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                                 # %%                          何况 一般默认 is_self_colorbar = 1...
                                 z=Z, )

    # %%
    # 对比 fget("G") 与 G2_Z 的 （绝对）误差

    U_compare(fget("G"), fft2(U2_Z), U_key2_Z.replace("U", "G"), Z,
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
    # 对比 U2_Z_Superposition 与 U2_Z 的 （绝对）误差

    U_compare(fget("U"), U2_Z, U_key2_Z, Z,
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


if __name__ == '__main__':
    kwargs = \
        {"img_full_name": "Grating.png",
         "is_phase_only": 0,
         # %%
         "z_pump": 0,
         "is_LG": 0, "is_Gauss": 0, "is_OAM": 0,
         "l": 0, "p": 0,
         "theta_x": 0, "theta_y": 0,
         # %%
         "is_random_phase": 0,
         "is_H_l": 0, "is_H_theta": 0, "is_H_random_phase": 0,
         # %% 生成横向结构
         "U_name_Structure": '',
         "structure_size_Enlarge": 0.1,
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
         "z1": 1, "z2": 2,
         # %% 不关心
         "z0_structure_frontface_expect": 0, "deff_structure_length_expect": 10,
         "sheets_stored_num": 10, "z0_section_1_expect": 0, "z0_section_2_expect": 0,
         "X": 0, "Y": 0,
         # %% 不关心
         "is_bulk": 1, "is_no_backgroud": 0,
         "is_stored": 0, "is_show_structure_face": 0, "is_energy_evolution_on": 1,
         # %%
         "lam1": 0.8, "is_air_pump": 1, "is_air": 0, "T": 25,
         "lam_structure": 1, "is_air_pump_structure": 1, "T_structure": 25,
         "deff": 30,
         # %%
         "Tx": 10, "Ty": 10, "Tz": "2*lc",
         "mx": 0, "my": 0, "mz": 0,
         "is_stripe": 0, "is_NLAST": 1,  # 不关心 is_stripe
         # %% 生成横向结构
         "Duty_Cycle_x": 0.5, "Duty_Cycle_y": 0.5, "Duty_Cycle_z": 0.5,
         "Depth": 2, "ssi_zoomout_times": 5, "structure_xy_mode": 'x',
         # %%
         "is_continuous": 0, "is_target_far_field": 1, "is_transverse_xy": 0,
         "is_reverse_xy": 0, "is_positive_xy": 1,
         # %%
         "is_save": 0, "is_save_txt": 0, "dpi": 100,
         # %%
         "color_1d": 'b', "cmap_2d": 'viridis',
         # %% 不关心
         "cmap_3d": 'rainbow', "elev": 10, "azim": -65, "alpha": 2,
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
         "is_colorbar_on": 1, "is_energy": 1,
         # %% 不关心
         "plot_group": "UGa", "is_animated": 1,
         "loop": 0, "duration": 0.033, "fps": 5,
         # %% 不关心
         "is_plot_3d_XYz": 0, "is_plot_selective": 0,
         "is_plot_YZ_XZ": 1, "is_plot_3d_XYZ": 0,
         # %%
         "is_print": 1, "is_contours": 66, "n_TzQ": 1,
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
         "polar3": "e",
         }

    kwargs = init_GLV_DICT(**kwargs)
    consistency_SHG_ssi__AST(**kwargs)

    # consistency_SFG_ssi__AST(img_full_name="Grating.png",
    #                          is_phase_only=0,
    #                          # %%
    #                          z_pump=0,
    #                          is_LG=0, is_Gauss=0, is_OAM=0,
    #                          l=0, p=0,
    #                          theta_x=0, theta_y=0,
    #                          # %%
    #                          is_random_phase=0,
    #                          is_H_l=0, is_H_theta=0, is_H_random_phase=0,
    #                          # %% 生成横向结构
    #                          U_name_Structure='',
    #                          structure_size_Enlarge=0.1,
    #                          is_phase_only_Structure=0,
    #                          # %%
    #                          w0_Structure=0, z_pump_Structure=0,
    #                          is_LG_Structure=0, is_Gauss_Structure=1, is_OAM_Structure=0,
    #                          l_Structure=0, p_Structure=0,
    #                          theta_x_Structure=0, theta_y_Structure=0,
    #                          # %%
    #                          is_random_phase_Structure=0,
    #                          is_H_l_Structure=0, is_H_theta_Structure=0, is_H_random_phase_Structure=0,
    #                          # %%
    #                          U_NonZero_size=1, w0=0.3,
    #                          z1=1, z2=2,
    #                          # %% 不关心
    #                          z0_structure_frontface_expect=0, deff_structure_length_expect=10,
    #                          sheets_stored_num=10, z0_section_1_expect=0, z0_section_2_expect=0,
    #                          X=0, Y=0,
    #                          # %% 不关心
    #                          is_bulk=1, is_no_backgroud=0,
    #                          is_stored=0, is_show_structure_face=0, is_energy_evolution_on=1,
    #                          # %%
    #                          lam1=0.8, is_air_pump=0, is_air=0, T=25,
    #                          deff=30,
    #                          # %%
    #                          Tx=10, Ty=10, Tz="2*lc",
    #                          mx=0, my=0, mz=0,
    #                          is_stripe=0, is_NLAST=1,  # 不关心 is_stripe
    #                          # %% 生成横向结构
    #                          Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
    #                          Depth=2, ssi_zoomout_times=5, structure_xy_mode='x',
    #                          # %%
    #                          is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
    #                          is_reverse_xy=0, is_positive_xy=1,
    #                          # %%
    #                          is_save=0, is_save_txt=0, dpi=100,
    #                          # %%
    #                          color_1d='b', cmap_2d='viridis',
    #                          # %% 不关心
    #                          cmap_3d='rainbow', elev=10, azim=-65, alpha=2,
    #                          # %%
    #                          sample=1, ticks_num=6, is_contourf=0,
    #                          is_title_on=1, is_axes_on=1, is_mm=1,
    #                          # %%
    #                          fontsize=9,
    #                          font={'family': 'serif',
    #                                'style': 'normal',  # 'normal', 'italic', 'oblique'
    #                                'weight': 'normal',
    #                                'color': 'black',  # 'black','gray','darkred'
    #                                },
    #                          # %%
    #                          is_colorbar_on=1, is_energy=1,
    #                          # %% 不关心
    #                          plot_group="UGa", is_animated=1,
    #                          loop=0, duration=0.033, fps=5,
    #                          # %% 不关心
    #                          is_plot_3d_XYz=0, is_plot_selective=0,
    #                          is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
    #                          # %%
    #                          is_print=1, is_contours=66, n_TzQ=1,
    #                          Gz_max_Enhance=1, match_mode=1,
    #                          # %% 该程序 独有 -------------------------------
    #                          is_NLA=1, is_amp_relative=1,
    #                          # %% 该程序 作为 主入口时
    #                          root_dir=r'',
    #                          border_percentage=0.1, is_end=-1,
    #                          size_fig_x_scale=10, size_fig_y_scale=1, )

# 注意 colorbar 上的数量级
