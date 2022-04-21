# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:57 2021

@author: Xcz
"""

# %%

import numpy as np
from fun_os import img_squared_bordered_Read, U_plot_save
from fun_img_Resize import if_image_Add_black_border
from fun_global_var import tree_print, init_GLV_rmw, fset, fget, fkey
from b_1_AST import AST
from B_3_SHG_NLA_SSI_chi2 import SHG_NLA_SSI
from B_3_SHG_SSF_SSI_chi2 import SHG_SSF_SSI
np.seterr(divide='ignore', invalid='ignore')


def consistency_SHG_SSI__AST(img_full_name = "Grating.png",
                            is_phase_only = 0,
                            #%%
                            z_pump = 0,
                            is_LG = 0, is_Gauss = 0, is_OAM = 0,
                            l = 0, p = 0,
                            theta_x = 0, theta_y = 0,
                            #%%
                            is_random_phase = 0,
                            is_H_l = 0, is_H_theta = 0, is_H_random_phase = 0,
                            #%%
                            U_NonZero_size = 1, w0 = 0.3,
                            z_AST = 1, z_SSI = 5,
                            is_energy_evolution_on = 1,
                            #%%
                            lam1 = 0.8, is_air_pump = 0, is_air = 0, T = 25,
                            deff = 30,
                            #%%
                            Tx = 10, Ty = 10, Tz = "2*lc",
                            mx = 0, my = 0, mz = 0,
                            is_NLAST = 0,
                            #%%
                            is_save = 0, is_save_txt = 0, dpi = 100,
                            #%%
                            color_1d = 'b', cmap_2d = 'viridis',
                            #%%
                            sample = 2, ticks_num = 6, is_contourf = 0,
                            is_title_on = 1, is_axes_on = 1, is_mm = 1,
                            #%%
                            fontsize = 9,
                            font = {'family': 'serif',
                                    'style': 'normal', # 'normal', 'italic', 'oblique'
                                    'weight': 'normal',
                                    'color': 'black', # 'black','gray','darkred'
                                    },
                            #%%
                            is_colorbar_on = 1, is_energy = 1,
                            #%%
                            is_print = 1, is_contours = 1, n_TzQ = 1,
                            Gz_max_Enhance = 1, match_mode = 1,
                            #%%
                            is_NLA = 1,
                            # %%
                            **kwargs, ):
    info = "利用 SHG 描边：SSI"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs["is_end"], kwargs["add_level"] = 0, 0  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    # %%
    # 非线性 惠更斯 菲涅尔 原理

    if_image_Add_black_border("", img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%
    # 先衍射 z_AST 后倍频 z_SSI

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

    def args_SSI(z_SSI):
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
                # 生成横向结构
                '', 0.1, 0,
                0, 0,
                0, 1, 0,
                0, 0,
                0, 0,
                # %%
                0, 0, 0, 0,
                # %%
                U_NonZero_size, w0,
                z_SSI, 0, 10,
                10,
                0, 0, 0, 0,
                # %%
                1, 0,
                0, 0, is_energy_evolution_on,
                # %%
                lam1, is_air_pump, is_air, T,
                deff,
                # %%
                Tx, Ty, Tz,
                mx, my, mz,
                0, is_NLAST,
                # %%
                # 生成横向结构
                0.5, 0.5, 0.5,
                2, 'x',
                0, 1, 0,
                0, 1,
                # %%
                is_save, is_save_txt, dpi,
                # %%
                color_1d, cmap_2d, 'rainbow',
                10, -65, 2,
                # %%
                sample, ticks_num, is_contourf,
                is_title_on, is_axes_on, is_mm,
                # %%
                fontsize, font,
                # %%
                is_colorbar_on, is_energy,
                # %%
                "UGa", 1,
                0, 0.033, 5,
                # %%
                0, 0,
                1, 0,
                # %%
                is_print, is_contours, n_TzQ,
                Gz_max_Enhance, match_mode, ]

    U1_z_AST, G1_z_AST, ray1_z_AST, method_and_way1_z_AST, U_key1_z_AST = \
        AST(*args_AST(z_AST), )

    U1_z_SSI, G1_z_SSI, ray1_z_SSI, method_and_way1_z_SSI, U_key1_z_SSI = \
        SHG_NLA_SSI(*args_SSI(z_SSI), U=U1_z_AST, ray=ray1_z_AST) if is_NLA == 1 else \
            SHG_SSF_SSI(*args_SSI(z_SSI), U=U1_z_AST, ray=ray1_z_AST)

    # %%
    # 先倍频 z_AST 后衍射 z_SSI

    U2_z_SSI, G2_z_SSI, ray2_z_SSI, method_and_way2_z_SSI, U_key2_z_SSI = \
        SHG_NLA_SSI(*args_SSI(z_SSI), ) if is_NLA == 1 else \
            SHG_SSF_SSI(*args_SSI(z_SSI), )

    U2_z_AST, G2_z_AST, ray2_z_AST, method_and_way2_z_AST, U_key2_z_AST = \
        AST(*args_AST(z_AST), U=U2_z_SSI, ray=ray2_z_SSI)

    # %%
    # 直接倍频 Z = z_AST + z_SSI

    Z = z_AST + z_SSI

    # %%
    # 加和 U1_NLA 与 U2_AST = U2_Z_Superposition

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, U = \
        img_squared_bordered_Read(img_full_name,
                                  U_NonZero_size, dpi,
                                  is_phase_only)

    U2_Z_ADD = U1_z_SSI + U2_z_AST
    init_GLV_rmw("", "a", "ADD", "SSI", **kwargs)
    fset("U", U2_Z_ADD)

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
                                 z=Z, is_end=1, )

    # %%

if __name__ == '__main__':
    consistency_SHG_SSI__AST(img_full_name = "Grating.png",
                                is_phase_only = 0,
                                #%%
                                z_pump = 0,
                                is_LG = 0, is_Gauss = 0, is_OAM = 0,
                                l = 0, p = 0,
                                theta_x = 0, theta_y = 0,
                                #%%
                                is_random_phase = 0,
                                is_H_l = 0, is_H_theta = 0, is_H_random_phase = 0,
                                #%%
                                U_NonZero_size = 1, w0 = 0.3,
                                z_AST = 1, z_SSI = 2,
                                is_energy_evolution_on = 1,
                                #%%
                                lam1 = 0.8, is_air_pump = 0, is_air = 0, T = 25,
                                deff = 30,
                                #%%
                                Tx = 10, Ty = 10, Tz = "2*lc",
                                mx = 0, my = 0, mz = 0,
                                is_NLAST = 1,
                                #%%
                                is_save = 0, is_save_txt = 0, dpi = 100,
                                #%%
                                color_1d = 'b', cmap_2d = 'viridis',
                                #%%
                                sample = 2, ticks_num = 6, is_contourf = 0,
                                is_title_on = 1, is_axes_on = 1, is_mm = 1,
                                #%%
                                fontsize = 9,
                                font = {'family': 'serif',
                                        'style': 'normal', # 'normal', 'italic', 'oblique'
                                        'weight': 'normal',
                                        'color': 'black', # 'black','gray','darkred'
                                        },
                                #%%
                                is_colorbar_on = 1, is_energy = 1,
                                #%%
                                is_print = 1, is_contours = 66, n_TzQ = 1,
                                Gz_max_Enhance = 1, match_mode = 1,
                                #%%
                                is_NLA = 1,
                                # %%
                                border_percentage=0.1, is_end=-1, )

# 注意 colorbar 上的数量级
