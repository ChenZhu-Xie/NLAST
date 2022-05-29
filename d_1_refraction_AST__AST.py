# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:57 2021

@author: Xcz
"""

# %%

import copy
from fun_global_var import init_GLV_DICT, tree_print
from fun_img_Resize import if_image_Add_black_border
from b_1_AST import AST


def refraction_AST__AST(img_full_name="Grating.png",
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
                        U_NonZero_size=1, w0=0.3,
                        z1=1, zn=5,
                        # %%
                        lam1=0.8, is_air_pump=0, is_air=0, T=25,
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
                        is_colorbar_on=1, is_energy=1,
                        # %%
                        is_print=1,
                        # %%
                        **kwargs, ):
    # %%
    info = "利用 折射 检验：AST"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    # %%
    # 线性 惠更斯 菲涅尔 原理

    if_image_Add_black_border("", img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    def args_AST(z_AST, is_air):
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

    # %%
    # 先以 n 衍射 zn 后 以 1 衍射 z1

    # U_zn, G_zn, ray_zn, method_and_way_zn, U_key_zn = \
    #     AST(*args_AST(zn, is_air), )

    # U_z1, G_z1, ray_z1, method_and_way_z1, U_key_z1 = \
    #     AST(*args_AST(z1, 1), U=U_zn, ray=ray_zn, is_end=1, )

    # %%
    # 先以 1 衍射 z1 后 以 n 衍射 zn

    kwargs_AST = copy.deepcopy(kwargs)
    kwargs_AST.update({"ray": "1", })
    U_z1, G_z1, ray_z1, method_and_way_z1, U_key_z1 = \
        AST(*args_AST(z1, 1), **kwargs_AST, )

    kwargs_AST = copy.deepcopy(kwargs)
    kwargs_AST.update({"U": U_z1, "ray": ray_z1, "is_end": 1, })
    U_zn, G_zn, ray_zn, method_and_way_zn, U_key_zn = \
        AST(*args_AST(zn, is_air), **kwargs_AST, )

    # %%


if __name__ == '__main__':
    kwargs = \
        {"img_full_name": "Grating.png",
         "is_phase_only": 0,
         # %%
         "z_pump": 0,
         "is_LG": 1, "is_Gauss": 1, "is_OAM": 1,
         "l": 1, "p": 1,
         "theta_x": 1, "theta_y": 0,
         # %%
         "is_random_phase": 0,
         "is_H_l": 0, "is_H_theta": 0, "is_H_random_phase": 0,
         # %%
         "U_NonZero_size": 1, "w0": 0.1,
         "z1": 5, "zn": 5,
         # %%
         "lam1": 1.5, "is_air_pump": 1, "is_air": 0, "T": 25,
         # %%
         "is_save": 0, "is_save_txt": 0, "dpi": 100,
         # %%
         "cmap_2d": 'viridis',
         # %%
         "ticks_num": 6, "is_contourf": 0,
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
         "is_print": 1,
         # %% 该程序 作为 主入口时 -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "border_percentage": 0.1, "is_end": -1,
         # %%
         "theta_z": 90, "phi_z": 0, "phi_c": 24.3,  # KTP deff 最高： 90, ~, 24.3 ———— 1994 ：68.8, ~, 90 ———— LN ：90, ~, ~
         "polar": "e",
         }

    kwargs = init_GLV_DICT(**kwargs)
    refraction_AST__AST(**kwargs)

    # refraction_AST__AST(img_full_name = "Grating.png",
    #                     is_phase_only = 0,
    #                     #%%
    #                     z_pump = 0,
    #                     is_LG = 1, is_Gauss = 1, is_OAM = 1,
    #                     l = 1, p = 1,
    #                     theta_x = 1, theta_y = 0,
    #                     #%%
    #                     is_random_phase = 0,
    #                     is_H_l = 0, is_H_theta = 0, is_H_random_phase = 0,
    #                     #%%
    #                     U_NonZero_size = 1, w0 = 0.1,
    #                     z1 = 5, zn = 5,
    #                     #%%
    #                     lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25,
    #                     #%%
    #                     is_save = 0, is_save_txt = 0, dpi = 100,
    #                     #%%
    #                     cmap_2d = 'viridis',
    #                     #%%
    #                     ticks_num = 6, is_contourf = 0,
    #                     is_title_on = 1, is_axes_on = 1, is_mm = 1,
    #                     #%%
    #                     fontsize = 9,
    #                     font = {'family': 'serif',
    #                             'style': 'normal', # 'normal', 'italic', 'oblique'
    #                             'weight': 'normal',
    #                             'color': 'black', # 'black','gray','darkred'
    #                             },
    #                     #%%
    #                     is_colorbar_on = 1, is_energy = 1,
    #                     #%%
    #                     is_print = 1,
    #                     # %%
    #                     root_dir=r'',
    #                     border_percentage=0.1, is_end=-1, )
