# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:57 2021

@author: Xcz
"""

# %%

import copy
import math
import numpy as np
from fun_global_var import init_GLV_DICT, tree_print, init_GLV_rmw, end_STD, fGHU_plot_save
from fun_img_Resize import if_image_Add_black_border
from fun_pump import pump_pic_or_U
from fun_linear import init_AST
from b_1_AST import AST

np.seterr(divide='ignore', invalid='ignore')


def interference_AST__AST(img_full_name="Grating.png",
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
                          z=1, dz_expect=5,
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
    info = "利用 干涉 描边：AST"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    # %%

    if_image_Add_black_border("", img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%

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

    kwargs_AST = copy.deepcopy(kwargs)
    kwargs_AST.update({"ray": "1", })
    U_z, G_z, ray_z, method_and_way_z, U_key_z = \
        AST(*args_AST(z), **kwargs_AST, )

    # %%
    # 获取 size_PerPixel，方便 后续计算 n1, k1，以及 为了生成 U1_0 和 g1_shift

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

    n1_inc, n1, k1_inc, k1, k1_z, k1_xy = init_AST(Ix, Iy, size_PerPixel,
                                                   lam1, is_air, T,
                                                   theta_x, theta_y,
                                                   kwargs.get("polar", "e"), **kwargs)

    # %%

    dz_min = math.pi / (k1 / size_PerPixel)
    is_print and print(tree_print() + "dz_min = {} mm".format(dz_min))

    delay_min_nums = dz_expect // dz_min
    delay_min_nums_odd = delay_min_nums + np.mod(delay_min_nums + 1, 2)
    dz = dz_min * delay_min_nums_odd
    is_print and print(tree_print() + "dz = {} mm".format(dz))

    Z = z + dz

    # %%

    kwargs_AST = copy.deepcopy(kwargs)
    kwargs_AST.update({"ray": "1", })
    U_Z, G_Z, ray_Z, method_and_way_Z, U_key_Z = \
        AST(*args_AST(Z), **kwargs_AST, )

    # %%

    kwargs.update({"ray": "1", })
    init_GLV_rmw("", "a", "ADD", "", **kwargs)

    end_STD(U_z + U_Z, G_z,
            is_energy, n_sigma=3, )

    fGHU_plot_save(0,  # 默认 全自动 is_auto = 1
                   img_name_extension, is_print,
                   # %%
                   [], 1, size_PerPixel,
                   is_save, is_save_txt, dpi, size_fig,
                   # %%
                   "b", cmap_2d,
                   ticks_num, is_contourf,
                   is_title_on, is_axes_on, is_mm,
                   fontsize, font,
                   # %%
                   is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                   # %%                          何况 一般默认 is_self_colorbar = 1...
                   z, is_end=1, )

    # %%


if __name__ == '__main__':
    kwargs = \
        {"img_full_name": "grating.png",
         "is_phase_only": 0,
         # %%
         "z_pump": 0,
         "is_LG": 0, "is_Gauss": 0, "is_OAM": 0,
         "l": 0, "p": 0,
         "theta_x": 0, "theta_y": 0,
         "is_random_phase": 0,
         "is_H_l": 0, "is_H_theta": 0, "is_H_random_phase": 0,
         # %%
         "U1_0_NonZero_size": 1, "w0": 0,  # 传递函数 是 等倾干涉图...
         "z": 0, "dz_expect": 0,  # z 越大，描边能量不变，但会越糊；dz_expect 越大，描边 能量越高，但也越糊
         # %%
         "lam1": 1, "is_air_pump": 1, "is_air": 0, "T": 25,
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
         "is_colorbar_on": 1, "is_energy": 1,
         # %%
         "is_print": 1,
         # %% 该程序 作为 主入口时 -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "border_percentage": 0.1, "is_end": -1,
         # %%
         "theta_z": 90, "phi_z": 0, "phi_c": 24.3,
         # KTP 25 度 ：deff 最高： 90, ~, 24.3，（24.3 - 2002, 24.8 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.7 - 2000）
         # LN 25 度 ：90, ~, ~
         "polar": "e",
         }

    kwargs = init_GLV_DICT(**kwargs)
    interference_AST__AST(**kwargs)

    # interference_AST__AST(img_full_name="grating.png",
    #                       is_phase_only=0,
    #                       # %%
    #                       z_pump=0,
    #                       is_LG=0, is_Gauss=0, is_OAM=0,
    #                       l=0, p=0,
    #                       theta_x=0, theta_y=0,
    #                       is_random_phase=0,
    #                       is_H_l=0, is_H_theta=0, is_H_random_phase=0,
    #                       # %%
    #                       U1_0_NonZero_size=1, w0=0,  # 传递函数 是 等倾干涉图...
    #                       z=0, dz_expect=0,  # z 越大，描边能量不变，但会越糊；dz_expect 越大，描边 能量越高，但也越糊
    #                       # %%
    #                       lam1=1, is_air_pump=0, is_air=0, T=25,
    #                       # %%
    #                       is_save=0, is_save_txt=0, dpi=100,
    #                       # %%
    #                       cmap_2d='viridis',
    #                       # %%
    #                       ticks_num=6, is_contourf=0,
    #                       is_title_on=1, is_axes_on=1, is_mm=1,
    #                       # %%
    #                       fontsize=9,
    #                       font={'family': 'serif',
    #                             'style': 'normal',  # 'normal', 'italic', 'oblique'
    #                             'weight': 'normal',
    #                             'color': 'black',  # 'black','gray','darkred'
    #                             },
    #                       # %%
    #                       is_colorbar_on=1, is_energy=1,
    #                       # %%
    #                       is_print=1,
    #                       # %%
    #                       root_dir=r'',
    #                       border_percentage=0.1, is_end=-1, )
