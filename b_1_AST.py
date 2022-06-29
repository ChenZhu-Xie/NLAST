# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
from fun_img_Resize import if_image_Add_black_border
from fun_global_var import init_GLV_DICT, tree_print, init_GLV_rmw, end_AST, Get, fget, fkey, fGHU_plot_save
from fun_pump import pump_pic_or_U
from fun_linear import init_AST

np.seterr(divide='ignore', invalid='ignore')


# %%

def AST(U_name="",
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
        U_size=1, w0=0.3,
        z0=1,
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
        is_colorbar_on=1, is_energy=0,
        # %%
        is_print=1,
        # %%
        **kwargs, ):
    # %%

    if_image_Add_black_border(U_name, img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%

    info = "AST_线性角谱"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # kwargs['ray'] = init_GLV_rmw(U_name, "~", "", "AST", **kwargs)
    init_GLV_rmw(U_name, "l", "AST", "", **kwargs)

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
                                 U_size, w0,
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

    if "h" in Get("ray"):  # 如果 ray 中含有 倍频 标识符
        lam1 = lam1 / 2
    if "lam3" in kwargs:
        lam1 = kwargs["lam3"]

    n1_inc, n1, k1_inc, k1, k1_z, k1_xy = init_AST(Ix, Iy, size_PerPixel,
                                                   lam1, is_air, T,
                                                   theta_x, theta_y,
                                                   **kwargs)

    # %%

    end_AST(z0, size_PerPixel,
            g_shift, k1_z, )

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
                   z0, is_end=1, )

    return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U")


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",
         "img_full_name": "Grating.png",
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
         # %%
         "U_size": 1, "w0": 0.1,
         "z0": 1,
         # %%
         "lam1": 0.8, "is_air_pump": 1, "is_air": 0, "T": 25,
         # %%
         "is_save": 1, "is_no_data_save": 0,
         "is_save_txt": 0, "dpi": 100,
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
         "is_print": 1,
         # %% 该程序 作为 主入口时 -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "border_percentage": 0.1, "is_end": -1,
         # %%
         "theta_z": 90, "phi_z": 0, "phi_c": 24.3,
         # KTP 50 度 ：deff 最高： 90, ~, 24.3，（24.3 - 2002, 25.3 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.9 - 2000）
         # KTP 25 度 ：deff 最高： 90, ~, 23.7，（23.7 - 2002, 24.8 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.7 - 2000）
         # LN 25 度 ：90, ~, ~
         "polar": "e", "ray": "1",
         }

    kwargs = init_GLV_DICT(**kwargs)
    AST(**kwargs)

    # AST(U_name="",
    #     img_full_name="Grating.png",
    #     is_phase_only=0,
    #     # %%
    #     z_pump=0,
    #     is_LG=0, is_Gauss=0, is_OAM=0,
    #     l=0, p=0,
    #     theta_x=0, theta_y=0,
    #     # %%
    #     is_random_phase=0,
    #     is_H_l=0, is_H_theta=0, is_H_random_phase=0,
    #     # %%
    #     U_size=1, w0=0.1,
    #     z0=1,
    #     # %%
    #     lam1=0.8, is_air_pump=0, is_air=0, T=25,
    #     # %%
    #     is_save=1, is_save_txt=0, dpi=100,
    #     # %%
    #     cmap_2d='viridis',
    #     # %%
    #     ticks_num=6, is_contourf=0,
    #     is_title_on=1, is_axes_on=1, is_mm=1,
    #     # %%
    #     fontsize=9,
    #     font={'family': 'serif',
    #           'style': 'normal',  # 'normal', 'italic', 'oblique'
    #           'weight': 'normal',
    #           'color': 'black',  # 'black','gray','darkred'
    #           },
    #     # %%
    #     is_colorbar_on=1, is_energy=0,
    #     # %%
    #     is_print=1,
    #     # %%
    #     root_dir=r'',
    #     border_percentage=0.1, ray="1", is_end=-1, )
