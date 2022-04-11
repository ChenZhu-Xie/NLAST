# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:57 2021

@author: Xcz
"""

# %%

import math
import numpy as np
from fun_global_var import init_GLV_DICT, end_STD, fGHU_plot_save
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
                          U_0_NonZero_size=1, w0=0.3,
                          z0=1, z0_delay_expect=5,
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

    if_image_Add_black_border("", img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%

    U_z0, G_z0, ray, method_and_way, U_key = \
        AST('',
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
                       U_0_NonZero_size, w0,
                       z0,
                       # %%
                       lam1, is_air_pump, is_air, T,
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
                       is_print, )

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
                                 U_0_NonZero_size, w0,
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

    # %%

    z0_delay_min = math.pi / (k1 / size_PerPixel)
    is_print and print("z0_delay_min = {} mm".format(z0_delay_min))

    delay_min_nums = z0_delay_expect // z0_delay_min
    delay_min_nums_odd = delay_min_nums + np.mod(delay_min_nums + 1, 2)
    z0_delay = z0_delay_min * delay_min_nums_odd
    is_print and print("z0_delay = {} mm".format(z0_delay))

    z1 = z0 + z0_delay

    # %%

    U_z1, G_z1, ray, method_and_way, U_key = AST('',
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
                       U_0_NonZero_size, w0,
                       z1,
                       # %%
                       lam1, is_air_pump, is_air, T,
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
                       is_print, )

    # %%

    init_GLV_DICT("", "a", "ADD", "", **kwargs)

    end_STD(U_z0 + U_z1, G_z0,
            is_energy, n_sigma=3, )

    fGHU_plot_save(0,  # 默认 全自动 is_auto = 1
                   img_name_extension,
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
                   z0, )

    # %%


if __name__ == '__main__':
    interference_AST__AST(img_full_name="grating.png",
                          is_phase_only=0,
                          # %%
                          z_pump=0,
                          is_LG=0, is_Gauss=0, is_OAM=0,
                          l=0, p=0,
                          theta_x=0, theta_y=0,
                          is_random_phase=0,
                          is_H_l=0, is_H_theta=0, is_H_random_phase=0,
                          # %%
                          U1_0_NonZero_size=1, w0=0,  # 传递函数 是 等倾干涉图...
                          z0=0, z0_delay_expect=0,  # z0 越大，描边能量不变，但会越糊；z0_delay_expect 越大，描边 能量越高，但也越糊
                          # %%
                          lam1=1, is_air_pump=0, is_air=0, T=25,
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
                          border_percentage=0.1, )
