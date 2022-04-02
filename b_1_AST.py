# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
import math
from fun_os import GHU_plot_save
from fun_img_Resize import image_Add_black_border
from fun_pump import pump_pic_or_U
from fun_linear import Cal_n, Cal_kz
np.seterr(divide='ignore', invalid='ignore')


# %%

def AST(U1_name="",
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
        U1_0_NonZero_size=1, w0=0.3,
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

    if (type(U1_name) != str) or U1_name == "" and ("U" not in kwargs):
        if __name__ == "__main__":
            border_percentage = kwargs["border_percentage"] if len(kwargs) != 0 else 0.1

            image_Add_black_border(img_full_name,  # 预处理 导入图片 为方形，并加边框
                                   border_percentage,
                                   is_print, )

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, I1_x, I1_y, \
    U1_0, g1_shift = pump_pic_or_U(U1_name,
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
                                   U1_0_NonZero_size, w0,
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

    if U1_name.find("U2") != -1:  # 如果找到了 U2 字样
        lam1 = lam1 / 2

    n1, k1 = Cal_n(size_PerPixel,
                   is_air,
                   lam1, T, p="e")

    k1_z_shift, mesh_k1_x_k1_y_shift = Cal_kz(I1_x, I1_y, k1)

    # %%
    # g1_shift = { g1_shift(k1_x, k1_y) } → 每个元素，乘以，频域 传递函数 e^{i*k1_z*z0} → G1_z0(k1_x, k1_y) = G1_z0

    z1_0 = z0
    i1_z0 = z1_0 / size_PerPixel

    H1_z0_shift = np.power(math.e, k1_z_shift * i1_z0 * 1j)

    # %%

    G1_z0_shift = g1_shift * H1_z0_shift

    # %%
    # G1_z0 = G1_z0(k1_x, k1_y) → IFFT2 → U1(x0, y0, z0) = U1_z0 ，毕竟 标量场 整体，是个 数组，就不写成 U1_x0_y0_z0 了

    G1_z0 = np.fft.ifftshift(G1_z0_shift)
    U1_z0 = np.fft.ifft2(G1_z0)

    GHU_plot_save(U1_name, 0,  # 默认 全自动 is_auto = 1
                  G1_z0_shift, "G1_z0", 'AST',
                  0,
                  H1_z0_shift, "H1_z0",
                  U1_z0, "U1_z0",
                  0,
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

    return U1_z0, G1_z0_shift


if __name__ == '__main__':
    AST(U1_name="",
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
        U1_0_NonZero_size=1, w0=0.1,
        z0=1,
        # %%
        lam1=0.8, is_air_pump=0, is_air=0, T=25,
        # %%
        is_save=1, is_save_txt=0, dpi=100,
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
        border_percentage=0.1, )
