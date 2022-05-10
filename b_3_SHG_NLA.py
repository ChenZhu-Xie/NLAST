# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import math
import numpy as np
from fun_img_Resize import if_image_Add_black_border
from fun_array_Transform import Rotate_180, Roll_xy
from fun_pump import pump_pic_or_U
from fun_linear import init_AST, init_SHG
from fun_nonlinear import args_SHG, Eikz, C_m, Cal_dk_zQ_SHG, Cal_roll_xy, \
    G2_z_modulation_NLAST, G2_z_modulation_3D_NLAST, G2_z_NLAST, G2_z_NLAST_false, Info_find_contours_SHG
from fun_thread import noop, my_thread
from fun_CGH import structure_chi2_Generate_2D
from fun_global_var import init_GLV_DICT, tree_print, init_GLV_rmw, end_SSI, Get, dset, dget, fget, fkey, fGHU_plot_save
np.seterr(divide='ignore', invalid='ignore')
# %%

def SHG_NLA(U_name="",
            img_full_name="Grating.png",
            is_phase_only=0,
            # %%
            z_pump=0,
            is_LG=0, is_Gauss=0, is_OAM=0,
            l=0, p=0,
            theta_x=0, theta_y=0,
            #%%
            is_random_phase=0,
            is_H_l=0, is_H_theta=0, is_H_random_phase=0,
            # %%
            # 生成横向结构
            U_name_Structure = '',
            structure_size_Enlarge = 0.1,
            is_phase_only_Structure = 0,
            # %%
            w0_Structure = 0, z_pump_Structure = 0,
            is_LG_Structure = 0, is_Gauss_Structure = 0, is_OAM_Structure = 0,
            l_Structure = 0, p_Structure = 0,
            theta_x_Structure = 0, theta_y_Structure = 0,
            # %%
            is_random_phase_Structure = 0,
            is_H_l_Structure = 0, is_H_theta_Structure = 0, is_H_random_phase_Structure = 0,
            # %%
            U_NonZero_size=1, w0=0.3,
            z0=1,
            # %%
            lam1=0.8, is_air_pump=0, is_air=0, T=25,
            deff=30, is_fft = 1, fft_mode = 0,
            is_sum_Gm=0, mG=0,
            is_linear_convolution = 0,
            #%%
            Tx=10, Ty=10, Tz="2*lc",
            mx=0, my=0, mz=0,
            # %%
            # 生成横向结构
            Duty_Cycle_x = 0.5, Duty_Cycle_y = 0.5, Duty_Cycle_z = 0.5,
            Depth = 2, structure_xy_mode = 'x',
            #%%
            is_continuous = 0, is_target_far_field = 1, is_transverse_xy = 0,
            is_reverse_xy = 0, is_positive_xy = 1, is_no_backgroud = 0,
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
            # %%
            **kwargs, ):

    # %%

    if_image_Add_black_border(U_name, img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    #%%

    info = "NLAST_一拳超人"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None); kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # kwargs['ray'] = init_GLV_rmw(U_name, "^", "", "NLA", **kwargs)
    init_GLV_rmw(U_name, "h", "NLA", "", **kwargs)

    #%%

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
                                   **kwargs, )
    # %%

    n1, k1, k1_z, k1_xy = init_AST(Ix, Iy, size_PerPixel,
                                   lam1, is_air, T, )

    lam2, n2, k2, k2_z, k2_xy = init_SHG(Ix, Iy, size_PerPixel,
                                         lam1, is_air, T, )

    # %%
    # 提供描边信息，并覆盖值

    z0, Tz, deff_structure_length_expect = Info_find_contours_SHG(g_shift, k1_z, k2_z, Tz, mz,
                                                                  z0, size_PerPixel, z0,
                                                                  is_print, is_contours, n_TzQ, Gz_max_Enhance,
                                                                  match_mode, )

    # %%
    # 引入 倒格矢，对 k2 的 方向 进行调整，其实就是对 k2 的 k2x, k2y, k2z 网格的 中心频率 从 (0, 0, k2z) 移到 (Gx, Gy, k2z + Gz)

    dk, lc, Tz, \
    Gx, Gy, Gz = args_SHG(k1, k2, size_PerPixel,
                          mx, my, mz,
                          Tx, Ty, Tz,
                          is_print, )

    # %%

    iz = z0 / size_PerPixel

    if is_fft == 0:

        const = (k2 / size_PerPixel / n2) ** 2 * C_m(mx) * C_m(my) * C_m(mz) * deff * 1e-12  # pm / V 转换成 m / V
        integrate_z0 = np.zeros((Ix, Iy), dtype=np.complex128())

        g_rotate_180 = Rotate_180(g_shift)

        def fun1(for_th, fors_num, *args, **kwargs, ):
            for n2_y in range(Iy):
                dk_zQ = Cal_dk_zQ_SHG(k1,
                                    k1_z, k2_z,
                                    k1_xy, k2_xy,
                                    for_th, n2_y,
                                    Gx, Gy, Gz, )

                roll_x, roll_y = Cal_roll_xy(Gx, Gy,
                                             Ix, Iy,
                                             for_th, n2_y, )

                g_shift_dk_x_dk_y = Roll_xy(g_rotate_180,
                                             roll_x, roll_y,
                                             is_linear_convolution, )

                integrate_z0[for_th, n2_y] = np.sum(
                    g_shift * g_shift_dk_x_dk_y * Eikz(dk_zQ * iz) * iz * size_PerPixel \
                    * (2 / (dk_zQ / k2_z[for_th, n2_y] + 2)))

        my_thread(10, Ix,
                  fun1, noop, noop,
                  is_ordered=1, is_print=is_print, )
        
        g2_z = const * integrate_z0 / k2_z * size_PerPixel

        dset("G", g2_z * np.power(math.e, k2_z * iz * 1j))

    else:

        Const = (k2 / size_PerPixel / n2) ** 2 * deff * 1e-12  # pm / V 转换成 m / V
        dset("G", np.zeros((Ix, Iy), dtype=np.complex128()))

        if fft_mode == 0:
            # %% generate structure

            n1, k1, k1_z, lam2, n2, k2, k2_z, \
            dk, lc, Tz, Gx, Gy, Gz, \
            size_PerPixel, U_0_structure, g_shift_structure, \
            structure, structure_opposite, modulation, modulation_opposite, modulation_squared, modulation_opposite_squared \
                = structure_chi2_Generate_2D(U_name_Structure,
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
                                             U_NonZero_size, w0_Structure,
                                             structure_size_Enlarge,
                                             Duty_Cycle_x, Duty_Cycle_y,
                                             structure_xy_mode, Depth,
                                             # %%
                                             is_continuous, is_target_far_field,
                                             is_transverse_xy, is_reverse_xy,
                                             is_positive_xy,
                                             0, is_no_backgroud,
                                             # %%
                                             lam1, is_air_pump, is_air, T,
                                             Tx, Ty, Tz,
                                             mx, my, mz,
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
                                             is_print,
                                             # %%
                                             **kwargs, )

            if is_sum_Gm == 0:
                addition_dict = {"Tz": Tz if kwargs.get("is_NLAST_sum", 0) else None} # 若 is_NLAST_sum 有且非 0，则 Tz
                dset("G", G2_z_modulation_NLAST(k1, k2,
                                                modulation_squared, U_0, iz, Const,
                                                Gz=Gz, **addition_dict, ))
            elif is_sum_Gm == 1:
                def fun1(for_th, fors_num, *args, **kwargs, ):
                    m_z = for_th - mG
                    Gz_m = 2 * math.pi * m_z * size_PerPixel / (Tz / 1000)
                    # print(m_z, C_m(m_z), "\n")

                    # 注意这个系数 C_m(m_z) 只对应 Duty_Cycle_z = 50% 占空比...
                    Const = (k2 / size_PerPixel / n2) ** 2 * C_m(mx) * C_m(my) * C_m(m_z) * deff * 1e-12
                    G2_z0_Gm = G2_z_modulation_NLAST(k1, k2,
                                                     modulation_squared, U_0, iz, Const,
                                                     Gz=Gz_m, ) if m_z != 0 else 0
                    return G2_z0_Gm

                def fun2(for_th, fors_num, G2_z0_Gm, *args, **kwargs, ):

                    dset("G", dget("G") + G2_z0_Gm )

                    return dget("G")

                my_thread(10, 2 * mG + 1,
                          fun1, fun2, noop,
                          is_ordered=1, is_print=is_print, )
            else:
                dset("G", G2_z_modulation_3D_NLAST(k1, k2,
                                                   modulation_squared, U_0, iz, Const,
                                                   Tz=Tz, ))

        elif fft_mode == 1:
            if is_sum_Gm == 0:
                dset("G", G2_z_NLAST(k1, k2, Gx, Gy, Gz,
                                     U_0, iz, Const,
                                     is_linear_convolution, ))
            else:
                def fun1(for_th, fors_num, *args, **kwargs, ):
                    m_x = for_th - mG
                    Gx_m = 2 * math.pi * m_x * size_PerPixel / (Tx / 1000)
                    # print(m_x, C_m(m_x), "\n")

                    # 注意这个系数 C_m(m_x) 只对应 Duty_Cycle_x = 50% 占空比...
                    Const = (k2 / size_PerPixel / n2) ** 2 * C_m(m_x) * C_m(my) * C_m(mz) * deff * 1e-12
                    G2_z0_Gm = G2_z_NLAST(k1, k2, Gx_m, Gy, Gz,
                                          U_0, iz, Const,
                                          is_linear_convolution, ) if m_x != 0 else 0
                    return G2_z0_Gm

                def fun2(for_th, fors_num, G2_z0_Gm, *args, **kwargs, ):

                    dset("G", dget("G") + G2_z0_Gm)

                    return dget("G")

                my_thread(10, 2 * mG + 1,
                          fun1, fun2, noop,
                          is_ordered=1, is_print=is_print, )

        elif fft_mode == 2:

            dset("G", G2_z_NLAST_false(k1, k2, Gx, Gy, Gz,
                                           U_0, iz, Const,
                                           is_linear_convolution, ))

    # %%

    end_SSI(g_shift, is_energy, n_sigma=3, )

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
        {"U_name": "", # 要么从 U_name 里传 ray 和 U 进来，要么 单独传个 U 和 ray
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
        # %%
        # 生成横向结构
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
        "U_NonZero_size": 0.9, "w0": 0.1,
        "z0": 10,
        # %%
        "lam1": 1.064, "is_air_pump": 0, "is_air": 0, "T": 25,
        "deff": 30, "is_fft": 1, "fft_mode": 0,
        "is_sum_Gm": 0, "mG": 0, 'is_NLAST_sum': 1, 
        "is_linear_convolution": 0,
        # %%
        "Tx": 10, "Ty": 10, "Tz": 3,
        "mx": 1, "my": 0, "mz": 0,
        # %%
        # 生成横向结构
        "Duty_Cycle_x": 0.5, "Duty_Cycle_y": 0.5, "Duty_Cycle_z": 0.5,
        "Depth": 2, "structure_xy_mode": 'x',
        # %%
        "is_continuous": 0, "is_target_far_field": 1, "is_transverse_xy": 0,
        "is_reverse_xy": 0, "is_positive_xy": 1, "is_no_backgroud": 0,
        # %%
        "is_save": 1, "is_save_txt": 0, "dpi": 100,
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
        "is_colorbar_on": 1, "is_energy": 0,
        # %%
        "is_print": 1, "is_contours": 0, "n_TzQ": 1,
        "Gz_max_Enhance": 1, "match_mode": 1,
        # %%
        "kwargs_seq": 0, "root_dir": r'1',
        "border_percentage": 0.1, "is_end": -1,
        "ray": "2", }
        
        # 要不要把 “要不要（是否）使用 上一次使用 的 参数”
        # or “这次 使用了 哪一次 使用的 参数” 传进去记录呢？—— 也不是不行。

    # kwargs.update(init_GLV_DICT(**kwargs))
    kwargs = init_GLV_DICT(**kwargs)
    SHG_NLA(**kwargs)

    # SHG_NLA(U_name="", # 要么从 U_name 里传 ray 和 U 进来，要么 单独传个 U 和 ray
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
