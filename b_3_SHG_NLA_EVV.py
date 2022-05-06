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
from fun_nonlinear import args_SHG, Eikz, C_m, Cal_dk_zQ_SHG, Cal_roll_xy, G2_z_modulation_NLAST, \
    G2_z_NLAST, G2_z_NLAST_false, Info_find_contours_SHG, G2_z_modulation_3D_NLAST
from fun_thread import noop, my_thread
from fun_CGH import structure_chi2_Generate_2D
from fun_global_var import init_GLV_DICT, tree_print, Set, Get, init_GLV_rmw, init_EVV, Fun3, end_SSI, \
    dset, dget, fget, fkey, fGHU_plot_save, fU_EVV_plot
np.seterr(divide='ignore', invalid='ignore')


# %%

def SHG_NLA_EVV(U_name="",
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
                z0=1, sheets_stored_num=10,
                # %%
                lam1=0.8, is_air_pump=0, is_air=0, T=25,
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
                is_colorbar_on=1, is_energy=0, is_plot_3d_XYz = 0,
                # %%
                plot_group = "UGa", is_animated = 1,
                loop = 0, duration = 0.033, fps = 5,
                # %%
                is_print=1, is_contours=1, n_TzQ=1,
                Gz_max_Enhance=1, match_mode=1,
                # %%
                **kwargs, ):
    # %%

    if_image_Add_black_border(U_name, img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    #%%

    info = "NLAST_演化版_EVV"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None); kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # kwargs['ray'] = init_GLV_rmw(U_name, "^", "EVV", "NLA", **kwargs)
    init_GLV_rmw(U_name, "h", "NLA", "EVV", **kwargs)

    # %%

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

    # %%

    iz = z0 / size_PerPixel
    # zj = kwargs.get("zj", np.linspace(0, z0, sheets_stored_num + 1)) \
    #     if is_stored==1 else np.linspace(0, z0, sheets_stored_num + 1)
    zj = kwargs.get("zj", np.linspace(0, z0, sheets_stored_num + 1))
    izj = zj / size_PerPixel
    Set("zj", zj)
    Set("izj", izj)

    sheets_stored_num = len(zj) - 1
    init_EVV(g_shift, U_0,
             is_energy_evolution_on, is_stored,
             sheets_stored_num, sheets_stored_num,
             iz, size_PerPixel, )
    
    def Fun1(for_th2, fors_num2, *args, **kwargs, ):

        Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
            np.zeros((Ix, Iy), dtype=np.complex128()))
        # 加上 Fun3 中的 sset 其实共 储存了 2 次 这些 G2_zm 层（m = for_th2），有点浪费
        # 但这里的又是必须存在的，因为之后得用它来在 fun2 里累加。。
        # 所以最多也是之后的 Fun3 中的 sset 可以不必存在

        # for_th2 == 0 时也要算，因为 zj[0] 不一定是 0：外部可能传入 zj
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
                        g_shift * g_shift_dk_x_dk_y * Eikz(dk_zQ * izj[for_th2]) * izj[for_th2] * size_PerPixel \
                        * (2 / (dk_zQ / k2_z[for_th, n2_y] + 2)))

            my_thread(10, Ix,
                      fun1, noop, noop,
                      is_ordered=1, is_print=is_print, )

            g2_z = const * integrate_z0 / k2_z * size_PerPixel

            Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                g2_z * np.power(math.e, k2_z * izj[for_th2] * 1j))

        else:

            Const = (k2 / size_PerPixel / n2) ** 2 * deff * 1e-12  # pm / V 转换成 m / V

            if fft_mode == 0:

                if is_sum_Gm == 0:
                    Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                        G2_z_modulation_NLAST(k1, k2, Gz,
                                              modulation_squared, U_0, izj[for_th2], Const, ))
                elif is_sum_Gm == 1:
                    def fun1(for_th, fors_num, *args, **kwargs, ):
                        m_z = for_th - mG
                        Gz_m = 2 * math.pi * m_z * size_PerPixel / (Tz / 1000)
                        # print(m_z, C_m(m_z), "\n")

                        # 注意这个系数 C_m(m_z) 只对应 Duty_Cycle_z = 50% 占空比...
                        Const = (k2 / size_PerPixel / n2) ** 2 * C_m(mx) * C_m(my) * C_m(m_z) * deff * 1e-12
                        G2_z0_Gm = G2_z_modulation_NLAST(k1, k2, Gz_m,
                                                         modulation_squared, U_0, izj[for_th2],
                                                         Const, ) if m_z != 0 else 0
                        return G2_z0_Gm

                    def fun2(for_th, fors_num, G2_z0_Gm, *args, **kwargs, ):
                        # print("forth = {}".format(for_th))

                        Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                            Get("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way")) + G2_z0_Gm)

                        return Get("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"))

                    my_thread(10, 2 * mG + 1,
                              fun1, fun2, noop,
                              is_ordered=1, is_print=is_print, )
                else:
                    Tz_unit = (Tz / 1000) / size_PerPixel

                    Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                        G2_z_modulation_3D_NLAST(k1, k2, Tz_unit,
                                                 modulation_squared, U_0,
                                                 izj[for_th2], Const, ))

            elif fft_mode == 1:

                if is_sum_Gm == 0:
                    Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                        G2_z_NLAST(k1, k2, Gx, Gy, Gz,
                                   U_0, izj[for_th2], Const,
                                   is_linear_convolution, ))
                else:
                    def fun1(for_th, fors_num, *args, **kwargs, ):
                        m_x = for_th - mG
                        Gx_m = 2 * math.pi * m_x * size_PerPixel / (Tx / 1000)
                        # print(m_x, C_m(m_x), "\n")

                        # 注意这个系数 C_m(m_x) 只对应 Duty_Cycle_x = 50% 占空比...
                        Const = (k2 / size_PerPixel / n2) ** 2 * C_m(m_x) * C_m(my) * C_m(mz) * deff * 1e-12
                        G2_z0_Gm = G2_z_NLAST(k1, k2, Gx_m, Gy, Gz,
                                              U_0, izj[for_th2], Const,
                                              is_linear_convolution, ) if m_x != 0 else 0
                        return G2_z0_Gm

                    def fun2(for_th, fors_num, G2_z0_Gm, *args, **kwargs, ):

                        Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                            Get("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way")) + G2_z0_Gm)

                        return Get("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"))

                    my_thread(10, 2 * mG + 1,
                              fun1, fun2, noop,
                              is_ordered=1, is_print=is_print, )

            elif fft_mode == 2:

                Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                    G2_z_NLAST_false(k1, k2, Gx, Gy, Gz,
                                     U_0, izj[for_th2], Const,
                                     is_linear_convolution, ))

        return Get("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"))

    def Fun2(for_th2, fors_num, G2_zm, *args, **kwargs, ):

        dset("G", G2_zm)  # 主要是为了 end_SSI 中的 dget("G") 而存在的，否则 直接 返回 G2_zm 了

        return dget("G")

    my_thread(10, sheets_stored_num + 1,
              Fun1, Fun2, Fun3,
              is_ordered=1, is_print=is_print, )

    # %%

    end_SSI(g_shift, is_energy, n_sigma=3, )

    fGHU_plot_save(is_energy_evolution_on,  # 默认 全自动 is_auto = 1
                   img_name_extension, is_print,
                   # %%
                   zj, sample, size_PerPixel,
                   is_save, is_save_txt, dpi, size_fig,
                   # %%
                   color_1d, cmap_2d,
                   ticks_num, is_contourf,
                   is_title_on, is_axes_on, is_mm,
                   fontsize, font,
                   # %%
                   is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                   # %%                          何况 一般默认 is_self_colorbar = 1...
                   z0, is_end=1, )

    # %%

    fU_EVV_plot(img_name_extension,
                # %%
                sample, size_PerPixel,
                is_save, dpi, size_fig,
                elev, azim, alpha,
                # %%
                cmap_2d, cmap_3d,
                ticks_num, is_contourf,
                is_title_on, is_axes_on, is_mm,
                fontsize, font,
                # %%
                is_colorbar_on, is_energy,
                # %%
                plot_group, is_animated,
                loop, duration, fps,
                # %%
                is_plot_3d_XYz,
                # %%
                z0, )

    return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U")


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",
        "img_full_name": "lena1.png",
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
        "U_NonZero_size": 0.9, "w0": 0.3,
        "z0": 10, "sheets_stored_num": 10,
        # %%
        "lam1": 1.064, "is_air_pump": 0, "is_air": 0, "T": 25,
        "deff": 30, "is_fft": 1, "fft_mode": 0,
        "is_sum_Gm": 0, "mG": 0,
        "is_linear_convolution": 0,
        # %%
        "Tx": 10, "Ty": 10, "Tz": 0,
        "mx": 1, "my": 0, "mz": 0,
        # %%
        # 生成横向结构
        "Duty_Cycle_x": 0.5, "Duty_Cycle_y": 0.5, "Duty_Cycle_z": 0.5,
        "Depth": 2, "structure_xy_mode": 'x',
        # %%
        "is_continuous": 0, "is_target_far_field": 1, "is_transverse_xy": 0,
        "is_reverse_xy": 0, "is_positive_xy": 1, "is_no_backgroud": 0,
        "is_stored": 0, "is_energy_evolution_on": 1,
        # %%
        "is_save": 0, "is_save_txt": 0, "dpi": 100,
        # %%
        "color_1d": 'b', "cmap_2d": 'viridis', "cmap_3d": 'rainbow',
        "elev": 10, "azim": -65, "alpha": 2,
        # %%
        "sample": 2, "ticks_num": 6, "is_contourf": 0,
        "is_title_on": 1, "is_axes_on": 1, "is_mm": 1,
        # %%
        "fontsize": 9,
        "font": {'family': 'serif',
              'style': 'normal',  # 'normal', 'italic', 'oblique'
              'weight': 'normal',
              'color': 'black',  # 'black','gray','darkred'
              },
        # %%
        "is_colorbar_on": 1, "is_energy": 0, "is_plot_3d_XYz": 0,
        # %%
        "plot_group": "UGa", "is_animated": 1,
        "loop": 0, "duration": 0.033, "fps": 5,
        # %%
        "is_print": 1, "is_contours": 66, "n_TzQ": 1,
        "Gz_max_Enhance": 1, "match_mode": 1,
        # %%
        "kwargs_seq": 0, "root_dir": r'',
        "border_percentage": 0.1, "is_end": -1,
        "size_fig_x_scale": 10, "size_fig_y_scale": 1,
        "ray": "2", }

    kwargs = init_GLV_DICT(**kwargs)
    SHG_NLA_EVV(**kwargs)

    # SHG_NLA_EVV(U_name="",
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
    #             sample=2, ticks_num=6, is_contourf=0,
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
