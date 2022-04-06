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
from fun_nonlinear import args_SHG, Eikz, C_m, Cal_dk_zQ_SHG, Cal_roll_xy, G2_z_modulation_NLAST, G2_z_NLAST, G2_z_NLAST_false, Info_find_contours_SHG
from fun_thread import noop, my_thread
from fun_CGH import structure_chi2_Generate_2D
from fun_global_var import Set, Get, init_GLV_DICT, init_SSI, end_SSI, dset, dget, fget, fGHU_plot_save
np.seterr(divide='ignore', invalid='ignore')
# %%

def NLA(U1_name="",
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
        U1_name_Structure = '',
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
        U1_0_NonZero_size=1, w0=0.3,
        z0=1, sheets_stored_num=10,
        # %%
        lam1=0.8, is_air_pump=0, is_air=0, T=25,
        deff=30, is_fft = 1, fft_mode = 0, 
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
        is_stored = 1, is_energy_evolution_on = 1,
        # %%
        is_save=0, is_save_txt=0, dpi=100,
        # %%
        cmap_2d='viridis',
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
        is_colorbar_on=1, is_energy=0,
        # %%
        is_print=1, is_contours=1, n_TzQ=1, 
        Gz_max_Enhance=1, match_mode=1,
        # %%
        **kwargs, ):

    # %%

    if_image_Add_black_border(U1_name, img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    ray = init_GLV_DICT(U1_name, "2", "EVV", "NLA", **kwargs)

    #%%

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, \
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
                                   ray=ray, **kwargs, )
    # %%

    n1, k1, k1_z, k1_xy = init_AST(Ix, Iy, size_PerPixel,
                                   lam1, is_air, T, )

    lam2, n2, k2, k2_z, k2_xy = init_SHG(Ix, Iy, size_PerPixel,
                                         lam1, is_air, T, )

    # %%
    # 提供描边信息，并覆盖值

    z0, Tz, deff_structure_length_expect = Info_find_contours_SHG(g1_shift, k1_z, k2_z, Tz, mz,
                                                                  z0, size_PerPixel, z0,
                                                                  is_print, is_contours, n_TzQ, Gz_max_Enhance,
                                                                  match_mode, )

    # %%
    # 引入 倒格矢，对 k2 的 方向 进行调整，其实就是对 k2 的 k2x, k2y, k2z 网格的 中心频率 从 (0, 0, k2z) 移到 (Gx, Gy, k2z + Gz)

    dk, lc, Tz, \
    Gx, Gy, Gz = args_SHG(k1, k2, size_PerPixel,
                          mx, my, mz,
                          Tx, Ty, Tz,
                          is_print=0, )

    if fft_mode == 0:
        # %% generate structure

        n1, k1, k1_z, lam2, n2, k2, k2_z, \
        dk, lc, Tz, Gx, Gy, Gz, \
        size_PerPixel, U1_0_structure, g1_shift_structure, \
        structure, structure_opposite, modulation, modulation_opposite, modulation_squared, modulation_opposite_squared \
            = structure_chi2_Generate_2D(U1_name_Structure,
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
                                         U1_0_NonZero_size, w0_Structure,
                                         structure_size_Enlarge,
                                         Duty_Cycle_x, Duty_Cycle_y,
                                         structure_xy_mode, Depth,
                                         # %%
                                         is_continuous, is_target_far_field,
                                         is_transverse_xy, is_reverse_xy,
                                         is_positive_xy, is_no_backgroud,
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
                                         is_print, )

    # %%
    # const

    const = (k2 / size_PerPixel / n2) ** 2 * C_m(mx) * C_m(my) * C_m(mz) * deff * 1e-12  # pm / V 转换成 m / V

    # %%

    iz = z0 / size_PerPixel
    zj = np.linspace(0, z0, sheets_stored_num + 1)
    izj = zj / size_PerPixel

    init_SSI(g1_shift, U1_0,
             is_energy_evolution_on, is_stored,
             sheets_stored_num, sheets_stored_num,
             0, 0, iz, size_PerPixel, )

    def Fun1(for_th2, fors_num2, *arg, **kwargs, ):

        Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
            np.zeros((Ix, Iy), dtype=np.complex128()))

        if for_th2 != 0:
            if is_fft == 0:

                integrate_z0 = np.zeros((Ix, Iy), dtype=np.complex128())

                g1_rotate_180 = Rotate_180(g1_shift)

                def fun1(for_th, fors_num, *arg, **kwargs, ):
                    for n2_y in range(Iy):
                        dk_zQ = Cal_dk_zQ_SHG(k1,
                                            k1_z, k2_z,
                                            k1_xy, k2_xy,
                                            for_th, n2_y,
                                            Gx, Gy, Gz, )

                        roll_x, roll_y = Cal_roll_xy(Gx, Gy,
                                                     Ix, Iy,
                                                     for_th, n2_y, )

                        g1_shift_dk_x_dk_y = Roll_xy(g1_rotate_180,
                                                     roll_x, roll_y,
                                                     is_linear_convolution, )

                        integrate_z0[for_th, n2_y] = np.sum(
                            g1_shift * g1_shift_dk_x_dk_y * Eikz(dk_zQ * izj[for_th2]) * izj[for_th2] * size_PerPixel \
                            * (2 / (dk_zQ / k2_z[for_th, n2_y] + 2)))

                my_thread(10, Ix,
                          fun1, noop, noop,
                          is_ordered=1, is_print=is_print, )

                g2_z = const * integrate_z0 / k2_z * size_PerPixel

                Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                    g2_z * np.power(math.e, k2_z * izj[for_th2] * 1j))

            else:

                if fft_mode == 0:

                    Const = (k2 / size_PerPixel / n2) ** 2 * deff * 1e-12  # pm / V 转换成 m / V
                    if is_sum_Gm == 0:
                        Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                            G2_z_modulation_NLAST(k1, k2, Gz,
                                                    modulation_squared, U1_0, izj[for_th2], const, ))
                    elif is_sum_Gm == 1:
                        def fun1(for_th, fors_num, *arg, **kwargs, ):
                            m_z = for_th - mG
                            Gz_m = 2 * math.pi * m_z * size_PerPixel / (Tz / 1000)
                            # print(m_z, C_m(m_z), "\n")

                            # 注意这个系数 C_m(m_z) 只对应 Duty_Cycle_z = 50% 占空比...
                            Const = (k2 / size_PerPixel / n2) ** 2 * C_m(mx) * C_m(my) * C_m(m_z) * deff * 1e-12
                            G2_z0_Gm = G2_z_modulation_NLAST(k1, k2, Gz_m,
                                                                   modulation_squared, U1_0, izj[for_th2],
                                                                   Const, ) if m_z != 0 else 0
                            return G2_z0_Gm

                        def fun2(for_th, fors_num, G2_z0_Gm, *arg, **kwargs, ):
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
                                                     modulation_squared, U1_0,
                                                     izj[for_th2], Const, ))

                elif fft_mode == 1:

                    if is_sum_Gm == 0:
                        Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                            G2_z_NLAST(k1, k2, Gx, Gy, Gz,
                                       U1_0, izj[for_th2], const,
                                       is_linear_convolution, ))
                    else:
                        def fun1(for_th, fors_num, *arg, **kwargs, ):
                            m_x = for_th - mG
                            Gx_m = 2 * math.pi * m_x * size_PerPixel / (Tx / 1000)
                            # print(m_x, C_m(m_x), "\n")

                            # 注意这个系数 C_m(m_x) 只对应 Duty_Cycle_x = 50% 占空比...
                            Const = (k2 / size_PerPixel / n2) ** 2 * C_m(m_x) * C_m(my) * C_m(mz) * deff * 1e-12
                            G2_z0_Gm = G2_z_NLAST(k1, k2, Gx_m, Gy, Gz,
                                                        U1_0, izj[for_th2], Const,
                                                        is_linear_convolution, ) if m_x != 0 else 0
                            return G2_z0_Gm

                        def fun2(for_th, fors_num, G2_z0_Gm, *arg, **kwargs, ):

                            Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                                Get("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way")) + G2_z0_Gm)

                            return Get("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"))

                        my_thread(10, 2 * mG + 1,
                                  fun1, fun2, noop,
                                  is_ordered=1, is_print=is_print, )

                elif fft_mode == 2:

                    Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                        G2_z_NLAST_false(k1, k2, Gx, Gy, Gz,
                                         U1_0, izj[for_th2], const,
                                         is_linear_convolution, ))

        return Get("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"))

    def Fun2(for_th2, fors_num, G2_zm, *args, **kwargs, ):

        dset("G", G2_zm)

        return dget("G") # 好像不需要 return ?

    my_thread(10, sheets_stored_num + 1,
              Fun1, Fun2, Fun3,
              is_ordered=1, is_print=is_print, )

    # %%

    end_SSI(g1_shift, is_energy, n_sigma=3, )

    fGHU_plot_save(U1_name, 0,  # 默认 全自动 is_auto = 1
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

    return fget("U"), fget("G")

if __name__ == '__main__':

    NLA(U1_name="",
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
        U1_name_Structure='',
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
        U1_0_NonZero_size=1, w0=0.3,
        z0=1,
        # %%
        lam1=0.8, is_air_pump=0, is_air=0, T=25,
        deff=30, is_fft=1, fft_mode=0,
        is_linear_convolution=0,
        # %%
        Tx=10, Ty=10, Tz="2*lc",
        mx=0, my=0, mz=0,
        # %%
        # 生成横向结构
        Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
        Depth=2, structure_xy_mode='x',
        is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
        is_reverse_xy=0, is_positive_xy=1, is_no_backgroud=0,
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
        border_percentage=0.1, )
