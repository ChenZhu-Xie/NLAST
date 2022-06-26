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
from fun_linear import init_AST, ifft2
from fun_nonlinear import accurate_args_SFG, Eikz, C_m, Cal_dk_zQ_SFG, Cal_roll_xy, G3_z_modulation_NLAST, \
    G3_z_NLAST, G3_z_NLAST_false, G3_z_modulation_3D_NLAST
from fun_thread import noop, my_thread
from fun_CGH import structure_chi2_Generate_2D
from fun_global_var import init_GLV_DICT, tree_print, Set, Get, init_GLV_rmw, init_EVV, Fun3, end_SSI, \
    dset, dget, fget, fkey, fGHU_plot_save, fU_EVV_plot

np.seterr(divide='ignore', invalid='ignore')


# %%

def SFG_NLA_EVV(U_name="",
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
                structure_size_Shrink=0.1,
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
                U_size=1, w0=0.3,
                z0=1, sheets_stored_num=10,
                # %%
                lam1=0.8, is_air_pump=0, is_air=0, T=25,
                is_air_pump_structure=0,
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
                is_colorbar_on=1, is_colorbar_log=0,
                is_energy=0,
                # %%
                is_plot_EVV=1, is_plot_3d_XYz=0, is_plot_selective=0,
                X=0, Y=0, is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
                # %%
                plot_group="UGa", is_animated=1,
                loop=0, duration=0.033, fps=5,
                # %%
                is_print=1, is_contours=1, n_TzQ=1,
                Gz_max_Enhance=1, match_mode=1,
                # %% 该程序 独有 -------------------------------
                is_EVV_SSI=1,
                # %%
                **kwargs, ):
    # %%
    if_image_Add_black_border(U_name, img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%
    ray_tag = "f" if kwargs.get('ray', "2") == "3" else "h"
    # if ray_tag == "f":
    U2_name = kwargs.get("U2_name", U_name)
    img2_full_name = kwargs.get("img2_full_name", img_full_name)
    is_phase_only_2 = kwargs.get("is_phase_only_2", is_phase_only)
    # %%
    z_pump2 = kwargs.get("z_pump2", z_pump)
    is_LG_2 = kwargs.get("is_LG_2", is_LG)
    is_Gauss_2 = kwargs.get("is_Gauss_2", is_Gauss)
    is_OAM_2 = kwargs.get("is_OAM_2", is_OAM)
    # %%
    l2 = kwargs.get("l2", l)
    p2 = kwargs.get("p2", p)
    theta2_x = kwargs.get("theta2_x", theta_x)
    theta2_y = kwargs.get("theta2_y", theta_y)
    # %%
    is_random_phase_2 = kwargs.get("is_random_phase_2", is_random_phase)
    is_H_l2 = kwargs.get("is_H_l2", is_H_l)
    is_H_theta2 = kwargs.get("is_H_theta2", is_H_theta)
    is_H_random_phase_2 = kwargs.get("is_H_random_phase_2", is_H_random_phase)
    # %%
    w0_2 = kwargs.get("w0_2", w0)
    lam2 = kwargs.get("lam2", lam1)
    is_air_pump2 = kwargs.get("is_air_pump2", is_air_pump)
    T2 = kwargs.get("T2", T)
    polar2 = kwargs.get("polar2", 'e')
    if ray_tag == "f":
        # %%
        pump2_keys = kwargs["pump2_keys"]
        # %%
        [kwargs.pop(key) for key in kwargs["pump2_keys"]]  # 及时清理 kwargs ，尽量 保持 其干净
        kwargs.pop("pump2_keys")  # 这个有点意思， "pump2_keys" 这个键本身 也会被删除。

    # %%

    info = "NLAST_演化版_EVV"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # kwargs['ray'] = init_GLV_rmw(U_name, "^", "EVV", "NLA", **kwargs)
    init_GLV_rmw(U_name, ray_tag, "NLA", "EVV", **kwargs)

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

    if ray_tag == "f":
        from fun_pump import pump_pic_or_U2
        U2_0, g2 = pump_pic_or_U2(U2_name,
                                  img2_full_name,
                                  is_phase_only_2,
                                  # %%
                                  z_pump2,
                                  is_LG_2, is_Gauss_2, is_OAM_2,
                                  l2, p2,
                                  theta2_x, theta2_y,
                                  # %%
                                  is_random_phase_2,
                                  is_H_l2, is_H_theta2, is_H_random_phase_2,
                                  # %%
                                  U_size, w0_2,
                                  # %%
                                  lam2, is_air_pump, T,
                                  polar2,
                                  # %%
                                  is_save, is_save_txt, dpi,
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
                                  ray_pump='2', **kwargs, )
    else:
        U2_0, g2 = U_0, g_shift

    # %%

    n1_inc, n1, k1_inc, k1, k1_z, k1_xy = init_AST(Ix, Iy, size_PerPixel,
                                                   lam1, is_air, T,
                                                   theta_x, theta_y,
                                                   **kwargs)

    if ray_tag == "f":
        n2_inc, n2, k2_inc, k2, k2_z, k2_xy = init_AST(Ix, Iy, size_PerPixel,
                                                       lam2, is_air, T,
                                                       theta2_x, theta2_y,
                                                       polar2=polar2, **kwargs)
    else:
        n2_inc, n2, k2_inc, k2, k2_z, k2_xy = n1_inc, n1, k1_inc, k1, k1_z, k1_xy

    theta3_x, theta3_y, lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, \
    dk, lc, Tz, \
    Gx, Gy, Gz, \
    z0, Tz, deff_structure_length_expect = accurate_args_SFG(Ix, Iy, size_PerPixel,
                                                             lam1, lam2, is_air, T,
                                                             k1_inc, k2_inc,
                                                             g_shift, k1_z,
                                                             z0, z0,
                                                             mx, my, mz,
                                                             Tx, Ty, Tz,
                                                             is_contours, n_TzQ,
                                                             Gz_max_Enhance, match_mode,
                                                             is_print,
                                                             theta_x, theta2_x,
                                                             theta_y, theta2_y,
                                                             **kwargs)

    is_NLAST_sum = kwargs.get("is_NLAST_sum", 0)  # 得写在外面，否则会传进 Fun 等的 kwargs...而这一般是空的
    if fft_mode == 0:
        # %% generate structure

        if ray_tag == "f":
            for key in pump2_keys:
                kwargs[key] = locals()[key]
                kwargs["pump2_keys"] = locals()["pump2_keys"]
        n1_inc, n1, k1_inc, k1, k1_z, n2_inc, n2, k2_inc, k2, k2_z, lam3, n3_inc, n3, k3_inc, k3, k3_z, \
        theta3_x, theta3_y, z0, deff_structure_length_expect, dk, lc, Tz, Gx, Gy, Gz, folder_address, \
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
                                         U_size, w0_Structure,
                                         structure_size_Shrink,
                                         Duty_Cycle_x, Duty_Cycle_y,
                                         structure_xy_mode, Depth,
                                         # %%
                                         is_continuous, is_target_far_field,
                                         is_transverse_xy, is_reverse_xy,
                                         is_positive_xy,
                                         0, is_no_backgroud,
                                         # %%
                                         lam1, is_air_pump_structure, is_air, T,
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
                                         # %% --------------------- for Info_find_contours_SHG
                                         deff_structure_length_expect,
                                         is_contours, n_TzQ,
                                         Gz_max_Enhance, match_mode,
                                         L0_Crystal=z0, g_shift=g_shift,
                                         # %%
                                         **kwargs, )
        if ray_tag == "f":
            [kwargs.pop(key) for key in kwargs["pump2_keys"]]  # 及时清理 kwargs ，尽量 保持 其干净
            kwargs.pop("pump2_keys")  # 这个有点意思， "pump2_keys" 这个键本身 也会被删除。

    # %%

    iz = z0 / size_PerPixel
    # zj = kwargs.get("zj", np.linspace(0, z0, sheets_stored_num + 1)) \
    #     if is_stored==1 else np.linspace(0, z0, sheets_stored_num + 1)
    zj = kwargs.get("zj_EVV", np.linspace(0, z0, sheets_stored_num + 1))  # 防止 后续函数 接收的 kwargs 里 出现 关键字 zj 后重名
    # kwargs.pop("zj", None) # 防止 后续函数 接收的 kwargs 里 出现 关键字 zj 后重名
    izj = zj / size_PerPixel
    # print(izj)
    if is_EVV_SSI == 1:
        izj_delay_dz = [0] + list(izj)
        dizj = list(np.array(izj_delay_dz)[1:] - np.array(izj_delay_dz)[:-1])  # 为循环 里使用
        izj_delay_dz.pop(-1) # 可 pop 可不 pop 掉 最后一个元素，反正没啥用
        # dizj = [izj[0] - 0] + dizj
        # Set("is_EVV_SSI", 1)
        # print(izj_delay_dz)
        # print(dizj)
    Set("zj", zj)
    Set("izj", izj)

    sheets_stored_num = len(zj) - 1
    init_EVV(g_shift, U_0,
             is_energy_evolution_on, is_stored,
             sheets_stored_num, sheets_stored_num,
             X, Y, iz, size_PerPixel, )

    if is_EVV_SSI == 1:
        def H3_zdz(diz):
            # return np.power(math.e, (k1_z + k2_z + Gz) * diz * 1j)
            # return np.power(math.e, ((k1_z + k2_z + Gz)/2 + k3_z/2) * diz * 1j)
            return np.power(math.e, k3_z * diz * 1j)
            # return np.power(math.e, k3 * diz * 1j)


    def Fun1(for_th2, fors_num2, *args, **kwargs, ):

        Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
            np.zeros((Ix, Iy), dtype=np.complex128()))
        # 加上 Fun3 中的 sset 其实共 储存了 2 次 这些 G3_zm 层（m = for_th2），有点浪费
        # 但这里的又是必须存在的，因为之后得用它来在 fun2 里累加。。
        # 所以最多也是之后的 Fun3 中的 sset 可以不必存在

        if is_EVV_SSI == 1:
            iz = izj_delay_dz[for_th2]
            H1_z = np.power(math.e, k1_z * iz * 1j)
            G1_z = g_shift * H1_z
            U_z = ifft2(G1_z)

            if ray_tag == "f":
                H2_z = np.power(math.e, k2_z * iz * 1j)
                G2_z = g2 * H2_z
                U2_z = ifft2(G2_z)
            else:
                G2_z = G1_z
                U2_z = U_z
            diz = dizj[for_th2]
        else:
            iz = izj[for_th2]
            G1_z = g_shift
            U_z = U_0
            G2_z = g2
            U2_z = U2_0
            diz = iz
        # print(iz)
        # print(diz)

        # for_th2 == 0 时也要算，因为 zj[0] 不一定是 0：外部可能传入 zj
        if is_fft == 0:

            const = (k3_inc / size_PerPixel / n3_inc) ** 2 * C_m(mx) * C_m(my) * C_m(
                mz) * deff * 1e-12  # pm / V 转换成 m / V
            integrate_z0 = np.zeros((Ix, Iy), dtype=np.complex128())

            G2_rotate_180 = Rotate_180(G2_z)

            def fun1(for_th, fors_num, *args, **kwargs, ):
                for n3_y in range(Iy):
                    dk_zQ = Cal_dk_zQ_SFG(k2,
                                          k1_z, k3_z,
                                          k1_xy, k3_xy,
                                          for_th, n3_y,
                                          Gx, Gy, Gz, )

                    roll_x, roll_y = Cal_roll_xy(Gx, Gy,
                                                 Ix, Iy,
                                                 for_th, n3_y, )

                    G2_shift_dk_x_dk_y = Roll_xy(G2_rotate_180,
                                                 roll_x, roll_y,
                                                 is_linear_convolution, )

                    integrate_z0[for_th, n3_y] = np.sum(
                        G1_z * G2_shift_dk_x_dk_y * Eikz(dk_zQ * diz) * diz * size_PerPixel \
                        * (2 / (dk_zQ / k3_z[for_th, n3_y] + 2)))

            my_thread(10, Ix,
                      fun1, noop, noop,
                      is_ordered=1, is_print=is_print, )

            G3_z = const * integrate_z0 / k3_z * size_PerPixel

            Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                G3_z * np.power(math.e, k3_z * diz * 1j))

        else:

            Const = (k3_inc / size_PerPixel / n3_inc) ** 2 * deff * 1e-12  # pm / V 转换成 m / V

            if fft_mode == 0:

                if is_sum_Gm == 0:
                    # print(kwargs)
                    addition_dict = {"Tz": Tz if is_NLAST_sum else None}  # 若 is_NLAST_sum 有且非 0，则 Tz
                    Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                        G3_z_modulation_NLAST(k1, k2, k3,
                                              modulation_squared, U_z, U2_z, diz, Const,
                                              Gz=Gz, is_customized=1, **addition_dict, ))
                elif is_sum_Gm == 1:
                    def fun1(for_th, fors_num, *args, **kwargs, ):
                        m_z = for_th - mG
                        Gz_m = 2 * math.pi * m_z * size_PerPixel / (Tz / 1000)
                        # print(m_z, C_m(m_z), "\n")

                        # 注意这个系数 C_m(m_z) 只对应 Duty_Cycle_z = 50% 占空比...
                        Const = (k3_inc / size_PerPixel / n3_inc) ** 2 * C_m(mx) * C_m(my) * C_m(m_z) * deff * 1e-12
                        G3_z0_Gm = G3_z_modulation_NLAST(k1, k2, k3,
                                                         modulation_squared, U_z, U2_z, diz, Const,
                                                         Gz=Gz_m, is_customized=1, ) if m_z != 0 else 0
                        return G3_z0_Gm

                    def fun2(for_th, fors_num, G3_z0_Gm, *args, **kwargs, ):
                        # print("forth = {}".format(for_th))

                        Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                            Get("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way")) + G3_z0_Gm)

                        return Get("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"))

                    my_thread(10, 2 * mG + 1,
                              fun1, fun2, noop,
                              is_ordered=1, is_print=is_print, )
                else:
                    Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                        G3_z_modulation_3D_NLAST(k1, k2, k3,
                                                 modulation_squared, U_z, U2_z,
                                                 diz, Const,
                                                 Tz=Tz, ))

            elif fft_mode == 1:

                if is_sum_Gm == 0:
                    Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                        G3_z_NLAST(k1, k2, k3, Gx, Gy, Gz,
                                   U_z, U2_z, diz, Const,
                                   is_linear_convolution, ))
                else:
                    def fun1(for_th, fors_num, *args, **kwargs, ):
                        m_x = for_th - mG
                        Gx_m = 2 * math.pi * m_x * size_PerPixel / (Tx / 1000)
                        # print(m_x, C_m(m_x), "\n")

                        # 注意这个系数 C_m(m_x) 只对应 Duty_Cycle_x = 50% 占空比...
                        Const = (k3_inc / size_PerPixel / n3_inc) ** 2 * C_m(m_x) * C_m(my) * C_m(mz) * deff * 1e-12
                        G3_z0_Gm = G3_z_NLAST(k1, k2, k3, Gx_m, Gy, Gz,
                                              U_z, U2_z, diz, Const,
                                              is_linear_convolution, ) if m_x != 0 else 0
                        return G3_z0_Gm

                    def fun2(for_th, fors_num, G3_z0_Gm, *args, **kwargs, ):

                        Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                            Get("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way")) + G3_z0_Gm)

                        return Get("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"))

                    my_thread(10, 2 * mG + 1,
                              fun1, fun2, noop,
                              is_ordered=1, is_print=is_print, )

            elif fft_mode == 2:

                Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
                    G3_z_NLAST_false(k1, k2, k3, Gx, Gy, Gz,
                                     U_z, U2_z, diz, Const,
                                     is_linear_convolution, ))

        return Get("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"))

    def Fun2(for_th2, fors_num, G3_zm, *args, **kwargs, ):

        # print(dizj[for_th2])
        # print(np.sum(np.abs(dget("G"))))
        if is_EVV_SSI == 1:
            dset("G", dget("G") * H3_zdz(dizj[for_th2]) + G3_zm)  # is_EVV_SSI == 1 时， 此时 G3_zm 是 dG3_zdz
        else:
            dset("G", G3_zm)  # 主要是为了 end_SSI 中的 dget("G") 而存在的，否则 直接 返回 G3_zm 了
        # print(np.sum(np.abs(dget("G"))))
        # dset("G", G3_zm)

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
                   z0,
                   # %%
                   is_end=1, **kwargs, )

    # %%

    fU_EVV_plot(img_name_extension,
                is_save_txt,
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
                is_colorbar_on, is_colorbar_log,
                is_energy,
                # %%
                plot_group, is_animated,
                loop, duration, fps,
                # %%
                is_plot_EVV, is_plot_3d_XYz, is_plot_selective,
                is_plot_YZ_XZ, is_plot_3d_XYZ,
                # %%
                z0,
                # %%
                **kwargs, )

    import inspect
    if inspect.stack()[1][3] == "SFG_NLA_EVV__AST_EVV":
        Set("k3", k3)
        Set("lam3", lam3)

    return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U")


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",
         "img_full_name": "lena1.png",
         "U_pixels_x": 0, "U_pixels_y": 0,
         "is_phase_only": 0,
         # %%
         "z_pump": 0,
         "is_LG": 1, "is_Gauss": 1, "is_OAM": 1,
         "l": 3, "p": 0,
         "theta_x": 0, "theta_y": 0,
         # %%
         "is_random_phase": 0,
         "is_H_l": 0, "is_H_theta": 0, "is_H_random_phase": 0,
         # %%
         # 生成横向结构
         "U_name_Structure": '',
         "structure_size_Shrink": 0.1, "structure_size_Shrinker": 0,
         "is_U_size_x_structure_side_y": 1,
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
         "U_size": 1, "w0": 0.05,
         "z0": 15, "sheets_stored_num": 10,
         # %%
         "lam1": 1.064, "is_air_pump": 1, "is_air": 0, "T": 25,
         "lam_structure": 1, "is_air_pump_structure": 1, "T_structure": 25,
         "deff": 30, "is_fft": 1, "fft_mode": 1,
         "is_sum_Gm": 0, "mG": 0, 'is_NLAST_sum': 0,
         "is_linear_convolution": 0,
         # %%
         "Tx": 13, "Ty": 20, "Tz": 10,
         "mx": 1, "my": 0, "mz": 1,
         # %%
         # 生成横向结构
         "Duty_Cycle_x": 0.5, "Duty_Cycle_y": 0.5, "Duty_Cycle_z": 0.5,
         "Depth": 2, "structure_xy_mode": 'x',
         # %%
         "is_continuous": 0, "is_target_far_field": 1, "is_transverse_xy": 0,
         "is_reverse_xy": 0, "is_positive_xy": 1, "is_no_backgroud": 0,
         "is_stored": 1, "is_energy_evolution_on": 1,
         # %%
         "is_save": 0, "is_no_data_save": 0,
         "is_save_txt": 0, "dpi": 100,
         # %%
         "color_1d": 'b', "cmap_2d": 'viridis', "cmap_3d": 'rainbow',
         "elev": 10, "azim": -65, "alpha": 2,
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
         "is_colorbar_on": 1, "is_colorbar_log": 0,
         "is_energy": 1,
         # %%
         "is_plot_EVV": 0, "is_plot_3d_XYz": 0, "is_plot_selective": 1,
         "X": 0, "Y": 0, "is_plot_YZ_XZ": 1, "is_plot_3d_XYZ": 1,
         # %%
         "plot_group": "UGa", "is_animated": 1,
         "loop": 0, "duration": 0.033, "fps": 5,
         # %%
         "is_print": 1, "is_contours": 0, "n_TzQ": 1,
         "Gz_max_Enhance": 1, "match_mode": 1,
         # %% 该程序 独有 -------------------------------
         "is_EVV_SSI": 0,
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
         "polar": "o",
         "ray": "3", "polar3": "o", 
         }

    if kwargs.get("ray", "2") == "3":  # 如果 ray == 3，则 默认 双泵浦 is_twin_pumps == 1
        pump2_kwargs = {
            "U2_name": "",
            "img2_full_name": "lena1.png",
            "is_phase_only_2": 0,
            # %%
            "z_pump2": 0,
            "is_LG_2": 1, "is_Gauss_2": 1, "is_OAM_2": 1,
            "l2": 3, "p2": 0,
            "theta2_x": 0, "theta2_y": 0,
            # %%
            "is_random_phase_2": 0,
            "is_H_l2": 0, "is_H_theta2": 0, "is_H_random_phase_2": 0,
            # %%
            "w0_2": 0.05,
            # %%
            "lam2": 1.064, "is_air_pump2": 1, "T2": 25,
            "polar2": 'e',
        }
        pump2_kwargs.update({"pump2_keys": list(pump2_kwargs.keys())})
        # Object of type dict_keys is not JSON serializable，所以 得转为 list
        kwargs.update(pump2_kwargs)

    kwargs = init_GLV_DICT(**kwargs)
    SFG_NLA_EVV(**kwargs)

    # SFG_NLA_EVV(U_name="",
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
    #             structure_size_Shrink=0.1,
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
    #             U_size=0.9, w0=0.3,
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
    #             sample=1, ticks_num=6, is_contourf=0,
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
