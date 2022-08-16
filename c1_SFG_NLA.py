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
from fun_linear import init_AST_12oe
from fun_nonlinear import accurate_args_SFG, Eikz, C_m, Cal_dk_zQ_SFG, Cal_roll_xy, \
    G3_z_modulation_NLAST, G3_z_modulation_3D_NLAST, G3_z_NLAST, G3_z_NLAST_false
from fun_thread import noop, my_thread
from fun_CGH import chi2_2D
from fun_global_var import init_GLV_DICT, tree_print, init_GLV_rmw, end_SSI, Get, dset, dget, fget, fkey, fGHU_plot_save
from a1_AST import Gan_gp_p, Gan_gp_VH, gan_nkgE_oe, gan_nkgE_VHoe, plot_n_VHoe

np.seterr(divide='ignore', invalid='ignore')


# %%

def define_n(**kwargs):
    n_name = "n"
    if "f" in Get("ray"):
        n_name += "3"
    elif "h" in Get("ray"):  # 如果 ray 中含有 倍频 标识符
        n_name += "2"
    else:
        n_name += "1"
    return n_name


# %%

def gan_args_SFG(Ix, Iy, size_PerPixel,
                 lam1, is_air, T,
                 theta_x, theta_y,
                 is_twin_pump, is_air_pump, is_print,
                 lam2, theta2_x, theta2_y,
                 z0, deff_structure_length_expect,
                 mx, my, mz,
                 Tx, Ty, Tz,
                 is_contours, n_TzQ,
                 Gz_max_Enhance, match_mode,
                 gp_1=0, gp_2=0, p_2=0,
                 is_end_3=0, **kwargs):
    # print(kwargs["polar2"])  # 里面应该是没有 polar2 关键字的

    kwargs_1 = {} if type(gp_1) != np.ndarray else {"gp": gp_1}
    kwargs.update(kwargs_1)
    n1_inc, n1, k1_inc, k1, k1_z, k1_xy, g1, E1_u = \
        init_AST_12oe(Ix, Iy, size_PerPixel,
                      lam1, is_air, T,
                      theta_x, theta_y, is_print,
                      is_air_pump=is_air_pump,
                      is_end2=-1, **kwargs, )

    kwargs_21 = {} if type(gp_2) != np.ndarray else {"gp": gp_2}
    kwargs_22 = {} if type(p_2) != str else {"polar2": p_2}
    kwargs.update(kwargs_21)
    kwargs.update(kwargs_22)
    # print(kwargs["polar2"])
    # print(kwargs["polar3"])
    if is_twin_pump == 1:
        n2_inc, n2, k2_inc, k2, k2_z, k2_xy, g2, E2_u = \
            init_AST_12oe(Ix, Iy, size_PerPixel,
                          lam2, is_air, T,
                          theta2_x, theta2_y, is_print,
                          is_air_pump=is_air_pump,
                          add_level=1, **kwargs, )
    else:
        n2_inc, n2, k2_inc, k2, k2_z, k2_xy = n1_inc, n1, k1_inc, k1, k1_z, k1_xy

    kwargs_31 = {} if type(gp_1) != np.ndarray else {"g1": gp_1}
    kwargs_32 = {} if type(gp_2) != np.ndarray else {"g2": gp_2}
    kwargs.update(kwargs_31)
    kwargs.update(kwargs_32)
    lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
    dk_z, lc, Gx, Gy, Gz, \
    z0, Tz, deff_structure_length_expect = accurate_args_SFG(Ix, Iy, size_PerPixel,
                                                             lam1, lam2, is_air, T,
                                                             k1_inc, k2_inc,
                                                             k1, k2, k1_z,
                                                             z0, deff_structure_length_expect,
                                                             mx, my, mz,
                                                             Tx, Ty, Tz,
                                                             is_contours, n_TzQ,
                                                             Gz_max_Enhance, match_mode,
                                                             is_print,
                                                             Get("theta_x"), Get("theta2_x"),  # 把晶体内的 角度 传进去
                                                             Get("theta_y"), Get("theta2_y"),
                                                             is_air_pump=is_air_pump,
                                                             is_end=is_end_3, **kwargs)
    return n1_inc, n1, k1_inc, k1, k1_z, k1_xy, E1_u, \
           n2_inc, n2, k2_inc, k2, k2_z, k2_xy, E2_u, \
           lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
           dk_z, lc, Gx, Gy, Gz, \
           z0, Tz, deff_structure_length_expect


# %%

def gan_args_SHG_oe(Ix, Iy, size_PerPixel,
                    lam1, is_air, T,
                    theta_x, theta_y,
                    z0, deff_structure_length_expect,
                    mx, my, mz,
                    Tx, Ty, Tz,
                    is_contours, n_TzQ,
                    Gz_max_Enhance, match_mode,
                    g_p, p_p, is_print,
                    is_air_pump=1, g_shift=0,
                    is_end_3=0, **kwargs):
    args_init_AST = [Ix, Iy, size_PerPixel,
                     lam1, is_air, T,
                     theta_x, theta_y]

    kwargs_init_AST = {"is_air_pump": is_air_pump, "gp": g_shift, }
    if "gp" in kwargs: kwargs.pop("gp")
    if "polar2" in kwargs: kwargs.pop("polar2")
    n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
    n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue \
        = gan_nkgE_oe(g_p, p_p, is_print,
                      args_init_AST, kwargs_init_AST, **kwargs)

    k1_z = (k1o_z + k1e_z) / 2

    kwargs["g1"] = g_o
    kwargs["g2"] = g_e
    lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
    dk_z, lc, Gx, Gy, Gz, \
    z0, Tz, deff_structure_length_expect = accurate_args_SFG(Ix, Iy, size_PerPixel,
                                                             lam1, lam1, is_air, T,
                                                             k1o_inc, k1e_inc,
                                                             k1o, k1e, k1_z,
                                                             z0, deff_structure_length_expect,
                                                             mx, my, mz,
                                                             Tx, Ty, Tz,
                                                             is_contours, n_TzQ,
                                                             Gz_max_Enhance, match_mode,
                                                             is_print,
                                                             Get("theta_x"), Get("theta2_x"),  # 把晶体内的 角度 传进去
                                                             Get("theta_y"), Get("theta2_y"),
                                                             is_air_pump=is_air_pump,
                                                             is_end=is_end_3, **kwargs)

    return n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
           n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue, \
           lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
           dk_z, lc, Gx, Gy, Gz, \
           z0, Tz, deff_structure_length_expect


# %%

def gan_args_SHG_VHoe(Ix, Iy, size_PerPixel,
                      lam1, is_air, T,
                      theta_x, theta_y,
                      z0, deff_structure_length_expect,
                      mx, my, mz,
                      Tx, Ty, Tz,
                      is_contours, n_TzQ,
                      Gz_max_Enhance, match_mode,
                      g_V, p_V, g_H, p_H, is_print,
                      is_air_pump=1, g_shift=0,
                      is_end_3=0, **kwargs):
    args_init_AST = [Ix, Iy, size_PerPixel,
                     lam1, is_air, T,
                     theta_x, theta_y]

    kwargs_init_AST = {"is_air_pump": is_air_pump, "gp": g_shift, }
    if "gp" in kwargs: kwargs.pop("gp")
    if "polar2" in kwargs: kwargs.pop("polar2")
    n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
    n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
    n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
    n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He = \
        gan_nkgE_VHoe(g_V, p_V, g_H, p_H, is_print,
                      args_init_AST, kwargs_init_AST, **kwargs)

    g_o = g_Vo + g_Ho
    g_e = g_Ve + g_He
    k1o_inc = (k1_Vo_inc + k1_Ho_inc) / 2
    k1e_inc = (k1_Ve_inc + k1_He_inc) / 2
    k1o = (k1_Vo + k1_Ho) / 2
    k1e = (k1_Ve + k1_He) / 2
    k1o_z = (k1_Vo_z + k1_Ho_z) / 2
    k1e_z = (k1_Ve_z + k1_He_z) / 2
    k1_z = (k1o_z + k1e_z) / 2

    kwargs["g1"] = g_o
    kwargs["g2"] = g_e
    lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
    dk_z, lc, Gx, Gy, Gz, \
    z0, Tz, deff_structure_length_expect = accurate_args_SFG(Ix, Iy, size_PerPixel,
                                                             lam1, lam1, is_air, T,
                                                             k1o_inc, k1e_inc,
                                                             k1o, k1e, k1_z,
                                                             z0, deff_structure_length_expect,
                                                             mx, my, mz,
                                                             Tx, Ty, Tz,
                                                             is_contours, n_TzQ,
                                                             Gz_max_Enhance, match_mode,
                                                             is_print,
                                                             Get("theta_x"), Get("theta2_x"),  # 把晶体内的 角度 传进去
                                                             Get("theta_y"), Get("theta2_y"),
                                                             is_air_pump=is_air_pump,
                                                             is_end=is_end_3, **kwargs)

    return n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
           n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
           n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
           n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He, \
           lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
           dk_z, lc, Gx, Gy, Gz, \
           z0, Tz, deff_structure_length_expect


# %%

def plot_n_123(ray_tag, is_save,
               n1, n2, n3,
               args_U_amp_plot_save,
               kwargs_U_amp_plot_save, **kwargs, ):
    from fun_os import U_dir, U_amp_plot_save
    kwargs['p_dir'] = 'PUMP'
    method = "DIFF"
    # %% 晶体内 o 光 折射率 分布

    n_p1_name = method + " - " + "n" + "p1"
    folder_address = U_dir(n_p1_name, is_save, **kwargs, )
    U_amp_plot_save(*args_U_amp_plot_save(folder_address, n1, n_p1_name),
                    **kwargs_U_amp_plot_save, **kwargs, )
    # kwargs['p_dir'] = 'PUMP - 2'
    n_p2_name = method + " - " + "n" + "p2"
    folder_address = U_dir(n_p2_name, is_save, **kwargs, )
    U_amp_plot_save(*args_U_amp_plot_save(folder_address, n2, n_p2_name),
                    **kwargs_U_amp_plot_save, **kwargs, )
    n_p2_name = method + " - " + "n" + ray_tag + "3"
    folder_address = U_dir(n_p2_name, is_save, **kwargs, )
    U_amp_plot_save(*args_U_amp_plot_save(folder_address, n3, n_p2_name),
                    **kwargs_U_amp_plot_save, **kwargs, )


# %%

def gan_gpnkE_123VHoe_xyzinc_SFG(is_birefringence_deduced, is_air,
                                 is_add_polarizer, is_HOPS,
                                 is_save, is_print,
                                 ray_tag, is_twin_pump, is_air_pump,
                                 lam_2, theta2_X, theta2_Y,  # 为了与 kwargs 里 的名称不重复，需要单独设计 位置参数名称 如 g_1, g_2
                                 g_1, g_2, U_0, U2_0, polar_2,
                                 args_init_AST, args_gan_args_SFG,
                                 args_U_amp_plot_save,
                                 kwargs_init_AST, kwargs_U_amp_plot_save,
                                 is_plot_n=1, is_end=0, **kwargs):
    from a1_AST import init_locals
    g_p, p_p, g_V, g_H, p_V, p_H, \
    n1_inc, n1, k1_inc, k1, k1_z, k1_xy, E1_u, \
    n2_inc, n2, k2_inc, k2, k2_z, k2_xy, E2_u, \
    lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
    n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
    n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue, \
    n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
    n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
    n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
    n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He, \
    dk_z, lc, Gx, Gy, Gz, \
    z0, Tz, deff_structure_length_expect = \
        init_locals("g_p, p_p, g_V, g_H, p_V, p_H, \
                    n1_inc, n1, k1_inc, k1, k1_z, k1_xy, E1_u, \
                    n2_inc, n2, k2_inc, k2, k2_z, k2_xy, E2_u, \
                    lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
                    n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
                    n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue, \
                    n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
                    n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
                    n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
                    n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He, \
                    dk_z, lc, Gx, Gy, Gz, \
                    z0, Tz, deff_structure_length_expect")
    # 主要是 is_add_polarizer 和 def 导致的，有些变量 没声明，却在 def 的 形参中 出现了，以致于 实参 在用到时 报错
    # 其实就是 把 pycharm 所提示的 “可能在赋值前引用” 的 局部变量 先赋好值

    if is_birefringence_deduced == 1:  # 考虑 偏振态 的 条件；is_air == 1 时 也可以 有偏振态，与是否 所处介质 无关
        # %% 起偏

        if is_add_polarizer == 1:
            g_p, p_p = Gan_gp_p(is_HOPS, g_1,
                                U_0, U2_0, polar_2, **kwargs)
        else:
            g_V, g_H, p_V, p_H = Gan_gp_VH(is_HOPS, U_0, U2_0, polar_2, **kwargs)

        # %% 空气中，偏振状态 与 入射方向 无关/独立，因此 无论 theta_x 怎么取，U 中所有点 偏振状态 均为 V，且 g 中 所有点的 偏振状态也 均为 V
        # 但晶体中，折射后的 偏振状态 与 g 中各点 kx,ky 对应的 入射方向 就有关了，因此得 在倒空间中 投影操作，且每个点都 分别考虑。
        if is_add_polarizer == 1:
            n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
            n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue, \
            lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
            dk_z, lc, Gx, Gy, Gz, \
            z0, Tz, deff_structure_length_expect \
                = gan_args_SHG_oe(*args_init_AST,
                                  *args_gan_args_SFG,
                                  g_p, p_p, is_print,
                                  **kwargs_init_AST,
                                  is_end_3=is_end, **kwargs)
        else:
            n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
            n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
            n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
            n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He, \
            lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
            dk_z, lc, Gx, Gy, Gz, \
            z0, Tz, deff_structure_length_expect = \
                gan_args_SHG_VHoe(*args_init_AST,
                                  *args_gan_args_SFG,
                                  g_V, p_V, g_H, p_H, is_print,
                                  **kwargs_init_AST,
                                  is_end_3=is_end, **kwargs)

        # %% 晶体内 oe 光 折射率 分布
        if is_plot_n == 1:
            plot_n_VHoe("np1", is_save,
                        is_add_polarizer,
                        n1o, n1_Vo, n1_Ho,
                        n1e, n1_Ve, n1_He,
                        args_U_amp_plot_save,
                        kwargs_U_amp_plot_save, **kwargs, )

    else:
        n1_inc, n1, k1_inc, k1, k1_z, k1_xy, E1_u, \
        n2_inc, n2, k2_inc, k2, k2_z, k2_xy, E2_u, \
        lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
        dk_z, lc, Gx, Gy, Gz, \
        z0, Tz, deff_structure_length_expect = \
            gan_args_SFG(*args_init_AST,
                         is_twin_pump, is_air_pump, is_print,
                         lam_2, theta2_X, theta2_Y,
                         *args_gan_args_SFG,
                         gp_1=g_1, gp_2=g_2, p_2=polar_2,
                         is_end_3=is_end, **kwargs)

        if is_plot_n == 1:
            plot_n_123(ray_tag, is_save,
                       n1, n2, n3,
                       args_U_amp_plot_save,
                       kwargs_U_amp_plot_save, **kwargs, )

        # print(n1_inc, n2_inc, n3_inc)
        # print(n1_inc + n2_inc - 2 * n3_inc)
    return g_p, p_p, g_V, g_H, p_V, p_H, \
           n1_inc, n1, k1_inc, k1, k1_z, k1_xy, E1_u, \
           n2_inc, n2, k2_inc, k2, k2_z, k2_xy, E2_u, \
           lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
           n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
           n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue, \
           n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
           n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
           n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
           n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He, \
           dk_z, lc, Gx, Gy, Gz, \
           z0, Tz, deff_structure_length_expect


# %%

def G3_z_NLCOV(iz, size_PerPixel,
               Ix, Iy, k3_inc, n3_inc,
               k1_z, k1_xy, k2, k3_z, k3_xy,
               mx, my, mz, deff,
               Gx, Gy, Gz,
               g_shift, g2,
               is_print, is_linear_convolution, ):
    const = (k3_inc / size_PerPixel / n3_inc) ** 2 * C_m(mx) * C_m(my) * C_m(mz) * deff * 1e-12  # pm / V 转换成 m / V
    integrate_z0 = np.zeros((Ix, Iy), dtype=np.complex128())

    g2_rotate_180 = Rotate_180(g2)

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

            g2_shift_dk_x_dk_y = Roll_xy(g2_rotate_180,
                                         roll_x, roll_y,
                                         is_linear_convolution, )

            integrate_z0[for_th, n3_y] = np.sum(
                g_shift * g2_shift_dk_x_dk_y * Eikz(dk_zQ * iz) * iz * size_PerPixel \
                * (2 / (dk_zQ / k3_z[for_th, n3_y] + 2)))

    my_thread(10, Ix,
              fun1, noop, noop,
              is_ordered=1, is_print=is_print, )

    g3_z = const * integrate_z0 / k3_z * size_PerPixel
    G3_z = g3_z * np.power(math.e, k3_z * iz * 1j)

    return G3_z


# %%

def sum_Gm_z_fun1(for_th, mG, size_PerPixel, Tz,
                  k3_inc, n3_inc, mx, my, deff,
                  k1, k2, k3, modulation_squared,
                  U_0, U2_0, iz, ):
    m_z = for_th - mG
    Gz_m = 2 * math.pi * m_z * size_PerPixel / (Tz / 1000)
    # print(m_z, C_m(m_z), "\n")

    # 注意这个系数 C_m(m_z) 只对应 Duty_Cycle_z = 50% 占空比...
    Const = (k3_inc / size_PerPixel / n3_inc) ** 2 * C_m(mx) * C_m(my) * C_m(m_z) * deff * 1e-12
    G3_z0_Gm = G3_z_modulation_NLAST(k1, k2, k3,
                                     modulation_squared, U_0, U2_0, iz, Const,
                                     Gz=Gz_m, is_customized=1, ) if m_z != 0 else 0
    return G3_z0_Gm


def sum_Gm_x_fun1(for_th, mG, size_PerPixel, Tx,
                  k3_inc, n3_inc, my, mz, deff,
                  k1, k2, k3, Gy, Gz,
                  U_0, U2_0, iz, is_linear_convolution, ):
    m_x = for_th - mG
    Gx_m = 2 * math.pi * m_x * size_PerPixel / (Tx / 1000)
    # print(m_x, C_m(m_x), "\n")

    # 注意这个系数 C_m(m_x) 只对应 Duty_Cycle_x = 50% 占空比...
    Const = (k3_inc / size_PerPixel / n3_inc) ** 2 * C_m(m_x) * C_m(my) * C_m(mz) * deff * 1e-12
    G3_z0_Gm = G3_z_NLAST(k1, k2, k3, Gx_m, Gy, Gz,
                          U_0, U2_0, iz, Const,
                          is_linear_convolution, ) if m_x != 0 else 0
    return G3_z0_Gm


# %%

def dset_or_Set(G3_z, for_th2):
    from fun_global_var import Set
    if type(for_th2) == str:
        dset("G", G3_z)
    else:
        Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"), G3_z)


def dsget_or_SGet(G3_z0_Gm, for_th2):
    from fun_global_var import Set
    if type(for_th2) == str:
        G3_z = dget("G") + G3_z0_Gm
        dset("G", G3_z)
    else:
        G3_z = Get("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way")) + G3_z0_Gm
        Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"), G3_z)
    return G3_z


def NLA(iz, is_fft, fft_mode,
        Ix, Iy, size_PerPixel,
        k3_inc, n3_inc,
        k1_z, k1_xy,
        k3_z, k3_xy,
        k1, k2, k3,
        mx, my, mz,
        Gx, Gy, Gz,
        Tx, Tz, deff,
        mG, is_sum_Gm, is_NLAST_sum,
        g_shift, g2,
        U_0, U2_0, modulation_squared,
        is_print, is_linear_convolution,
        for_th2="", ):
    if is_fft == 0:

        G3_z = G3_z_NLCOV(iz, size_PerPixel,
                          Ix, Iy, k3_inc, n3_inc,
                          k1_z, k1_xy, k2, k3_z, k3_xy,
                          mx, my, mz, deff,
                          Gx, Gy, Gz,
                          g_shift, g2,
                          is_print, is_linear_convolution, )

        dset_or_Set(G3_z, for_th2)

    else:

        Const = (k3_inc / size_PerPixel / n3_inc) ** 2 * deff * 1e-12  # pm / V 转换成 m / V

        if fft_mode == 0:

            if is_sum_Gm == 0:
                addition_dict = {"Tz": Tz if is_NLAST_sum else None}  # 若 is_NLAST_sum 有且非 0，则 Tz
                # print(Gz,iz, Const,)
                G3_z = G3_z_modulation_NLAST(k1, k2, k3,
                                             modulation_squared, U_0, U2_0, iz, Const,
                                             Gz=Gz, is_customized=1, **addition_dict, )

                dset_or_Set(G3_z, for_th2)

            elif is_sum_Gm == 1:
                def fun1(for_th, fors_num, *args, **kwargs, ):
                    G3_z0_Gm = sum_Gm_z_fun1(for_th, mG, size_PerPixel, Tz,
                                             k3_inc, n3_inc, mx, my, deff,
                                             k1, k2, k3, modulation_squared,
                                             U_0, U2_0, iz, )
                    return G3_z0_Gm

                def fun2(for_th, fors_num, G3_z0_Gm, *args, **kwargs, ):

                    G3_z = dsget_or_SGet(G3_z0_Gm, for_th2)

                    return G3_z

                my_thread(10, 2 * mG + 1,
                          fun1, fun2, noop,
                          is_ordered=1, is_print=is_print, )
            else:
                G3_z = G3_z_modulation_3D_NLAST(k1, k2, k3,
                                                modulation_squared, U_0, U2_0, iz, Const,
                                                Tz=Tz, )

                dset_or_Set(G3_z, for_th2)

        elif fft_mode == 1:
            if is_sum_Gm == 0:
                G3_z = G3_z_NLAST(k1, k2, k3, Gx, Gy, Gz,
                                  U_0, U2_0, iz, Const,
                                  is_linear_convolution, )
                dset_or_Set(G3_z, for_th2)
            else:
                def fun1(for_th, fors_num, *args, **kwargs, ):
                    G3_z0_Gm = sum_Gm_x_fun1(for_th, mG, size_PerPixel, Tx,
                                             k3_inc, n3_inc, my, mz, deff,
                                             k1, k2, k3, Gy, Gz,
                                             U_0, U2_0, iz, is_linear_convolution, )
                    return G3_z0_Gm

                def fun2(for_th, fors_num, G3_z0_Gm, *args, **kwargs, ):

                    G3_z = dsget_or_SGet(G3_z0_Gm, for_th2)

                    return G3_z

                my_thread(10, 2 * mG + 1,
                          fun1, fun2, noop,
                          is_ordered=1, is_print=is_print, )

        elif fft_mode == 2:
            G3_z = G3_z_NLAST_false(k1, k2, k3, Gx, Gy, Gz,
                                    U_0, U2_0, iz, Const,
                                    is_linear_convolution, )

            dset_or_Set(G3_z, for_th2)

    return G3_z


# %%

def gan_U_VHoe(g_o, g_e, g_Vo, g_Ho, g_Ve, g_He):
    from fun_linear import ifft2
    U_o, U_e = ifft2(g_o) if type(g_o) == np.ndarray else 0, ifft2(g_e) if type(g_e) == np.ndarray else 0
    U_Vo, U_Ve = ifft2(g_Vo) if type(g_Vo) == np.ndarray else 0, ifft2(g_Ve) if type(g_Ve) == np.ndarray else 0
    U_Ho, U_He = ifft2(g_Ho) if type(g_Ho) == np.ndarray else 0, ifft2(g_He) if type(g_He) == np.ndarray else 0
    return U_o, U_e, U_Vo, U_Ve, U_Ho, U_He


def NLA_123VHoe(is_birefringence_deduced, is_air,
                is_add_polarizer, match_type,
                iz, is_fft, fft_mode,
                Ix, Iy, size_PerPixel,
                k3_inc, n3_inc,
                k1_z, k1_xy,
                k1o_z, k1o_xy, k1e_z, k1e_xy,
                k1_Vo_z, k1_Vo_xy, k1_Ho_z, k1_Ho_xy,
                k1_Ve_z, k1_Ve_xy, k1_He_z, k1_He_xy,
                k3_z, k3_xy,
                k1o, k1e, k1_Vo, k1_Ve, k1_Ho, k1_He,
                k1, k2, k3,
                mx, my, mz,
                Gx, Gy, Gz,
                Tx, Tz, deff,
                mG, is_sum_Gm, is_NLAST_sum,
                g_shift, g2,
                g_o, g_e, g_Vo, g_Ve, g_Ho, g_He,
                U_0, U2_0,
                U_o, U_e, U_Vo, U_Ve, U_Ho, U_He,
                modulation_squared,
                is_print, is_linear_convolution,
                for_th2="", ):
    def gan_args_NLA(k1_z, k1_xy,
                     k1, k2,
                     g_shift, g2,
                     U_0, U2_0, ):
        return [iz, is_fft, fft_mode,
                Ix, Iy, size_PerPixel,
                k3_inc, n3_inc,
                k1_z, k1_xy,
                k3_z, k3_xy,
                k1, k2, k3,
                mx, my, mz,
                Gx, Gy, Gz,
                Tx, Tz, deff,
                mG, is_sum_Gm, is_NLAST_sum,
                g_shift, g2,
                U_0, U2_0, modulation_squared,
                is_print, is_linear_convolution, ]

    if is_birefringence_deduced == 1:  # 考虑 偏振态 的 条件；is_air == 1 时 也可以 有偏振态，与是否 所处介质 无关
        if is_add_polarizer == 1:
            if match_type == "oe" or match_type == "eo":
                NLA(*gan_args_NLA(k1o_z, k1o_xy,
                                  k1o, k1e,
                                  g_o, g_e,
                                  U_o, U_e, ), for_th2)
            elif match_type == "oo":
                NLA(*gan_args_NLA(k1o_z, k1o_xy,
                                  k1o, k1o,
                                  g_o, g_o,
                                  U_o, U_o, ), for_th2)
            elif match_type == "ee":
                NLA(*gan_args_NLA(k1e_z, k1e_xy,
                                  k1e, k1e,
                                  g_e, g_e,
                                  U_e, U_e, ), for_th2)
        else:
            if match_type == "oe" or match_type == "eo":
                # 组内 和频
                G3_z_VoVe = NLA(*gan_args_NLA(k1_Vo_z, k1_Vo_xy,
                                              k1_Vo, k1_Ve,
                                              g_Vo, g_Ve,
                                              U_Vo, U_Ve, ), for_th2)
                G3_z_HoHe = NLA(*gan_args_NLA(k1_Ho_z, k1_Ho_xy,
                                              k1_Ho, k1_He,
                                              g_Ho, g_He,
                                              U_Ho, U_He, ), for_th2)
                # 组间 和频
                G3_z_VoHe = NLA(*gan_args_NLA(k1_Vo_z, k1_Vo_xy,
                                              k1_Vo, k1_He,
                                              g_Vo, g_He,
                                              U_Vo, U_He, ), for_th2)
                G3_z_HoVe = NLA(*gan_args_NLA(k1_Ho_z, k1_Ho_xy,
                                              k1_Ho, k1_Ve,
                                              g_Ho, g_Ve,
                                              U_Ho, U_Ve, ), for_th2)
                G3_z = G3_z_VoVe + G3_z_HoHe + G3_z_VoHe + G3_z_HoVe
                dset_or_Set(G3_z, for_th2)
            elif match_type == "oo":
                # 组内 和频
                G3_z_VoVo = NLA(*gan_args_NLA(k1_Vo_z, k1_Vo_xy,
                                              k1_Vo, k1_Vo,
                                              g_Vo, g_Vo,
                                              U_Vo, U_Vo, ), for_th2)
                G3_z_HoHo = NLA(*gan_args_NLA(k1_Ho_z, k1_Ho_xy,
                                              k1_Ho, k1_Ho,
                                              g_Ho, g_Ho,
                                              U_Ho, U_Ho, ), for_th2)
                # 组间 和频
                G3_z_VoHo = NLA(*gan_args_NLA(k1_Vo_z, k1_Vo_xy,
                                              k1_Vo, k1_Ho,
                                              g_Vo, g_Ho,
                                              U_Vo, U_Ho, ), for_th2)
                G3_z = G3_z_VoVo + G3_z_HoHo + G3_z_VoHo
                dset_or_Set(G3_z, for_th2)
            elif match_type == "ee":
                # 组内 和频
                G3_z_VeVe = NLA(*gan_args_NLA(k1_Ve_z, k1_Ve_xy,
                                              k1_Ve, k1_Ve,
                                              g_Ve, g_Ve,
                                              U_Ve, U_Ve, ), for_th2)
                G3_z_HeHe = NLA(*gan_args_NLA(k1_He_z, k1_He_xy,
                                              k1_He, k1_He,
                                              g_He, g_He,
                                              U_He, U_He, ), for_th2)
                # 组间 和频
                G3_z_VeHe = NLA(*gan_args_NLA(k1_Ve_z, k1_Ve_xy,
                                              k1_Ve, k1_He,
                                              g_Ve, g_He,
                                              U_Ve, U_He, ), for_th2)
                G3_z = G3_z_VeVe + G3_z_HeHe + G3_z_VeHe
                dset_or_Set(G3_z, for_th2)
    else:
        NLA(*gan_args_NLA(k1_z, k1_xy,
                          k1, k2,
                          g_shift, g2,
                          U_0, U2_0, ), for_th2)


# %%

def SFG_NLA(U_name="",
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
            z0=1,
            # %%
            lam1=1.064, is_air_pump=0, is_air=0, T=25,
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
    # print(kwargs)
    # %%
    if_image_Add_black_border(U_name, img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%
    is_HOPS = kwargs.get("is_HOPS_SHG", 0)
    is_twin_pump_degenerate = int(is_HOPS >= 1)  # is_HOPS == 0.x 的情况 仍是单泵浦
    is_single_pump_birefringence = int(0 <= is_HOPS < 1 and kwargs.get("polar", "e") in "VvHhRrLl")
    is_birefringence_deduced = int(is_twin_pump_degenerate == 1 or is_single_pump_birefringence == 1)
    kwargs['ray'] = "2" if is_birefringence_deduced == 1 else kwargs.get('ray', "2")
    ray_tag = "f" if kwargs['ray'] == "3" else "h"
    is_twin_pump = int(ray_tag == "f" or is_twin_pump_degenerate == 1)
    is_add_polarizer = int(is_HOPS > 0 and type(is_HOPS) != int)
    is_add_analyzer = int(type(kwargs.get("phi_a", 0)) != str)
    # %%
    # if is_twin_pump == 1:
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
    theta2_x = kwargs.get("theta2_x", theta_x) if is_HOPS == 0 or is_HOPS >= 2 else theta_x
    theta2_y = kwargs.get("theta2_y", theta_y) if is_HOPS == 0 or is_HOPS >= 2 else theta_y
    # %%
    is_random_phase_2 = kwargs.get("is_random_phase_2", is_random_phase)
    is_H_l2 = kwargs.get("is_H_l2", is_H_l)
    is_H_theta2 = kwargs.get("is_H_theta2", is_H_theta)
    is_H_random_phase_2 = kwargs.get("is_H_random_phase_2", is_H_random_phase)
    # %%
    w0_2 = kwargs.get("w0_2", w0)
    lam2 = kwargs.get("lam2", lam1) if is_HOPS == 0 else lam1
    is_air_pump2 = kwargs.get("is_air_pump2", is_air_pump)
    T2 = kwargs.get("T2", T)
    polar2 = kwargs.get("polar2", 'e')
    # %%
    if is_twin_pump == 1:
        # %%
        pump2_keys = kwargs["pump2_keys"]
        # %%
        [kwargs.pop(key) for key in kwargs["pump2_keys"]]  # 及时清理 kwargs ，尽量 保持 其干净
        kwargs.pop("pump2_keys")  # 这个有点意思， "pump2_keys" 这个键本身 也会被删除。

    # %%

    info = "NLAST_一拳超人"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # kwargs['ray'] = init_GLV_rmw(U_name, "^", "", "NLA", **kwargs)
    init_GLV_rmw(U_name, ray_tag, "NLA", "", **kwargs)

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

    if is_twin_pump == 1:
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

    if "U" in kwargs:  # 防止对 U_amp_plot_save 造成影响
        kwargs.pop("U")

    # %% 确定 公有参数

    args_init_AST = \
        [Ix, Iy, size_PerPixel,
         lam1, is_air, T,
         theta_x, theta_y, ]
    kwargs_init_AST = {"is_air_pump": is_air_pump, "gp": g_shift, }

    args_gan_args_SFG = \
        [z0, z0,
         mx, my, mz,
         Tx, Ty, Tz,
         is_contours, n_TzQ,
         Gz_max_Enhance, match_mode, ]

    def args_U_amp_plot_save(folder_address, U, U_name):
        return [U, U_name,
                [], folder_address,
                Get("img_name_extension"), is_save_txt,
                # %%
                size_PerPixel, dpi, Get("size_fig"),  # is_save = 1 - is_bulk 改为 不储存，因为 反正 都储存了
                # %%
                cmap_2d, ticks_num, is_contourf,
                is_title_on, is_axes_on, is_mm,
                fontsize, font,
                # %%
                is_colorbar_on, is_save,
                1, 0, 1, 0, ]  # 折射率分布差别很小，而 is_self_colorbar = 0 只看前 3 位小数的差异，因此用自动 colorbar。

    kwargs_U_amp_plot_save = {"suffix": ""}

    # %%

    g_p, p_p, g_V, g_H, p_V, p_H, \
    n1_inc, n1, k1_inc, k1, k1_z, k1_xy, E1_u, \
    n2_inc, n2, k2_inc, k2, k2_z, k2_xy, E2_u, \
    lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
    n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
    n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue, \
    n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
    n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
    n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
    n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He, \
    dk_z, lc, Gx, Gy, Gz, \
    z0, Tz, deff_structure_length_expect \
        = gan_gpnkE_123VHoe_xyzinc_SFG(is_birefringence_deduced, is_air,
                                       is_add_polarizer, is_HOPS,
                                       is_save, is_print,
                                       ray_tag, is_twin_pump, is_air_pump,
                                       lam2, theta2_x, theta2_y,
                                       g_shift, g2, U_0, U2_0, polar2,
                                       args_init_AST, args_gan_args_SFG,
                                       args_U_amp_plot_save,
                                       kwargs_init_AST, kwargs_U_amp_plot_save,
                                       is_plot_n=1, **kwargs)

    # if is_fft != 0 and fft_mode == 0:  # 之后 总要传入 一些参数，所以 无论如何 都要 赋值
    # %% generate structure
    if is_twin_pump == 1:
        for key in pump2_keys:
            kwargs[key] = locals()[key]
            kwargs["pump2_keys"] = locals()["pump2_keys"]
    folder_address, size_PerPixel, U_0_structure, g_shift_structure, \
    g_p, p_p, g_V, g_H, p_V, p_H, \
    n1_inc, n1, k1_inc, k1, k1_z, k1_xy, E1_u, \
    n2_inc, n2, k2_inc, k2, k2_z, k2_xy, E2_u, \
    lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u, \
    n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
    n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue, \
    n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
    n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
    n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
    n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He, \
    dk_z, lc, Gx, Gy, Gz, z0, Tz, deff_structure_length_expect, \
    structure, structure_opposite, \
    modulation, modulation_opposite, \
    modulation_squared, modulation_opposite_squared \
        = chi2_2D(U_name_Structure,
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
                                     L0_Crystal=z0, g1=g_shift, g2=g2,
                                     # %%
                                     is_air_pump=is_air_pump, **kwargs, )
    if is_twin_pump == 1:
        [kwargs.pop(key) for key in kwargs["pump2_keys"]]  # 及时清理 kwargs ，尽量 保持 其干净
        kwargs.pop("pump2_keys")  # 这个有点意思， "pump2_keys" 这个键本身 也会被删除。

    # %%

    iz = z0 / size_PerPixel
    dset("G", np.zeros((Ix, Iy), dtype=np.complex128()))

    U_o, U_e, U_Vo, U_Ve, U_Ho, U_He = \
        gan_U_VHoe(g_o, g_e, g_Vo, g_Ho, g_Ve, g_He)

    is_NLAST_sum = kwargs.get("is_NLAST_sum", 0)
    match_type = kwargs.get("match_type", "oe")
    NLA_123VHoe(is_birefringence_deduced, is_air,
                is_add_polarizer, match_type,
                iz, is_fft, fft_mode,
                Ix, Iy, size_PerPixel,
                k3_inc, n3_inc,
                k1_z, k1_xy,
                k1o_z, k1o_xy, k1e_z, k1e_xy,
                k1_Vo_z, k1_Vo_xy, k1_Ho_z, k1_Ho_xy,
                k1_Ve_z, k1_Ve_xy, k1_He_z, k1_He_xy,
                k3_z, k3_xy,
                k1o, k1e, k1_Vo, k1_Ve, k1_Ho, k1_He,
                k1, k2, k3,
                mx, my, mz,
                Gx, Gy, Gz,
                Tx, Tz, deff,
                mG, is_sum_Gm, is_NLAST_sum,
                g_shift, g2,
                g_o, g_e, g_Vo, g_Ve, g_Ho, g_He,
                U_0, U2_0,
                U_o, U_e, U_Vo, U_Ve, U_Ho, U_He,
                modulation_squared,
                is_print, is_linear_convolution, )

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
                   z0, is_end=1, **kwargs, )

    import inspect
    if inspect.stack()[1][3] == "SFG_NLA_reverse":
        from fun_statistics import find_Kxyz
        from fun_linear import fft2
        K1_z, K1_xy = find_Kxyz(fft2(U_0), k1)
        K2_z, K2_xy = find_Kxyz(fft2(U2_0), k2)
        kiizQ = K1_z + K2_z + Gz
        # print(np.max(np.abs(fft2(fget("U")) / Get("size_PerPixel") ** 2)))

        Const = (k3_inc / size_PerPixel / n3_inc) ** 2 * deff * 1e-12  # pm / V 转换成 m / V
        return fget("U"), U_0, U2_0, modulation_squared, k1_inc, k2_inc, \
               theta_x, theta_y, theta2_x, theta2_y, kiizQ, \
               k1, k2, k3, Const, iz, Gz
    else:
        return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U")


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",  # 要么从 U_name 里传 ray 和 U 进来，要么 单独传个 U 和 ray
         "img_full_name": "lena1.png",
         "U_pixels_x": 300, "U_pixels_y": 300,
         "is_phase_only": 0,
         # %%
         "z_pump": 0,
         "is_LG": 1, "is_Gauss": 1, "is_OAM": 1,
         "l": -50, "p": 0,
         "theta_x": 0, "theta_y": 0,
         # %%
         "is_random_phase": 0,
         "is_H_l": 0, "is_H_theta": 0, "is_H_random_phase": 0,
         # %%
         # 生成横向结构
         "U_name_Structure": '',
         "structure_size_Shrink": 0, "structure_size_Shrinker": 0,
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
         "z0": 10,
         # %%
         "lam1": 1.064, "is_air_pump": 1, "is_air": 2, "T": 25,
         "lam_structure": 1.064, "is_air_pump_structure": 1, "T_structure": 25,
         # %%  控制 单双泵浦 和 绘图方式：0 代表 无双折射 "is_birefringence_SHG": 0 是否 考虑 双折射
         "is_HOPS_SHG": 0,  # 0.x 代表 单泵浦，1.x 代表 高阶庞加莱球，2.x 代表 最广义情况：2 个 线偏 标量场 叠加；这些都是在 左手系下，且都是 线偏基
         "Theta": 0, "Phi": 0,  # 是否 采用 高阶加莱球、若采用，请给出 极角 和 方位角
         # 是否 使用 起偏器（"is_HOPS": 整数 即不使用）、若使用，请给出 其相对于 H (水平 x) 方向的 逆时针 转角 phi_p
         "phi_p": "45", "phi_a": "45",  # 是否 使用 检偏器（"phi_a": str 则不使用）、若使用，请给出 其相对于 H (水平 x) 方向的 逆时针 转角 phi_a
         # %%
         "deff": 30, "is_fft": 1, "fft_mode": 0,
         "is_sum_Gm": 0, "mG": 0, 'is_NLAST_sum': 0,
         "is_linear_convolution": 0,
         # %%
         "Tx": 20, "Ty": 30, "Tz": 3,
         "mx": 0, "my": 0, "mz": 0,
         # %%
         # 生成横向结构
         "Duty_Cycle_x": 0.5, "Duty_Cycle_y": 0.5, "Duty_Cycle_z": 0.5,
         "Depth": 2, "structure_xy_mode": 'x',
         # %%
         "is_continuous": 0, "is_target_far_field": 1, "is_transverse_xy": 0,
         "is_reverse_xy": 0, "is_positive_xy": 1, "is_no_backgroud": 0,
         # %%
         "is_save": 0, "is_no_data_save": 0,
         "is_save_txt": 0, "dpi": 100,
         # %%
         "cmap_2d": 'viridis',
         # %%
         "ticks_num": 6, "is_contourf": 0,
         "is_title_on": 1, "is_axes_on": 1, "is_mm": 1,
         # %%
         "fontsize": 10.0,
         "font": {'family': 'serif',
                  'style': 'normal',  # 'normal', 'italic', 'oblique'
                  'weight': 'normal',
                  'color': 'black',  # 'black','gray','darkred'
                  },
         # %%
         "is_colorbar_on": 1, "is_energy": 1,
         # %%
         "is_print": 1, "is_contours": 66, "n_TzQ": 1,
         "Gz_max_Enhance": 1, "match_mode": 1,
         # %% 该程序 作为 主入口时 -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "border_percentage": 0.1, "is_end": -1,
         # %%
         "theta_z": 90, "phi_z": 90, "phi_c": 23.7,
         # KTP 50 度 ：deff 最高： 90, ~, 24.3，（24.3 - 2002, 25.3 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.9 - 2000）
         # KTP 25 度 ：deff 最高： 90, ~, 23.7，（23.7 - 2002, 24.8 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.7 - 2000）
         # LN 25 度 ：90, ~, ~
         "polar": "o", "match_type": "oe",
         "polar3": "o", "ray": "3",
         }

    if kwargs.get("ray", "2") == "3" or kwargs.get("is_HOPS_SHG", 0) >= 1:  # 如果 is_HOPS >= 1，则 默认 双泵浦 is_twin_pumps == 1
        pump2_kwargs = {
            "U2_name": "",
            "img2_full_name": "spaceship.png",
            "is_phase_only_2": 0,
            # %%
            "z_pump2": 0,
            "is_LG_2": 1, "is_Gauss_2": 1, "is_OAM_2": 1,
            "l2": 50, "p2": 0,
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

    # 要不要把 “要不要（是否）使用 上一次使用 的 参数”
    # or “这次 使用了 哪一次 使用的 参数” 传进去记录呢？—— 也不是不行。

    # kwargs.update(init_GLV_DICT(**kwargs))
    kwargs = init_GLV_DICT(**kwargs)
    SFG_NLA(**kwargs)
