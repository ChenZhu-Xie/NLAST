# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
from fun_os import U_dir
from fun_img_Resize import if_image_Add_black_border
from fun_pump import pump_pic_or_U
from fun_SSI import slice_ssi
from fun_linear import fft2, ifft2
from fun_nonlinear import Eikz
from fun_thread import my_thread
from fun_global_var import init_GLV_DICT, tree_print, init_GLV_rmw, init_SSI, end_SSI, Get, dset, dget, fun3, \
    fget, fkey, fGHU_plot_save, fU_SSI_plot
from a2_AST_EVV import H_zdz
from c1_SFG_NLA import gan_gpnkE_123VHoe_xyzinc_SFG
from C1x_SFG_NLA_ssi import gan_modulation_squared_z_ssi, gan_U_12VHoe_iterate, gan_U_12VHoe_z_iterate

np.seterr(divide='ignore', invalid='ignore')


# %%

def H3_z(cal_mode, k3, diz, size_PerPixel,
         k1_z, k2_z, k3_z, dk_z, ):
    if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版
        dk_z = k1_z + k2_z - k3_z
        return H_zdz(k3_z, diz) / (k3_z / size_PerPixel) * \
               Eikz(dk_z * diz) * diz * size_PerPixel * \
               (2 / (dk_z / k3_z + 2))
    else:
        return H_zdz(k3, diz) / (k3 / size_PerPixel) * \
               Eikz(dk_z * diz) * diz * size_PerPixel * \
               (2 / (dk_z / k3 + 2))


# %%

def SSF_ssi(diz, cal_mode, size_PerPixel,
            k1_z, k2_z, k3_z, k3, dk_z,
            U_z, U2_z, const,
            modulation_squared_z, ):
    def args_H3_z(diz):
        return [cal_mode, diz, size_PerPixel,
                k1_z, k2_z, k3_z, k3, dk_z, ]

    if cal_mode[0] == 1:  # 如果以 G 算

        if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版
            if cal_mode[1] == 1:  # 若 源项 也衍射
                Q3_z = fft2(modulation_squared_z * U_z * U2_z * H3_z(*args_H3_z(diz)) / H_zdz(k3_z, diz))
            else:
                Q3_z = fft2(modulation_squared_z * U_z * U2_z * H3_z(*args_H3_z(diz)))
        else:
            Q3_z = fft2(modulation_squared_z * U_z * U2_z)

        if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版
            dG3_zdz = const * Q3_z
        else:
            if cal_mode[1] == 1:  # 若 源项 也衍射
                dG3_zdz = const * Q3_z * H3_z(*args_H3_z(diz)) / H_zdz(k3, diz)
            else:
                dG3_zdz = const * Q3_z * H3_z(*args_H3_z(diz))

        return dG3_zdz

    else:

        S3_z = modulation_squared_z * U_z * U2_z

        if cal_mode[1] == 1:  # 若 源项 也衍射
            if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版
                dU3_zdz = const * S3_z * H3_z(*args_H3_z(diz)) / H_zdz(k3_z, diz)
            else:
                dU3_zdz = const * S3_z * H3_z(*args_H3_z(diz)) / H_zdz(k3, diz)
        else:
            dU3_zdz = const * S3_z * H3_z(*args_H3_z(diz))

        return dU3_zdz


# %%

def SSF_iterate_123VHoe(is_birefringence_deduced, is_air,
                        is_add_polarizer, cal_mode,
                        diz, size_PerPixel,
                        const, match_type,
                        k1o_z, k1e_z, k1_Vo_z, k1_Ve_z, k1_Ho_z, k1_He_z,
                        k1_z, k2_z, k3_z, k3, dk_z,
                        U_z, U2_z,
                        U_o, U_e, U_Vo, U_Ve, U_Ho, U_He,
                        modulation_squared_z, ):
    def gan_args_SSF_ssi(k1_z, k2_z,
                         U_z, U2_z, ):
        return [diz, cal_mode, size_PerPixel,
                k1_z, k2_z, k3_z, k3, dk_z,
                U_z, U2_z, const,
                modulation_squared_z, ]  # dk_z 本身与 k1_inc, k2_inc, theta_x, theta_y 有关而需要 作为参数 传入 和 改变的
        # 但 VHoe 的 情况 实在太复杂，以至于 甚至每个 k3_inc、k3_inc_z 都不同，要单独算完各个 dk_z 是很难的，计算量太大。
        # 而且这个 SSF 本来就是错的，搞它干嘛。

    if is_birefringence_deduced == 1:  # 考虑 偏振态 的 条件；is_air == 1 时 也可以 有偏振态，与是否 所处介质 无关
        if is_add_polarizer == 1:
            if match_type == "oe" or match_type == "eo":
                dG3_zdz = SSF_ssi(*gan_args_SSF_ssi(k1o_z, k1e_z,
                                                    U_o, U_e, ), )
            elif match_type == "oo":
                dG3_zdz = SSF_ssi(*gan_args_SSF_ssi(k1o_z, k1o_z,
                                                    U_o, U_o, ), )
            elif match_type == "ee":
                dG3_zdz = SSF_ssi(*gan_args_SSF_ssi(k1e_z, k1e_z,
                                                    U_e, U_e, ), )
        else:
            if match_type == "oe" or match_type == "eo":
                # 组内 和频
                dG3_zdz_VoVe = SSF_ssi(*gan_args_SSF_ssi(k1_Vo_z, k1_Ve_z,
                                                         U_Vo, U_Ve, ), )

                dG3_zdz_HoHe = SSF_ssi(*gan_args_SSF_ssi(k1_Ho_z, k1_He_z,
                                                         U_Ho, U_He, ), )

                # 组间 和频
                dG3_zdz_VoHe = SSF_ssi(*gan_args_SSF_ssi(k1_Vo_z, k1_He_z,
                                                         U_Vo, U_He, ), )

                dG3_zdz_HoVe = SSF_ssi(*gan_args_SSF_ssi(k1_Ho_z, k1_Ve_z,
                                                         U_Ho, U_Ve, ), )

                dG3_zdz = dG3_zdz_VoVe + dG3_zdz_HoHe + dG3_zdz_VoHe + dG3_zdz_HoVe
            elif match_type == "oo":
                # 组内 和频
                dG3_zdz_VoVo = SSF_ssi(*gan_args_SSF_ssi(k1_Vo_z, k1_Vo_z,
                                                         U_Vo, U_Vo, ), )

                dG3_zdz_HoHo = SSF_ssi(*gan_args_SSF_ssi(k1_Ho_z, k1_Ho_z,
                                                         U_Ho, U_Ho, ), )

                # 组间 和频
                dG3_zdz_VoHo = SSF_ssi(*gan_args_SSF_ssi(k1_Vo_z, k1_Ho_z,
                                                         U_Vo, U_Ho, ), )

                dG3_zdz = dG3_zdz_VoVo + dG3_zdz_HoHo + dG3_zdz_VoHo
            elif match_type == "ee":
                # 组内 和频
                dG3_zdz_VeVe = SSF_ssi(*gan_args_SSF_ssi(k1_Ve_z, k1_Ve_z,
                                                         U_Ve, U_Ve, ), )

                dG3_zdz_HeHe = SSF_ssi(*gan_args_SSF_ssi(k1_He_z, k1_He_z,
                                                         U_He, U_He, ), )

                # 组间 和频
                dG3_zdz_VeHe = SSF_ssi(*gan_args_SSF_ssi(k1_Ve_z, k1_He_z,
                                                         U_Ve, U_He, ), )

                dG3_zdz = dG3_zdz_VeVe + dG3_zdz_HeHe + dG3_zdz_VeHe
    else:
        dG3_zdz = SSF_ssi(*gan_args_SSF_ssi(k1_z, k2_z,
                                            U_z, U2_z, ), )
    return dG3_zdz


# %%

def SSF_fun2(cal_mode, dG3_zdz,
             k3_z, diz):
    if cal_mode[0] == 1:  # 如果以 G 算

        if cal_mode[1] == 1:  # 若 源项 也衍射
            dset("G", (dget("G") + dG3_zdz) * H_zdz(k3_z, diz))
        else:
            dset("G", dget("G") * H_zdz(k3_z, diz) + dG3_zdz)

        return dget("G")

    else:

        dU3_zdz = dG3_zdz

        if cal_mode[1] == 1:  # 若 源项 也衍射
            dset("U", ifft2(fft2(dget("U") + dU3_zdz) * H_zdz(k3_z, diz)))
        else:
            dset("U", ifft2(fft2(dget("U")) * H_zdz(k3_z, diz)) + dU3_zdz)

        return dget("U")


# %%

def SFG_SSF_ssi(U_name="",
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
                L0_Crystal=5, z0_structure_frontface_expect=0.5, deff_structure_length_expect=2,
                Duty_Cycle_z=0.5, ssi_zoomout_times=5, sheets_stored_num=10,
                z0_section_1_expect=1, z0_section_2_expect=1,
                X=0, Y=0,
                # %%
                is_bulk=1, is_no_backgroud=0,
                is_stored=0, is_show_structure_face=1, is_energy_evolution_on=1,
                # %%
                lam1=0.8, is_air_pump=0, is_air=0, T=25,
                deff=30,
                # %%
                Tx=10, Ty=10, Tz="2*lc",
                mx=0, my=0, mz=0,
                is_NLAST=0,
                # %%
                is_save=0, is_save_txt=0, dpi=100,
                # %%
                color_1d='b', cmap_2d='viridis', cmap_3d='rainbow',
                elev=10, azim=-65, alpha=2,
                # %%
                sample=1, ticks_num=6, is_contourf=0,
                is_title_on=1, is_axes_on=1,
                is_mm=1,
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
                plot_group="UGa", is_animated=1,
                loop=0, duration=0.033, fps=5,
                # %%
                is_plot_EVV=1, is_plot_3d_XYz=0, is_plot_selective=0,
                is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
                # %%
                is_print=1, is_contours=1, n_TzQ=1,
                Gz_max_Enhance=1, match_mode=1,
                # %%
                **kwargs, ):
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

    info = "SSF_小步长_ssi"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # kwargs['ray'] = init_GLV_rmw(U_name, "^", "SSI", "Sfm", **kwargs)
    init_GLV_rmw(U_name, ray_tag, "SSF", "ssi", **kwargs)

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
        [L0_Crystal, deff_structure_length_expect,
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

    # %%

    diz, deff_structure_sheet, \
    sheet_th_frontface, sheets_num_frontface, Iz_frontface, z0_front, \
    sheets_num_structure, Iz_structure, deff_structure_length, \
    sheets_num, Iz, z0, \
    dizj, izj, zj, \
    sheet_th_endface, sheets_num_endface, Iz_endface, z0_end, \
    sheet_th_sec1, sheets_num_sec1, iz_1, z0_1, \
    sheet_th_sec2, sheets_num_sec2, iz_2, z0_2 \
        = slice_ssi(L0_Crystal, Duty_Cycle_z,
                    z0_structure_frontface_expect, deff_structure_length_expect,
                    z0_section_1_expect, z0_section_2_expect,
                    Tz, ssi_zoomout_times, size_PerPixel,
                    is_print, )

    # %%
    # const

    const = (k3_inc / size_PerPixel / n3_inc) ** 2 * deff * 1e-12  # pm / V 转换成 m / V

    # %%
    # G3_z0_shift

    method = "MOD"
    folder_name = method + " - " + "χ2_modulation_squared"
    folder_address = U_dir(folder_name, 1 - is_bulk, )

    init_SSI(g_shift, U_0,
             is_energy_evolution_on, is_stored,
             sheets_num, sheets_stored_num,
             X, Y, Iz, size_PerPixel, )

    cal_mode = [1, 1, 0]

    # 以 G 算 还是以 U 算、源项 是否 也衍射、k_2z 是否是 matrix 版
    # 用 G 算 会快很多
    # 不管是 G 还是 U，matrix 版的能量 总是要低一些，只不过 U 低得少些，没有数量级差异，而 G 少得很多

    # %%

    gan_U_oe, gan_U_VHoe, gan_U12 = \
        gan_U_12VHoe_iterate(is_birefringence_deduced, is_air,
                             is_twin_pump, is_add_polarizer, izj,
                             g_shift, g2, g_o, g_e,
                             g_Vo, g_Ve, g_Ho, g_He,
                             k1_z, k2_z, k1o_z, k1e_z,
                             k1_Vo_z, k1_Ve_z, k1_Ho_z, k1_He_z, )

    # %%

    match_type = kwargs.get("match_type", "oe")

    # %%

    def fun1(for_th, fors_num, *args, **kwargs, ):
        modulation_squared_z = \
            gan_modulation_squared_z_ssi(is_bulk, for_th, Ix, Iy,
                                         sheets_num_endface, sheets_num_frontface,
                                         is_no_backgroud, folder_address, is_save_txt, )

        U1_z, U2_z, U1o_z, U1e_z, \
        U1_Vo_z, U1_Ve_z, U1_Ho_z, U1_He_z = \
            gan_U_12VHoe_z_iterate(is_birefringence_deduced, is_air,
                                   for_th, is_add_polarizer,
                                   gan_U_oe, gan_U_VHoe, gan_U12, )

        dG3_zdz = SSF_iterate_123VHoe(is_birefringence_deduced, is_air,
                                      is_add_polarizer, cal_mode,
                                      dizj[for_th], size_PerPixel,
                                      const, match_type,
                                      k1o_z, k1e_z, k1_Vo_z, k1_Ve_z, k1_Ho_z, k1_He_z,
                                      k1_z, k2_z, k3_z, k3, dk_z,
                                      U1_z, U2_z,
                                      U1o_z, U1e_z, U1_Vo_z, U1_Ve_z, U1_Ho_z, U1_He_z,
                                      modulation_squared_z, )
        return dG3_zdz

    def fun2(for_th, fors_num, dG3_zdz, *args, **kwargs, ):

        G3_zdz = SSF_fun2(cal_mode, dG3_zdz,
                          k3_z, dizj[for_th])

        return G3_zdz

    # %%

    is_U = 0 if cal_mode[0] == 1 else 1  # 如以 G 算，则 is_U = 0

    my_thread(10, sheets_num,
              fun1, fun2, fun3,
              is_ordered=1, is_print=is_print,
              is_U=is_U, )

    # %%

    end_SSI(g_shift, is_energy, n_sigma=3,
            is_U=is_U, )

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

    fU_SSI_plot(sheets_num_frontface, sheets_num_endface,
                img_name_extension,
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
                is_energy, is_show_structure_face,
                # %%
                plot_group, is_animated,
                loop, duration, fps,
                # %%
                is_plot_EVV, is_plot_3d_XYz, is_plot_selective,
                is_plot_YZ_XZ, is_plot_3d_XYZ,
                # %%
                z0_1, z0_2,
                z0_front, z0_end, z0,
                # %%
                **kwargs, )

    return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U")


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",
         "img_full_name": "Grating.png",
         "U_pixels_x": 300, "U_pixels_y": 300,
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
         "U_size": 1, "w0": 0.3,
         "L0_Crystal": 1, "z0_structure_frontface_expect": 0.5, "deff_structure_length_expect": 2,
         "Duty_Cycle_z": 0.5, "ssi_zoomout_times": 5, "sheets_stored_num": 10,
         "z0_section_1_expect": 1, "z0_section_2_expect": 1,
         "X": 0, "Y": 0,
         # %%
         "is_bulk": 1, "is_no_backgroud": 0,
         "is_stored": 0, "is_show_structure_face": 1, "is_energy_evolution_on": 1,
         # %%
         "lam1": 0.8, "is_air_pump": 1, "is_air": 0, "T": 25,
         "deff": 30,
         # %%  控制 单双泵浦 和 绘图方式：0 代表 无双折射 "is_birefringence_SHG": 0 是否 考虑 双折射
         "is_HOPS_SHG": 0,  # 0.x 代表 单泵浦，1.x 代表 高阶庞加莱球，2.x 代表 最广义情况：2 个 线偏 标量场 叠加；这些都是在 左手系下，且都是 线偏基
         "Theta": 0, "Phi": 0,  # 是否 采用 高阶加莱球、若采用，请给出 极角 和 方位角
         # 是否 使用 起偏器（"is_HOPS": 整数 即不使用）、若使用，请给出 其相对于 H (水平 x) 方向的 逆时针 转角 phi_p
         "phi_p": "45", "phi_a": "45",  # 是否 使用 检偏器（"phi_a": str 则不使用）、若使用，请给出 其相对于 H (水平 x) 方向的 逆时针 转角 phi_a
         # %%
         "Tx": 10, "Ty": 10, "Tz": "2*lc",
         "mx": 0, "my": 0, "mz": 0,
         "is_NLAST": 0,
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
         "fontsize": 10.0,
         "font": {'family': 'serif',
                  'style': 'normal',  # 'normal', 'italic', 'oblique'
                  'weight': 'normal',
                  'color': 'black',  # 'black','gray','darkred'
                  },
         # %%
         "is_colorbar_on": 1, "is_colorbar_log": 0,
         "is_energy": 0,
         # %%
         "plot_group": "UGa", "is_animated": 1,
         "loop": 0, "duration": 0.033, "fps": 5,
         # %%
         "is_plot_EVV": 1, "is_plot_3d_XYz": 0, "is_plot_selective": 0,
         "is_plot_YZ_XZ": 1, "is_plot_3d_XYZ": 0,
         # %%
         "is_print": 1, "is_contours": 0, "n_TzQ": 1,
         "Gz_max_Enhance": 1, "match_mode": 1,
         # %% 该程序 作为 主入口时 -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "border_percentage": 0.1, "is_end": -1,
         # %%
         "size_fig_x_scale": 10, "size_fig_y_scale": 2,
         # %%
         "theta_z": 90, "phi_z": 90, "phi_c": 23.7,
         # KTP 50 度 ：deff 最高： 90, ~, 24.3，（24.3 - 2002, 25.3 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.9 - 2000）
         # KTP 25 度 ：deff 最高： 90, ~, 23.7，（23.7 - 2002, 24.8 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.7 - 2000）
         # LN 25 度 ：90, ~, ~
         "polar": "e", "match_type": "oe",
         "polar3": "e", "ray": "2",
         }

    if kwargs.get("ray", "2") == "3" or kwargs.get("is_HOPS_SHG", 0) >= 1:  # 如果 is_HOPS >= 1，则 默认 双泵浦 is_twin_pumps == 1
        pump2_kwargs = {
            "U2_name": "",
            "img2_full_name": "lena.png",
            "is_phase_only_2": 0,
            # %%
            "z_pump2": 0,
            "is_LG_2": 0, "is_Gauss_2": 1, "is_OAM_2": 0,
            "l2": 0, "p2": 0,
            "theta2_x": 0, "theta2_y": 0,
            # %%
            "is_random_phase_2": 0,
            "is_H_l2": 0, "is_H_theta2": 0, "is_H_random_phase_2": 0,
            # %%
            "w0_2": 0.3,
            # %%
            "lam2": 1, "is_air_pump2": 1, "T2": 25,
            "polar2": 'e',
        }
        pump2_kwargs.update({"pump2_keys": list(pump2_kwargs.keys())})
        # Object of type dict_keys is not JSON serializable，所以 得转为 list
        kwargs.update(pump2_kwargs)

    kwargs = init_GLV_DICT(**kwargs)
    SFG_SSF_ssi(**kwargs)
