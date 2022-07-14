# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
from fun_img_Resize import if_image_Add_black_border
from fun_pump import pump_pic_or_U
from fun_linear import ifft2
from fun_thread import my_thread
from fun_CGH import structure_chi2_Generate_2D
from fun_global_var import init_GLV_DICT, tree_print, Set, Get, init_GLV_rmw, init_EVV, Fun3, end_SSI, \
    dset, dget, fget, fkey, fGHU_plot_save, fU_EVV_plot
from b_1_AST import init_locals
from b_1_AST_EVV import H_zdz, gan_iz_diz
from b_3_SFG_NLA import gan_gpnkE_123VHoe_xyzinc_SFG, NLA_123VHoe, gan_U_VHoe

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
    is_HOPS = kwargs.get("is_HOPS_SHG", 0)
    is_twin_pump_degenerate = int(is_HOPS >= 1)  # is_HOPS == 0.x 的情况 仍是单泵浦
    is_single_pump_birefringence = int(is_HOPS > 0 and is_HOPS < 1)
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
        return [folder_address,
                # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
                U, U_name,
                Get("img_name_extension"),
                is_save_txt,
                # %%
                [], 1, size_PerPixel,
                0, dpi, Get("size_fig"),  # is_save = 1 - is_bulk 改为 不储存，因为 反正 都储存了
                # %%
                cmap_2d, ticks_num, is_contourf,
                is_title_on, is_axes_on, is_mm, 0,  # 1, 1 或 0, 0
                fontsize, font,
                # %%
                1, is_colorbar_on, 0, ]  # 折射率分布差别很小，而 is_self_colorbar = 0 只看前 3 位小数的差异，因此用自动 colorbar。

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
                                     L0_Crystal=z0, g1=g_shift, g2=g2,
                                     # %%
                                     is_air_pump=is_air_pump, **kwargs, )
    if is_twin_pump == 1:
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
        izj_delay_dz.pop(-1)  # 可 pop 可不 pop 掉 最后一个元素，反正没啥用
        # dizj = [izj[0] - 0] + dizj
        # Set("is_EVV_SSI", 1)
        # print(izj_delay_dz)
        # print(dizj)
    else:
        izj_delay_dz = 0  # 无论如何还是要赋值的，因为之后要用
        dizj = 0
    Set("zj", zj)
    Set("izj", izj)

    sheets_stored_num = len(zj) - 1
    init_EVV(g_shift, U_0,
             is_energy_evolution_on, is_stored,
             sheets_stored_num, sheets_stored_num,
             X, Y, iz, size_PerPixel, )

    # %%

    U_o, U_e, U_Vo, U_Ve, U_Ho, U_He = \
        gan_U_VHoe(g_o, g_e, g_Vo, g_Ho, g_Ve, g_He)  # 得写在外面，否则会传进 Fun 等的 kwargs...而这一般是空的

    if is_birefringence_deduced == 1 and is_air != 1:
        # %%

        if is_add_polarizer == 1:

            def gan_g_oe(for_th2):
                diz, iz = gan_iz_diz(is_EVV_SSI, for_th2,
                                     izj_delay_dz, dizj, izj)
                if is_EVV_SSI == 1:
                    G1o_z = g_o * H_zdz(k1o_z, iz)
                    G1e_z = g_e * H_zdz(k1e_z, iz)
                    from fun_linear import fft2
                    U1o_z, U1e_z = fft2(G1o_z), fft2(G1e_z)
                else:
                    G1o_z = g_o
                    G1e_z = g_e
                    U1o_z = U_o
                    U1e_z = U_e
                return G1o_z, G1e_z, diz, \
                       U1o_z, U1e_z
        else:

            def gan_g_VHoe(for_th2):
                diz, iz = gan_iz_diz(is_EVV_SSI, for_th2,
                                     izj_delay_dz, dizj, izj)
                if is_EVV_SSI == 1:
                    G1_Vo_z = g_Vo * H_zdz(k1_Vo_z, iz)
                    G1_Ve_z = g_Ve * H_zdz(k1_Ve_z, iz)
                    G1_Ho_z = g_Ho * H_zdz(k1_Ho_z, iz)
                    G1_He_z = g_He * H_zdz(k1_He_z, iz)
                    from fun_linear import fft2
                    U1_Vo_z, U1_Ve_z = fft2(G1_Vo_z), fft2(G1_Ve_z)
                    U1_Ho_z, U1_He_z = fft2(G1_Ho_z), fft2(G1_He_z)
                else:
                    G1_Vo_z = g_Vo
                    G1_Ve_z = g_Ve
                    G1_Ho_z = g_Ho
                    G1_He_z = g_He
                    U1_Vo_z = U_Vo
                    U1_Ve_z = U_Ve
                    U1_Ho_z = U_Ho
                    U1_He_z = U_He
                return G1_Vo_z, G1_Ve_z, G1_Ho_z, G1_He_z, diz, \
                       U1_Vo_z, U1_Ve_z, U1_Ho_z, U1_He_z

    else:

        def gan_g12(for_th2):
            diz, iz = gan_iz_diz(is_EVV_SSI, for_th2,
                                 izj_delay_dz, dizj, izj)
            if is_EVV_SSI == 1:
                G1_z = g_shift * H_zdz(k1_z, iz)
                U_z = ifft2(G1_z)

                if is_twin_pump == 1:
                    G2_z = g2 * H_zdz(k2_z, iz)
                    U2_z = ifft2(G2_z)
                else:
                    G2_z = G1_z
                    U2_z = U_z
            else:
                G1_z = g_shift
                U_z = U_0
                G2_z = g2
                U2_z = U2_0
            # print(iz)
            # print(diz)
            return G1_z, G2_z, diz, \
                   U_z, U2_z

    is_NLAST_sum = kwargs.get("is_NLAST_sum", 0)
    match_type = kwargs.get("match_type", "oe")

    def Fun1(for_th2, fors_num2, *args, **kwargs, ):

        Set("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"),
            np.zeros((Ix, Iy), dtype=np.complex128()))
        # 加上 Fun3 中的 sset 其实共 储存了 2 次 这些 G3_zm 层（m = for_th2），有点浪费
        # 但这里的又是必须存在的，因为之后得用它来在 fun2 里累加。。
        # 所以最多也是之后的 Fun3 中的 sset 可以不必存在

        G1_z, G2_z, \
        G1o_z, G1e_z, G1_Vo_z, G1_Ve_z, G1_Ho_z, G1_He_z, \
        U1_z, U2_z, \
        U1o_z, U1e_z, U1_Vo_z, U1_Ve_z, U1_Ho_z, U1_He_z = \
            init_locals("G1_z, G2_z, \
                        G1o_z, G1e_z, G1_Vo_z, G1_Ve_z, G1_Ho_z, G1_He_z, \
                        U1_z, U2_z, \
                        U1o_z, U1e_z, U1_Vo_z, U1_Ve_z, U1_Ho_z, U1_He_z")

        if is_birefringence_deduced == 1 and is_air != 1:
            if is_add_polarizer == 1:
                G1o_z, G1e_z, diz, \
                U1o_z, U1e_z = gan_g_oe(for_th2)
            else:
                G1_Vo_z, G1_Ve_z, G1_Ho_z, G1_He_z, diz, \
                U1_Vo_z, U1_Ve_z, U1_Ho_z, U1_He_z = gan_g_VHoe(for_th2)
        else:
            G1_z, G2_z, diz, \
            U1_z, U2_z = gan_g12(for_th2)

        # for_th2 == 0 时也要算，因为 zj[0] 不一定是 0：外部可能传入 zj
        NLA_123VHoe(is_birefringence_deduced, is_air,
                    is_add_polarizer, match_type,
                    diz, is_fft, fft_mode,
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
                    G1_z, G2_z,
                    G1o_z, G1e_z, G1_Vo_z, G1_Ve_z, G1_Ho_z, G1_He_z,
                    U1_z, U2_z,
                    U1o_z, U1e_z, U1_Vo_z, U1_Ve_z, U1_Ho_z, U1_He_z,
                    modulation_squared,
                    is_print, is_linear_convolution,
                    for_th2=for_th2, )

        return Get("G" + Get("ray") + "_z" + str(for_th2) + "_" + Get("way"))

    def Fun2(for_th2, fors_num, G3_zm, *args, **kwargs, ):

        # print(dizj[for_th2])
        # print(np.sum(np.abs(dget("G"))))
        if is_EVV_SSI == 1:
            dset("G", dget("G") * H_zdz(k3_z, dizj[for_th2]) + G3_zm)  # is_EVV_SSI == 1 时， 此时 G3_zm 是 dG3_zdz
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
         "z0": 10, "sheets_stored_num": 10,
         # %%
         "lam1": 1.064, "is_air_pump": 1, "is_air": 2, "T": 25,
         "lam_structure": 1.064, "is_air_pump_structure": 1, "T_structure": 25,
         # %%  控制 单双泵浦 和 绘图方式：0 代表 无双折射 "is_birefringence_SHG": 0 是否 考虑 双折射
         "is_HOPS_SHG": 0,  # 0.x 代表 单泵浦，1 代表 高阶庞加莱球，2 代表 最广义情况：2 个 线偏 标量场 叠加；这些都是在 左手系下，且都是 线偏基
         "Theta": 0, "Phi": 0,  # 是否 采用 高阶加莱球、若采用，请给出 极角 和 方位角
         # 是否 使用 起偏器（0 即不使用）、若使用，请给出 其相对于 H (水平 x) 方向的 逆时针 转角 phi_p
         "phi_p": "45", "phi_a": "45",  # 是否 使用 检偏器、若使用，请给出 其相对于 H (水平 x) 方向的 逆时针 转角 phi_a
         # %%
         "deff": 30, "is_fft": 1, "fft_mode": 0,
         "is_sum_Gm": 0, "mG": 0, 'is_NLAST_sum': 0,
         "is_linear_convolution": 0,
         # %%
         "Tx": 13, "Ty": 20, "Tz": 10,
         "mx": 0, "my": 0, "mz": 0,
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
         "is_plot_EVV": 1, "is_plot_3d_XYz": 0, "is_plot_selective": 0,
         "X": 0, "Y": 0, "is_plot_YZ_XZ": 0, "is_plot_3d_XYZ": 0,
         # %%
         "plot_group": "Ua", "is_animated": 1,
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
         "theta_z": 90, "phi_z": 90, "phi_c": 23.8,
         # KTP 50 度 ：deff 最高： 90, ~, 24.3，（24.3 - 2002, 25.3 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.9 - 2000）
         # KTP 25 度 ：deff 最高： 90, ~, 23.7，（23.7 - 2002, 24.8 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.7 - 2000）
         # LN 25 度 ：90, ~, ~
         "polar": "R", "match_type": "oe",
         "polar3": "o", "ray": "3",
         }

    if kwargs.get("ray", "2") == "3" or kwargs.get("is_HOPS_SHG", 0) > 0:  # 如果 ray == 3，则 默认 双泵浦 is_twin_pumps == 1
        pump2_kwargs = {
            "U2_name": "",
            "img2_full_name": "lena1.png",
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
            "polar2": 'L',
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
