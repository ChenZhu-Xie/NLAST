# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import math
import numpy as np
from fun_img_Resize import if_image_Add_black_border
from fun_global_var import init_GLV_DICT, tree_print, init_GLV_rmw, end_AST, g_oea_vs_g_AST, Get, Set, init_EVV, \
    fget, fkey, fGHU_plot_save, dset, dget, Fun3, fU_EVV_plot
from fun_thread import my_thread
from fun_pump import pump_pic_or_U
from fun_linear import ifft2
from a1_AST import define_lam_n_AST, gan_g_eoa, plot_GU_oe_energy_add, gan_gpnkE_VHoe_xyzinc_AST

np.seterr(divide='ignore', invalid='ignore')


def H_zdz(kz, diz):
    return np.power(math.e, kz * diz * 1j)


def gan_iz_diz(is_EVV_SSI, for_th2,
               izj_delay_dz, dizj, izj):
    if is_EVV_SSI == 1:
        iz = izj_delay_dz[for_th2]
        diz = dizj[for_th2]
    else:
        iz = izj[for_th2]
        diz = iz
    return diz, iz


# %%

def AST_EVV(U_name="",
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
            is_colorbar_on=1, is_colorbar_log=0,
            is_energy=0,
            # %%
            is_print=1,
            # %% 该程序 独有 -------------------------------
            is_EVV_SSI=1, is_stored=1, sheets_stored_num=10,
            # %%
            sample=1, cmap_3d='rainbow',
            elev=10, azim=-65, alpha=2,
            # %%
            is_plot_EVV=1, is_plot_3d_XYz=0, is_plot_selective=0,
            X=0, Y=0, is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
            # %%
            plot_group="UGa", is_animated=1,
            loop=0, duration=0.033, fps=5,
            # %%
            **kwargs, ):
    # %%

    if_image_Add_black_border(U_name, img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%
    is_HOPS = kwargs.get("is_HOPS_AST", 0)
    is_twin_pump_degenerate = int(is_HOPS >= 1)  # is_HOPS == 0.x 的情况 仍是单泵浦
    is_single_pump_birefringence = int(0 <= is_HOPS < 1 and kwargs.get("polar", "e") in "VvHhRrLl")
    is_birefringence_deduced = int(is_twin_pump_degenerate == 1 or is_single_pump_birefringence == 1)
    is_add_polarizer = int(is_HOPS > 0 and type(is_HOPS) != int)
    is_add_analyzer = int(type(kwargs.get("phi_a", 0)) != str)
    # %%
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
    theta2_x = kwargs.get("theta2_x", theta_x) if is_HOPS == 2 else theta_x  # 只有是 2 时，才能自由设定 theta2_x
    theta2_y = kwargs.get("theta2_y", theta_y) if is_HOPS == 2 else theta_y  # 只有是 2 时，才能自由设定 theta2_y
    # %%
    is_random_phase_2 = kwargs.get("is_random_phase_2", is_random_phase)
    is_H_l2 = kwargs.get("is_H_l2", is_H_l)
    is_H_theta2 = kwargs.get("is_H_theta2", is_H_theta)
    is_H_random_phase_2 = kwargs.get("is_H_random_phase_2", is_H_random_phase)
    # %%
    w0_2 = kwargs.get("w0_2", w0)
    # lam2 = kwargs.get("lam2", lam1)
    lam2 = lam1
    is_air_pump2 = kwargs.get("is_air_pump2", is_air_pump)
    T2 = kwargs.get("T2", T)
    polar2 = kwargs.get("polar2", 'H')
    # %%
    if is_twin_pump_degenerate == 1:
        # %%
        pump2_keys = kwargs["pump2_keys"]
        # %%
        [kwargs.pop(key) for key in kwargs["pump2_keys"]]  # 及时清理 kwargs ，尽量 保持 其干净
        kwargs.pop("pump2_keys")  # 这个有点意思， "pump2_keys" 这个键本身 也会被删除。

    # %%

    info = "AST_线性角谱_EVV"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # kwargs['ray'] = init_GLV_rmw(U_name, "~", "", "AST", **kwargs)
    init_GLV_rmw(U_name, "l", "AST", "", **kwargs)

    # %% 确定 波长（得在 所有用 lam1 的 函数 之前）

    lam1, n_name = define_lam_n_AST(lam1, **kwargs)

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

    if is_twin_pump_degenerate == 1:
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
        U2_0, g2 = 0, 0  # 之后总会 引用到，所以这里 先在 locals() 里加上

    # %%

    if "U" in kwargs:  # 防止对 U_amp_plot_save 造成影响
        kwargs.pop("U")

    # %%
    iz = z0 / size_PerPixel
    zj = kwargs.get("zj_AST_EVV", np.linspace(0, z0, sheets_stored_num + 1))  # 防止 后续函数 接收的 kwargs 里 出现 关键字 zj 后重名
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
             0, is_stored,
             sheets_stored_num, sheets_stored_num,
             X, Y, iz, size_PerPixel, )

    # %% 确定 公有参数

    args_init_AST = \
        [Ix, Iy, size_PerPixel,
         lam1, is_air, T,
         theta_x, theta_y, ]
    kwargs_init_AST = {"is_air_pump": is_air_pump, "gp": g_shift, }

    kwargs_gan_g_eoa = {"phi_a": kwargs.get("phi_a", 0), }

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

    args_fGHU_plot_save = \
        [0,  # 默认 全自动 is_auto = 1
         img_name_extension, is_print,
         # %%
         zj, sample, size_PerPixel,
         is_save, is_save_txt, dpi, size_fig,
         # %%
         "b", cmap_2d,
         ticks_num, is_contourf,
         is_title_on, is_axes_on, is_mm,
         fontsize, font,
         # %%
         is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
         # %%                          何况 一般默认 is_self_colorbar = 1...
         z0, ]

    def args_fU_EVV_plot(is_energy):
        return [img_name_extension,
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
                z0, ]

    # print(kwargs.get("g_o", 0))

    # %% 折射

    g_p, p_p, g_V, g_H, p_V, p_H, \
    n1_inc, n1, k1_inc, k1, k1_z, k1_xy, E1_u, \
    n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
    n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue, \
    n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
    n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
    n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
    n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He = \
        gan_gpnkE_VHoe_xyzinc_AST(is_birefringence_deduced, is_air,
                                  is_add_polarizer, is_HOPS,
                                  is_save, is_print, n_name,
                                  g_shift, U_0, U2_0, polar2,
                                  args_init_AST, args_U_amp_plot_save,
                                  kwargs_init_AST, kwargs_U_amp_plot_save,
                                  is_plot_n=1, **kwargs)
    # print(g_shift)

    if is_birefringence_deduced == 1:  # 考虑 偏振态 的 条件；is_air == 1 时 也可以 有偏振态，与是否 所处介质 无关
        # %%

        if is_add_polarizer == 1:

            def gan_g_oe(for_th2):
                diz, iz = gan_iz_diz(is_EVV_SSI, for_th2,
                                     izj_delay_dz, dizj, izj)
                if is_EVV_SSI == 1:
                    G1o_z = g_o * H_zdz(k1o_z, iz)
                    G1e_z = g_e * H_zdz(k1e_z, iz)
                else:
                    G1o_z = g_o
                    G1e_z = g_e
                return G1o_z, G1e_z, diz

            def gan_Gz_oe(G1o_z, G1e_z, diz):
                Go_z = G1o_z * H_zdz(k1o_z, diz)
                Ge_z = G1e_z * H_zdz(k1e_z, diz)
                return Go_z, Ge_z

            def Gan_Gz_oe(for_th2):
                G1o_z, G1e_z, diz = gan_g_oe(for_th2)
                Go_z, Ge_z = gan_Gz_oe(G1o_z, G1e_z, diz)
                return Go_z, Ge_z
        else:

            def gan_g_VHoe(for_th2):
                diz, iz = gan_iz_diz(is_EVV_SSI, for_th2,
                                     izj_delay_dz, dizj, izj)
                if is_EVV_SSI == 1:
                    G1_Vo_z = g_Vo * H_zdz(k1_Vo_z, iz)
                    G1_Ve_z = g_Ve * H_zdz(k1_Ve_z, iz)
                    G1_Ho_z = g_Ho * H_zdz(k1_Ho_z, iz)
                    G1_He_z = g_He * H_zdz(k1_He_z, iz)
                else:
                    G1_Vo_z = g_Vo
                    G1_Ve_z = g_Ve
                    G1_Ho_z = g_Ho
                    G1_He_z = g_He
                return G1_Vo_z, G1_Ve_z, G1_Ho_z, G1_He_z, diz

            def gan_Gz_VHoe(G1_Vo_z, G1_Ve_z, G1_Ho_z, G1_He_z, diz):
                Gz_Vo = G1_Vo_z * H_zdz(k1_Vo_z, diz)
                Gz_Ve = G1_Ve_z * H_zdz(k1_Ve_z, diz)
                Gz_Ho = G1_Ho_z * H_zdz(k1_Ho_z, diz)
                Gz_He = G1_He_z * H_zdz(k1_He_z, diz)
                return Gz_Vo, Gz_Ve, Gz_Ho, Gz_He

            def Gan_Gz_VHoe(for_th2):
                G1_Vo_z, G1_Ve_z, G1_Ho_z, G1_He_z, diz = gan_g_VHoe(for_th2)
                Gz_Vo, Gz_Ve, Gz_Ho, Gz_He = gan_Gz_VHoe(G1_Vo_z, G1_Ve_z, G1_Ho_z, G1_He_z, diz)
                return Gz_Vo, Gz_Ve, Gz_Ho, Gz_He

        if is_add_analyzer == 1:
            # %% 开始 EVV

            def Fun1(for_th2, fors_num2, *args, **kwargs, ):

                if is_add_polarizer == 1:
                    Gz_o, Gz_e = Gan_Gz_oe(for_th2)
                    Set("Gz_o", Gz_o)
                    Set("Gz_e", Gz_e)
                    
                    Gz_a = gan_g_eoa(Gz_o, Gz_e, E_uo, E_ue, **kwargs_gan_g_eoa)
                else:
                    Gz_Vo, Gz_Ve, Gz_Ho, Gz_He = Gan_Gz_VHoe(for_th2)
                    Set("Gz_Vo", Gz_Vo)
                    Set("Gz_Ve", Gz_Ve)
                    Set("Gz_Ho", Gz_Ho)
                    Set("Gz_He", Gz_He)
                    
                    Gz_Va = gan_g_eoa(Gz_Vo, Gz_Ve, E_u_Vo, E_u_Ve, **kwargs_gan_g_eoa)
                    Gz_Ha = gan_g_eoa(Gz_Ho, Gz_He, E_u_Ho, E_u_He, **kwargs_gan_g_eoa)
                    Gz_a = Gz_Va + Gz_Ha

                return Gz_a

            def Fun2(for_th2, fors_num, Gz_a, *args, **kwargs, ):

                dset("G", Gz_a)

                return dget("G")

            my_thread(10, sheets_stored_num + 1,
                      Fun1, Fun2, Fun3,
                      is_ordered=1, is_print=is_print, )

            # %%

            g_oea_vs_g_AST(dget("G"), dget("G"))

            fGHU_plot_save(*args_fGHU_plot_save, part_z="_oea_z", is_end=1, **kwargs, )

            # %%

            fU_EVV_plot(*args_fU_EVV_plot(is_energy), part_z="_oea_z", **kwargs, )

        else:
            # 如果 传进来的 phi_a 不是数字，则说明 没加 偏振片，则 正交线偏 oe 直接叠加后，gUH 的 相位 就 没用了；只有 gU 的 能量分布 才有用
            # 并且 二者的 的 复场 和 能量，根本不满足 傅立叶变换对 的 关系；

            # %% 开始 EVV

            def Fun1(for_th2, fors_num2, *args, **kwargs, ):

                if is_add_polarizer == 1:
                    Gz_o, Gz_e = Gan_Gz_oe(for_th2)
                    Set("Gz_o", Gz_o)
                    Set("Gz_e", Gz_e)
                    # 下面这个不行，不知道为啥...偏要像有 analyzer 一样，再 gan_g_eoa 投影一下，哪怕 再投影到 它自己上
                    # G_oe_energy_add = np.abs(Gz_o) ** 2 + np.abs(Gz_e) ** 2  # 远场 平方和
                    # U_oe_energy_add = np.abs(ifft2(Gz_o)) ** 2 + np.abs(ifft2(Gz_e)) ** 2  # 近场 平方和
                    
                    Gz_V = gan_g_eoa(Gz_o, Gz_e, E_uo, E_ue, phi_a=90)
                    Gz_H = gan_g_eoa(Gz_o, Gz_e, E_uo, E_ue, phi_a=0)
                    
                    G_oe_energy_add = np.abs(Gz_V) ** 2 + np.abs(Gz_H) ** 2  # 远场 模方和
                    U_oe_energy_add = np.abs(ifft2(Gz_V)) ** 2 + np.abs(ifft2(Gz_H)) ** 2  # 近场 模方和
                else:
                    Gz_Vo, Gz_Ve, Gz_Ho, Gz_He = Gan_Gz_VHoe(for_th2)
                    Set("Gz_Vo", Gz_Vo)
                    Set("Gz_Ve", Gz_Ve)
                    Set("Gz_Ho", Gz_Ho)
                    Set("Gz_He", Gz_He)
                    
                    Gz_o = Gz_Vo + Gz_Ho
                    Gz_e = Gz_Ve + Gz_He  # 先求和，再模方和，而不是 先模方和，再求和
                    # 下面这个不行，不知道为啥...偏要像有 analyzer 一样，再 gan_g_eoa 投影一下，哪怕 再投影到 它自己上
                    # G_oe_energy_add = np.abs(Gz_o) ** 2 + np.abs(Gz_e) ** 2  # 远场 平方和
                    # U_oe_energy_add = np.abs(ifft2(Gz_o)) ** 2 + np.abs(ifft2(Gz_e)) ** 2  # 近场 平方和
                    
                    Gz_V = gan_g_eoa(Gz_o, Gz_e, E_u_Vo, E_u_Ve, phi_a=90)
                    Gz_H = gan_g_eoa(Gz_o, Gz_e, E_u_Ho, E_u_He, phi_a=0)
                    # 下面 与 上面两条语句 等效
                    # Gz_VV = gan_g_eoa(Gz_Vo, Gz_Ve, E_u_Vo, E_u_Ve, phi_a=90)
                    # Gz_HV = gan_g_eoa(Gz_Ho, Gz_He, E_u_Ho, E_u_He, phi_a=90)
                    # Gz_V = Gz_VV + Gz_HV
                    # Gz_VH = gan_g_eoa(Gz_Vo, Gz_Ve, E_u_Vo, E_u_Ve, phi_a=0)
                    # Gz_HH = gan_g_eoa(Gz_Ho, Gz_He, E_u_Ho, E_u_He, phi_a=0)
                    # Gz_H = Gz_VH + Gz_HH
                    
                    G_oe_energy_add = np.abs(Gz_V) ** 2 + np.abs(Gz_H) ** 2  # 远场 模方和
                    U_oe_energy_add = np.abs(ifft2(Gz_V)) ** 2 + np.abs(ifft2(Gz_H)) ** 2  # 近场 模方和

                return G_oe_energy_add, U_oe_energy_add

            def Fun2(for_th2, fors_num, G_z_add, *args, **kwargs, ):

                dset("G", G_z_add)
                dset("U", args[0])

                return dget("G"), dget("U")

            my_thread(10, sheets_stored_num + 1,
                      Fun1, Fun2, Fun3,
                      is_ordered=1, is_print=is_print, )

            # %%

            # 如果 传进来的 phi_a 不是数字，则说明 没加 偏振片，则 正交线偏 oe 直接叠加后，gUH 的 相位 就 没用了；只有 gU 的 能量分布 才有用
            # 并且 二者的 的 复场 和 能量，根本不满足 傅立叶变换对 的 关系；

            # %%
            plot_GU_oe_energy_add(dget("G"), dget("U"),
                                  is_save, is_print,
                                  args_U_amp_plot_save,
                                  kwargs_U_amp_plot_save,
                                  z=z0, is_end=1, **kwargs, )

            # %%

            from fun_global_var import fset
            fset("G", dget("G"))  # fU_EVV_plot 里要用到，但其实那里 也可以用 dget 就没有这档子事了
            fset("U", dget("U"))  # 那里不用 dget 是因为，dget 被设计来 只储存 中间结果，不一定像 fset 只储存 最后结果。
            fU_EVV_plot(*args_fU_EVV_plot(0), part_z="_oe_z_energy_add", **kwargs, )

        return Get("Gz_o"), Get("Gz_e"), E_uo, E_ue, \
               Get("Gz_Vo"), Get("Gz_Ve"), E_u_Vo, E_u_Ve, \
               Get("Gz_Ho"), Get("Gz_He"), E_u_Ho, E_u_He, \
               Get("ray"), Get("method_and_way"), fkey("U")  # 加不加 起偏 或 检偏 器，都可能 会有 VH 两个分量，所以 将 return 写在外面

    else:
        # %% 开始 EVV

        def gan_g(for_th2):
            diz, iz = gan_iz_diz(is_EVV_SSI, for_th2,
                                 izj_delay_dz, dizj, izj)
            if is_EVV_SSI == 1:
                G1_z = g_shift * H_zdz(k1_z, iz)
                # U_z = ifft2(G1_z)
            else:
                G1_z = g_shift
                # U_z = U_0
            return G1_z, diz

        def gan_Gz(G1_z, diz):
            G_z = G1_z * H_zdz(k1_z, diz)
            return G_z

        def Gan_Gz(for_th2):
            G1_z, diz = gan_g(for_th2)
            G_z = gan_Gz(G1_z, diz)
            return G_z

        def Fun1(for_th2, fors_num2, *args, **kwargs, ):

            G_z = Gan_Gz(for_th2)

            return G_z

        def Fun2(for_th2, fors_num, G_z, *args, **kwargs, ):

            dset("G", G_z)

            return dget("G")

        my_thread(10, sheets_stored_num + 1,
                  Fun1, Fun2, Fun3,
                  is_ordered=1, is_print=is_print, )

        # %% 后续绘图

        end_AST(z0, size_PerPixel,
                g_shift, k1_z, )

        fGHU_plot_save(*args_fGHU_plot_save, is_end=1, **kwargs, )

        # %%

        fU_EVV_plot(*args_fU_EVV_plot(is_energy), **kwargs, )

        return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U")


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",
         "img_full_name": "lena1.png",
         "U_pixels_x": 500, "U_pixels_y": 500,
         "is_phase_only": 0,
         # %%
         "z_pump": -5,
         "is_LG": 1, "is_Gauss": 1, "is_OAM": 1,
         "l": 50, "p": 0,
         "theta_x": 0, "theta_y": 0,
         # %%
         "is_random_phase": 0,
         "is_H_l": 0, "is_H_theta": 0, "is_H_random_phase": 0,
         # %%
         "U_size": 1.5, "w0": 0.05,
         "z0": 10,
         # %%  控制 单双泵浦 和 绘图方式："is_HOPS": 0 代表 无双折射，即 "is_linear_birefringence": 0
         "is_HOPS_AST": 0.1,  # 0.x 代表 单泵浦，1.x 代表 高阶庞加莱球，2.x 代表 最广义情况：2 个 线偏 标量场 叠加；这些都是在 左手系下，且都是 线偏基
         "Theta": 0, "Phi": 0,  # 是否 采用 高阶加莱球、若采用，请给出 极角 和 方位角
         # 是否 使用 起偏器（"is_HOPS": 整数 即不使用）、若使用，请给出 其相对于 H (水平 x) 方向的 逆时针 转角 phi_p
         "phi_p": "45", "phi_a": "45",  # 是否 使用 检偏器（"phi_a": str 则不使用）、若使用，请给出 其相对于 H (水平 x) 方向的 逆时针 转角 phi_a
         # %%
         "lam1": 1.064, "is_air_pump": 1, "is_air": 2, "T": 25,
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
         "is_colorbar_on": 1, "is_colorbar_log": -1,
         "is_energy": 1,
         # %%
         "is_print": 1,
         # %% 该程序 独有 -------------------------------
         "is_EVV_SSI": 0, "is_stored": 1, "sheets_stored_num": 10,
         # %%
         "sample": 1, "cmap_3d": 'rainbow',
         "elev": 10, "azim": -65, "alpha": 2,
         # %%
         "is_plot_EVV": 1, "is_plot_3d_XYz": 0, "is_plot_selective": 0,
         "X": 0, "Y": 0, "is_plot_YZ_XZ": 0, "is_plot_3d_XYZ": 0,
         # %%
         "plot_group": "Ua", "is_animated": 1,
         "loop": 0, "duration": 0.033, "fps": 5,
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
         "polar": "V", "ray": "1",
         }

    if kwargs.get("is_HOPS_AST", 0) >= 1:  # 如果 is_HOPS >= 1，则 默认 双泵浦 is_twin_pumps == 1
        pump2_kwargs = {
            "U2_name": "",
            "img2_full_name": "spaceship.png",
            "is_phase_only_2": 0,
            # %%
            "z_pump2": -5,
            "is_LG_2": 1, "is_Gauss_2": 1, "is_OAM_2": 1,
            "l2": -50, "p2": 0,
            "theta2_x": 0, "theta2_y": 0,
            # %%
            "is_random_phase_2": 0,
            "is_H_l2": 0, "is_H_theta2": 0, "is_H_random_phase_2": 0,
            # %%
            "w0_2": 0.05,
            # %%
            "lam2": 1.064, "is_air_pump2": 1, "T2": 25,
            "polar2": 'V',
            # 有双泵浦，则必然考虑偏振、起偏，和检偏，且原 "polar2": 'e'、 "polar": "e" 已再不起作用
            # 取而代之的是，既然原 "polar": "e" 不再 work 但还存在，就不能浪费 它的存在，让其 重新规定 第一束光
            # 偏振方向 为 "VHRL" 中的一个，而不再规定其 极化方向 为 “oe” 中的一个；这里 第二束 泵浦的 偏振方向 默认与之 正交，因而可以 不用填写
            # 但仍然可以 规定第 2 个泵浦 为其他偏振，比如 2 个 同向 线偏叠加，也就是 2 个图叠加，或者 一个 线偏基，另一个 圆偏基
        }
        pump2_kwargs.update({"pump2_keys": list(pump2_kwargs.keys())})
        # Object of type dict_keys is not JSON serializable，所以 得转为 list
        kwargs.update(pump2_kwargs)

    kwargs = init_GLV_DICT(**kwargs)
    AST_EVV(**kwargs)
