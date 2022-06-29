# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
import math
from fun_img_Resize import if_image_Add_black_border
from fun_pump import pump_pic_or_U
from fun_SSI import slice_SSI
from fun_linear import fft2, ifft2
from fun_nonlinear import Eikz, Info_find_contours_SHG
from fun_thread import my_thread
from fun_CGH import structure_chi2_Generate_2D
from fun_global_var import init_GLV_DICT, tree_print, init_GLV_rmw, init_SSI, end_SSI, Get, Set, dset, dget, fun3, \
    fget, fkey, fGHU_plot_save, fU_SSI_plot

np.seterr(divide='ignore', invalid='ignore')


# %%

def SFG_SSF_SSI(U_name="",
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
                L0_Crystal=5, z0_structure_frontface_expect=0.5, deff_structure_length_expect=2,
                SSI_zoomout_times=1, sheets_stored_num=10,
                z0_section_1_expect=1, z0_section_2_expect=1,
                X=0, Y=0,
                # %%
                is_bulk=1, is_no_backgroud=0,
                is_stored=0, is_show_structure_face=1, is_energy_evolution_on=1,
                # %%
                lam1=0.8, is_air_pump=0, is_air=0, T=25,
                is_air_pump_structure=0,
                deff=30,
                # %%
                Tx=10, Ty=10, Tz="2*lc",
                mx=0, my=0, mz=0,
                is_stripe=0, is_NLAST=0,
                # %%
                # 生成横向结构
                Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
                Depth=2, structure_xy_mode='x',
                is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
                is_reverse_xy=0, is_positive_xy=1,
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

    info = "SSF_大步长_SSI"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # kwargs['ray'] = init_GLV_rmw(U_name, "^", "SSI", "SFM", **kwargs)
    init_GLV_rmw(U_name, ray_tag, "SSF", "SSI", **kwargs)

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

    if ray_tag == "f":
        for key in pump2_keys:
            kwargs[key] = locals()[key]
            kwargs["pump2_keys"] = locals()["pump2_keys"]
    n1_inc, n1, k1_inc, k1, k1_z, n2_inc, n2, k2_inc, k2, k2_z, lam3, n3_inc, n3, k3_inc, k3, k3_z, \
    theta3_x, theta3_y, L0_Crystal, deff_structure_length_expect, dk, lc, Tz, Gx, Gy, Gz, folder_address, \
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
                                     is_bulk, is_no_backgroud,
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
                                     L0_Crystal=L0_Crystal, g_shift=g_shift,
                                     # %%
                                     **kwargs, )
    if ray_tag == "f":
        [kwargs.pop(key) for key in kwargs["pump2_keys"]]  # 及时清理 kwargs ，尽量 保持 其干净
        kwargs.pop("pump2_keys")  # 这个有点意思， "pump2_keys" 这个键本身 也会被删除。

    # L0_Crystal, Tz, deff_structure_length_expect = Info_find_contours_SHG(g_shift, k1_z, k3_z, dk, Tz, mz,
    #                                                                       L0_Crystal, size_PerPixel,
    #                                                                       deff_structure_length_expect,
    #                                                                       is_print, is_contours, n_TzQ, Gz_max_Enhance,
    #                                                                       match_mode, )

    # %%

    diz, deff_structure_sheet, \
    sheet_th_frontface, sheets_num_frontface, Iz_frontface, z0_front, \
    sheets_num_structure, Iz_structure, deff_structure_length, \
    sheets_num, Iz, z0, \
    mj, mj_structure, dizj, izj, zj, zj_structure, \
    sheet_th_endface, sheets_num_endface, Iz_endface, z0_end, \
    sheet_th_sec1, sheets_num_sec1, iz_1, z0_1, \
    sheet_th_sec2, sheets_num_sec2, iz_2, z0_2 \
        = slice_SSI(L0_Crystal, SSI_zoomout_times, size_PerPixel,
                    z0_structure_frontface_expect, deff_structure_length_expect,
                    z0_section_1_expect, z0_section_2_expect,
                    is_stripe, mx, my, Tx, Ty, Tz, Duty_Cycle_z, structure_xy_mode,
                    is_print)

    # %%

    if is_stripe > 0:
        from fun_os import U_amp_plot_save
        sheets_stored_num_structure = sheets_stored_num
        for_th_first = int(mj[0] == '0') * SSI_zoomout_times
        for_th_stored = list(np.int64(np.round(np.linspace(0 + for_th_first, len(mj_structure) - 1 + for_th_first,
                                                           sheets_stored_num_structure))))
        # print(for_th_stored, sheets_num, len(for_th_stored))
        m_list = []
        mod_name_list = []
    if is_stripe == 2.2:
        from fun_CGH import structure_nonrect_chi2_Generate_2D
        # if structure_xy_mode == 'x':
        #     Ix_structure, Iy_structure = len(mj_structure), Get("Iy")
        # elif structure_xy_mode == 'y':
        #     Ix_structure, Iy_structure = Get("Ix"), len(mj_structure)
        # Ix_structure, Iy_structure = len(mj_structure), Get("Iy")
        Ix_structure, Iy_structure = len(mj_structure), modulation.shape[1]
        # print(Ix_structure, Iy_structure)
        modulation_lie_down, folder_address = \
            structure_nonrect_chi2_Generate_2D(z_pump_Structure,
                                               is_LG_Structure, is_Gauss_Structure, is_OAM_Structure,
                                               l_Structure, p_Structure,
                                               theta_x_Structure, theta_y_Structure,
                                               # %%
                                               is_random_phase_Structure,
                                               is_H_l_Structure, is_H_theta_Structure, is_H_random_phase_Structure,
                                               # %%
                                               Ix_structure, Iy_structure, w0_Structure,
                                               Duty_Cycle_x, Duty_Cycle_y, structure_xy_mode, Depth,
                                               # %%
                                               is_continuous, is_target_far_field, is_transverse_xy,
                                               is_reverse_xy, is_positive_xy,
                                               0, is_no_backgroud,
                                               # %%
                                               lam1, is_air_pump_structure, T,
                                               # %%
                                               is_save, is_save_txt, dpi,
                                               # %%
                                               cmap_2d,
                                               # %%
                                               ticks_num, is_contourf,
                                               is_title_on, is_axes_on, is_mm, zj_structure[:-1],
                                               # %%
                                               fontsize, font,
                                               # %%
                                               is_colorbar_on, is_energy,
                                               # %%
                                               **kwargs, )
    elif is_stripe == 2 or is_stripe == 2.1:  # 躺下 的 插值算法
        from fun_CGH import structure_nonrect_chi2_interp2d_2D
        modulation_lie_down = structure_nonrect_chi2_interp2d_2D(folder_address, modulation,
                                                                 len(mj_structure),
                                                                 # %%
                                                                 is_save_txt, dpi,
                                                                 # %%
                                                                 cmap_2d,
                                                                 # %%
                                                                 ticks_num, is_contourf,
                                                                 is_title_on, is_axes_on, is_mm, zj_structure[:-1],
                                                                 # %%
                                                                 fontsize, font,
                                                                 # %%
                                                                 is_colorbar_on,
                                                                 # %%
                                                                 **kwargs, )

    # %%
    # const

    const = (k3_inc / size_PerPixel / n3_inc) ** 2 * deff * 1e-12  # pm / V 转换成 m / V

    # %%
    # G3_z0_shift

    init_SSI(g_shift, U_0,
             is_energy_evolution_on, is_stored,
             sheets_num, sheets_stored_num,
             X, Y, Iz, size_PerPixel, )

    cal_mode = [1, 1, 0]

    # 以 G 算 还是以 U 算、源项 是否 也衍射、k_2z 是否是 matrix 版
    # 用 G 算 会快很多
    # 不管是 G 还是 U，matrix 版的能量 总是要低一些，只不过 U 低得少些，没有数量级差异，而 G 少得很多

    border_width_x, border_width_y = Get("border_width_x"), Get("border_width_y")

    def H3_zdz(diz):
        return np.power(math.e, k3_z * diz * 1j)

    def H3_z(diz):
        if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版
            dk_z = 2 * k1_z - k3_z
            return np.power(math.e, k3_z * diz * 1j) / (k3_z / size_PerPixel) * Eikz(
                dk_z * diz) * diz * size_PerPixel \
                   * (2 / (dk_z / k3_z + 2))
        else:
            return np.power(math.e, k3 * diz * 1j) / (k3 / size_PerPixel) * Eikz(
                dk * diz) * diz * size_PerPixel * (2 / (dk / k3 + 2))

    def fun1(for_th, fors_num, *args, **kwargs, ):
        iz = izj[for_th]

        H1_z = np.power(math.e, k1_z * iz * 1j)
        G1_z = g_shift * H1_z
        U_z = ifft2(G1_z)

        if ray_tag == "f":
            H2_z = np.power(math.e, k2_z * iz * 1j)
            G2_z = g2 * H2_z
            U2_z = ifft2(G2_z)
        else:
            U2_z = U_z

        if is_bulk == 0:
            if for_th >= sheets_num_frontface and for_th <= sheets_num_endface - 1:
                if mj[for_th] == '0':
                    # print("???????????????")
                    modulation_squared_z = np.ones((Ix, Iy), dtype=np.int64()) - is_no_backgroud
                elif is_stripe == 0:
                    if mj[for_th] == '1':
                        modulation_squared_z = modulation_squared
                    elif mj[for_th] == '-1':
                        modulation_squared_z = modulation_opposite_squared
                elif is_stripe == 1:
                    if structure_xy_mode == 'x':  # 往右（列） 线性平移 mj[for_th] 像素
                        modulation_z = np.roll(modulation, mj[for_th], axis=1)
                    elif structure_xy_mode == 'y':  # 往下（行） 线性平移 mj[for_th] 像素
                        modulation_z = np.roll(modulation, mj[for_th], axis=0)
                    elif structure_xy_mode == 'xy':  # 往右（列） 线性平移 mj[for_th] 像素
                        modulation_z = np.roll(modulation, mj[for_th], axis=1)
                        # modulation_z = np.roll(modulation_squared_z, mj[for_th] / (mx * Tx) * (my * Ty), axis=0)
                        modulation_z = np.roll(modulation_z, int(my * Ty / Tz * (izj[for_th] - Iz_frontface)), axis=0)
                    modulation_squared_z = np.pad(modulation_z,
                                                  ((border_width_x, border_width_x),
                                                   (border_width_y, border_width_y)),
                                                  'constant',
                                                  constant_values=(1 - is_no_backgroud, 1 - is_no_backgroud))

                    if for_th in for_th_stored:
                        m_list.append(modulation_squared_z)
                        mod_name_list.append("χ2_" + "tran_shift_" + str(for_th))

                elif is_stripe == 2 or is_stripe == 2.1 or is_stripe == 2.2:
                    # if structure_xy_mode == 'x':
                    #     modulation_squared_z = np.tile(modulation_lie_down[for_th], (Get("Ix"), 1))
                    #     # 按行复制 多行，成一个方阵
                    # elif structure_xy_mode == 'y':
                    #     modulation_squared_z = np.tile(modulation_lie_down[:, for_th],
                    #                                    (Get("Iy"), 1))  # 按列复制 多列，成一个方阵
                    # modulation_squared_z = np.tile(modulation_lie_down[for_th], (Get("Ix"), 1))
                    modulation_new_z = np.tile(modulation_lie_down[for_th], (modulation.shape[0], 1))
                    # 按行复制 多行，成一个 与 modulation 尺寸相同 的 矩阵
                    modulation_squared_z = np.pad(modulation_new_z,
                                                  ((border_width_x, border_width_x),
                                                   (border_width_y, border_width_y)),
                                                  'constant',
                                                  constant_values=(1 - is_no_backgroud, 1 - is_no_backgroud))

                    if for_th in for_th_stored:
                        m_list.append(modulation_squared_z)
                        mod_name_list.append("χ2_" + "lie_down_" + str(for_th))


            else:
                modulation_squared_z = np.ones((Ix, Iy), dtype=np.int64()) - is_no_backgroud
        else:
            modulation_squared_z = np.ones((Ix, Iy), dtype=np.int64()) - is_no_backgroud

        if cal_mode[0] == 1:  # 如果以 G 算
            from fun_linear import fft2

            if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版
                if cal_mode[1] == 1:  # 若 源项 也衍射
                    Q2_z = fft2(
                        modulation_squared_z * U_z * U2_z * H3_z(dizj[for_th])
                        / np.power(math.e, k3_z * diz * 1j))
                else:
                    Q2_z = fft2(modulation_squared_z * U_z * U2_z * H3_z(dizj[for_th]))
            else:
                Q2_z = fft2(modulation_squared_z * U_z * U2_z)

            if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版
                dG3_zdz = const * Q2_z
            else:
                if cal_mode[1] == 1:  # 若 源项 也衍射
                    dG3_zdz = const * Q2_z * H3_z(dizj[for_th]) \
                              / np.power(math.e, k3 * diz * 1j)
                else:
                    dG3_zdz = const * Q2_z * H3_z(dizj[for_th])

            return dG3_zdz

        else:

            S2_z = modulation_squared_z * U_z ** 2

            if cal_mode[1] == 1:  # 若 源项 也衍射
                if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版
                    dU2_zdz = const * S2_z * H3_z(dizj[for_th]) / \
                              np.power(math.e, k3_z * diz * 1j)
                else:
                    dU2_zdz = const * S2_z * H3_z(dizj[for_th]) / np.power(math.e, k3 * diz * 1j)
            else:
                dU2_zdz = const * S2_z * H3_z(dizj[for_th])

            return dU2_zdz

    def fun2(for_th, fors_num, dG3_zdz, *args, **kwargs, ):

        if cal_mode[0] == 1:  # 如果以 G 算

            if cal_mode[1] == 1:  # 若 源项 也衍射
                dset("G", (dget("G") + dG3_zdz) * H3_zdz(dizj[for_th]))
            else:
                dset("G", dget("G") * H3_zdz(dizj[for_th]) + dG3_zdz)

            return dget("G")

        else:

            dU2_zdz = dG3_zdz

            if cal_mode[1] == 1:  # 若 源项 也衍射
                dset("U", ifft2(fft2(dget("U") + dU2_zdz) * H3_zdz(dizj[for_th])))
            else:
                dset("U", ifft2(fft2(dget("U")) * H3_zdz(dizj[for_th])) + dU2_zdz)

            return dget("U")

    # %%

    is_U = 0 if cal_mode[0] == 1 else 1  # 如以 G 算，则 is_U = 0

    my_thread(10, sheets_num,
              fun1, fun2, fun3,
              is_ordered=1, is_print=is_print,
              is_U=is_U, )

    if is_stripe > 0:
        for i in range(sheets_stored_num_structure):
            U_amp_plot_save(folder_address,
                            # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
                            m_list[i], mod_name_list[i],
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
                            0, is_colorbar_on, 0,
                            # %%
                            suffix="", **kwargs, )

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

    import inspect
    if inspect.stack()[1][3] == "SFG_NLA_reverse":
        from fun_statistics import find_Kxyz
        from fun_linear import fft2
        K1_z, K1_xy = find_Kxyz(fft2(U_0), k1)
        K2_z, K2_xy = find_Kxyz(fft2(U2_0), k2)
        kiizQ = K1_z + K2_z + Gz
        # print(np.max(np.abs(fft2(fget("U")) / Get("size_PerPixel") ** 2)))

        return fget("U"), U_0, U2_0, modulation_squared, k1_inc, k2_inc, \
               theta_x, theta_y, theta2_x, theta2_y, kiizQ, \
               k1, k2, k3, const, Iz, Gz
    elif inspect.stack()[1][3] == "SFG_NLA_SSI__AST_EVV":
        Set("k3", k3)
        Set("lam3", lam3)

        return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U")
    else:
        return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U")


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",
         "img_full_name": "lena1.png",
         "U_pixels_x": 0, "U_pixels_y": 0,
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
         "structure_size_Shrink": 0.1, "structure_size_Shrinker": 0,
         "is_U_size_x_structure_side_y": 1,
         "is_phase_only_Structure": 0,
         # %%
         "w0_Structure": 0, "z_pump_Structure": 0,
         "is_LG_Structure": 0, "is_Gauss_Structure": 1, "is_OAM_Structure": 1,
         "l_Structure": 1, "p_Structure": 0,
         "theta_x_Structure": 0, "theta_y_Structure": 0,
         # %%
         "is_random_phase_Structure": 0,
         "is_H_l_Structure": 0, "is_H_theta_Structure": 0, "is_H_random_phase_Structure": 0,
         # %%
         "U_size": 1, "w0": 0.3,
         "L0_Crystal": 2, "z0_structure_frontface_expect": 0, "deff_structure_length_expect": 2,
         "SSI_zoomout_times": 1, "sheets_stored_num": 10,
         "z0_section_1_expect": 1, "z0_section_2_expect": 1,
         "X": 0, "Y": 0,
         # %%
         "is_bulk": 0, "is_no_backgroud": 0,
         "is_stored": 0, "is_show_structure_face": 1, "is_energy_evolution_on": 1,
         # %%
         "lam1": 1, "is_air_pump": 1, "is_air": 0, "T": 25,
         "lam_structure": 1, "is_air_pump_structure": 1, "T_structure": 25,
         "deff": 30,
         # %%
         "Tx": 18, "Ty": 10, "Tz": 8,
         "mx": 1, "my": 1, "mz": 1,
         "is_stripe": 2.1, "is_NLAST": 0,
         # %%
         # 生成横向结构
         "Duty_Cycle_x": 0.5, "Duty_Cycle_y": 0.5, "Duty_Cycle_z": 0.5,
         "Depth": 2, "structure_xy_mode": 'x',
         "is_continuous": 0, "is_target_far_field": 1, "is_transverse_xy": 0,
         "is_reverse_xy": 0, "is_positive_xy": 1,
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
         "size_fig_x_scale": 10, "size_fig_y_scale": 1,
         # %%
         "theta_z": 90, "phi_z": 0, "phi_c": 24.3,
         # KTP 50 度 ：deff 最高： 90, ~, 24.3，（24.3 - 2002, 25.3 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.9 - 2000）
         # KTP 25 度 ：deff 最高： 90, ~, 23.7，（23.7 - 2002, 24.8 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.7 - 2000）
         # LN 25 度 ：90, ~, ~
         "polar": "o",
         "ray": "3", "polar3": "o",
         }

    if kwargs.get("ray", "2") == "3":  # 如果 ray == 3，则 默认 双泵浦 is_twin_pumps == 1
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
    SFG_SSF_SSI(**kwargs)

    # SFG_SSF_SSI(U_name="",
    #              img_full_name="Grating.png",
    #              is_phase_only=0,
    #              # %%
    #              z_pump=0,
    #              is_LG=0, is_Gauss=0, is_OAM=0,
    #              l=0, p=0,
    #              theta_x=0, theta_y=0,
    #              # %%
    #              is_random_phase=0,
    #              is_H_l=0, is_H_theta=0, is_H_random_phase=0,
    #              # %%
    #              # 生成横向结构
    #              U_name_Structure='',
    #              structure_size_Shrink=0.1,
    #              is_phase_only_Structure=0,
    #              # %%
    #              w0_Structure=0, z_pump_Structure=0,
    #              is_LG_Structure=0, is_Gauss_Structure=0, is_OAM_Structure=0,
    #              l_Structure=0, p_Structure=0,
    #              theta_x_Structure=0, theta_y_Structure=0,
    #              # %%
    #              is_random_phase_Structure=0,
    #              is_H_l_Structure=0, is_H_theta_Structure=0, is_H_random_phase_Structure=0,
    #              # %%
    #              U_size=1, w0=0.3,
    #              L0_Crystal=5, z0_structure_frontface_expect=0, deff_structure_length_expect=2,
    #              sheets_stored_num=10, z0_section_1_expect=1, z0_section_2_expect=1,
    #              X=0, Y=0,
    #              # %%
    #              is_bulk=1, is_no_backgroud=0,
    #              is_stored=0, is_show_structure_face=1, is_energy_evolution_on=1,
    #              # %%
    #              lam1=0.8, is_air_pump=0, is_air=0, T=25,
    #              deff=30,
    #              # %%
    #              Tx=10, Ty=10, Tz="2*lc",
    #              mx=1, my=0, mz=0,
    #              is_stripe=0, is_NLAST=0,
    #              # %%
    #              # 生成横向结构
    #              Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
    #              Depth=2, structure_xy_mode='x',
    #              is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
    #              is_reverse_xy=0, is_positive_xy=1,
    #              # %%
    #              is_save=1, is_save_txt=0, dpi=100,
    #              # %%
    #              color_1d='b', cmap_2d='viridis', cmap_3d='rainbow',
    #              elev=10, azim=-65, alpha=2,
    #              # %%
    #              sample=1, ticks_num=6, is_contourf=0,
    #              is_title_on=1, is_axes_on=1, is_mm=1,
    #              # %%
    #              fontsize=9,
    #              font={'family': 'serif',
    #                    'style': 'normal',  # 'normal', 'italic', 'oblique'
    #                    'weight': 'normal',
    #                    'color': 'black',  # 'black','gray','darkred'
    #                    },
    #              # %%
    #              is_colorbar_on=1, is_energy=0,
    #              # %%
    #              plot_group="UGa", is_animated=1,
    #              loop=0, duration=0.033, fps=5,
    #              # %%
    #              is_plot_3d_XYz=0, is_plot_selective=0,
    #              is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
    #              # %%
    #              is_print=1, is_contours=1, n_TzQ=1,
    #              Gz_max_Enhance=1, match_mode=1,
    #              # %%
    #              root_dir=r'',
    #              border_percentage=0.1, ray="2", is_end=-1,
    #              size_fig_x_scale=10, size_fig_y_scale=1, )
