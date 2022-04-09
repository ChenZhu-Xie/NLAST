# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
import math
from scipy.io import loadmat
from fun_os import U_dir
from fun_img_Resize import if_image_Add_black_border
from fun_pump import pump_pic_or_U
from fun_SSI import Slice_SSI
from fun_linear import init_AST, init_SHG, fft2, ifft2
from fun_nonlinear import Eikz, args_SHG, Info_find_contours_SHG
from fun_thread import my_thread
from fun_global_var import init_GLV_DICT, init_SSI, end_SSI, dset, dget, fun3, fget, fGHU_plot_save, fU_SSI_plot

np.seterr(divide='ignore', invalid='ignore')


# %%

def Sfm_SSI(U1_name="",
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
            L0_Crystal=5, z0_structure_frontface_expect=0.5, deff_structure_length_expect=2,
            deff_structure_sheet_expect=1.8, sheets_stored_num=10,
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
            sample=2, ticks_num=6, is_contourf=0,
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
            is_colorbar_on=1, is_energy=0,
            # %%
            plot_group="UGa", is_animated=1,
            loop=0, duration=0.033, fps=5,
            # %%
            is_plot_3d_XYz=0, is_plot_selective=0,
            is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
            # %%
            is_print=1, is_contours=1, n_TzQ=1,
            Gz_max_Enhance=1, match_mode=1,
            # %%
            **kwargs, ):
    # %%

    if_image_Add_black_border(U1_name, img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    ray = init_GLV_DICT(U1_name, "2", "SSI", "Sfm", **kwargs)

    # %%

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

    n1, k1, k1_z, k1_xy = init_AST(Ix, Iy, size_PerPixel,
                                   lam1, is_air, T, )

    lam2, n2, k2, k2_z, k2_xy = init_SHG(Ix, Iy, size_PerPixel,
                                         lam1, is_air, T, )

    L0_Crystal, Tz, deff_structure_length_expect = Info_find_contours_SHG(g1_shift, k1_z, k2_z, Tz, mz,
                                                                          L0_Crystal, size_PerPixel,
                                                                          deff_structure_length_expect,
                                                                          is_print, is_contours, n_TzQ, Gz_max_Enhance,
                                                                          match_mode, )

    dk, lc, Tz, \
    Gx, Gy, Gz = args_SHG(k1, k2, size_PerPixel,
                          mx, my, mz,
                          Tx, Ty, Tz,
                          is_print=0, )

    # %%

    diz, deff_structure_sheet, \
    sheet_th_frontface, sheets_num_frontface, Iz_frontface, z0_front, \
    sheets_num_structure, Iz_structure, deff_structure_length, \
    sheets_num, Iz, z0, \
    dizj, izj, zj, \
    sheet_th_endface, sheets_num_endface, Iz_endface, z0_end, \
    sheet_th_sec1, sheets_num_sec1, iz_1, z0_1, \
    sheet_th_sec2, sheets_num_sec2, iz_2, z0_2 \
        = Slice_SSI(L0_Crystal, deff_structure_sheet_expect,
                    z0_structure_frontface_expect, deff_structure_length_expect,
                    z0_section_1_expect, z0_section_2_expect,
                    Tz, mz, size_PerPixel,
                    is_print)

    # %%
    # const

    const = (k2 / size_PerPixel / n2) ** 2 * deff * 1e-12  # pm / V 转换成 m / V

    # %%
    # G2_z0_shift

    folder_address = U_dir("", "0.χ2_modulation_squared", 0,
                           is_save - 0.5 * is_bulk, )

    init_SSI(g1_shift, U1_0,
             is_energy_evolution_on, is_stored,
             sheets_num, sheets_stored_num,
             X, Y, Iz, size_PerPixel, )

    cal_mode = [1, 1, 0]
    # 以 G 算 还是以 U 算、源项 是否 也衍射、k_2z 是否是 matrix 版
    # 用 G 算 会快很多
    # 不管是 G 还是 U，matrix 版的能量 总是要低一些，只不过 U 低得少些，没有数量级差异，而 G 少得很多

    def H2_zdz(diz):
        return np.power(math.e, k2_z * diz * 1j)

    def H2_z(diz):
        if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版
            dk_z = 2 * k1_z - k2_z
            return np.power(math.e, k2_z * diz * 1j) / (k2_z / size_PerPixel) * Eikz(
                    dk_z * diz) * diz * size_PerPixel \
                   * (2 / (dk_z / k2_z + 2))
        else:
            return np.power(math.e, k2 * diz * 1j) / (k2 / size_PerPixel) * Eikz(
                    dk * diz) * diz * size_PerPixel * (2 / (dk / k2 + 2))

    def fun1(for_th, fors_num, *args, **kwargs, ):
        iz = izj[for_th]

        H1_z = np.power(math.e, k1_z * iz * 1j)
        G1_z = g1_shift * H1_z
        U1_z = ifft2(G1_z)

        if is_bulk == 0:
            if for_th >= sheets_num_frontface and for_th <= sheets_num_endface - 1:
                modulation_squared_full_name = str(for_th - sheets_num_frontface) + ".mat"
                modulation_squared_address = folder_address + "\\" + modulation_squared_full_name
                modulation_squared_z = loadmat(modulation_squared_address)['chi2_modulation_squared']
            else:
                modulation_squared_z = np.ones((Ix, Iy), dtype=np.int64()) - is_no_backgroud
        else:
            modulation_squared_z = np.ones((Ix, Iy), dtype=np.int64()) - is_no_backgroud

        if cal_mode[0] == 1:  # 如果以 G 算

            if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版
                if cal_mode[1] == 1:  # 若 源项 也衍射
                    Q2_z = fft2(
                        modulation_squared_z * U1_z ** 2 * H2_z(dizj[for_th])
                        / np.power(math.e, k2_z * diz * 1j))
                else:
                    Q2_z = fft2(modulation_squared_z * U1_z ** 2 * H2_z(dizj[for_th]))
            else:
                Q2_z = fft2(modulation_squared_z * U1_z ** 2)

            if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版
                dG2_zdz = const * Q2_z
            else:
                if cal_mode[1] == 1:  # 若 源项 也衍射
                    dG2_zdz = const * Q2_z * H2_z(dizj[for_th]) \
                                          / np.power(math.e, k2 * diz * 1j)
                else:
                    dG2_zdz = const * Q2_z * H2_z(dizj[for_th])

            return dG2_zdz

        else:

            S2_z = modulation_squared_z * U1_z ** 2

            if cal_mode[1] == 1:  # 若 源项 也衍射
                if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版
                    dU2_zdz = const * S2_z * H2_z(dizj[for_th]) / \
                              np.power(math.e, k2_z * diz * 1j)
                else:
                    dU2_zdz = const * S2_z * H2_z(dizj[for_th]) / np.power(math.e, k2 * diz * 1j)
            else:
                dU2_zdz = const * S2_z * H2_z(dizj[for_th])

            return dU2_zdz

    def fun2(for_th, fors_num, dG2_zdz, *args, **kwargs, ):

        if cal_mode[0] == 1:  # 如果以 G 算

            if cal_mode[1] == 1:  # 若 源项 也衍射
                dset("G", (dget("G") + dG2_zdz) * H2_zdz(dizj[for_th]))
            else:
                dset("G", dget("G") * H2_zdz(dizj[for_th]) + dG2_zdz)

            return dget("G")

        else:

            dU2_zdz = dG2_zdz

            if cal_mode[1] == 1:  # 若 源项 也衍射
                dset("U", ifft2(fft2(dget("U") + dU2_zdz) * H2_zdz(dizj[for_th])))
            else:
                dset("U", ifft2(fft2(dget("U")) * H2_zdz(dizj[for_th])) + dU2_zdz)

            return dget("U")

    # %%

    is_U = 0 if cal_mode[0] == 1 else 1 # 如以 G 算，则 is_U = 0

    my_thread(10, sheets_num,
              fun1, fun2, fun3,
              is_ordered=1, is_print=is_print,
              is_U=is_U, )

    # %%

    end_SSI(g1_shift, is_energy, n_sigma=3,
            is_U=is_U, )

    fGHU_plot_save(U1_name, is_energy_evolution_on,  # 默认 全自动 is_auto = 1
                   img_name_extension,
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
                   z0, )

    # %%

    fU_SSI_plot(U1_name,
                sheets_num_frontface, sheets_num_endface,
                img_name_extension,
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
                is_colorbar_on, is_energy, is_show_structure_face,
                # %%
                plot_group, is_animated,
                loop, duration, fps,
                # %%
                is_plot_3d_XYz, is_plot_selective,
                is_plot_YZ_XZ, is_plot_3d_XYZ,
                # %%
                z0_1, z0_2,
                z0_front, z0_end, z0, )

    return fget("U"), fget("G")


if __name__ == '__main__':
    Sfm_SSI(U1_name="",
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
            L0_Crystal=5, z0_structure_frontface_expect=0.5, deff_structure_length_expect=2,
            deff_structure_sheet_expect=1.8, sheets_stored_num=10,
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
            plot_group="UGa", is_animated=1,
            loop=0, duration=0.033, fps=5,
            # %%
            is_plot_3d_XYz=0, is_plot_selective=0,
            is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
            # %%
            is_print=1, is_contours=1, n_TzQ=1,
            Gz_max_Enhance=1, match_mode=1,
            # %%
            border_percentage=0.1, )
