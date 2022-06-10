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
from fun_SSI import slice_ssi
from fun_linear import init_AST, fft2, ifft2
from fun_nonlinear import Eikz, accurate_args_SFG, Info_find_contours_SHG
from fun_thread import my_thread
from fun_global_var import init_GLV_DICT, tree_print, init_GLV_rmw, init_SSI, end_SSI, Get, dset, dget, fun3, \
    fget, fkey, fGHU_plot_save, fU_SSI_plot

np.seterr(divide='ignore', invalid='ignore')


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
                U_NonZero_size=1, w0=0.3,
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
                                U_NonZero_size, w0_2,
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
    L0_Crystal, Tz, deff_structure_length_expect = accurate_args_SFG(Ix, Iy, size_PerPixel,
                                                                     lam1, lam2, is_air, T,
                                                                     k1_inc, k2_inc,
                                                                     g_shift, k1_z,
                                                                     L0_Crystal, deff_structure_length_expect,
                                                                     mx, my, mz,
                                                                     Tx, Ty, Tz,
                                                                     is_contours, n_TzQ,
                                                                     Gz_max_Enhance, match_mode,
                                                                     is_print,
                                                                     theta_x, theta2_x,
                                                                     theta_y, theta2_y,
                                                                     **kwargs)

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
                modulation_squared_full_name = str(for_th - sheets_num_frontface) + (is_save_txt and ".txt" or ".mat")
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
                   z0, is_end=1, )

    # %%

    fU_SSI_plot(sheets_num_frontface, sheets_num_endface,
                img_name_extension,
                kwargs.get("is_no_data_save", 0), is_save_txt,
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

    return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U")


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",
         "img_full_name": "Grating.png",
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
         "U_NonZero_size": 1, "w0": 0.3,
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
         # %%
         "Tx": 10, "Ty": 10, "Tz": "2*lc",
         "mx": 0, "my": 0, "mz": 0,
         "is_NLAST": 0,
         # %%
         "is_save": 0, "is_save_txt": 0, "dpi": 100,
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
         "is_colorbar_on": 1, "is_energy": 0,
         # %%
         "plot_group": "UGa", "is_animated": 1,
         "loop": 0, "duration": 0.033, "fps": 5,
         # %%
         "is_plot_3d_XYz": 0, "is_plot_selective": 0,
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
         # KTP 25 度 ：deff 最高： 90, ~, 24.3，（24.3 - 2002, 24.8 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.7 - 2000）
         # LN 25 度 ：90, ~, ~
         "polar": "e",
         "ray": "2", "polar3": "e",
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
    SFG_SSF_ssi(**kwargs)

    # SFG_SSF_ssi(U_name="",
    #         img_full_name="Grating.png",
    #         is_phase_only=0,
    #         # %%
    #         z_pump=0,
    #         is_LG=0, is_Gauss=0, is_OAM=0,
    #         l=0, p=0,
    #         theta_x=0, theta_y=0,
    #         # %%
    #         is_random_phase=0,
    #         is_H_l=0, is_H_theta=0, is_H_random_phase=0,
    #         # %%
    #         U_NonZero_size=1, w0=0.3,
    #         L0_Crystal=1, z0_structure_frontface_expect=0.5, deff_structure_length_expect=2,
    #         Duty_Cycle_z=0.5, ssi_zoomout_times=5, sheets_stored_num=10,
    #         z0_section_1_expect=1, z0_section_2_expect=1,
    #         X=0, Y=0,
    #         # %%
    #         is_bulk=1, is_no_backgroud=0,
    #         is_stored=0, is_show_structure_face=1, is_energy_evolution_on=1,
    #         # %%
    #         lam1=0.8, is_air_pump=0, is_air=0, T=25,
    #         deff=30,
    #         # %%
    #         Tx=10, Ty=10, Tz="2*lc",
    #         mx=0, my=0, mz=0,
    #         is_NLAST=0,
    #         # %%
    #         is_save=0, is_save_txt=0, dpi=100,
    #         # %%
    #         color_1d='b', cmap_2d='viridis', cmap_3d='rainbow',
    #         elev=10, azim=-65, alpha=2,
    #         # %%
    #         sample=1, ticks_num=6, is_contourf=0,
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
    #         plot_group="UGa", is_animated=1,
    #         loop=0, duration=0.033, fps=5,
    #         # %%
    #         is_plot_3d_XYz=0, is_plot_selective=0,
    #         is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
    #         # %%
    #         is_print=1, is_contours=1, n_TzQ=1,
    #         Gz_max_Enhance=1, match_mode=1,
    #         # %%
    #         root_dir=r'',
    #         border_percentage=0.1, ray="2", is_end=-1,
    #         size_fig_x_scale=10, size_fig_y_scale=1, )
