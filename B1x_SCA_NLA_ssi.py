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
from fun_linear import init_AST_12oe, fft2, ifft2
from fun_nonlinear import args_SFG
from fun_thread import my_thread
from fun_global_var import init_GLV_DICT, tree_print, init_GLV_rmw, init_SSI, end_SSI, Get, dset, dget, fun3, \
    fget, fkey, sget, skey, fGHU_plot_save, fU_SSI_plot

np.seterr(divide='ignore', invalid='ignore')


# %%

def SCA_NLA_ssi(U_name="",
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
                L0_Crystal=1, z0_structure_frontface_expect=0.5, deff_structure_length_expect=2,
                Duty_Cycle_z=0.5, ssi_zoomout_times=5, sheets_stored_num=10,
                z0_section_1_expect=1, z0_section_2_expect=1,
                X=0, Y=0,
                # %%
                is_bulk=1,
                is_stored=0, is_show_structure_face=1, is_energy_evolution_on=1,
                # %%
                lam1=0.8, is_air_pump=0, is_air=0, T=25,
                deff=30,
                # %%
                Tx=10, Ty=10, Tz="2*lc",
                mx=0, my=0, mz=0,
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
                is_print=1,
                # %%
                **kwargs, ):
    # %%

    if_image_Add_black_border(U_name, img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%

    info = "NLA_???????????????"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # ??? def ????????? ???????????? is_end = 0????????? kwargs ????????? ???????????? ?????????

    # kwargs['ray'] = init_GLV_rmw(U_name, "~", "SSI", "nla", **kwargs)
    init_GLV_rmw(U_name, "l", "nLA", "ssi", **kwargs)

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

    n1_inc, n1, k1_inc, k1, k1_z, k1_xy, g_shift, E1_u = \
        init_AST_12oe(Ix, Iy, size_PerPixel,
                      lam1, is_air, T,
                      theta_x, theta_y, is_print,
                      is_air_pump=is_air_pump,
                      gp=g_shift, **kwargs, )

    dk_z, lc, Tz, \
    Gx, Gy, Gz, \
    lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy, E3_u \
        = args_SFG(Ix, Iy, size_PerPixel,
                   is_air, T, lam1, lam1,
                   k1, k1_inc, k1, k1_inc,
                   theta_x, theta_y, theta_x, theta_y,
                   mx, my, mz,
                   Tx, Ty, Tz,
                   is_print,
                   g1=g_shift, g2=g_shift, **kwargs)

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

    const = deff * 1e-12  # pm / V ????????? m / V

    # %%
    # G2_z0_shift

    method = "MOD"
    folder_name = method + " - " + "n1_modulation_squared"
    folder_address = U_dir(folder_name, 1 - is_bulk, )

    init_SSI(g_shift, U_0,
             is_energy_evolution_on, is_stored,
             sheets_num, sheets_stored_num,
             X, Y, Iz, size_PerPixel, )

    def H1_zdz(diz):
        return np.power(math.e, k1_z * diz * 1j)
        # ?????? ????????? ???????????? ??? ????????? ?????? ?????????

    def H1_z(diz):
        return (np.power(math.e, k1_z * diz * 1j) - 1) / k1_z ** 2 * size_PerPixel ** 2
        # ?????? ????????? ???????????? ??? ????????? ?????? ?????????

    def fun1(for_th, fors_num, *args, **kwargs, ):

        if is_bulk == 0:
            if for_th >= sheets_num_frontface and for_th <= sheets_num_endface - 1:
                modulation_squared_full_name = str(for_th - sheets_num_frontface) + (is_save_txt and ".txt" or ".mat")
                modulation_squared_address = folder_address + "\\" + modulation_squared_full_name
                modulation_squared_z = loadmat(modulation_squared_address)['n1_modulation_squared']
            else:
                modulation_squared_z = np.ones((Ix, Iy), dtype=np.int64()) * n1
        else:
            modulation_squared_z = np.ones((Ix, Iy), dtype=np.int64()) * n1

        return modulation_squared_z

    def fun2(for_th, fors_num, modulation_squared_z, *args, **kwargs, ):

        U_z = ifft2(dget("G"))
        Q1_z = fft2((k1 / size_PerPixel / n1) ** 2 * (modulation_squared_z ** 2 - n1 ** 2) * U_z)

        dset("G", dget("G") * H1_zdz(dizj[for_th]) +
             const * Q1_z * H1_z(dizj[for_th]))

        return dget("G")

    my_thread(10, sheets_num,
              fun1, fun2, fun3,
              is_ordered=1, is_print=is_print, )

    # %%

    end_SSI(g_shift, is_energy, n_sigma=3, )

    fGHU_plot_save(is_energy_evolution_on,  # ?????? ????????? is_auto = 1
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
                   is_colorbar_on, is_energy,  # ???????????? ???????????? vmax ??? vmin????????? ????????? ?????? ??? ?????? ??? ?????? 2*2 ??? v
                   # %%                          ?????? ???????????? is_self_colorbar = 1...
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

    if abs(is_stored) != 1:
        return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U")
    else:
        return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U"), \
               sget("U"), sget("G"), skey("U"),


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",
         "img_full_name": "lena.png",
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
         "L0_Crystal": 1, "z0_structure_frontface_expect": 0, "deff_structure_length_expect": 1,
         "Duty_Cycle_z": 0.5, "ssi_zoomout_times": 5, "sheets_stored_num": 10,
         "z0_section_1_expect": 1, "z0_section_2_expect": 1,
         "X": 0, "Y": 0,
         # %%
         "is_bulk": 1,
         "is_stored": 1, "is_show_structure_face": 1, "is_energy_evolution_on": 1,
         # %%
         "lam1": 0.8, "is_air_pump": 1, "is_air": 0, "T": 25,
         "deff": 30,
         # %%
         "Tx": 10, "Ty": 10, "Tz": "2*lc",
         "mx": 0, "my": 0, "mz": 0,
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
         "is_print": 1,
         # %% ????????? ?????? ???????????? -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "border_percentage": 0.1, "is_end": -1,
         # %%
         "size_fig_x_scale": 10, "size_fig_y_scale": 2,
         # %%
         "theta_z": 90, "phi_z": 90, "phi_c": 23.7,
         # KTP 50 ??? ???deff ????????? 90, ~, 24.3??????24.3 - 2002, 25.3 - 2000???
         #                1994 ???68.8, ~, 90??????68.8 - 2002, 68.9 - 2000???
         # KTP 25 ??? ???deff ????????? 90, ~, 23.7??????23.7 - 2002, 24.8 - 2000???
         #                1994 ???68.8, ~, 90??????68.8 - 2002, 68.7 - 2000???
         # LN 25 ??? ???90, ~, ~
         "polar": "e",
         "ray": "1",
         }

    kwargs = init_GLV_DICT(**kwargs)
    SCA_NLA_ssi(**kwargs)
