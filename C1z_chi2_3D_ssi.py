# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
from scipy.io import savemat
from fun_global_var import Get, init_GLV_DICT, tree_print
from fun_algorithm import gcd_of_float
from fun_img_Resize import if_image_Add_black_border
from fun_SSI import slice_structure_ssi
from fun_thread import noop, my_thread
from fun_CGH import chi2_2D

np.seterr(divide='ignore', invalid='ignore')


# %%

def chi2_3D_ssi(U_name="",
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
                U_size=1, w0=0.3, structure_size_Shrink=0.1,
                deff_structure_length_expect=2,
                # %%
                Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
                structure_xy_mode='x', Depth=2, ssi_zoomout_times=5,
                # %%
                is_continuous=1, is_target_far_field=1, is_transverse_xy=0,
                is_reverse_xy=0, is_positive_xy=1, is_no_backgroud=1,
                is_bulk=1,
                # %%
                lam1=0.8, is_air_pump_structure=0, is_air=0, T=25,
                is_air_pump=1,
                # %%
                Tx=10, Ty=10, Tz="2*lc",
                mx=0, my=0, mz=0,
                is_stripe=0,
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
    # %%
    # ????????? ???????????? ????????????????????????

    if_image_Add_black_border(U_name, img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%

    info = "??2_3D_????????????"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # ??? def ????????? ???????????? is_end = 0????????? kwargs ????????? ???????????? ?????????
    # %%
    # ?????????????????????????????????

    # ?????? ??? deff_structure_length_expect ?????? ????????? z0????????????????????????????????? ?????? Tz ?????? NLA_SSI ?????????????????????????????????
    # ????????? deff_structure_length_expect < NLA_SSI ?????? z0 ??? ??????????????? > deff_structure_length_expect ??? ???????????????????????? A_to_B_3_NLA_SSI ????????? deff_structure_length_expect ??? z0 ???
    # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????? NLA_SSI ??????????????????????????? ?????????

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
    dk_z, lc, Gx, Gy, Gz, z0_recommend, Tz, deff_structure_length_expect, \
    structure, structure_opposite, \
    modulation, modulation_opposite, \
    modulation_squared, modulation_opposite_squared \
        = chi2_2D(U_name,
                  img_full_name,
                  is_phase_only,
                  # %%
                  z_pump,
                  is_LG, is_Gauss, is_OAM,
                  l, p,
                  theta_x, theta_y,
                  # %%s
                  is_random_phase,
                  is_H_l,
                  is_H_theta,
                  is_H_random_phase,
                  # %%
                  U_size, w0,
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
                  is_title_on, is_axes_on, is_mm,
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
                  # %%
                  is_air_pump=is_air_pump,
                  is_plot_n=1, is_print2=1, **kwargs, )
    if kwargs.get('ray', "2") == "3":
        [kwargs.pop(key) for key in kwargs["pump2_keys"]]  # ???????????? kwargs ????????? ?????? ?????????
        kwargs.pop("pump2_keys")  # ????????????????????? "pump2_keys" ??????????????? ??????????????????

    # %%
    # ?????? ???????????????????????? ??? ????????????????????????????????????????????? ??? ??????????????????
    # ?????? ???????????? ??? ????????????????????????????????? ??? ??????????????????
    # Tz_Unit

    diz, deff_structure_sheet, sheets_num, \
    Iz, deff_structure_length, Tz_unit, zj_structure = \
        slice_structure_ssi(Duty_Cycle_z, deff_structure_length_expect,
                            Tz, ssi_zoomout_times, size_PerPixel,
                            is_print)
    # print(sheets_num, len(zj_structure))

    # %%

    if is_stripe > 0:
        from fun_os import U_amp_plot_save
        sheets_stored_num = 10
        for_th_stored = list(np.int64(np.round(np.linspace(0, sheets_num - 1, sheets_stored_num))))
        # print(for_th_stored, sheets_num, len(for_th_stored))
        m_list = []
        mod_name_list = []
    if is_stripe == 2.2:
        from fun_CGH import nonrect_chi2_2D
        # if structure_xy_mode == 'x':
        #     Ix_structure, Iy_structure = sheets_num, Get("Iy")
        # elif structure_xy_mode == 'y':
        #     Ix_structure, Iy_structure = Get("Ix"), sheets_num
        # Ix_structure, Iy_structure = sheets_num, Get("Iy")
        Ix_structure, Iy_structure = sheets_num, modulation.shape[1]  # ????????? Get("Iy")
        modulation_lie_down, folder_address = \
            nonrect_chi2_2D(z_pump,
                                               is_LG, is_Gauss, is_OAM,
                                               l, p,
                                               theta_x, theta_y,
                                               # %%
                                               is_random_phase,
                                               is_H_l, is_H_theta, is_H_random_phase,
                                               # %%
                                               Ix_structure, Iy_structure, w0,
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
    elif is_stripe == 2 or is_stripe == 2.1:  # ?????? ??? ????????????
        from fun_CGH import interp2d_nonrect_chi2_2D
        modulation_lie_down = interp2d_nonrect_chi2_2D(folder_address, modulation,
                                                                 sheets_num,
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
    # print(modulation_lie_down.shape)

    # %%
    # ?????? ?????? ??? ?????? structure

    mj = []
    border_width_x, border_width_y = Get("border_width_x"), Get("border_width_y")

    def fun1(for_th, fors_num, *args, **kwargs, ):
        iz = for_th * diz
        step_nums_left, step_nums_right, step_nums_total = gcd_of_float(Duty_Cycle_z)[1]

        if mz != 0:  # ?????? ?????? Tz???????????? ?????????

            if is_stripe == 0:
                # print(iz - iz // Tz_unit * Tz_unit)
                # if iz - iz // Tz_unit * Tz_unit < Tz_unit * Duty_Cycle_z:  # ?????? ????????? ?????? ????????? ????????????????????????????????? diz / 10??????????????? ??????????????? ????????? ???????????????
                if np.mod(for_th, step_nums_total * ssi_zoomout_times) < step_nums_left * ssi_zoomout_times:
                    m = modulation_squared
                    mj.append("1")
                else:  # ?????? ????????? ???????????? ?????????????????? ??????????????? ????????? ???????????????
                    m = modulation_opposite_squared
                    mj.append("-1")
            elif is_stripe == 1:  # ??? modulation_squared ?????? z ??? ?????? ??? ???????????????x - y ????????? ??????
                if structure_xy_mode == 'x':  # ??????????????? ???????????? mj[for_th] ??????
                    mj.append(int(mx * Tx / Tz * iz))
                    m = np.roll(modulation, mj[-1], axis=1)
                elif structure_xy_mode == 'y':  # ??????????????? ???????????? mj[for_th] ??????
                    mj.append(int(my * Ty / Tz * iz))
                    m = np.roll(modulation, mj[-1], axis=0)
                elif structure_xy_mode == 'xy':  # ??????????????? ???????????? mj[for_th] ??????
                    mj.append(int(mx * Tx / Tz * iz))
                    m = np.roll(modulation, mj[-1], axis=1)
                    m = np.roll(modulation, int(my * Ty / Tz * iz), axis=0)
                m = np.pad(m, ((border_width_x, border_width_x), (border_width_y, border_width_y)),
                           'constant', constant_values=(1 - is_no_backgroud, 1 - is_no_backgroud))

                if for_th in for_th_stored:
                    m_list.append(m)
                    mod_name_list.append("??2_" + "tran_shift_" + str(for_th))

            elif is_stripe == 2 or is_stripe == 2.1 or is_stripe == 2.2:  # ?????? ??? ???????????? & ?????? CGH ??????
                # if structure_xy_mode == 'x':
                #     modulation_squared_new = np.tile(modulation_lie_down[for_th], (Get("Ix"), 1))  # ???????????? ????????????????????????
                # elif structure_xy_mode == 'y':
                #     modulation_squared_new = np.tile(modulation_lie_down[:, for_th], (Get("Iy"), 1))  # ???????????? ????????????????????????
                # modulation_squared_new = np.tile(modulation_lie_down[for_th], (Get("Ix"), 1))  # ????????? Get("Ix")
                modulation_new = np.tile(modulation_lie_down[for_th], (modulation.shape[0], 1))
                # ???????????? ?????????????????? ??? modulation ???????????? ??? ??????
                m = np.pad(modulation_new, ((border_width_x, border_width_x), (border_width_y, border_width_y)),
                           'constant', constant_values=(1 - is_no_backgroud, 1 - is_no_backgroud))

                if for_th in for_th_stored:
                    m_list.append(m)
                    mod_name_list.append("??2_" + "lie_down_" + str(for_th))

            modulation_squared_full_name = str(for_th) + (is_save_txt and ".txt" or ".mat")
            modulation_squared_address = folder_address + "\\" + modulation_squared_full_name

            if is_bulk == 0:  # ?????? U_save ?????????????????????????????????????????????????????????????????????????????????????????????
                np.savetxt(modulation_squared_address, m, fmt='%i') if is_save_txt else savemat(
                    modulation_squared_address, {'chi2_modulation_squared': m})

        else:  # ???????????? Tz?????? z ??? ??????????????????????????? ?????????

            m = modulation_squared
            mj.append("0")
            modulation_squared_full_name = str(for_th) + (is_save_txt and ".txt" or ".mat")
            modulation_squared_address = folder_address + "\\" + modulation_squared_full_name

            if is_bulk == 0:
                np.savetxt(modulation_squared_address, m, fmt='%i') if is_save_txt else savemat(
                    modulation_squared_address, {'chi2_modulation_squared': m})

    my_thread(10, sheets_num,
              fun1, noop, noop,
              is_ordered=1, is_print=is_print, is_end=1)

    # print(len(m_list))
    if is_stripe > 0:
        for i in range(sheets_stored_num):
            U_amp_plot_save(m_list[i], mod_name_list[i],
                            [], folder_address,
                            Get("img_name_extension"), is_save_txt,
                            # %%
                            size_PerPixel, dpi, Get("size_fig"),  # is_save = 1 - is_bulk ?????? ?????????????????? ?????? ????????????
                            # %%
                            cmap_2d, ticks_num, is_contourf,
                            is_title_on, is_axes_on, is_mm,
                            fontsize, font,
                            # %%
                            is_colorbar_on, 0,
                            1, 0, 0, 0,
                            # %%
                            suffix="", **kwargs, )

    # print(mj)
    # print(len(mj))


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",
         "img_full_name": "lena1.png",
         "U_pixels_x": 300, "U_pixels_y": 300,
         "is_phase_only": 0,
         # %%
         "z_pump": 0,
         "is_LG": 0, "is_Gauss": 1, "is_OAM": 1,
         "l": 3, "p": 0,
         "theta_x": 0, "theta_y": 0,
         # %%
         "is_random_phase": 0,
         "is_H_l": 0, "is_H_theta": 0, "is_H_random_phase": 0,
         # %%
         "U_size": 1, "w0": 0,
         "structure_size_Shrink": 0, "structure_size_Shrinker": 0,
         "is_U_size_x_structure_side_y": 1,
         "deff_structure_length_expect": 1,
         # %%
         "Duty_Cycle_x": 0.5, "Duty_Cycle_y": 0.5, "Duty_Cycle_z": 0.5,
         "structure_xy_mode": 'x', "Depth": 2, "ssi_zoomout_times": 1,
         # %%
         "is_continuous": 0, "is_target_far_field": 1, "is_transverse_xy": 0,
         "is_reverse_xy": 0, "is_positive_xy": 1, "is_no_backgroud": 0,
         "is_bulk": 0,
         # %%
         "lam1": 0.8, "is_air_pump_structure": 1, "is_air": 0, "T": 25,
         "is_air_pump": 1,
         # %%
         "Tx": 30, "Ty": 20, "Tz": 0,
         "mx": 1, "my": 1, "mz": 1,
         "is_stripe": 1,
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
         "is_colorbar_on": 1, "is_energy": 0,
         # %%
         "is_print": 1, "is_contours": 0, "n_TzQ": 1,
         "Gz_max_Enhance": 1, "match_mode": 1,
         # %% ????????? ?????? ???????????? -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "border_percentage": 0.1, "is_end": -1,
         # %%
         "ray": "2", "polar_structure": "e",
         }

    if kwargs.get("ray", "2") == "3":  # ?????? is_HOPS >= 1?????? ?????? ????????? is_twin_pumps == 1
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
        # Object of type dict_keys is not JSON serializable????????? ????????? list
        kwargs.update(pump2_kwargs)

    kwargs = init_GLV_DICT(**kwargs)
    chi2_3D_ssi(**kwargs)
