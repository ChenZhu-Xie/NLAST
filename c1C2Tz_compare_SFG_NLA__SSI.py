# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:57 2021

@author: Xcz
"""

# %%

import math
import copy
import numpy as np
from fun_global_var import init_GLV_DICT, Get, tree_print, GU_error_energy_plot_save
from fun_img_Resize import if_image_Add_black_border
from fun_pump import pump_pic_or_U
from c1C2_compare_SFG_NLA__SSI import compare_SFG_NLA__SSI
from c1_SFG_NLA import gan_gpnkE_123VHoe_xyzinc_SFG

np.seterr(divide='ignore', invalid='ignore')


def Tz_compare_SFG_NLA__SSI(U_name_Structure="",
                            is_phase_only_Structure=0,
                            # %%
                            z_pump_Structure=0,
                            is_LG_Structure=0, is_Gauss_Structure=1, is_OAM_Structure=1,
                            l_Structure=0, p_Structure=0,
                            theta_x_Structure=0, theta_y_Structure=0,
                            # %%
                            is_random_phase_Structure=0,
                            is_H_l_Structure=0, is_H_theta_Structure=0, is_H_random_phase_Structure=0,
                            # %%
                            U_name="",
                            img_full_name="l=1.png",
                            is_phase_only=0,
                            # %%
                            z_pump=0,
                            is_LG=0, is_Gauss=1, is_OAM=1,
                            l=1, p=0,
                            theta_x=1, theta_y=0,
                            # %%
                            is_random_phase=0,
                            is_H_l=0, is_H_theta=0, is_H_random_phase=0,
                            # %%---------------------------------------------------------------------
                            # %%
                            U_size=0.5, w0=0.1, w0_Structure=5, structure_size_Shrink=0.1,
                            L0_Crystal=2, z0_structure_frontface_expect=0.5, deff_structure_length_expect=1,
                            SSI_zoomout_times=1, sheets_stored_num=10,
                            z0_section_1_expect=1, z0_section_2_expect=1,
                            X=0, Y=0,
                            # %%
                            Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
                            structure_xy_mode='x', Depth=2,
                            # %%
                            is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
                            is_reverse_xy=0, is_positive_xy=1,
                            # %%
                            is_bulk=1, is_no_backgroud=1,
                            is_stored=0, is_show_structure_face=1, is_energy_evolution_on=1,
                            # %%
                            lam1=1.5, is_air_pump=0, is_air=0, T=25,
                            is_air_pump_structure=0,
                            deff=30, is_fft=1, fft_mode=0,
                            is_sum_Gm=0, mG=0,
                            is_linear_convolution=0,
                            # %%
                            Tx=19.769, Ty=20, Tz=8.139,
                            mx=-1, my=0, mz=1,
                            is_stripe=0, is_NLAST=0,
                            # %%
                            is_save=0, is_save_txt=0, dpi=100,
                            # %%
                            color_1d='b', color_1d2='r', cmap_2d='viridis', cmap_3d='rainbow',
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
                            is_energy=1,
                            # %%
                            plot_group="UGa", is_animated=1,
                            loop=0, duration=0.033, fps=5,
                            # %%
                            is_plot_EVV=1, is_plot_3d_XYz=0, is_plot_selective=0,
                            is_plot_YZ_XZ=1, is_plot_3d_XYZ=0,
                            # %%
                            is_print=1, is_contours=1, n_TzQ=1,
                            Gz_max_Enhance=1, match_mode=1,
                            # %% ????????? ?????? -------------------------------
                            is_NLA=1, is_amp_relative=1,
                            num_data_points=3, center_times=40, shift_right=3,
                            # %%
                            **kwargs, ):
    # %%
    is_HOPS = kwargs.get("is_HOPS_SHG", 0)
    is_twin_pump_degenerate = int(is_HOPS >= 1)  # is_HOPS == 0.x ????????? ???????????????
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
        [kwargs.pop(key) for key in kwargs["pump2_keys"]]  # ???????????? kwargs ????????? ?????? ?????????
        kwargs.pop("pump2_keys")  # ????????????????????? "pump2_keys" ??????????????? ??????????????????

    # %%
    info = "?????? Tz??????????????????NLA ??? SSI"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # ??? def ????????? ???????????? is_end = 0????????? kwargs ????????? ???????????? ?????????
    # %%

    if_image_Add_black_border("", img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

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

    if "U" in kwargs:  # ????????? U_amp_plot_save ????????????
        kwargs.pop("U")

    # %% ?????? ????????????

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
                size_PerPixel, dpi, Get("size_fig"),  # is_save = 1 - is_bulk ?????? ?????????????????? ?????? ????????????
                # %%
                cmap_2d, ticks_num, is_contourf,
                is_title_on, is_axes_on, is_mm,
                fontsize, font,
                # %%
                is_colorbar_on, is_save,
                1, 0, 1, 0, ]  # ????????????????????????????????? is_self_colorbar = 0 ????????? 3 ???????????????????????????????????? colorbar???

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

    Tc = 2 * lc

    ticks_Num = num_data_points
    ticks_Num += 1

    # center_times >= 1???????????? Tz ????????????
    samples_zoomout_times = (ticks_Num + 1) // 2 * center_times  # ??????????????????????????? ?????? * ?????? ????????????????????? Tz ????????????
    # print(Tz, Tc)
    if Tz != Tc:
        Gz = 2 * math.pi * mz / (Tz / 1000)  # Tz / 1000 ?????? mm ?????????
        delta_k = abs(dk_z / size_PerPixel + Gz)  # Unit: 1 / mm
    else:
        delta_k = abs(dk_z / size_PerPixel) / samples_zoomout_times  # delta_k ??? ?????????

    # ????????? shift_right ??????????????? ?????? ??? ??????????????????????????? ?????? ????????? x ??????????????????????????? - shift_right
    # x ?????? ??? ????????????????????? ????????? ?????? ?????????????????? Tz ?????? ????????????
    array_1d = np.arange(0, ticks_Num, 1) - (ticks_Num - 1) // 2 - shift_right  # ???????????? ????????????????????????????????????????????? Tz ????????????

    array_dkQ = array_1d * delta_k
    array_Gz = array_dkQ - dk_z / size_PerPixel  # Unit: 1 / mm
    array_Tz = 2 * math.pi * mz / array_Gz  # Unit: mm

    array_dkQ /= 1000  # Unit: 1 / ??m
    array_Tz *= 1000  # Unit: ??m
    # print(array_dkQ)
    print(array_Tz)

    G_energy = []
    G0_energy = []
    G_error_energy = []
    U_energy = []
    U0_energy = []
    U_error_energy = []

    def args_compare_SFG_NLA__SSI(Tz):
        return [
            U_name_Structure,
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
            U_name,
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
            # %%---------------------------------------------------------------------
            # %%
            U_size, w0, w0_Structure, structure_size_Shrink,
            L0_Crystal, z0_structure_frontface_expect, deff_structure_length_expect,
            SSI_zoomout_times, sheets_stored_num,
            z0_section_1_expect, z0_section_2_expect,
            X, Y,
            # %%
            Duty_Cycle_x, Duty_Cycle_y, Duty_Cycle_z,
            structure_xy_mode, Depth,
            # %%
            is_continuous, is_target_far_field, is_transverse_xy,
            is_reverse_xy, is_positive_xy,
            # %%
            is_bulk, is_no_backgroud,
            is_stored, is_show_structure_face, is_energy_evolution_on,
            # %%
            lam1, is_air_pump, is_air, T,
            is_air_pump_structure,
            deff, is_fft, fft_mode,
            is_sum_Gm, mG,
            is_linear_convolution,
            # %%
            Tx, Ty, Tz,
            mx, my, mz,
            is_stripe, is_NLAST,
            # %%
            is_save, is_save_txt, dpi,
            # %%
            color_1d, cmap_2d, cmap_3d,
            elev, azim, alpha,
            # %%
            sample, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm,
            # %%
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
            is_print, is_contours, n_TzQ,
            Gz_max_Enhance, match_mode,
            # %%
            is_NLA, is_amp_relative,
        ]

    if is_twin_pump == 1:
        for key in pump2_keys:
            kwargs[key] = locals()[key]
            kwargs["pump2_keys"] = locals()["pump2_keys"]
    kwargs_compare_SFG_NLA__SSI = copy.deepcopy(kwargs)
    kwargs_compare_SFG_NLA__SSI.update({"ray": kwargs.get("ray", "2"),
                                        "is_end": 0, })
    if is_twin_pump == 1:
        [kwargs.pop(key) for key in kwargs["pump2_keys"]]  # ???????????? kwargs ????????? ?????? ?????????
        kwargs.pop("pump2_keys")  # ????????????????????? "pump2_keys" ??????????????? ??????????????????

    # print(kwargs_compare_SFG_NLA__SSI)
    from fun_global_var import Set
    Set("is_cover_Tz", False)
    for i in range(ticks_Num):
        tuple_temp = \
            compare_SFG_NLA__SSI(*args_compare_SFG_NLA__SSI(array_Tz[i]),
                                 **kwargs_compare_SFG_NLA__SSI, )

        G_energy.append(tuple_temp[0][0])
        G0_energy.append(tuple_temp[0][1])
        G_error_energy.append(tuple_temp[0][2])
        U_energy.append(tuple_temp[1][0])
        U0_energy.append(tuple_temp[1][1])
        U_error_energy.append(tuple_temp[1][2])

    G_energy = np.array(G_energy)  # ????????? list ????????? array
    G0_energy = np.array(G0_energy)
    G_error_energy = np.array(G_error_energy)
    U_energy = np.array(U_energy)
    U0_energy = np.array(U0_energy)
    U_error_energy = np.array(U_error_energy)

    is_end = [0] * (ticks_Num - 1)
    is_end.append(-1)

    is_print and print(tree_print(add_level=1) + "G_energy ??? G_error")
    for i in range(ticks_Num):
        is_print and print(tree_print(is_end[i]) + "Tz, G_error, dkQ, G0_energy, G_energy = {}, {}, {}, {}, {}"
                           .format(format(array_Tz[i], Get("F_f")), format(G_error_energy[i], Get("F_E")),
                                   format(array_dkQ[i], Get("F_E")), format(G0_energy[i], Get("F_E")),
                                   format(G_energy[i], Get("F_E")), ))

    is_print and print(tree_print(is_end=1, add_level=1) + "U_energy ??? U_error")
    for i in range(ticks_Num):
        is_print and print(tree_print(is_end[i]) + "Tz, U_error, dkQ, U0_energy, U_energy = {}, {}, {}, {}, {}"
                           .format(format(array_Tz[i], Get("F_f")), format(U_error_energy[i], Get("F_E")),
                                   format(array_dkQ[i], Get("F_E")), format(U0_energy[i], Get("F_E")),
                                   format(U_energy[i], Get("F_E")), ))

    # if kwargs.get('ray', "2") == "3":  #  ?????? l2 ????????? ??? U_twin_energy_error_plot_save ?????? ??? line2 ??????
    #     [kwargs.pop(key) for key in kwargs["pump2_keys"]]  # ???????????? kwargs ????????? ?????? ?????????
    #     kwargs.pop("pump2_keys")

    GU_error_energy_plot_save(G0_energy, G_energy, G_error_energy,
                              U0_energy, U_energy, U_error_energy,
                              img_name_extension, is_save_txt,
                              # %%
                              array_dkQ, array_Tz, sample, size_PerPixel,
                              is_save, dpi, Get("size_fig_x"), Get("size_fig_y"),
                              # %%
                              color_1d, color_1d2,
                              ticks_num, is_title_on, is_axes_on, is_mm,
                              fontsize, font,  # ???????????? ????????????????????? ???????????? y ??? max ??? min ???????????? ??? ?????? colorbar???????????? is_energy
                              # %%
                              L0_Crystal, **kwargs, )

    # %%


if __name__ == '__main__':
    kwargs = \
        {"U_name_Structure": "",
         "is_phase_only_Structure": 0,
         # %%
         "z_pump_Structure": 0,
         "is_LG_Structure": 0, "is_Gauss_Structure": 1, "is_OAM_Structure": 1,
         "l_Structure": 2, "p_Structure": 0,
         "theta_x_Structure": 0, "theta_y_Structure": 0,
         # %%
         "is_random_phase_Structure": 0,
         "is_H_l_Structure": 0, "is_H_theta_Structure": 0, "is_H_random_phase_Structure": 0,
         # %%
         "U_name": "",
         "img_full_name": "lena1.png",
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
         # %%---------------------------------------------------------------------
         # %%
         "U_size": 1, "w0": 0, "w0_Structure": 0,
         "structure_size_Shrink": 0, "structure_size_Shrinker": 0,
         "is_U_size_x_structure_side_y": 1,
         "L0_Crystal": 1, "z0_structure_frontface_expect": 0, "deff_structure_length_expect": 1,
         # %%
         "SSI_zoomout_times": 1, "sheets_stored_num": 10,
         "z0_section_1_expect": 0, "z0_section_2_expect": 0,
         "X": 0, "Y": 0,
         # %%
         "Duty_Cycle_x": 0.5, "Duty_Cycle_y": 0.5, "Duty_Cycle_z": 0.5,
         "structure_xy_mode": 'x', "Depth": 2,
         # %%
         "is_continuous": 0, "is_target_far_field": 1, "is_transverse_xy": 0,
         "is_reverse_xy": 0, "is_positive_xy": 1,
         # %%
         "is_bulk": 0, "is_no_backgroud": 0,
         "is_stored": 0, "is_show_structure_face": 0, "is_energy_evolution_on": 1,
         # %%
         "lam1": 1.064, "is_air_pump": 1, "is_air": 0, "T": 25,
         "lam_structure": 1.064, "is_air_pump_structure": 1, "T_structure": 25,
         # %%  ?????? ???????????? ??? ???????????????0 ?????? ???????????? "is_birefringence_SHG": 0 ?????? ?????? ?????????
         "is_HOPS_SHG": 0,  # 0.x ?????? ????????????1.x ?????? ?????????????????????2.x ?????? ??????????????????2 ??? ?????? ????????? ???????????????????????? ???????????????????????? ?????????
         "Theta": 0, "Phi": 0,  # ?????? ?????? ??????????????????????????????????????? ?????? ??? ?????????
         # ?????? ?????? ????????????"is_HOPS": ?????? ??????????????????????????????????????? ???????????? H (?????? x) ????????? ????????? ?????? phi_p
         "phi_p": "45", "phi_a": "45",  # ?????? ?????? ????????????"phi_a": str ??????????????????????????????????????? ???????????? H (?????? x) ????????? ????????? ?????? phi_a
         # %%
         "deff": 30, "is_fft": 1, "fft_mode": 0,
         "is_sum_Gm": 0, "mG": 0, 'is_NLAST_sum': 0,
         "is_linear_convolution": 0,
         # %%
         "Tx": 18.769, "Ty": 20, "Tz": 0,
         "mx": 1, "my": 0, "mz": 1,
         "is_stripe": 0, "is_NLAST": 1,
         # %%
         "is_save": 2, "is_no_data_save": 0,
         "is_save_txt": 0, "dpi": 100,
         # %%
         "color_1d": 'b', "color_1d2": 'r', "cmap_2d": 'viridis', "cmap_3d": 'rainbow',
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
         "is_energy": 1,
         # %%
         "plot_group": "UGa", "is_animated": 1,
         "loop": 0, "duration": 0.033, "fps": 5,
         # %%
         "is_plot_EVV": 1, "is_plot_3d_XYz": 0, "is_plot_selective": 0,
         "is_plot_YZ_XZ": 1, "is_plot_3d_XYZ": 0,
         # %%
         "is_print": 1, "is_contours": 0, "n_TzQ": 1,
         "Gz_max_Enhance": 1, "match_mode": 1,
         # %% ????????? ?????? -------------------------------
         "is_NLA": 1, "is_amp_relative": 1,
         "num_data_points": 40, "center_times": 1.5, "shift_right": 0,
         # %% ????????? ?????? ???????????? -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "border_percentage": 0.1, "is_end": -1,
         # %%
         "size_fig_x_scale": 10, "size_fig_y_scale": 4,
         "ax_yscale": 'linear', "xticklabels_rotate": 45,
         # %%
         "theta_z": 90, "phi_z": 90, "phi_c": 23.7,
         # KTP 50 ??? ???deff ????????? 90, ~, 24.3??????24.3 - 2002, 25.3 - 2000???
         #                1994 ???68.8, ~, 90??????68.8 - 2002, 68.9 - 2000???
         # KTP 25 ??? ???deff ????????? 90, ~, 23.7??????23.7 - 2002, 24.8 - 2000???
         #                1994 ???68.8, ~, 90??????68.8 - 2002, 68.7 - 2000???
         # LN 25 ??? ???90, ~, ~
         "polar": "e", "match_type": "oe",
         "polar3": "e", "ray": '3',
         }

    if kwargs.get("ray", "2") == "3" or kwargs.get("is_HOPS_SHG", 0) >= 1:  # ?????? is_HOPS >= 1?????? ?????? ????????? is_twin_pumps == 1
        pump2_kwargs = {
            "U2_name": "",
            "img2_full_name": "lena1.png",
            "is_phase_only_2": 0,
            # %%
            "z_pump2": 0,
            "is_LG_2": 0, "is_Gauss_2": 0, "is_OAM_2": 0,
            "l2": 0, "p2": 0,
            "theta2_x": 0, "theta2_y": 0,
            # %%
            "is_random_phase_2": 0,
            "is_H_l2": 0, "is_H_theta2": 0, "is_H_random_phase_2": 0,
            # %%
            "w0_2": 0,
            # %%
            "lam2": 1.064, "is_air_pump2": 1, "T2": 25,
            "polar2": 'e',
        }
        pump2_kwargs.update({"pump2_keys": list(pump2_kwargs.keys())})
        # Object of type dict_keys is not JSON serializable????????? ????????? list
        kwargs.update(pump2_kwargs)

    kwargs = init_GLV_DICT(**kwargs)
    Tz_compare_SFG_NLA__SSI(**kwargs)
