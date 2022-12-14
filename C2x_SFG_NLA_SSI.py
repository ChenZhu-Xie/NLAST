# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
from fun_img_Resize import if_image_Add_black_border
from fun_pump import pump_pic_or_U
from fun_SSI import slice_SSI
from fun_thread import my_thread
from fun_CGH import chi2_2D
from fun_global_var import init_GLV_DICT, tree_print, init_GLV_rmw, init_SSI, end_SSI, Get, Set, dset, dget, fun3, \
    fget, fkey, fGHU_plot_save, fU_SSI_plot
from a1_AST import init_locals
from a2_AST_EVV import H_zdz
from C1x_SFG_NLA_ssi import gan_U_12VHoe_iterate, gan_U_12VHoe_z_iterate, NLA_iterate_123VHoe

np.seterr(divide='ignore', invalid='ignore')


# %%

def gan_modulation_squared_z_SSI(for_th, for_th_stored, mj, izj,
                                 Ix, Iy, my, Ty, Tz, Iz_frontface,
                                 sheets_num_frontface, sheets_num_endface,
                                 is_bulk, is_stripe, is_no_backgroud,
                                 structure_xy_mode, modulation,
                                 modulation_squared, modulation_opposite_squared,
                                 modulation_lie_down,
                                 m_list, mod_name_list, ):
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
                                              ((Get("border_width_x"), Get("border_width_y")),
                                               (Get("border_width_x"), Get("border_width_y"))),
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
                                              ((Get("border_width_x"), Get("border_width_y")),
                                               (Get("border_width_x"), Get("border_width_y"))),
                                              'constant',
                                              constant_values=(1 - is_no_backgroud, 1 - is_no_backgroud))

                if for_th in for_th_stored:
                    m_list.append(modulation_squared_z)
                    mod_name_list.append("χ2_" + "lie_down_" + str(for_th))

        else:
            modulation_squared_z = np.ones((Ix, Iy), dtype=np.int64()) - is_no_backgroud
    else:
        modulation_squared_z = np.ones((Ix, Iy), dtype=np.int64()) - is_no_backgroud

    return modulation_squared_z, m_list, mod_name_list


# %%

def SFG_NLA_SSI(U_name="",
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
                is_print=1, is_contours=0, n_TzQ=1,
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

    info = "NLA_大步长_SSI"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # kwargs['ray'] = init_GLV_rmw(U_name, "^", "SSI", "NLA", **kwargs) # 更新的传入的 ray 键的值
    init_GLV_rmw(U_name, ray_tag, "NLA", "SSI", **kwargs)  # 不更新 并传入 pump_pic_or_U

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
    dk_z, lc, Gx, Gy, Gz, L0_Crystal, Tz, deff_structure_length_expect, \
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
                                     L0_Crystal=L0_Crystal, g1=g_shift, g2=g2,
                                     # %%
                                     is_air_pump=is_air_pump,
                                     is_plot_n=1, is_print2=1, **kwargs, )
    if is_twin_pump == 1:
        [kwargs.pop(key) for key in kwargs["pump2_keys"]]  # 及时清理 kwargs ，尽量 保持 其干净
        kwargs.pop("pump2_keys")  # 这个有点意思， "pump2_keys" 这个键本身 也会被删除。

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

    # print(dizj)
    # print(zj)

    # %%
    global m_list, mod_name_list
    for_th_stored, m_list, mod_name_list, modulation_lie_down \
        = init_locals("for_th_stored, m_list, mod_name_list, modulation_lie_down")

    if is_stripe > 0:
        from fun_os import U_amp_plot_save
        sheets_stored_num_structure = sheets_stored_num
        # print(len(zj_structure), len(zj), sheets_num)  # sheets_num = len(zj) - 1，因为 最后一层 的 结构 没用于 产生 非线性波
        # print(zj, zj_structure)
        for_th_first = int(mj[0] == '0') * SSI_zoomout_times
        # print(for_th_first)
        for_th_stored = list(np.int64(np.round(np.linspace(0 + for_th_first, len(mj_structure) - 1 + for_th_first,
                                                           sheets_stored_num_structure))))
        # print(for_th_stored, sheets_num-1)
        # print(len(mj_structure), sheets_num)
        m_list, mod_name_list = [], []  # 无论 is_stripe 如何，都初始化他们，因为以后总要用

    if is_stripe == 2.2:
        from fun_CGH import nonrect_chi2_2D
        # if structure_xy_mode == 'x':
        #     Ix_structure, Iy_structure = len(mj_structure), Get("Iy")
        # elif structure_xy_mode == 'y':
        #     Ix_structure, Iy_structure = Get("Ix"), len(mj_structure)
        # Ix_structure, Iy_structure = len(mj_structure), Get("Iy")
        Ix_structure, Iy_structure = len(mj_structure), modulation.shape[1]
        modulation_lie_down, folder_address = \
            nonrect_chi2_2D(z_pump_Structure,
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
        from fun_CGH import interp2d_nonrect_chi2_2D
        modulation_lie_down = interp2d_nonrect_chi2_2D(folder_address, modulation,
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
        global m_list, mod_name_list
        modulation_squared_z, m_list, mod_name_list = \
            gan_modulation_squared_z_SSI(for_th, for_th_stored, mj, izj,
                                         Ix, Iy, my, Ty, Tz, Iz_frontface,
                                         sheets_num_frontface, sheets_num_endface,
                                         is_bulk, is_stripe, is_no_backgroud,
                                         structure_xy_mode, modulation,
                                         modulation_squared, modulation_opposite_squared,
                                         modulation_lie_down,
                                         m_list, mod_name_list, )

        U1_z, U2_z, U1o_z, U1e_z, \
        U1_Vo_z, U1_Ve_z, U1_Ho_z, U1_He_z = \
            gan_U_12VHoe_z_iterate(is_birefringence_deduced, is_air,
                                   for_th, is_add_polarizer,
                                   gan_U_oe, gan_U_VHoe, gan_U12, )

        dG3_zdz = NLA_iterate_123VHoe(is_birefringence_deduced, is_air,
                                      is_add_polarizer, is_NLAST,
                                      dizj[for_th], size_PerPixel,
                                      const, match_type,
                                      k1o, k1e, k1_Vo, k1_Ve, k1_Ho, k1_He,
                                      k1, k2, k3, k3_z,
                                      U1_z, U2_z,
                                      U1o_z, U1e_z, U1_Vo_z, U1_Ve_z, U1_Ho_z, U1_He_z,
                                      modulation_squared_z, )
        return dG3_zdz

    def fun2(for_th, fors_num, dG3_zdz, *args, **kwargs, ):

        dset("G", dget("G") * H_zdz(k3_z, dizj[for_th]) + dG3_zdz)

        return dget("G")

    my_thread(10, sheets_num,
              fun1, fun2, fun3,
              is_ordered=1, is_print=is_print, )

    if is_stripe > 0:
        for i in range(sheets_stored_num_structure):
            U_amp_plot_save(m_list[i], mod_name_list[i],
                            [], folder_address,
                            Get("img_name_extension"), is_save_txt,
                            # %%
                            size_PerPixel, dpi, Get("size_fig"),  # is_save = 1 - is_bulk 改为 不储存，因为 反正 都储存了
                            # %%
                            cmap_2d, ticks_num, is_contourf,
                            is_title_on, is_axes_on, is_mm,
                            fontsize, font,
                            # %%
                            is_colorbar_on, 0,
                            1, 0, 0, 0,
                            # %%
                            suffix="", **kwargs, )

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
    if kwargs.get("p_inspect", inspect.stack()[1][3]) == "SFG_NLA_reverse":
        from fun_statistics import find_Kxyz
        from fun_linear import fft2
        K1_z, K1_xy = find_Kxyz(fft2(U_0), k1)
        K2_z, K2_xy = find_Kxyz(fft2(U2_0), k2)
        kiizQ = K1_z + K2_z + Gz
        # print(np.max(np.abs(fft2(fget("U")) / Get("size_PerPixel") ** 2)))

        return fget("U"), U_0, U2_0, modulation_squared, k1_inc, k2_inc, \
               theta_x, theta_y, theta2_x, theta2_y, kiizQ, \
               k1, k2, k3, const, Iz, Gz
    elif kwargs.get("p_inspect", inspect.stack()[1][3]) == "SFG_NLA_SSI__AST_EVV":
        Set("k3", k3)
        Set("lam3", lam3)

        return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U")
    else:
        return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U")


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",
         "img_full_name": "lena1.png",
         "U_pixels_x": 300, "U_pixels_y": 300,
         "is_phase_only": 0,
         # %%
         "z_pump": -5,
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
         "L0_Crystal": 10, "z0_structure_frontface_expect": 0, "deff_structure_length_expect": 2,
         "SSI_zoomout_times": 1, "sheets_stored_num": 10,
         "z0_section_1_expect": 1, "z0_section_2_expect": 1,
         "X": 0, "Y": 0,
         # %%
         "is_bulk": 0, "is_no_backgroud": 0,
         "is_stored": 1, "is_show_structure_face": 1, "is_energy_evolution_on": 1,
         # %%
         "lam1": 1.064, "is_air_pump": 1, "is_air": 2, "T": 25,
         "lam_structure": 1.064, "is_air_pump_structure": 1, "T_structure": 25,
         "deff": 30,
         # %%  控制 单双泵浦 和 绘图方式：0 代表 无双折射 "is_birefringence_SHG": 0 是否 考虑 双折射
         "is_HOPS_SHG": 1,  # 0.x 代表 单泵浦，1.x 代表 高阶庞加莱球，2.x 代表 最广义情况：2 个 线偏 标量场 叠加；这些都是在 左手系下，且都是 线偏基
         "Theta": 0, "Phi": 0,  # 是否 采用 高阶加莱球、若采用，请给出 极角 和 方位角
         # 是否 使用 起偏器（"is_HOPS": 整数 即不使用）、若使用，请给出 其相对于 H (水平 x) 方向的 逆时针 转角 phi_p
         "phi_p": "45", "phi_a": "45",  # 是否 使用 检偏器（"phi_a": str 则不使用）、若使用，请给出 其相对于 H (水平 x) 方向的 逆时针 转角 phi_a
         # %%
         "Tx": 18.769, "Ty": 20, "Tz": 500,
         "mx": 0, "my": 0, "mz": 0,
         "is_stripe": 0, "is_NLAST": 1,  # 注意，如果 z 向有周期，或是 z 向 无周期的 2d PPLN，这个不能填 0，也就是必须用 NLAST，否则不准；
         # 如果 斜条纹，则 根本不能用这个 py 文件， 因为 z 向无周期了，必须 划分细小周期
         # %%
         # 生成横向结构
         "Duty_Cycle_x": 0.5, "Duty_Cycle_y": 0.5, "Duty_Cycle_z": 0.5,
         "Depth": 2, "structure_xy_mode": 'x',
         # %%
         "is_continuous": 0, "is_target_far_field": 1, "is_transverse_xy": 0,
         "is_reverse_xy": 0, "is_positive_xy": 1,
         # %%
         "is_save": 0, "is_no_data_save": 0,
         "is_save_txt": 0, "dpi": 100,
         # %%
         "color_1d": 'b', "cmap_2d": 'viridis', "cmap_3d": 'rainbow',
         "elev": 10, "azim": -65, "alpha": 2,
         # %%
         "sample": 1, "ticks_num": 7, "is_contourf": 0,
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
         "plot_group": "UGa", "is_animated": 1,
         "loop": 0, "duration": 0.033, "fps": 5,
         # %%
         "is_plot_EVV": 1, "is_plot_3d_XYz": 0, "is_plot_selective": 0,
         "is_plot_YZ_XZ": 0, "is_plot_3d_XYZ": 0,
         # %%
         "is_print": 1, "is_contours": 0, "n_TzQ": 1,
         "Gz_max_Enhance": 1, "match_mode": 1,
         # %% 该程序 作为 主入口时 -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "border_percentage": 0.1, "is_end": -1,
         # %%
         "size_fig_x_scale": 10, "size_fig_y_scale": 2,
         # %%
         "theta_z": 90, "phi_z": 90, "phi_c": 23.8,  # LN 的 phi_c 为什么 不能填 0
         # KTP 50 度 ：deff 最高： 90, ~, 24.3，（24.3 - 2002, 25.3 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.9 - 2000）
         # KTP 25 度 ：deff 最高： 90, ~, 23.7，（23.7 - 2002, 24.8 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.7 - 2000）
         # LN 25 度 ：90, ~, ~
         "polar": "R", "match_type": "oe",
         "polar3": "o", "ray": "3",
         }

    if kwargs.get("ray", "2") == "3" or kwargs.get("is_HOPS_SHG", 0) >= 1:  # 如果 is_HOPS >= 1，则 默认 双泵浦 is_twin_pumps == 1
        pump2_kwargs = {
            "U2_name": "",
            "img2_full_name": "lena1.png",
            "is_phase_only_2": 0,
            # %%
            "z_pump2": -5,
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
    SFG_NLA_SSI(**kwargs)
