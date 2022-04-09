# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
import math
from fun_os import U_dir
from fun_img_Resize import if_image_Add_black_border
from fun_pump import pump_pic_or_U
from fun_SSI import slice_SSI
from fun_linear import fft2, ifft2
from fun_nonlinear import Info_find_contours_SHG, G2_z_modulation_NLAST
from fun_thread import my_thread
from fun_CGH import structure_chi2_Generate_2D
from fun_global_var import init_GLV_DICT, init_SSI, end_SSI, dset, dget, fun3, fget, fGHU_plot_save, fU_SSI_plot

np.seterr(divide='ignore', invalid='ignore')


# %%

def NLA_SSI_chi2(U1_name="",
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
                 U1_name_Structure='',
                 structure_size_Enlarge=0.1,
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
                 U1_0_NonZero_size=1, w0=0.3,
                 L0_Crystal=5, z0_structure_frontface_expect=0.5, deff_structure_length_expect=2,
                 sheets_stored_num=10, z0_section_1_expect=1, z0_section_2_expect=1,
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
                 **kwargs, ):
    # %%

    if_image_Add_black_border(U1_name, img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    ray = init_GLV_DICT(U1_name, "2", "SSI", "NLA", **kwargs)

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

    n1, k1, k1_z, lam2, n2, k2, k2_z, \
    dk, lc, Tz, Gx, Gy, Gz, \
    size_PerPixel, U1_0_structure, g1_shift_structure, \
    structure, structure_opposite, modulation, modulation_opposite, modulation_squared, modulation_opposite_squared \
        = structure_chi2_Generate_2D(U1_name_Structure,
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
                                     U1_0_NonZero_size, w0_Structure,
                                     structure_size_Enlarge,
                                     Duty_Cycle_x, Duty_Cycle_y,
                                     structure_xy_mode, Depth,
                                     # %%
                                     is_continuous, is_target_far_field,
                                     is_transverse_xy, is_reverse_xy,
                                     is_positive_xy,
                                     is_bulk, is_no_backgroud,
                                     # %%
                                     lam1, is_air_pump, is_air, T,
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
                                     is_print, )

    L0_Crystal, Tz, deff_structure_length_expect = Info_find_contours_SHG(g1_shift, k1_z, k2_z, Tz, mz,
                                                                          L0_Crystal, size_PerPixel,
                                                                          deff_structure_length_expect,
                                                                          is_print, is_contours, n_TzQ, Gz_max_Enhance,
                                                                          match_mode, )

    # %%

    diz, deff_structure_sheet, \
    sheet_th_frontface, sheets_num_frontface, Iz_frontface, z0_front, \
    sheets_num_structure, Iz_structure, deff_structure_length, \
    sheets_num, Iz, z0, \
    mj, dizj, izj, zj, \
    sheet_th_endface, sheets_num_endface, Iz_endface, z0_end, \
    sheet_th_sec1, sheets_num_sec1, iz_1, z0_1, \
    sheet_th_sec2, sheets_num_sec2, iz_2, z0_2 \
        = slice_SSI(L0_Crystal, size_PerPixel,
                    z0_structure_frontface_expect, deff_structure_length_expect,
                    z0_section_1_expect, z0_section_2_expect,
                    is_stripe, mx, my, Tx, Ty, Tz, Duty_Cycle_z, structure_xy_mode,
                    is_print)

    # %%
    # const

    const = (k2 / size_PerPixel / n2) ** 2 * deff * 1e-12  # pm / V 转换成 m / V

    # %%
    # G2_z0_shift

    init_SSI(g1_shift, U1_0,
             is_energy_evolution_on, is_stored,
             sheets_num, sheets_stored_num,
             X, Y, Iz, size_PerPixel, )

    def H2_zdz(diz):
        return np.power(math.e, k2_z * diz * 1j)
        # 注意 这里的 传递函数 的 指数是 正的 ！！！

    def H2_z(diz):
        return (np.power(math.e, k2_z * diz * 1j) - 1) / k2_z ** 2 * size_PerPixel ** 2
        # 注意 这里的 传递函数 的 指数是 正的 ！！！

    def fun1(for_th, fors_num, *args, **kwargs, ):
        iz = izj[for_th]

        H1_z = np.power(math.e, k1_z * iz * 1j)
        G1_z = g1_shift * H1_z
        U1_z = ifft2(G1_z)

        if is_bulk == 0:
            if for_th >= sheets_num_frontface and for_th <= sheets_num_endface - 1:
                if mj[for_th] == '1':
                    modulation_squared_z = modulation_squared
                elif mj[for_th] == '-1':
                    modulation_squared_z = modulation_opposite_squared
                elif mj[for_th] == '0':
                    # print("???????????????")
                    modulation_squared_z = np.ones((Ix, Iy), dtype=np.int64()) - is_no_backgroud
                else:
                    if structure_xy_mode == 'x':  # 往右（列） 线性平移 mj[for_th] 像素
                        modulation_squared_z = np.roll(modulation_squared, mj[for_th], axis=1)
                    elif structure_xy_mode == 'y':  # 往下（行） 线性平移 mj[for_th] 像素
                        modulation_squared_z = np.roll(modulation_squared, mj[for_th], axis=0)
                    elif structure_xy_mode == 'xy':  # 往右（列） 线性平移 mj[for_th] 像素
                        modulation_squared_z = np.roll(modulation_squared, mj[for_th], axis=1)
                        # modulation_squared_z = np.roll(modulation_squared_z, mj[for_th] / (mx * Tx) * (my * Ty), axis=0)
                        modulation_squared_z = np.roll(modulation_squared_z,
                                                       int(my * Ty / Tz * (izj[for_th] - Iz_frontface)), axis=0)
            else:
                modulation_squared_z = np.ones((Ix, Iy), dtype=np.int64()) - is_no_backgroud
        else:
            modulation_squared_z = np.ones((Ix, Iy), dtype=np.int64()) - is_no_backgroud

        if is_NLAST == 1:
            dG2_zdz = G2_z_modulation_NLAST(k1, k2, 0,
                                            modulation_squared_z, U1_z, dizj[for_th], const, )
        else:
            Q2_z = fft2(modulation_squared_z * U1_z ** 2)
            dG2_zdz = const * Q2_z * H2_z(dizj[for_th])

        return dG2_zdz

    def fun2(for_th, fors_num, dG2_zdz, *args, **kwargs, ):

        dset("G", dget("G") * H2_zdz(dizj[for_th]) + dG2_zdz)

        return dget("G")

    my_thread(10, sheets_num,
              fun1, fun2, fun3,
              is_ordered=1, is_print=is_print, )

    # %%

    end_SSI(g1_shift, is_energy, n_sigma=3, )

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
    NLA_SSI_chi2(U1_name="",
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
                 U1_name_Structure='',
                 structure_size_Enlarge=0.1,
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
                 U1_0_NonZero_size=1, w0=0.3,
                 L0_Crystal=1, z0_structure_frontface_expect=0, deff_structure_length_expect=2,
                 sheets_stored_num=10, z0_section_1_expect=1, z0_section_2_expect=1,
                 X=0, Y=0,
                 # %%
                 is_bulk=0, is_no_backgroud=0,
                 is_stored=0, is_show_structure_face=1, is_energy_evolution_on=1,
                 # %%
                 lam1=0.8, is_air_pump=0, is_air=0, T=25,
                 deff=30,
                 # %%
                 Tx=10, Ty=10, Tz=2.66,
                 mx=0, my=0, mz=1,
                 is_stripe=0, is_NLAST=0,  # 注意，如果 z 向有周期，或是 z 向 无周期的 2d PPLN，这个不能填 0，也就是必须用 NLAST，否则不准；
                 # 如果 斜条纹，则 根本不能用这个 py 文件， 因为 z 向无周期了，必须 划分细小周期
                 # %%
                 # 生成横向结构
                 Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
                 Depth=2, structure_xy_mode='x',
                 #%%
                 is_continuous=0, is_target_far_field=1, is_transverse_xy=0,
                 is_reverse_xy=0, is_positive_xy=1,
                 # %%
                 is_save=1, is_save_txt=0, dpi=100,
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
                 is_print=1, is_contours=66, n_TzQ=1,
                 Gz_max_Enhance=1, match_mode=1,
                 # %%
                 border_percentage=0.1, )
