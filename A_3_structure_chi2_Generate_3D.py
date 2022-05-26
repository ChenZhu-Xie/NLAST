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
from fun_nonlinear import Info_find_contours_SHG
from fun_thread import noop, my_thread
from fun_CGH import structure_chi2_Generate_2D

np.seterr(divide='ignore', invalid='ignore')


# %%

def structure_chi2_3D(U_name="",
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
                      U_NonZero_size=1, w0=0.3, structure_size_Enlarge=0.1,
                      deff_structure_length_expect=2,
                      # %%
                      Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
                      structure_xy_mode='x', Depth=2, ssi_zoomout_times=5,
                      # %%
                      is_continuous=1, is_target_far_field=1, is_transverse_xy=0,
                      is_reverse_xy=0, is_positive_xy=1, is_no_backgroud=1,
                      # %%
                      lam1=0.8, is_air_pump_structure=0, is_air=0, T=25,
                      # %%
                      Tx=10, Ty=10, Tz="2*lc",
                      mx=0, my=0, mz=0,
                      is_stripe=0,
                      # %%
                      is_save=0, is_save_txt=0, dpi=100,
                      is_bulk=1,
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
    # 预处理 导入图片 为方形，并加边框

    if_image_Add_black_border(U_name, img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%

    info = "χ2_3D_生成结构"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    # %%
    # 提供描边信息，并覆盖值

    # 这里 传 deff_structure_length_expect 进去 而不是 z0，是有问题的，导致只有 周期 Tz 能与 NLA_SSI 保持一致，长度并不能，
    # 这样若 deff_structure_length_expect < NLA_SSI 中的 z0 则 无法读取到 > deff_structure_length_expect 的 结构，只能手动在 A_to_B_3_NLA_SSI 中设置 deff_structure_length_expect 比 z0 大
    # 并不打算改这一点，因为否则的话，需要向这个函数传入一个参数，而这个参数却是之后要引用的函数 NLA_SSI 才能给出的，违反了 因果律

    n1_inc, n1, k1_inc, k1, k1_z, n2_inc, n2, k2_inc, k2, k2_z, lam3, n3_inc, n3, k3_inc, k3, k3_z, \
    theta3_x, theta3_y, z0_recommend, deff_structure_length_expect, dk, lc, Tz, Gx, Gy, Gz, folder_address, \
    size_PerPixel, U_0, g_shift, \
    structure, structure_opposite, modulation, modulation_opposite, modulation_squared, modulation_opposite_squared \
        = structure_chi2_Generate_2D(U_name,
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
                                     U_NonZero_size, w0,
                                     structure_size_Enlarge,
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
                                     **kwargs, )
    if kwargs.get('ray', "2") == "3":
        [kwargs.pop(key) for key in kwargs["pump2_keys"]]  # 及时清理 kwargs ，尽量 保持 其干净
        kwargs.pop("pump2_keys")  # 这个有点意思， "pump2_keys" 这个键本身 也会被删除。

    # %%
    # 定义 调制区域切片厚度 的 纵向实际像素、调制区域切片厚度 的 实际纵向尺寸
    # 定义 调制区域 的 纵向实际像素、调制区域 的 实际纵向尺寸
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
        from fun_CGH import structure_nonrect_chi2_Generate_2D
        if structure_xy_mode == 'x':
            Ix_structure, Iy_structure = sheets_num, Get("Iy")
        elif structure_xy_mode == 'y':
            Ix_structure, Iy_structure = Get("Ix"), sheets_num
        modulation_lie_down, folder_address = \
            structure_nonrect_chi2_Generate_2D(z_pump,
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
    elif is_stripe == 2 or is_stripe == 2.1:  # 躺下 的 插值算法
        from fun_CGH import structure_nonrect_chi2_interp2d_2D
        modulation_lie_down = structure_nonrect_chi2_interp2d_2D(folder_address, modulation_squared,
                                                                 structure_xy_mode, sheets_num,
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
    # 逐层 绘制 并 输出 structure

    mj = []

    def fun1(for_th, fors_num, *args, **kwargs, ):
        iz = for_th * diz
        step_nums_left, step_nums_right, step_nums_total = gcd_of_float(Duty_Cycle_z)[1]

        if mz != 0:  # 如果 要用 Tz，则如下 分层；

            if is_stripe == 0:
                # print(iz - iz // Tz_unit * Tz_unit)
                # if iz - iz // Tz_unit * Tz_unit < Tz_unit * Duty_Cycle_z:  # 如果 左端面 小于 占空比 【减去一个微小量（比如 diz / 10）】，则以 正向畴结构 输出为 该端面结构
                if np.mod(for_th, step_nums_total * ssi_zoomout_times) < step_nums_left * ssi_zoomout_times:
                    m = modulation_squared
                    mj.append("1")
                else:  # 如果 左端面 大于等于 占空比，则以 反向畴结构 输出为 该端面结构
                    m = modulation_opposite_squared
                    mj.append("-1")
            elif is_stripe == 1:  # 将 modulation_squared 随着 z 的 增加 而 左右上下（x - y 面内） 滑动
                if structure_xy_mode == 'x':  # 往右（列） 线性平移 mj[for_th] 像素
                    mj.append(int(mx * Tx / Tz * iz))
                    m = np.roll(modulation_squared, mj[-1], axis=1)
                elif structure_xy_mode == 'y':  # 往下（行） 线性平移 mj[for_th] 像素
                    mj.append(int(my * Ty / Tz * iz))
                    m = np.roll(modulation_squared, mj[-1], axis=0)
                elif structure_xy_mode == 'xy':  # 往右（列） 线性平移 mj[for_th] 像素
                    mj.append(int(mx * Tx / Tz * iz))
                    m = np.roll(modulation_squared, mj[-1], axis=1)
                    m = np.roll(modulation_squared, int(my * Ty / Tz * iz), axis=0)

                if for_th in for_th_stored:
                    m_list.append(m)
                    mod_name_list.append("χ2_" + "tran_shift_" + str(for_th))

            elif is_stripe == 2 or is_stripe == 2.1 or is_stripe == 2.2:  # 躺下 的 插值算法 & 直接 CGH 算法
                if structure_xy_mode == 'x':
                    modulation_squared_new = np.tile(modulation_lie_down[for_th], (Get("Ix"), 1))  # 按行复制 多行，成一个方阵
                elif structure_xy_mode == 'y':
                    modulation_squared_new = np.tile(modulation_lie_down[:, for_th], (Get("Iy"), 1))  # 按列复制 多列，成一个方阵
                m = modulation_squared_new

                if for_th in for_th_stored:
                    m_list.append(m)
                    mod_name_list.append("χ2_" + "lie_down_" + str(for_th))

            modulation_squared_full_name = str(for_th) + (is_save_txt and ".txt" or ".mat")
            modulation_squared_address = folder_address + "\\" + modulation_squared_full_name

            if is_bulk == 0:  # 不用 U_save 也就不储存信息：直接输出；因为这个会大量输出结构，并且信息简单
                np.savetxt(modulation_squared_address, m, fmt='%i') if is_save_txt else savemat(
                    modulation_squared_address, {'chi2_modulation_squared': m})

        else:  # 如果不用 Tz，则 z 向 无结构，则一直输出 正向畴

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

    # print(mj)
    # print(len(mj))


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",
         "img_full_name": "lena1.png",
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
         "U_NonZero_size": 1, "w0": 0, "structure_size_Enlarge": 0.1,
         "deff_structure_length_expect": 1,
         # %%
         "Duty_Cycle_x": 0.5, "Duty_Cycle_y": 0.5, "Duty_Cycle_z": 0.5,
         "structure_xy_mode": 'x', "Depth": 2, "ssi_zoomout_times": 1,
         # %%
         "is_continuous": 0, "is_target_far_field": 1, "is_transverse_xy": 0,
         "is_reverse_xy": 0, "is_positive_xy": 1, "is_no_backgroud": 0,
         # %%
         "lam1": 0.8, "is_air_pump_structure": 0, "is_air": 0, "T": 25,
         # %%
         "Tx": 30, "Ty": 20, "Tz": 0,
         "mx": 1, "my": 0, "mz": 1,
         "is_stripe": 2.2,
         # %%
         "is_save": 0, "is_save_txt": 0, "dpi": 100,
         "is_bulk": 0,
         # %%
         "cmap_2d": 'viridis',
         # %%
         "ticks_num": 6, "is_contourf": 0,
         "is_title_on": 1, "is_axes_on": 1, "is_mm": 1,
         # %%
         "fontsize": 7,
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
         # %% 该程序 作为 主入口时 -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "border_percentage": 0.1, "is_end": -1,
         # %%
         "ray": "2", "polar_structure": "e",
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
            "lam2": 1, "is_air_pump2": 0, "T2": 25,
            "polar2": 'e',
        }
        pump2_kwargs.update({"pump2_keys": list(pump2_kwargs.keys())})
        # Object of type dict_keys is not JSON serializable，所以 得转为 list
        kwargs.update(pump2_kwargs)

    kwargs = init_GLV_DICT(**kwargs)
    structure_chi2_3D(**kwargs)

    # structure_chi2_3D(U_name="",
    #                   img_full_name="Grating.png",
    #                   is_phase_only=0,
    #                   # %%
    #                   z_pump=0,
    #                   is_LG=0, is_Gauss=0, is_OAM=0,
    #                   l=0, p=0,
    #                   theta_x=0, theta_y=0,
    #                   # %%
    #                   is_random_phase=0,
    #                   is_H_l=0, is_H_theta=0, is_H_random_phase=0,
    #                   # %%
    #                   U_NonZero_size=1, w0=0.3, structure_size_Enlarge=0.1,
    #                   deff_structure_length_expect=2,
    #                   # %%
    #                   Duty_Cycle_x=0.5, Duty_Cycle_y=0.5, Duty_Cycle_z=0.5,
    #                   structure_xy_mode='x', Depth=2, ssi_zoomout_times=5,
    #                   # %%
    #                   is_continuous=1, is_target_far_field=1, is_transverse_xy=0,
    #                   is_reverse_xy=0, is_positive_xy=1, is_no_backgroud=1,
    #                   # %%
    #                   lam1=0.8, is_air_pump_structure=0, is_air=0, T=25,
    #                   # %%
    #                   Tx=10, Ty=10, Tz="2*lc",
    #                   mx=0, my=0, mz=0,
    #                   is_stripe=0,
    #                   # %%
    #                   is_save=0, is_save_txt=0, dpi=100,
    #                   is_bulk=1,
    #                   # %%
    #                   cmap_2d='viridis',
    #                   # %%
    #                   ticks_num=6, is_contourf=0,
    #                   is_title_on=1, is_axes_on=1, is_mm=1,
    #                   # %%
    #                   fontsize=9,
    #                   font={'family': 'serif',
    #                         'style': 'normal',  # 'normal', 'italic', 'oblique'
    #                         'weight': 'normal',
    #                         'color': 'black',  # 'black','gray','darkred'
    #                         },
    #                   # %%
    #                   is_colorbar_on=1, is_energy=0,
    #                   # %%
    #                   is_print=1, is_contours=1, n_TzQ=1,
    #                   Gz_max_Enhance=1, match_mode=1,
    #                   # %%
    #                   root_dir=r'',
    #                   border_percentage=0.1, is_end=-1, )
