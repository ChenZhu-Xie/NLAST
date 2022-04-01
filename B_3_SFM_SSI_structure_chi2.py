# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import os
import numpy as np
import math
from scipy.io import savemat
from fun_os import img_squared_bordered_Read, U_Read, U_dir, U_energy_print, U_plot, U_save
from fun_img_Resize import image_Add_black_border
from fun_plot import plot_1d, plot_2d, plot_3d_XYZ, plot_3d_XYz
from fun_pump import pump_LG
from fun_SSI import cal_Iz_frontface, cal_Iz_endface_1, cal_Iz_endface_2, cal_Iz, cal_diz, cal_zj_mj_structure, \
    cal_zj_izj_dizj_mj, cal_iz_1, cal_iz_2
from fun_linear import Cal_n
from fun_nonlinear import Eikz, Info_find_contours_SHG
from fun_thread import my_thread
from fun_CGH import structure_chi2_Generate_2D
from fun_statistics import U_Drop_n_sigma

np.seterr(divide='ignore', invalid='ignore')


# %%

def SFM_SSI_chi2(U1_name="",
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
                 is_print=1, is_contours=1, n_TzQ=1,
                 Gz_max_Enhance=1, match_mode=1,
                 # %%
                 **kwargs, ):
    # %%

    location = os.path.dirname(os.path.abspath(__file__))  # 其实不需要，默认就是在 相对路径下 读，只需要 文件名 即可

    if (type(U1_name) != str) or U1_name == "":

        if __name__ == "__main__":
            border_percentage = kwargs["border_percentage"] if len(kwargs) != 0 else 0.1

            image_Add_black_border(img_full_name,  # 预处理 导入图片 为方形，并加边框
                                   border_percentage,
                                   is_print, )

        # %%
        # 导入 方形，以及 加边框 的 图片

        img_name, img_name_extension, img_squared, \
        size_PerPixel, size_fig, I1_x, I1_y, U1_0 = img_squared_bordered_Read(img_full_name,
                                                                              U1_0_NonZero_size, dpi,
                                                                              is_phase_only)

        # %%
        # 预处理 输入场

        n1, k1 = Cal_n(size_PerPixel,
                       is_air_pump,
                       lam1, T, p="e")

        U1_0, g1_shift = pump_LG(img_full_name,
                                 I1_x, I1_y, size_PerPixel,
                                 U1_0, w0, k1, z_pump,
                                 is_LG, is_Gauss, is_OAM,
                                 l, p,
                                 theta_x, theta_y,
                                 is_random_phase,
                                 is_H_l, is_H_theta, is_H_random_phase,
                                 is_save, is_save_txt, dpi,
                                 cmap_2d, ticks_num, is_contourf,
                                 is_title_on, is_axes_on, is_mm,
                                 fontsize, font,
                                 is_colorbar_on, is_energy,
                                 is_print, )

    else:

        # %%
        # 导入 方形 的 图片，以及 U

        img_name, img_name_extension, img_squared, \
        size_PerPixel, size_fig, I1_x, I1_y, U1_0 = U_Read(U1_name, img_full_name,
                                                           U1_0_NonZero_size, dpi,
                                                           is_save_txt, )

    # %%
    # 非线性 角谱理论 - SSI begin

    I2_x, I2_y = U1_0.shape[0], U1_0.shape[1]

    # %%
    # 生成横向结构

    n1, k1, k1_z_shift, lam2, n2, k2, k2_z_shift, \
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
                                     is_positive_xy, is_no_backgroud,
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

    # %%
    # 提供描边信息，并覆盖值

    L0_Crystal, Tz, deff_structure_length_expect = Info_find_contours_SHG(g1_shift, k1_z_shift, k2_z_shift, Tz, mz,
                                                                          L0_Crystal, size_PerPixel,
                                                                          deff_structure_length_expect,
                                                                          is_print, is_contours, n_TzQ, Gz_max_Enhance,
                                                                          match_mode, )

    # %%
    # 定义 结构前端面 距离 晶体前端面 的 纵向实际像素、结构前端面 距离 晶体前端面 的 实际纵向尺寸

    sheet_th_frontface, sheets_num_frontface, Iz_frontface, z0_structure_frontface \
        = cal_Iz_frontface(z0_structure_frontface_expect, L0_Crystal, size_PerPixel,
                           is_print, )

    # %%
    # 定义 结构后端面 距离 晶体前端面 的 纵向实际像素、结构后端面 距离 晶体前端面 的 实际纵向尺寸 1

    deff_structure_length, z0_structure_endface \
        = cal_Iz_endface_1(z0_structure_frontface, deff_structure_length_expect, L0_Crystal,
                           is_print, )

    # %%
    # 定义 调制区域 的 纵向实际像素、调制区域 的 实际纵向尺寸

    Iz_structure = deff_structure_length / size_PerPixel

    # %%
    # 定义 调制区域切片厚度 的 纵向实际像素、调制区域切片厚度 的 实际纵向尺寸

    sheets_num_structure, diz, deff_structure_sheet \
        = cal_diz(Duty_Cycle_z, Tz, Iz_structure, size_PerPixel,
                  is_print, )

    # %%
    # 定义 结构后端面 距离 晶体前端面 的 纵向实际像素、结构后端面 距离 晶体前端面 的 实际纵向尺寸 2

    sheet_th_endface, sheets_num_endface, Iz_endface \
        = cal_Iz_endface_2(sheets_num_frontface, sheets_num_structure, Iz_frontface, Iz_structure, )

    # %%
    # 定义 晶体 的 纵向实际像素、晶体 的 实际纵向尺寸

    sheets_num, Iz, z0 \
        = cal_Iz(sheets_num_endface, z0_structure_endface, L0_Crystal, size_PerPixel,
                 is_print, )

    # %%
    # 生成 structure 各层 z 序列，以及 正负畴 序列信息 mj

    zj_structure, mj_structure \
        = cal_zj_mj_structure(Duty_Cycle_z, deff_structure_sheet, sheets_num_structure, z0_structure_frontface,
                              z0_structure_endface,
                              is_stripe, mx, my, Tx, Ty, Tz, structure_xy_mode, size_PerPixel, )

    # %%
    # 生成 晶体内 各层 z 序列、izj、dizj，以及 正负畴 序列信息 mj

    zj, izj, dizj, mj \
        = cal_zj_izj_dizj_mj(zj_structure, mj_structure, z0_structure_frontface, z0_structure_endface, L0_Crystal,
                             size_PerPixel)

    # %%
    # 定义 需要展示的截面 1 距离晶体前端面 的 纵向实际像素、需要展示的截面 1 距离晶体前端面 的 实际纵向尺寸

    sheet_th_section_1, sheets_num_section_1, Iz_1, z0_1 \
        = cal_iz_1(zj, z0_section_1_expect, size_PerPixel,
                   is_print, )

    # %%
    # 定义 需要展示的截面 2 距离晶体后端面 的 纵向实际像素、需要展示的截面 2 距离晶体后端面 的 实际纵向尺寸

    sheet_th_section_2, sheets_num_section_2, Iz_2, z0_2 \
        = cal_iz_2(zj, deff_structure_length_expect, z0_section_2_expect, size_PerPixel,
                   is_print, )

    # %%
    # const

    deff = deff * 1e-12  # pm / V 转换成 m / V
    const = (k2 / size_PerPixel / n2) ** 2 * deff

    # %%
    # G2_z0_shift

    cal_mode = [1, 1, 0]
    # 以 G 算 还是以 U 算、源项 是否 也衍射、k_2z 是否是 matrix 版
    # 用 G 算 会快很多
    # 不管是 G 还是 U，matrix 版的能量 总是要低一些，只不过 U 低得少些，没有数量级差异，而 G 少得很多

    if cal_mode[0] == 1:  # 如果以 G 算
        global G2_z_plus_dz_shift
    else:
        global U2_z_plus_dz
    G2_z_plus_dz_shift = np.zeros((I2_x, I2_y), dtype=np.complex128())
    U2_z_plus_dz = np.zeros((I2_x, I2_y), dtype=np.complex128())

    if is_energy_evolution_on == 1:
        G2_z_shift_energy = np.zeros((sheets_num + 1), dtype=np.float64())
        U2_z_energy = np.zeros((sheets_num + 1), dtype=np.float64())
    G2_z_shift_energy[0] = np.sum(np.abs(G2_z_plus_dz_shift) ** 2)
    U2_z_energy[0] = np.sum(np.abs(U2_z_plus_dz) ** 2)

    # %%

    # H2_z_plus_dz_shift_k2_z = np.power(math.e, k2_z_shift * diz * 1j) # 注意 这里的 传递函数 的 指数是 正的 ！！！
    # H2_z_shift_k2_z = (np.power(math.e, k2_z_shift * diz * 1j) - 1) / k2_z_shift**2 * size_PerPixel**2 # 注意 这里的 传递函数 的 指数是 正的 ！！！
    # H2_z_plus_dz_shift_k2_z_temp = np.power(math.e, k2_z_shift * leftover * 1j) # 注意 这里的 传递函数 的 指数是 正的 ！！！
    # H2_z_shift_k2_z_temp = (np.power(math.e, k2_z_shift * leftover * 1j) - 1) / k2_z_shift**2 * size_PerPixel**2 # 注意 这里的 传递函数 的 指数是 正的 ！！！

    # %%

    def H2_z_plus_dz_shift_k2_z(diz):
        return np.power(math.e, k2_z_shift * diz * 1j)

    if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版

        dk_z_shift = 2 * k1_z_shift - k2_z_shift

        def H2_z_shift_k2_z(diz):
            return np.power(math.e, k2_z_shift * diz * 1j) / (k2_z_shift / size_PerPixel) * Eikz(
                dk_z_shift * diz) * diz * size_PerPixel \
                   * (2 / (dk_z_shift / k2_z_shift + 2))
    else:
        def H2_z_shift_k2_z(diz):
            return np.power(math.e, k2 * diz * 1j) / (k2 / size_PerPixel) * Eikz(dk * diz) * diz * size_PerPixel \
                   * (2 / (dk / k2 + 2))

    # %%

    if is_stored == 1:
        # sheet_stored_th = np.zeros( (sheets_stored_num + 1), dtype=np.int64() ) # 这个其实 就是 0123...
        sheet_th_stored = np.zeros(int(sheets_stored_num + 1), dtype=np.int64())
        iz_stored = np.zeros(int(sheets_stored_num + 1), dtype=np.float64())
        z_stored = np.zeros(int(sheets_stored_num + 1), dtype=np.float64())
        G2_z_shift_stored = np.zeros((I2_x, I2_y, int(sheets_stored_num + 1)), dtype=np.complex128())
        U2_z_stored = np.zeros((I2_x, I2_y, int(sheets_stored_num + 1)), dtype=np.complex128())

        # 小写的 x,y 表示 电脑中 矩阵坐标系，大写 X,Y 表示 笛卡尔坐标系
        # G2_shift_xz_stored = np.zeros( (I2_x, sheets_num + 1), dtype=np.complex128() )
        # G2_shift_yz_stored = np.zeros( (I2_y, sheets_num + 1), dtype=np.complex128() )
        # U2_xz_stored = np.zeros( (I2_x, sheets_num + 1), dtype=np.complex128() )
        # U2_yz_stored = np.zeros( (I2_y, sheets_num + 1), dtype=np.complex128() )
        G2_shift_YZ_stored = np.zeros((I2_x, sheets_num + 1), dtype=np.complex128())
        G2_shift_XZ_stored = np.zeros((I2_y, sheets_num + 1), dtype=np.complex128())
        U2_YZ_stored = np.zeros((I2_x, sheets_num + 1), dtype=np.complex128())
        U2_XZ_stored = np.zeros((I2_y, sheets_num + 1), dtype=np.complex128())

        G2_structure_frontface_shift = np.zeros((I2_x, I2_y), dtype=np.complex128())
        U2_structure_frontface = np.zeros((I2_x, I2_y), dtype=np.complex128())
        G2_structure_endface_shift = np.zeros((I2_x, I2_y), dtype=np.complex128())
        U2_structure_endface = np.zeros((I2_x, I2_y), dtype=np.complex128())
        G2_section_1_shift = np.zeros((I2_x, I2_y), dtype=np.complex128())
        U2_section_1 = np.zeros((I2_x, I2_y), dtype=np.complex128())
        G2_section_2_shift = np.zeros((I2_x, I2_y), dtype=np.complex128())
        U2_section_2 = np.zeros((I2_x, I2_y), dtype=np.complex128())

    if cal_mode[0] == 1:  # 如果以 G 算

        def Cal_dG2_z_plus_dz_shift(for_th, fors_num, *arg, ):
            iz = izj[for_th]

            H1_z_shift = np.power(math.e, k1_z_shift * iz * 1j)
            G1_z_shift = g1_shift * H1_z_shift
            G1_z = np.fft.ifftshift(G1_z_shift)
            U1_z = np.fft.ifft2(G1_z)

            if is_bulk == 0:
                if for_th >= sheets_num_frontface and for_th <= sheets_num_endface - 1:
                    if mj[for_th] == '1':
                        modulation_squared_z = modulation_squared
                    elif mj[for_th] == '-1':
                        modulation_squared_z = modulation_opposite_squared
                    elif mj[for_th] == '0':
                        # print("???????????????")
                        modulation_squared_z = np.ones((I2_x, I2_y), dtype=np.int64()) - is_no_backgroud
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
                    modulation_squared_z = np.ones((I2_x, I2_y), dtype=np.int64()) - is_no_backgroud
            else:
                modulation_squared_z = np.ones((I2_x, I2_y), dtype=np.int64()) - is_no_backgroud

            if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版
                if cal_mode[1] == 1:  # 若 源项 也衍射
                    Q2_z = np.fft.fft2(
                        modulation_squared_z * U1_z ** 2 * H2_z_shift_k2_z(dizj[for_th]) / np.power(math.e,
                                                                                                    k2_z_shift * diz * 1j))
                else:
                    Q2_z = np.fft.fft2(modulation_squared_z * U1_z ** 2 * H2_z_shift_k2_z(dizj[for_th]))
            else:
                Q2_z = np.fft.fft2(modulation_squared_z * U1_z ** 2)

            Q2_z_shift = np.fft.fftshift(Q2_z)

            if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版
                dG2_z_plus_dz_shift = const * Q2_z_shift
            else:
                if cal_mode[1] == 1:  # 若 源项 也衍射
                    dG2_z_plus_dz_shift = const * Q2_z_shift * H2_z_shift_k2_z(dizj[for_th]) / np.power(math.e,
                                                                                                        k2 * diz * 1j)
                else:
                    dG2_z_plus_dz_shift = const * Q2_z_shift * H2_z_shift_k2_z(dizj[for_th])

            return dG2_z_plus_dz_shift

        def Cal_G2_z_plus_dz_shift(for_th, fors_num, dG2_z_plus_dz_shift, *arg, ):

            global G2_z_plus_dz_shift

            if cal_mode[1] == 1:  # 若 源项 也衍射
                G2_z_plus_dz_shift = (G2_z_plus_dz_shift + dG2_z_plus_dz_shift) * H2_z_plus_dz_shift_k2_z(dizj[for_th])
            else:
                G2_z_plus_dz_shift = G2_z_plus_dz_shift * H2_z_plus_dz_shift_k2_z(dizj[for_th]) + dG2_z_plus_dz_shift

            return G2_z_plus_dz_shift

        def After_G2_z_plus_dz_shift_temp(for_th, fors_num, G2_z_plus_dz_shift_temp, *arg, ):

            if is_stored == 1:
                global G2_structure_frontface_shift, U2_structure_frontface, G2_structure_endface_shift, U2_structure_endface, G2_section_1_shift, U2_section_1, G2_section_2_shift, U2_section_2

            G2_z_plus_dz = np.fft.ifftshift(G2_z_plus_dz_shift_temp)
            U2_z_plus_dz = np.fft.ifft2(G2_z_plus_dz)

            if is_energy_evolution_on == 1:
                G2_z_shift_energy[for_th + 1] = np.sum(np.abs(G2_z_plus_dz_shift_temp) ** 2)
                U2_z_energy[for_th + 1] = np.sum(np.abs(U2_z_plus_dz) ** 2)

            if is_stored == 1:
                # 小写的 x,y 表示 电脑中 矩阵坐标系，大写 X,Y 表示 笛卡尔坐标系
                G2_shift_YZ_stored[:, for_th] = G2_z_plus_dz_shift_temp[:, I2_y // 2 + int(
                    X / size_PerPixel)]  # X 增加，则 从 G2_z_shift 中 读取的 列 向右移，也就是 YZ 面向 列 增加的方向（G2_z_shift 的 右侧）移动
                G2_shift_XZ_stored[:, for_th] = G2_z_plus_dz_shift_temp[I2_x // 2 - int(Y / size_PerPixel),
                                                :]  # Y 增加，则 从 G2_z_shift 中 读取的 行 向上移，也就是 XZ 面向 行 减小的方向（G2_z_shift 的 上侧）移动
                U2_YZ_stored[:, for_th] = U2_z_plus_dz[:, I2_y // 2 + int(X / size_PerPixel)]
                U2_XZ_stored[:, for_th] = U2_z_plus_dz[I2_x // 2 - int(Y / size_PerPixel), :]

                # %%

                if np.mod(for_th,
                          sheets_num // sheets_stored_num) == 0:  # 如果 for_th 是 sheets_num // sheets_stored_num 的 整数倍（包括零），则 储存之
                    sheet_th_stored[int(for_th // (sheets_num // sheets_stored_num))] = for_th + 1
                    iz_stored[int(for_th // (sheets_num // sheets_stored_num))] = izj[for_th + 1]
                    z_stored[int(for_th // (sheets_num // sheets_stored_num))] = zj[for_th + 1]
                    G2_z_shift_stored[:, :, int(for_th // (
                                sheets_num // sheets_stored_num))] = G2_z_plus_dz_shift_temp  # 储存的 第一层，实际上不是 G2_0，而是 G2_dz
                    U2_z_stored[:, :,
                    int(for_th // (sheets_num // sheets_stored_num))] = U2_z_plus_dz  # 储存的 第一层，实际上不是 U2_0，而是 U2_dz

                if for_th == sheet_th_frontface:  # 如果 for_th 是 sheet_th_frontface，则把结构 前端面 场分布 储存起来，对应的是 zj[sheets_num_frontface]
                    G2_structure_frontface_shift = G2_z_plus_dz_shift_temp
                    U2_structure_frontface = U2_z_plus_dz
                if for_th == sheet_th_endface:  # 如果 for_th 是 sheet_th_endface，则把结构 后端面 场分布 储存起来，对应的是 zj[sheets_num_endface]
                    G2_structure_endface_shift = G2_z_plus_dz_shift_temp
                    U2_structure_endface = U2_z_plus_dz
                if for_th == sheet_th_section_1:  # 如果 for_th 是 想要观察的 第一个面 前面那一层的 层序数，则 将储存之于 该层 前面那一层的 后端面（毕竟 算出来的是 z + dz） 分布中
                    G2_section_1_shift = G2_z_plus_dz_shift_temp  # 对应的是 zj[sheets_num_section_1]
                    U2_section_1 = U2_z_plus_dz
                if for_th == sheet_th_section_2:  # 如果 for_th 是 想要观察的 第二个面 前面那一层的 层序数，则 将储存之于 该层 前面那一层的 后端面（毕竟 算出来的是 z + dz） 分布中
                    G2_section_2_shift = G2_z_plus_dz_shift_temp  # 对应的是 zj[sheets_num_section_2]
                    U2_section_2 = U2_z_plus_dz

        my_thread(10, sheets_num,
                  Cal_dG2_z_plus_dz_shift, Cal_G2_z_plus_dz_shift, After_G2_z_plus_dz_shift_temp,
                  is_ordered=1, is_print=is_print, )

    else:

        def Cal_dU2_z_plus_dz(for_th, fors_num, *arg, ):
            iz = izj[for_th]

            H1_z_shift = np.power(math.e, k1_z_shift * iz * 1j)
            G1_z_shift = g1_shift * H1_z_shift
            G1_z = np.fft.ifftshift(G1_z_shift)
            U1_z = np.fft.ifft2(G1_z)

            if is_bulk == 0:
                if for_th >= sheets_num_frontface and for_th <= sheets_num_endface - 1:
                    if mj[for_th] == 1:
                        modulation_squared_z = modulation_squared
                    elif mj[for_th] == -1:
                        modulation_squared_z = modulation_opposite_squared
                    else:
                        modulation_squared_z = np.ones((I2_x, I2_y), dtype=np.int64()) - is_no_backgroud
                else:
                    modulation_squared_z = np.ones((I2_x, I2_y), dtype=np.int64()) - is_no_backgroud
            else:
                modulation_squared_z = np.ones((I2_x, I2_y), dtype=np.int64()) - is_no_backgroud

            S2_z = modulation_squared_z * U1_z ** 2

            if cal_mode[1] == 1:  # 若 源项 也衍射
                if cal_mode[2] == 1:  # dk_z, k_2z 若是 matrix 版
                    dU2_z_plus_dz = const * S2_z * H2_z_shift_k2_z(dizj[for_th]) / np.power(math.e,
                                                                                            k2_z_shift * diz * 1j)
                else:
                    dU2_z_plus_dz = const * S2_z * H2_z_shift_k2_z(dizj[for_th]) / np.power(math.e, k2 * diz * 1j)
            else:
                dU2_z_plus_dz = const * S2_z * H2_z_shift_k2_z(dizj[for_th])

            return dU2_z_plus_dz

        def Cal_U2_z_plus_dz(for_th, fors_num, dU2_z_plus_dz, *arg, ):

            global U2_z_plus_dz

            if cal_mode[1] == 1:  # 若 源项 也衍射
                U2_z_plus_dz = U2_z_plus_dz + dU2_z_plus_dz
                G2_z_shift = np.fft.fftshift(np.fft.fft2(U2_z_plus_dz))
                U2_z_plus_dz = np.fft.ifft2(np.fft.ifftshift(G2_z_shift * H2_z_plus_dz_shift_k2_z(dizj[for_th])))
            else:
                G2_z_shift = np.fft.fftshift(np.fft.fft2(U2_z_plus_dz))
                U2_z_plus_dz = np.fft.ifft2(np.fft.ifftshift(G2_z_shift * H2_z_plus_dz_shift_k2_z(dizj[for_th])))
                U2_z_plus_dz = U2_z_plus_dz + dU2_z_plus_dz

            return U2_z_plus_dz

        def After_U2_z_plus_dz(for_th, fors_num, U2_z_plus_dz, *arg, ):

            if is_stored == 1:
                global G2_structure_frontface_shift, U2_structure_frontface, G2_structure_endface_shift, U2_structure_endface, G2_section_1_shift, U2_section_1, G2_section_2_shift, U2_section_2

            G2_z_plus_dz = np.fft.fft2(U2_z_plus_dz)
            G2_z_plus_dz_shift_temp = np.fft.fftshift(G2_z_plus_dz)

            if is_energy_evolution_on == 1:
                G2_z_shift_energy[for_th + 1] = np.sum(np.abs(G2_z_plus_dz_shift_temp) ** 2)
                U2_z_energy[for_th + 1] = np.sum(np.abs(U2_z_plus_dz) ** 2)

            if is_stored == 1:
                # 小写的 x,y 表示 电脑中 矩阵坐标系，大写 X,Y 表示 笛卡尔坐标系
                G2_shift_YZ_stored[:, for_th] = G2_z_plus_dz_shift_temp[:, I2_y // 2 + int(
                    X / size_PerPixel)]  # X 增加，则 从 G2_z_shift 中 读取的 列 向右移，也就是 YZ 面向 列 增加的方向（G2_z_shift 的 右侧）移动
                G2_shift_XZ_stored[:, for_th] = G2_z_plus_dz_shift_temp[I2_x // 2 - int(Y / size_PerPixel),
                                                :]  # Y 增加，则 从 G2_z_shift 中 读取的 行 向上移，也就是 XZ 面向 行 减小的方向（G2_z_shift 的 上侧）移动
                U2_YZ_stored[:, for_th] = U2_z_plus_dz[:, I2_y // 2 + int(X / size_PerPixel)]
                U2_XZ_stored[:, for_th] = U2_z_plus_dz[I2_x // 2 - int(Y / size_PerPixel), :]

                # %%

                if np.mod(for_th,
                          sheets_num // sheets_stored_num) == 0:  # 如果 for_th 是 sheets_num // sheets_stored_num 的 整数倍（包括零），则 储存之
                    sheet_th_stored[int(for_th // (sheets_num // sheets_stored_num))] = for_th + 1
                    iz_stored[int(for_th // (sheets_num // sheets_stored_num))] = izj[for_th + 1]
                    z_stored[int(for_th // (sheets_num // sheets_stored_num))] = zj[for_th + 1]
                    G2_z_shift_stored[:, :, int(for_th // (
                                sheets_num // sheets_stored_num))] = G2_z_plus_dz_shift_temp  # 储存的 第一层，实际上不是 G2_0，而是 G2_dz
                    U2_z_stored[:, :,
                    int(for_th // (sheets_num // sheets_stored_num))] = U2_z_plus_dz  # 储存的 第一层，实际上不是 U2_0，而是 U2_dz

                if for_th == sheet_th_frontface:  # 如果 for_th 是 sheet_th_frontface，则把结构 前端面 场分布 储存起来，对应的是 zj[sheets_num_frontface]
                    G2_structure_frontface_shift = G2_z_plus_dz_shift_temp
                    U2_structure_frontface = U2_z_plus_dz
                if for_th == sheet_th_endface:  # 如果 for_th 是 sheet_th_endface，则把结构 后端面 场分布 储存起来，对应的是 zj[sheets_num_endface]
                    G2_structure_endface_shift = G2_z_plus_dz_shift_temp
                    U2_structure_endface = U2_z_plus_dz
                if for_th == sheet_th_section_1:  # 如果 for_th 是 想要观察的 第一个面 前面那一层的 层序数，则 将储存之于 该层 前面那一层的 后端面（毕竟 算出来的是 z + dz） 分布中
                    G2_section_1_shift = G2_z_plus_dz_shift_temp  # 对应的是 zj[sheets_num_section_1]
                    U2_section_1 = U2_z_plus_dz
                if for_th == sheet_th_section_2:  # 如果 for_th 是 想要观察的 第二个面 前面那一层的 层序数，则 将储存之于 该层 前面那一层的 后端面（毕竟 算出来的是 z + dz） 分布中
                    G2_section_2_shift = G2_z_plus_dz_shift_temp  # 对应的是 zj[sheets_num_section_2]
                    U2_section_2 = U2_z_plus_dz

        my_thread(10, sheets_num,
                  Cal_dU2_z_plus_dz, Cal_U2_z_plus_dz, After_U2_z_plus_dz,
                  is_ordered=1, is_print=is_print, )

    # %%

    if cal_mode[0] == 1:  # 如果以 G 算
        G2_z0_SSI_shift = G2_z_plus_dz_shift
        G2_z0_SSI = np.fft.ifftshift(G2_z0_SSI_shift)
        U2_z0_SSI = np.fft.ifft2(G2_z0_SSI)
    else:
        U2_z0_SSI = U2_z_plus_dz
        G2_z0_SSI = np.fft.fft2(U2_z0_SSI)
        G2_z0_SSI_shift = np.fft.fftshift(G2_z0_SSI)

    G2_z0_SSI_shift_amp = np.abs(G2_z0_SSI_shift)
    # print(np.max(G2_z0_SSI_shift_amp))
    G2_z0_SSI_shift_phase = np.angle(G2_z0_SSI_shift)
    if is_save == 1:
        if not os.path.isdir("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift"):
            os.makedirs("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift")

    # %%
    # 绘图：G2_z0_SSI_shift_amp

    G2_z0_SSI_shift_amp_address = location + "\\" + "5. G2_" + str(
        float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + "5.1. NLA - " + "G2_" + str(
        float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_amp" + img_name_extension

    plot_2d(zj, sample, size_PerPixel,
            G2_z0_SSI_shift_amp, G2_z0_SSI_shift_amp_address,
            "G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_amp",
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    # %%
    # 绘图：G2_z0_SSI_shift_phase

    G2_z0_SSI_shift_phase_address = location + "\\" + "5. G2_" + str(
        float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + "5.2. NLA - " + "G2_" + str(
        float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_phase" + img_name_extension

    plot_2d(zj, sample, size_PerPixel,
            G2_z0_SSI_shift_phase, G2_z0_SSI_shift_phase_address,
            "G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_phase",
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)

    # %%
    # 储存 G2_z0_SSI_shift 到 txt 文件

    if is_save == 1:
        G2_z0_SSI_shift_full_name = "5. NLA - G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + (
                    is_save_txt and ".txt" or ".mat")
        G2_z0_SSI_shift_txt_address = location + "\\" + "5. G2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + G2_z0_SSI_shift_full_name
        np.savetxt(G2_z0_SSI_shift_txt_address, G2_z0_SSI_shift) if is_save_txt else savemat(
            G2_z0_SSI_shift_txt_address, {'G': G2_z0_SSI_shift})

    # %%
    # 绘制 G2_z_shift_energy 随 z 演化的 曲线

    if is_energy_evolution_on == 1:
        vmax_G2_z_shift_energy = np.max(G2_z_shift_energy)
        vmin_G2_z_shift_energy = np.min(G2_z_shift_energy)

        G2_z_shift_energy_address = location + "\\" + "5. G2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + "5.1. NLA - " + "G2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_energy_evolution" + img_name_extension

        plot_1d(zj, sample, size_PerPixel,
                G2_z_shift_energy, G2_z_shift_energy_address,
                "G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_energy_evolution",
                is_save, dpi, size_fig * 10, size_fig,
                color_1d, ticks_num, is_title_on, is_axes_on, is_mm, 1,
                fontsize, font,
                0, vmax_G2_z_shift_energy, vmin_G2_z_shift_energy)

    # %%
    # % H2_z0

    H2_z0_SSI_shift = G2_z0_SSI_shift / np.max(np.abs(G2_z0_SSI_shift)) / (g1_shift / np.max(np.abs(g1_shift)))
    # 扔掉 amp 偏离 amp 均值 3 倍于 总体 标准差 以外 的 数据，保留 剩下的 3 倍 以内的 数据。
    H2_z0_SSI_shift = U_Drop_n_sigma(H2_z0_SSI_shift, 3, is_energy)

    if is_save == 1:
        if not os.path.isdir("4. H2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift"):
            os.makedirs("4. H2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift")

    # %%
    # % H2_z0_SSI_shift_amp

    H2_z0_SSI_shift_amp_address = location + "\\" + "4. H2_" + str(
        float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + "4.1. NLA - " + "H2_" + str(
        float('%.2g' % z0)) + "mm" + "_SSI_shift_amp" + img_name_extension

    plot_2d(zj, sample, size_PerPixel,
            np.abs(H2_z0_SSI_shift), H2_z0_SSI_shift_amp_address,
            "H2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift_amp",
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    # %%
    # 绘图：H2_z0_SSI_shift_phase

    H2_z0_SSI_shift_phase_address = location + "\\" + "4. H2_" + str(
        float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + "4.2. NLA - " + "H2_" + str(
        float('%.2g' % z0)) + "mm" + "_SSI_shift_phase" + img_name_extension

    plot_2d(zj, sample, size_PerPixel,
            np.angle(H2_z0_SSI_shift), H2_z0_SSI_shift_phase_address,
            "H2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift_phase",
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)

    # %%
    # 储存 H2_z0_SSI_shift 到 txt 文件

    if is_save == 1:
        H2_z0_SSI_shift_full_name = "4. NLA - H2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + (
                    is_save_txt and ".txt" or ".mat")
        H2_z0_SSI_shift_txt_address = location + "\\" + "4. H2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + H2_z0_SSI_shift_full_name
        np.savetxt(H2_z0_SSI_shift_txt_address, H2_z0_SSI_shift) if is_save_txt else savemat(
            H2_z0_SSI_shift_txt_address, {"H": H2_z0_SSI_shift})

    # %%
    # G2_z0_SSI = G2_z0_SSI(k1_x, k1_y) → IFFT2 → U2(x0, y0, z0) = U2_z0_SSI

    if is_stored == 1:

        sheet_th_stored[sheets_stored_num] = sheets_num
        iz_stored[sheets_stored_num] = Iz
        z_stored[sheets_stored_num] = Iz * size_PerPixel
        G2_z_shift_stored[:, :, sheets_stored_num] = G2_z0_SSI_shift  # 储存的 第一层，实际上不是 G2_0，而是 G2_dz
        U2_z_stored[:, :, sheets_stored_num] = U2_z0_SSI  # 储存的 第一层，实际上不是 U2_0，而是 U2_dz

        # %%

        if is_save == 1:
            if not os.path.isdir("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_stored"):
                os.makedirs("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_stored")
            if not os.path.isdir("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_stored"):
                os.makedirs("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_stored")

        # -------------------------

        vmax_G2_z_shift_stored_amp = np.max(np.abs(G2_z_shift_stored))
        vmin_G2_z_shift_stored_amp = np.min(np.abs(G2_z_shift_stored))

        for sheet_stored_th in range(sheets_stored_num + 1):
            G2_z_shift_sheet_stored_th_amp_address = location + "\\" + "5. G2_" + str(
                float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_stored" + "\\" + "5.1. NLA - " + "G2_" + str(
                float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI_shift" + "_amp" + img_name_extension

            plot_2d(zj, sample, size_PerPixel,
                    np.abs(G2_z_shift_stored[:, :, sheet_stored_th]), G2_z_shift_sheet_stored_th_amp_address,
                    "G2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI_shift" + "_amp",
                    is_save, dpi, size_fig,
                    cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    is_self_colorbar, is_colorbar_on, is_energy, vmax_G2_z_shift_stored_amp, vmin_G2_z_shift_stored_amp)

        vmax_G2_z_shift_stored_phase = np.max(np.angle(G2_z_shift_stored))
        vmin_G2_z_shift_stored_phase = np.min(np.angle(G2_z_shift_stored))

        for sheet_stored_th in range(sheets_stored_num + 1):
            G2_z_shift_sheet_stored_th_phase_address = location + "\\" + "5. G2_" + str(
                float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_stored" + "\\" + "5.2. NLA - " + "G2_" + str(
                float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI_shift" + "_phase" + img_name_extension

            plot_2d(zj, sample, size_PerPixel,
                    np.angle(G2_z_shift_stored[:, :, sheet_stored_th]), G2_z_shift_sheet_stored_th_phase_address,
                    "G2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI_shift" + "_phase",
                    is_save, dpi, size_fig,
                    cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    is_self_colorbar, is_colorbar_on, 0, vmax_G2_z_shift_stored_phase, vmin_G2_z_shift_stored_phase)

        # -------------------------

        vmax_U2_z_stored_amp = np.max(np.abs(U2_z_stored))
        vmin_U2_z_stored_amp = np.min(np.abs(U2_z_stored))

        for sheet_stored_th in range(sheets_stored_num + 1):
            U2_z_sheet_stored_th_amp_address = location + "\\" + "6. U2_" + str(
                float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_stored" + "\\" + "6.1. NLA - " + "U2_" + str(
                float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI" + "_amp" + img_name_extension

            plot_2d(zj, sample, size_PerPixel,
                    np.abs(U2_z_stored[:, :, sheet_stored_th]), U2_z_sheet_stored_th_amp_address,
                    "U2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI" + "_amp",
                    is_save, dpi, size_fig,
                    cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    is_self_colorbar, is_colorbar_on, is_energy, vmax_U2_z_stored_amp, vmin_U2_z_stored_amp)

        vmax_U2_z_stored_phase = np.max(np.angle(U2_z_stored))
        vmin_U2_z_stored_phase = np.min(np.angle(U2_z_stored))

        for sheet_stored_th in range(sheets_stored_num + 1):
            U2_z_sheet_stored_th_phase_address = location + "\\" + "6. U2_" + str(
                float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_stored" + "\\" + "6.2. NLA - " + "U2_" + str(
                float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI" + "_phase" + img_name_extension

            plot_2d(zj, sample, size_PerPixel,
                    np.angle(U2_z_stored[:, :, sheet_stored_th]), U2_z_sheet_stored_th_phase_address,
                    "U2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI" + "_phase",
                    is_save, dpi, size_fig,
                    cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    is_self_colorbar, is_colorbar_on, 0, vmax_U2_z_stored_phase, vmin_U2_z_stored_phase)

        # %%
        # 这 sheets_stored_num 层 也可以 画成 3D，就是太丑了，所以只 整个 U2_amp 示意一下即可

        # U2_z_sheets_stored_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_stored" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_sheets_stored" + "_amp" + img_name_extension

        # plot_3d_XYz(zj, sample, size_PerPixel, 
        #             sheets_stored_num, U2_z_stored, z_stored, 
        #             U2_z_sheets_stored_amp_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_sheets_stored" + "_amp", 
        #             is_save, dpi, size_fig, 
        #             cmap_3d, elev, azim, alpha, 
        #             ticks_num, is_title_on, is_axes_on, is_mm,  
        #             fontsize, font,
        #             is_self_colorbar, is_colorbar_on, is_energy, vmax_U2_z_stored_amp, vmin_U2_z_stored_amp)

        # %%

        # 小写的 x,y 表示 电脑中 矩阵坐标系，大写 X,Y 表示 笛卡尔坐标系
        # G2_shift_xz_stored[:, sheets_num] = G2_z0_SSI_shift[:, I2_y // 2 + int(X / size_PerPixel)]
        # G2_shift_yz_stored[:, sheets_num] = G2_z0_SSI_shift[I2_x // 2 - int(Y / size_PerPixel), :]
        # U2_xz_stored[:, sheets_num] = U2_z0_SSI[:, I2_y // 2 + int(X / size_PerPixel)]
        # U2_yz_stored[:, sheets_num] = U2_z0_SSI[I2_x // 2 - int(Y / size_PerPixel), :]
        G2_shift_YZ_stored[:, sheets_num] = G2_z0_SSI_shift[:, I2_y // 2 + int(X / size_PerPixel)]
        G2_shift_XZ_stored[:, sheets_num] = G2_z0_SSI_shift[I2_x // 2 - int(Y / size_PerPixel), :]
        U2_YZ_stored[:, sheets_num] = U2_z0_SSI[:, I2_y // 2 + int(X / size_PerPixel)]
        U2_XZ_stored[:, sheets_num] = U2_z0_SSI[I2_x // 2 - int(Y / size_PerPixel), :]

        # %%
        # 再算一下 初始的 场分布，之后 绘 3D 用，因为 开启 多线程后，就不会 储存 中间层 了

        # modulation_squared_0 = 1

        # if is_bulk == 0:
        #     modulation_squared_full_name = str(0) + ".mat"
        #     modulation_squared_address = location + "\\" + "0.χ2_modulation_squared" + "\\" + modulation_squared_full_name
        #     modulation_squared_0 = loadmat(modulation_squared_address)['chi2_modulation_squared']

        # Q2_0 = np.fft.fft2(modulation_squared_0 * U1_0**2)
        # Q2_0_shift = np.fft.fftshift(Q2_0)

        # H2_0_shift_k2_z = 1 / k2_z_shift * size_PerPixel # 注意 这里的 传递函数 的 指数是 负的 ！！！（但 z = iz = 0，所以 指数项 变成 1 了）
        # g2_dz_shift = const * Q2_0_shift * H2_0_shift_k2_z

        # H2_dz_shift_k2_z = np.power(math.e, k2_z_shift * diz * 1j) # 注意 这里的 传递函数 的 指数是 正的 ！！！
        # G2_dz_shift = g2_dz_shift * H2_dz_shift_k2_z #　每一步 储存的 实际上不是 G2_z，而是 G2_z+dz
        # G2_dz = np.fft.ifftshift(G2_dz_shift)
        # U2_dz = np.fft.ifft2(G2_dz)

        # %%

        if is_save == 1:
            if not os.path.isdir("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored"):
                os.makedirs("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored")
            if not os.path.isdir("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored"):
                os.makedirs("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored")

        # ========================= G2_shift_YZ_stored_amp、G2_shift_XZ_stored_amp

        vmax_G2_shift_YZ_XZ_stored_amp = np.max(
            [np.max(np.abs(G2_shift_YZ_stored)), np.max(np.abs(G2_shift_XZ_stored))])
        vmin_G2_shift_YZ_XZ_stored_amp = np.min(
            [np.min(np.abs(G2_shift_YZ_stored)), np.min(np.abs(G2_shift_XZ_stored))])

        G2_shift_YZ_stored_amp_address = location + "\\" + "5. G2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored" + "\\" + "5.1. NLA - " + "G2_" + str(
            float('%.2g' % X)) + "mm" + "_SSI_shift" + "_YZ" + "_amp" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.abs(G2_shift_YZ_stored), G2_shift_YZ_stored_amp_address,
                "G2_" + str(float('%.2g' % X)) + "mm" + "_SSI_shift" + "_YZ" + "_amp",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, is_energy, vmax_G2_shift_YZ_XZ_stored_amp,
                vmin_G2_shift_YZ_XZ_stored_amp)

        G2_shift_XZ_stored_amp_address = location + "\\" + "5. G2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored" + "\\" + "5.1. NLA - " + "G2_" + str(
            float('%.2g' % Y)) + "mm" + "_SSI_shift" + "_XZ" + "_amp" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.abs(G2_shift_XZ_stored), G2_shift_XZ_stored_amp_address,
                "G2_" + str(float('%.2g' % Y)) + "mm" + "_SSI_shift" + "_XZ" + "_amp",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, is_energy, vmax_G2_shift_YZ_XZ_stored_amp,
                vmin_G2_shift_YZ_XZ_stored_amp)

        # ------------------------- G2_shift_YZ_stored_phase、G2_shift_XZ_stored_phase

        vmax_G2_shift_YZ_XZ_stored_phase = np.max(
            [np.max(np.angle(G2_shift_YZ_stored)), np.max(np.angle(G2_shift_XZ_stored))])
        vmin_G2_shift_YZ_XZ_stored_phase = np.min(
            [np.min(np.angle(G2_shift_YZ_stored)), np.min(np.angle(G2_shift_XZ_stored))])

        G2_shift_YZ_stored_phase_address = location + "\\" + "5. G2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored" + "\\" + "5.2. NLA - " + "G2_" + str(
            float('%.2g' % X)) + "mm" + "_SSI_shift" + "_YZ" + "_phase" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.angle(G2_shift_YZ_stored), G2_shift_YZ_stored_phase_address,
                "G2_" + str(float('%.2g' % X)) + "mm" + "_SSI_shift" + "_YZ" + "_phase",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, 0, vmax_G2_shift_YZ_XZ_stored_phase, vmin_G2_shift_YZ_XZ_stored_phase)

        G2_shift_XZ_stored_phase_address = location + "\\" + "5. G2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored" + "\\" + "5.2. NLA - " + "G2_" + str(
            float('%.2g' % Y)) + "mm" + "_SSI_shift" + "_XZ" + "_phase" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.angle(G2_shift_XZ_stored), G2_shift_XZ_stored_phase_address,
                "G2_" + str(float('%.2g' % Y)) + "mm" + "_SSI_shift" + "_XZ" + "_phase",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, 0, vmax_G2_shift_YZ_XZ_stored_phase, vmin_G2_shift_YZ_XZ_stored_phase)

        # ========================= U2_YZ_stored_amp、U2_XZ_stored_amp

        vmax_U2_YZ_XZ_stored_amp = np.max([np.max(np.abs(U2_YZ_stored)), np.max(np.abs(U2_XZ_stored))])
        vmin_U2_YZ_XZ_stored_amp = np.min([np.min(np.abs(U2_YZ_stored)), np.min(np.abs(U2_XZ_stored))])

        U2_YZ_stored_amp_address = location + "\\" + "6. U2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored" + "\\" + "6.1. NLA - " + "U2_" + str(
            float('%.2g' % X)) + "mm" + "_SSI" + "_YZ" + "_amp" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.abs(U2_YZ_stored), U2_YZ_stored_amp_address,
                "U2_" + str(float('%.2g' % X)) + "mm" + "_SSI" + "_YZ" + "_amp",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, is_energy, vmax_U2_YZ_XZ_stored_amp, vmin_U2_YZ_XZ_stored_amp)

        U2_XZ_stored_amp_address = location + "\\" + "6. U2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored" + "\\" + "6.1. NLA - " + "U2_" + str(
            float('%.2g' % Y)) + "mm" + "_SSI" + "_XZ" + "_amp" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.abs(U2_XZ_stored), U2_XZ_stored_amp_address,
                "U2_" + str(float('%.2g' % Y)) + "mm" + "_SSI" + "_XZ" + "_amp",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, is_energy, vmax_U2_YZ_XZ_stored_amp, vmin_U2_YZ_XZ_stored_amp)

        # ------------------------- U2_YZ_stored_phase、U2_XZ_stored_phase

        vmax_U2_YZ_XZ_stored_phase = np.max([np.max(np.angle(U2_YZ_stored)), np.max(np.angle(U2_XZ_stored))])
        vmin_U2_YZ_XZ_stored_phase = np.min([np.min(np.angle(U2_YZ_stored)), np.min(np.angle(U2_XZ_stored))])

        U2_YZ_stored_phase_address = location + "\\" + "6. U2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored" + "\\" + "6.2. NLA - " + "U2_" + str(
            float('%.2g' % X)) + "mm" + "_SSI" + "_YZ" + "_phase" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.angle(U2_YZ_stored), U2_YZ_stored_phase_address,
                "U2_" + str(float('%.2g' % X)) + "mm" + "_SSI" + "_YZ" + "_phase",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, 0, vmax_U2_YZ_XZ_stored_phase, vmin_U2_YZ_XZ_stored_phase)

        U2_XZ_stored_phase_address = location + "\\" + "6. U2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored" + "\\" + "6.2. NLA - " + "U2_" + str(
            float('%.2g' % Y)) + "mm" + "_SSI" + "_XZ" + "_phase" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.angle(U2_XZ_stored), U2_XZ_stored_phase_address,
                "U2_" + str(float('%.2g' % Y)) + "mm" + "_SSI" + "_XZ" + "_phase",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, 0, vmax_U2_YZ_XZ_stored_phase, vmin_U2_YZ_XZ_stored_phase)

        # %%

        if is_save == 1:
            if not os.path.isdir("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored"):
                os.makedirs("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored")
            if not os.path.isdir("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored"):
                os.makedirs("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored")

        # ------------------------- 储存 G2_section_1_shift_amp、G2_section_2_shift_amp、G2_structure_frontface_shift_amp、G2_structure_endface_shift_amp

        if is_show_structure_face == 1:
            vmax_G2_section_1_2_front_end_shift_amp = np.max(
                [np.max(np.abs(G2_section_1_shift)), np.max(np.abs(G2_section_2_shift)),
                 np.max(np.abs(G2_structure_frontface_shift)), np.max(np.abs(G2_structure_endface_shift))])
            vmin_G2_section_1_2_front_end_shift_amp = np.min(
                [np.min(np.abs(G2_section_1_shift)), np.min(np.abs(G2_section_2_shift)),
                 np.min(np.abs(G2_structure_frontface_shift)), np.min(np.abs(G2_structure_endface_shift))])
        else:
            vmax_G2_section_1_2_front_end_shift_amp = np.max(
                [np.max(np.abs(G2_section_1_shift)), np.max(np.abs(G2_section_2_shift))])
            vmin_G2_section_1_2_front_end_shift_amp = np.min(
                [np.min(np.abs(G2_section_1_shift)), np.min(np.abs(G2_section_2_shift))])

        G2_section_1_shift_amp_address = location + "\\" + "5. G2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.1. NLA - " + "G2_" + str(
            float('%.2g' % z0_1)) + "mm" + "_SSI_shift" + "_amp" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.abs(G2_section_1_shift), G2_section_1_shift_amp_address,
                "G2_" + str(float('%.2g' % z0_1)) + "mm" + "_SSI_shift" + "_amp",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, is_energy, vmax_G2_section_1_2_front_end_shift_amp,
                vmin_G2_section_1_2_front_end_shift_amp)

        G2_section_2_shift_amp_address = location + "\\" + "5. G2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.1. NLA - " + "G2_" + str(
            float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_amp" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.abs(G2_section_2_shift), G2_section_2_shift_amp_address,
                "G2_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_amp",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, is_energy, vmax_G2_section_1_2_front_end_shift_amp,
                vmin_G2_section_1_2_front_end_shift_amp)

        if is_show_structure_face == 1:
            G2_structure_frontface_shift_amp_address = location + "\\" + "5. G2_" + str(float(
                '%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.1. NLA - " + "G2_" + str(
                float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI_shift" + "_amp" + img_name_extension

            plot_2d(zj, sample, size_PerPixel,
                    np.abs(G2_structure_frontface_shift), G2_structure_frontface_shift_amp_address,
                    "G2_" + str(float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI_shift" + "_amp",
                    is_save, dpi, size_fig,
                    cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    is_self_colorbar, is_colorbar_on, is_energy, vmax_G2_section_1_2_front_end_shift_amp,
                    vmin_G2_section_1_2_front_end_shift_amp)

            G2_structure_endface_shift_amp_address = location + "\\" + "5. G2_" + str(float(
                '%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.1. NLA - " + "G2_" + str(
                float('%.2g' % z0_structure_endface)) + "mm" + "_SSI_shift" + "_amp" + img_name_extension

            plot_2d(zj, sample, size_PerPixel,
                    np.abs(G2_structure_endface_shift), G2_structure_endface_shift_amp_address,
                    "G2_" + str(float('%.2g' % z0_structure_endface)) + "mm" + "_SSI_shift" + "_amp",
                    is_save, dpi, size_fig,
                    cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    is_self_colorbar, is_colorbar_on, is_energy, vmax_G2_section_1_2_front_end_shift_amp,
                    vmin_G2_section_1_2_front_end_shift_amp)

        # ------------------------- 储存 G2_section_1_shift_phase、G2_section_2_shift_phase、G2_structure_frontface_shift_phase、G2_structure_endface_shift_phase

        if is_show_structure_face == 1:
            vmax_G2_section_1_2_front_end_shift_phase = np.max(
                [np.max(np.angle(G2_section_1_shift)), np.max(np.angle(G2_section_2_shift)),
                 np.max(np.angle(G2_structure_frontface_shift)), np.max(np.angle(G2_structure_endface_shift))])
            vmin_G2_section_1_2_front_end_shift_phase = np.min(
                [np.min(np.angle(G2_section_1_shift)), np.min(np.angle(G2_section_2_shift)),
                 np.min(np.angle(G2_structure_frontface_shift)), np.min(np.angle(G2_structure_endface_shift))])
        else:
            vmax_G2_section_1_2_front_end_shift_phase = np.max(
                [np.max(np.angle(G2_section_1_shift)), np.max(np.angle(G2_section_2_shift))])
            vmin_G2_section_1_2_front_end_shift_phase = np.min(
                [np.min(np.angle(G2_section_1_shift)), np.min(np.angle(G2_section_2_shift))])

        G2_section_1_shift_phase_address = location + "\\" + "5. G2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.2. NLA - " + "G2_" + str(
            float('%.2g' % z0_1)) + "mm" + "_SSI_shift" + "_phase" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.angle(G2_section_1_shift), G2_section_1_shift_phase_address,
                "G2_" + str(float('%.2g' % z0_1)) + "mm" + "_SSI_shift" + "_phase",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, 0, vmax_G2_section_1_2_front_end_shift_phase,
                vmin_G2_section_1_2_front_end_shift_phase)

        G2_section_2_shift_phase_address = location + "\\" + "5. G2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.2. NLA - " + "G2_" + str(
            float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_phase" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.angle(G2_section_2_shift), G2_section_2_shift_phase_address,
                "G2_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_phase",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, 0, vmax_G2_section_1_2_front_end_shift_phase,
                vmin_G2_section_1_2_front_end_shift_phase)

        if is_show_structure_face == 1:
            G2_structure_frontface_shift_phase_address = location + "\\" + "5. G2_" + str(float(
                '%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.2. NLA - " + "G2_" + str(
                float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI_shift" + "_phase" + img_name_extension

            plot_2d(zj, sample, size_PerPixel,
                    np.angle(G2_structure_frontface_shift), G2_structure_frontface_shift_phase_address,
                    "G2_" + str(float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI_shift" + "_phase",
                    is_save, dpi, size_fig,
                    cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    is_self_colorbar, is_colorbar_on, 0, vmax_G2_section_1_2_front_end_shift_phase,
                    vmin_G2_section_1_2_front_end_shift_phase)

            G2_structure_endface_shift_phase_address = location + "\\" + "5. G2_" + str(float(
                '%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.2. NLA - " + "G2_" + str(
                float('%.2g' % z0_structure_endface)) + "mm" + "_SSI_shift" + "_phase" + img_name_extension

            plot_2d(zj, sample, size_PerPixel,
                    np.angle(G2_structure_endface_shift), G2_structure_endface_shift_phase_address,
                    "G2_" + str(float('%.2g' % z0_structure_endface)) + "mm" + "_SSI_shift" + "_phase",
                    is_save, dpi, size_fig,
                    cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    is_self_colorbar, is_colorbar_on, 0, vmax_G2_section_1_2_front_end_shift_phase,
                    vmin_G2_section_1_2_front_end_shift_phase)

        # ------------------------- 储存 U2_section_1_amp、U2_section_2_amp、U2_structure_frontface_amp、U2_structure_endface_amp

        if is_show_structure_face == 1:
            vmax_U2_section_1_2_front_end_shift_amp = np.max(
                [np.max(np.abs(U2_section_1)), np.max(np.abs(U2_section_2)), np.max(np.abs(U2_structure_frontface)),
                 np.max(np.abs(U2_structure_endface))])
            vmin_U2_section_1_2_front_end_shift_amp = np.min(
                [np.min(np.abs(U2_section_1)), np.min(np.abs(U2_section_2)), np.min(np.abs(U2_structure_frontface)),
                 np.min(np.abs(U2_structure_endface))])
        else:
            vmax_U2_section_1_2_front_end_shift_amp = np.max(
                [np.max(np.abs(U2_section_1)), np.max(np.abs(U2_section_2))])
            vmin_U2_section_1_2_front_end_shift_amp = np.min(
                [np.min(np.abs(U2_section_1)), np.min(np.abs(U2_section_2))])

        U2_section_1_amp_address = location + "\\" + "6. U2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.1. NLA - " + "U2_" + str(
            float('%.2g' % z0_1)) + "mm" + "_SSI" + "_amp" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.abs(U2_section_1), U2_section_1_amp_address,
                "U2_" + str(float('%.2g' % z0_1)) + "mm" + "_SSI" + "_amp",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, is_energy, vmax_U2_section_1_2_front_end_shift_amp,
                vmin_U2_section_1_2_front_end_shift_amp)

        U2_section_2_amp_address = location + "\\" + "6. U2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.1. NLA - " + "U2_" + str(
            float('%.2g' % z0_2)) + "mm" + "_SSI" + "_amp" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.abs(U2_section_2), U2_section_2_amp_address,
                "U2_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_amp",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, is_energy, vmax_U2_section_1_2_front_end_shift_amp,
                vmin_U2_section_1_2_front_end_shift_amp)

        if is_show_structure_face == 1:
            U2_structure_frontface_amp_address = location + "\\" + "6. U2_" + str(
                float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.1. NLA - " + "U2_" + str(
                float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI" + "_amp" + img_name_extension

            plot_2d(zj, sample, size_PerPixel,
                    np.abs(U2_structure_frontface), U2_structure_frontface_amp_address,
                    "U2_" + str(float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI" + "_amp",
                    is_save, dpi, size_fig,
                    cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    is_self_colorbar, is_colorbar_on, is_energy, vmax_U2_section_1_2_front_end_shift_amp,
                    vmin_U2_section_1_2_front_end_shift_amp)

            U2_structure_endface_amp_address = location + "\\" + "6. U2_" + str(
                float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.1. NLA - " + "U2_" + str(
                float('%.2g' % z0_structure_endface)) + "mm" + "_SSI" + "_amp" + img_name_extension

            plot_2d(zj, sample, size_PerPixel,
                    np.abs(U2_structure_endface), U2_structure_endface_amp_address,
                    "U2_" + str(float('%.2g' % z0_structure_endface)) + "mm" + "_SSI" + "_amp",
                    is_save, dpi, size_fig,
                    cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    is_self_colorbar, is_colorbar_on, is_energy, vmax_U2_section_1_2_front_end_shift_amp,
                    vmin_U2_section_1_2_front_end_shift_amp)

        # ------------------------- 储存 U2_section_1_phase、U2_section_2_phase、U2_structure_frontface_phase、U2_structure_endface_phase

        if is_show_structure_face == 1:
            vmax_U2_section_1_2_front_end_shift_phase = np.max(
                [np.max(np.angle(U2_section_1)), np.max(np.angle(U2_section_2)),
                 np.max(np.angle(U2_structure_frontface)), np.max(np.angle(U2_structure_endface))])
            vmin_U2_section_1_2_front_end_shift_phase = np.min(
                [np.min(np.angle(U2_section_1)), np.min(np.angle(U2_section_2)),
                 np.min(np.angle(U2_structure_frontface)), np.min(np.angle(U2_structure_endface))])
        else:
            vmax_U2_section_1_2_front_end_shift_phase = np.max(
                [np.max(np.angle(U2_section_1)), np.max(np.angle(U2_section_2))])
            vmin_U2_section_1_2_front_end_shift_phase = np.min(
                [np.min(np.angle(U2_section_1)), np.min(np.angle(U2_section_2))])

        U2_section_1_phase_address = location + "\\" + "6. U2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.2. NLA - " + "U2_" + str(
            float('%.2g' % z0_1)) + "mm" + "_SSI" + "_phase" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.angle(U2_section_1), U2_section_1_phase_address,
                "U2_" + str(float('%.2g' % z0_1)) + "mm" + "_SSI" + "_phase",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, 0, vmax_U2_section_1_2_front_end_shift_phase,
                vmin_U2_section_1_2_front_end_shift_phase)

        U2_section_2_phase_address = location + "\\" + "6. U2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.2. NLA - " + "U2_" + str(
            float('%.2g' % z0_2)) + "mm" + "_SSI" + "_phase" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                np.angle(U2_section_2), U2_section_2_phase_address,
                "U2_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_phase",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                fontsize, font,
                is_self_colorbar, is_colorbar_on, 0, vmax_U2_section_1_2_front_end_shift_phase,
                vmin_U2_section_1_2_front_end_shift_phase)

        if is_show_structure_face == 1:
            U2_structure_frontface_phase_address = location + "\\" + "6. U2_" + str(
                float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.2. NLA - " + "U2_" + str(
                float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI" + "_phase" + img_name_extension

            plot_2d(zj, sample, size_PerPixel,
                    np.angle(U2_structure_frontface), U2_structure_frontface_phase_address,
                    "U2_" + str(float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI" + "_phase",
                    is_save, dpi, size_fig,
                    cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    is_self_colorbar, is_colorbar_on, 0, vmax_U2_section_1_2_front_end_shift_phase,
                    vmin_U2_section_1_2_front_end_shift_phase)

            U2_structure_endface_phase_address = location + "\\" + "6. U2_" + str(
                float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.2. NLA - " + "U2_" + str(
                float('%.2g' % z0_structure_endface)) + "mm" + "_SSI" + "_phase" + img_name_extension

            plot_2d(zj, sample, size_PerPixel,
                    np.angle(U2_structure_endface), U2_structure_endface_phase_address,
                    "U2_" + str(float('%.2g' % z0_structure_endface)) + "mm" + "_SSI" + "_phase",
                    is_save, dpi, size_fig,
                    cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                    fontsize, font,
                    is_self_colorbar, is_colorbar_on, 0, vmax_U2_section_1_2_front_end_shift_phase,
                    vmin_U2_section_1_2_front_end_shift_phase)

        # %%
        # 绘制 G2_amp 的 侧面 3D 分布图，以及 初始 和 末尾的 G2_amp（现在 可以 任选位置 了）

        # vmax_G2_amp = np.max([vmax_G2_shift_YZ_XZ_stored_amp, vmax_G2_section_1_2_front_end_shift_amp])
        # vmin_G2_amp = np.min([vmin_G2_shift_YZ_XZ_stored_amp, vmin_G2_section_1_2_front_end_shift_amp])

        # G2_shift_XYZ_stored_amp_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored" + "\\" + "5.1. NLA - " + "G2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_XYZ" + "_amp" + img_name_extension

        # plot_3d_XYZ(zj, sample, size_PerPixel, 
        #             np.abs(G2_shift_YZ_stored), np.abs(G2_shift_XZ_stored), np.abs(G2_section_1_shift), np.abs(G2_section_2_shift), 
        #             np.abs(G2_structure_frontface_shift), np.abs(G2_structure_endface_shift), is_show_structure_face, 
        #             G2_shift_XYZ_stored_amp_address, "G2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_XYZ" + "_amp", 
        #             I2_y // 2 + int(X / size_PerPixel), I2_x // 2 + int(Y / size_PerPixel), sheets_num_section_1, sheets_num_section_2, 
        #             sheets_num_frontface, sheets_num_endface, 
        #             is_save, dpi, size_fig, 
        #             cmap_3d, elev, azim, alpha, 
        #             ticks_num, is_title_on, is_axes_on, is_mm,  
        #             fontsize, font, 
        #             is_self_colorbar, is_colorbar_on, is_energy, vmax_G2_amp, vmin_G2_amp)

        # %%
        # 绘制 G2_phase 的 侧面 3D 分布图，以及 初始 和 末尾的 G2_phase

        # vmax_G2_phase = np.max([vmax_G2_shift_YZ_XZ_stored_phase, vmax_G2_section_1_2_front_end_shift_phase])
        # vmin_G2_phase = np.min([vmin_G2_shift_YZ_XZ_stored_phase, vmin_G2_section_1_2_front_end_shift_phase])

        # G2_shift_XYZ_stored_phase_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored" + "\\" + "5.2. NLA - " + "G2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_XYZ" + "_phase" + img_name_extension

        # plot_3d_XYZ(zj, sample, size_PerPixel, 
        #             np.angle(G2_shift_YZ_stored), np.angle(G2_shift_XZ_stored), np.angle(G2_section_1_shift), np.angle(G2_section_2_shift), 
        #             np.angle(G2_structure_frontface_shift), np.angle(G2_structure_endface_shift), is_show_structure_face, 
        #             G2_shift_XYZ_stored_phase_address, "G2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_XYZ" + "_phase", 
        #             I2_y // 2 + int(X / size_PerPixel), I2_x // 2 + int(Y / size_PerPixel), sheets_num_section_1, sheets_num_section_2, 
        #             sheets_num_frontface, sheets_num_endface, 
        #             is_save, dpi, size_fig, 
        #             cmap_3d, elev, azim, alpha, 
        #             ticks_num, is_title_on, is_axes_on, is_mm,  
        #             fontsize, font, 
        #             is_self_colorbar, is_colorbar_on, 0, vmax_G2_phase, vmin_G2_phase)

        # %%
        # 绘制 U2_amp 的 侧面 3D 分布图，以及 初始 和 末尾的 U2_amp

        # vmax_U2_amp = np.max([vmax_U2_YZ_XZ_stored_amp, vmax_U2_section_1_2_front_end_shift_amp])
        # vmin_U2_amp = np.min([vmin_U2_YZ_XZ_stored_amp, vmin_U2_section_1_2_front_end_shift_amp])

        # U2_XYZ_stored_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_XYZ" + "_amp" + img_name_extension

        # plot_3d_XYZ(zj, sample, size_PerPixel, 
        #             np.abs(U2_YZ_stored), np.abs(U2_XZ_stored), np.abs(U2_section_1), np.abs(U2_section_2), 
        #             np.abs(U2_structure_frontface), np.abs(U2_structure_endface), is_show_structure_face, 
        #             U2_XYZ_stored_amp_address, "U2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_XYZ" + "_amp", 
        #             I2_y // 2 + int(X / size_PerPixel), I2_x // 2 + int(Y / size_PerPixel), sheets_num_section_1, sheets_num_section_2, 
        #             sheets_num_frontface, sheets_num_endface, 
        #             is_save, dpi, size_fig, 
        #             cmap_3d, elev, azim, alpha, 
        #             ticks_num, is_title_on, is_axes_on, is_mm,  
        #             fontsize, font, 
        #             is_self_colorbar, is_colorbar_on, is_energy, vmax_U2_amp, vmin_U2_amp)

        # %%
        # 绘制 U2_phase 的 侧面 3D 分布图，以及 初始 和 末尾的 U2_phase

        # vmax_U2_phase = np.max([vmax_U2_YZ_XZ_stored_phase, vmax_U2_section_1_2_front_end_shift_phase])
        # vmin_U2_phase = np.min([vmin_U2_YZ_XZ_stored_phase, vmin_U2_section_1_2_front_end_shift_phase])

        # U2_XYZ_stored_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored" + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_XYZ" + "_phase" + img_name_extension

        # plot_3d_XYZ(zj, sample, size_PerPixel, 
        #             np.angle(U2_YZ_stored), np.angle(U2_XZ_stored), np.angle(U2_section_1), np.angle(U2_section_2), 
        #             np.angle(U2_structure_frontface), np.angle(U2_structure_endface), is_show_structure_face, 
        #             U2_XYZ_stored_phase_address, "U2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_XYZ" + "_phase", 
        #             I2_y // 2 + int(X / size_PerPixel), I2_x // 2 + int(Y / size_PerPixel), sheets_num_section_1, sheets_num_section_2, 
        #             sheets_num_frontface, sheets_num_endface, 
        #             is_save, dpi, size_fig, 
        #             cmap_3d, elev, azim, alpha, 
        #             ticks_num, is_title_on, is_axes_on, is_mm,  
        #             fontsize, font, 
        #             is_self_colorbar, is_colorbar_on, 0, vmax_U2_phase, vmin_U2_phase)

    # %%

    U2_z0_SSI_amp = np.abs(U2_z0_SSI)
    # print(np.max(U2_z0_SSI_amp))
    U2_z0_SSI_phase = np.angle(U2_z0_SSI)

    print("NLA - U2_{}mm_SSI.total_energy = {}".format(z0, np.sum(U2_z0_SSI_amp ** 2)))

    if is_save == 1:
        if not os.path.isdir("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI"):
            os.makedirs("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI")

    # %%
    # 绘图：U2_z0_SSI_amp

    U2_z0_SSI_amp_address = location + "\\" + "6. U2_" + str(
        float('%.2g' % z0)) + "mm" + "_SSI" + "\\" + "6.1. NLA - " + "U2_" + str(
        float('%.2g' % z0)) + "mm" + "_SSI" + "_amp" + img_name_extension

    plot_2d(zj, sample, size_PerPixel,
            U2_z0_SSI_amp, U2_z0_SSI_amp_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_amp",
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    # %%
    # 绘图：U2_z0_SSI_phase

    U2_z0_SSI_phase_address = location + "\\" + "6. U2_" + str(
        float('%.2g' % z0)) + "mm" + "_SSI" + "\\" + "6.2. NLA - " + "U2_" + str(
        float('%.2g' % z0)) + "mm" + "_SSI" + "_phase" + img_name_extension

    plot_2d(zj, sample, size_PerPixel,
            U2_z0_SSI_phase, U2_z0_SSI_phase_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_phase",
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)

    # %%
    # 储存 U2_z0_SSI 到 txt 文件

    U2_z0_SSI_full_name = "6. NLA - U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + (is_save_txt and ".txt" or ".mat")
    if is_save == 1:
        U2_z0_SSI_txt_address = location + "\\" + "6. U2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI" + "\\" + U2_z0_SSI_full_name
        np.savetxt(U2_z0_SSI_txt_address, U2_z0_SSI) if is_save_txt else savemat(U2_z0_SSI_txt_address,
                                                                                 {'U': U2_z0_SSI})

        # %%
        # 再次绘图：U2_z0_SSI_amp

        U2_z0_SSI_amp_address = location + "\\" + "6.1. NLA - " + "U2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI" + "_amp" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                U2_z0_SSI_amp, U2_z0_SSI_amp_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_amp",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                fontsize, font,
                1, is_colorbar_on, is_energy, vmax, vmin)

        # 再次绘图：U2_z0_SSI_phase

        U2_z0_SSI_phase_address = location + "\\" + "6.2. NLA - " + "U2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI" + "_phase" + img_name_extension

        plot_2d(zj, sample, size_PerPixel,
                U2_z0_SSI_phase, U2_z0_SSI_phase_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_phase",
                is_save, dpi, size_fig,
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
                fontsize, font,
                1, is_colorbar_on, 0, vmax, vmin)

    # %%
    # 储存 U2_z0_SSI 到 txt 文件

    # if is_save == 1:
    np.savetxt(U2_z0_SSI_full_name, U2_z0_SSI) if is_save_txt else savemat(U2_z0_SSI_full_name, {'U': U2_z0_SSI})

    # %%
    # 绘制 U2_z_energy 随 z 演化的 曲线

    if is_energy_evolution_on == 1:
        vmax_U2_z_energy = np.max(U2_z_energy)
        vmin_U2_z_energy = np.min(U2_z_energy)

        U2_z_energy_address = location + "\\" + "6. U2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI" + "\\" + "6.1. NLA - " + "U2_" + str(
            float('%.2g' % z0)) + "mm" + "_SSI" + "_energy_evolution" + img_name_extension

        plot_1d(zj, sample, size_PerPixel,
                U2_z_energy, U2_z_energy_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_energy_evolution",
                is_save, dpi, size_fig * 10, size_fig,
                color_1d, ticks_num, is_title_on, is_axes_on, is_mm, 1,
                fontsize, font,
                0, vmax_U2_z_energy, vmin_U2_z_energy)

    return U2_z0_SSI, G2_z0_SSI_shift


if __name__ == '__main__':
    SFM_SSI_chi2(U1_name="",
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
                 is_print=1, is_contours=1, n_TzQ=1,
                 Gz_max_Enhance=1, match_mode=1,
                 # %%
                 border_percentage=0.1, )
