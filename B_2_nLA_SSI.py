# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
import math
from scipy.io import loadmat
from fun_os import img_squared_bordered_Read, U_Read, U_dir, U_energy_print, \
    U_plot, U_energy_plot, U_amps_z_plot, U_phases_z_plot, U_amp_plot_3d, U_slices_plot, U_selects_plot, U_save
from fun_img_Resize import image_Add_black_border
from fun_pump import pump_LG
from fun_SSI import Cal_diz, Cal_Iz_frontface, Cal_Iz_structure, Cal_Iz_endface, Cal_Iz, Cal_iz_1, Cal_iz_2
from fun_linear import Cal_n, Cal_kz
from fun_nonlinear import Cal_lc_SHG, Cal_GxGyGz
from fun_thread import my_thread
from fun_statistics import U_Drop_n_sigma
np.seterr(divide='ignore', invalid='ignore')

# %%

def nLA_SSI(U1_name="",
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
            is_print=1,
            # %%
            **kwargs, ):
    # %%

    if (type(U1_name) != str) or U1_name == "":

        if __name__ == "__main__":
            border_percentage = kwargs["border_percentage"] if len(kwargs) != 0 else 0.1

            image_Add_black_border(img_full_name,  # 预处理 导入图片 为方形，并加边框
                                   border_percentage,
                                   is_print, )

        # %%
        # 导入 方形，以及 加边框 的 图片

        img_name, img_name_extension, img_squared, \
        size_PerPixel, size_fig, I1_x, I1_y, U1_0 = img_squared_bordered_Read(
            img_full_name,
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
        size_PerPixel, size_fig, I1_x, I1_y, U1_0 = U_Read(U1_name,
                                                           img_full_name,
                                                           U1_0_NonZero_size,
                                                           dpi,
                                                           is_save_txt, )

    # %%

    n1, k1 = Cal_n(size_PerPixel,
                   is_air,
                   lam1, T, p="e")

    k1_z_shift, mesh_k1_x_k1_y_shift = Cal_kz(I1_x, I1_y, k1)

    # %%
    # 非线性 角谱理论 - SSI begin

    I1_x, I1_y = U1_0.shape[0], U1_0.shape[1]

    # %%
    # 引入 倒格矢，对 k2 的 方向 进行调整，其实就是对 k2 的 k2x, k2y, k2z 网格的 中心频率 从 (0, 0, k2z) 移到 (Gx, Gy, k2z + Gz)

    lam2 = lam1 / 2

    n2, k2 = Cal_n(size_PerPixel,
                   is_air,
                   lam2, T, p="e")

    # %%

    dk, lc, Tz = Cal_lc_SHG(k1, k2, Tz, size_PerPixel,
                            is_print=0)

    Gx, Gy, Gz = Cal_GxGyGz(mx, my, mz,
                            Tx, Ty, Tz, size_PerPixel,
                            is_print)

    # %%
    # 定义 调制区域切片厚度 的 纵向实际像素、调制区域切片厚度 的 实际纵向尺寸

    diz, deff_structure_sheet = Cal_diz(deff_structure_sheet_expect, deff_structure_length_expect, size_PerPixel,
                                        Tz, mz,
                                        is_print)

    # %%
    # 定义 结构前端面 距离 晶体前端面 的 纵向实际像素、结构前端面 距离 晶体前端面 的 实际纵向尺寸

    sheet_th_frontface, sheets_num_frontface, Iz_frontface, z0_structure_frontface = Cal_Iz_frontface(diz,
                                                                                                      z0_structure_frontface_expect,
                                                                                                      L0_Crystal,
                                                                                                      size_PerPixel,
                                                                                                      is_print)

    # %%
    # 定义 调制区域 的 纵向实际像素、调制区域 的 实际纵向尺寸

    sheets_num_structure, Iz_structure, deff_structure_length = Cal_Iz_structure(diz,
                                                                                 deff_structure_length_expect,
                                                                                 size_PerPixel,
                                                                                 is_print)

    # %%
    # 定义 结构后端面 距离 晶体前端面 的 纵向实际像素、结构后端面 距离 晶体前端面 的 实际纵向尺寸

    sheet_th_endface, sheets_num_endface, Iz_endface, z0_structure_endface = Cal_Iz_endface(sheets_num_frontface,
                                                                                            sheets_num_structure,
                                                                                            Iz_frontface, Iz_structure,
                                                                                            diz,
                                                                                            size_PerPixel,
                                                                                            is_print)

    # %%
    # 定义 晶体 的 纵向实际像素、晶体 的 实际纵向尺寸

    sheets_num, Iz = Cal_Iz(diz,
                            L0_Crystal, size_PerPixel,
                            is_print)
    z0 = L0_Crystal

    # zj = np.zeros( (sheets_num + 1), dtype=np.float64() )
    # for i in range(sheets_num + 1):
    #     iz = i * diz
    #     zj[i] = iz * size_PerPixel
    zj = np.arange(sheets_num + 1, dtype=np.float64()) * diz * size_PerPixel
    zj[sheets_num] = Iz * size_PerPixel

    izj = zj / size_PerPixel  # 为循环 里使用
    dizj = izj[1:] - izj[:-1]  # 为循环 里使用

    # print(np.mod(Iz,diz))
    # print(Iz - Iz//diz * diz)
    leftover = Iz - Iz // diz * diz
    if leftover == 0:  # 触底反弹：如果 不剩（整除），则最后一步 保持 diz 不动，否则 沿用 leftover
        leftover = diz
    # print(leftover)

    # %%
    # 定义 需要展示的截面 1 距离晶体前端面 的 纵向实际像素、需要展示的截面 1 距离晶体前端面 的 实际纵向尺寸

    sheet_th_section_1, sheets_num_section_1, iz_1, z0_1 = Cal_iz_1(diz,
                                                                    z0_section_1_expect, size_PerPixel,
                                                                    is_print)

    # %%
    # 定义 需要展示的截面 2 距离晶体后端面 的 纵向实际像素、需要展示的截面 2 距离晶体后端面 的 实际纵向尺寸

    sheet_th_section_2, sheets_num_section_2, iz_2, z0_2 = Cal_iz_2(sheets_num,
                                                                    Iz, diz,
                                                                    z0_section_2_expect, size_PerPixel,
                                                                    is_print)

    # %%
    # const

    deff = deff * 1e-12  # pm / V 转换成 m / V
    const = deff

    # %%
    # G1_z0_shift

    folder_address = U_dir("", "0.n1_modulation_squared", 0, )

    global G1_z_plus_dz_shift
    G1_z_plus_dz_shift = g1_shift
    U1_z_plus_dz = U1_0

    if is_energy_evolution_on == 1:
        G1_z_shift_energy = np.zeros((sheets_num + 1), dtype=np.float64())
        U1_z_energy = np.zeros((sheets_num + 1), dtype=np.float64())
    G1_z_shift_energy[0] = np.sum(np.abs(G1_z_plus_dz_shift) ** 2)
    U1_z_energy[0] = np.sum(np.abs(U1_z_plus_dz) ** 2)

    def H1_z_plus_dz_shift_k1_z(diz):
        return np.power(math.e, k1_z_shift * diz * 1j)  # 注意 这里的 传递函数 的 指数是 正的 ！！！

    def H1_z_shift_k1_z(diz):
        return (np.power(math.e,
                         k1_z_shift * diz * 1j) - 1) / k1_z_shift ** 2 * size_PerPixel ** 2  # 注意 这里的 传递函数 的 指数是 正的 ！！！

    if is_stored == 1:
        # sheet_stored_th = np.zeros( (sheets_stored_num + 1), dtype=np.int64() ) # 这个其实 就是 0123...
        sheet_th_stored = np.zeros(int(sheets_stored_num + 1), dtype=np.int64())
        iz_stored = np.zeros(int(sheets_stored_num + 1), dtype=np.float64())
        z_stored = np.zeros(int(sheets_stored_num + 1), dtype=np.float64())
        G1_z_shift_stored = np.zeros((I1_x, I1_y, int(sheets_stored_num + 1)), dtype=np.complex128())
        U1_z_stored = np.zeros((I1_x, I1_y, int(sheets_stored_num + 1)), dtype=np.complex128())

        # 小写的 x,y 表示 电脑中 矩阵坐标系，大写 X,Y 表示 笛卡尔坐标系
        # G1_shift_xz_stored = np.zeros( (I1_x, sheets_num + 1), dtype=np.complex128() )
        # G1_shift_yz_stored = np.zeros( (I1_y, sheets_num + 1), dtype=np.complex128() )
        # U1_xz_stored = np.zeros( (I1_x, sheets_num + 1), dtype=np.complex128() )
        # U1_yz_stored = np.zeros( (I1_y, sheets_num + 1), dtype=np.complex128() )
        G1_shift_YZ_stored = np.zeros((I1_x, sheets_num + 1), dtype=np.complex128())
        G1_shift_XZ_stored = np.zeros((I1_y, sheets_num + 1), dtype=np.complex128())
        U1_YZ_stored = np.zeros((I1_x, sheets_num + 1), dtype=np.complex128())
        U1_XZ_stored = np.zeros((I1_y, sheets_num + 1), dtype=np.complex128())

        G1_structure_frontface_shift = np.zeros((I1_x, I1_y), dtype=np.complex128())
        U1_structure_frontface = np.zeros((I1_x, I1_y), dtype=np.complex128())
        G1_structure_endface_shift = np.zeros((I1_x, I1_y), dtype=np.complex128())
        U1_structure_endface = np.zeros((I1_x, I1_y), dtype=np.complex128())
        G1_section_1_shift = np.zeros((I1_x, I1_y), dtype=np.complex128())
        U1_section_1 = np.zeros((I1_x, I1_y), dtype=np.complex128())
        G1_section_2_shift = np.zeros((I1_x, I1_y), dtype=np.complex128())
        U1_section_2 = np.zeros((I1_x, I1_y), dtype=np.complex128())

    def Cal_modulation_squared_z(for_th, fors_num, *arg, ):

        if is_bulk == 0:
            if for_th >= sheets_num_frontface and for_th <= sheets_num_endface - 1:
                modulation_squared_full_name = str(for_th - sheets_num_frontface) + ".mat"
                modulation_squared_address = folder_address + "\\" + modulation_squared_full_name
                modulation_squared_z = loadmat(modulation_squared_address)['n1_modulation_squared']
            else:
                modulation_squared_z = np.ones((I1_x, I1_y), dtype=np.int64()) * n1
        else:
            modulation_squared_z = np.ones((I1_x, I1_y), dtype=np.int64()) * n1

        return modulation_squared_z

    def Cal_G1_z_plus_dz_shift(for_th, fors_num, modulation_squared_z, *arg, ):

        global G1_z_plus_dz_shift

        G1_z = np.fft.ifftshift(G1_z_plus_dz_shift)
        U1_z = np.fft.ifft2(G1_z)

        Q1_z = np.fft.fft2((k1 / size_PerPixel / n1) ** 2 * (modulation_squared_z ** 2 - n1 ** 2) * U1_z)
        Q1_z_shift = np.fft.fftshift(Q1_z)

        G1_z_plus_dz_shift = G1_z_plus_dz_shift * H1_z_plus_dz_shift_k1_z(
            dizj[for_th]) + const * Q1_z_shift * H1_z_shift_k1_z(dizj[for_th])

        return G1_z_plus_dz_shift

    def After_G1_z_plus_dz_shift_temp(for_th, fors_num, G1_z_plus_dz_shift_temp, *arg, ):

        if is_stored == 1:
            global G1_structure_frontface_shift, U1_structure_frontface, G1_structure_endface_shift, \
                U1_structure_endface, G1_section_1_shift, U1_section_1, G1_section_2_shift, U1_section_2

        G1_z_plus_dz = np.fft.ifftshift(G1_z_plus_dz_shift_temp)
        U1_z_plus_dz = np.fft.ifft2(G1_z_plus_dz)

        if is_energy_evolution_on == 1:
            G1_z_shift_energy[for_th + 1] = np.sum(np.abs(G1_z_plus_dz_shift_temp) ** 2)
            U1_z_energy[for_th + 1] = np.sum(np.abs(U1_z_plus_dz) ** 2)

        if is_stored == 1:

            # 小写的 x,y 表示 电脑中 矩阵坐标系，大写 X,Y 表示 笛卡尔坐标系
            G1_shift_YZ_stored[:, for_th] = G1_z_plus_dz_shift_temp[:, I1_y // 2 + int(
                X / size_PerPixel)]  # X 增加，则 从 G1_z_shift 中 读取的 列 向右移，也就是 YZ 面向 列 增加的方向（G1_z_shift 的 右侧）移动
            G1_shift_XZ_stored[:, for_th] = G1_z_plus_dz_shift_temp[I1_x // 2 - int(Y / size_PerPixel),
                                            :]  # Y 增加，则 从 G1_z_shift 中 读取的 行 向上移，也就是 XZ 面向 行 减小的方向（G1_z_shift 的 上侧）移动
            U1_YZ_stored[:, for_th] = U1_z_plus_dz[:, I1_y // 2 + int(X / size_PerPixel)]
            U1_XZ_stored[:, for_th] = U1_z_plus_dz[I1_x // 2 - int(Y / size_PerPixel), :]

            # %%

            if np.mod(for_th,
                      sheets_num // sheets_stored_num) == 0:  # 如果 for_th 是 sheets_num // sheets_stored_num 的 整数倍（包括零），则 储存之
                sheet_th_stored[int(for_th // (sheets_num // sheets_stored_num))] = for_th + 1
                iz_stored[int(for_th // (sheets_num // sheets_stored_num))] = izj[for_th + 1]
                z_stored[int(for_th // (sheets_num // sheets_stored_num))] = zj[for_th + 1]
                G1_z_shift_stored[:, :, int(for_th // (
                        sheets_num // sheets_stored_num))] = G1_z_plus_dz_shift_temp  # 储存的 第一层，实际上不是 G1_0，而是 G1_dz
                U1_z_stored[:, :,
                int(for_th // (sheets_num // sheets_stored_num))] = U1_z_plus_dz  # 储存的 第一层，实际上不是 U1_0，而是 U1_dz

            if for_th == sheet_th_frontface:  # 如果 for_th 是 sheet_th_frontface，则把结构 前端面 场分布 储存起来，对应的是 zj[sheets_num_frontface]
                G1_structure_frontface_shift = G1_z_plus_dz_shift_temp
                U1_structure_frontface = U1_z_plus_dz
            if for_th == sheet_th_endface:  # 如果 for_th 是 sheet_th_endface，则把结构 后端面 场分布 储存起来，对应的是 zj[sheets_num_endface]
                G1_structure_endface_shift = G1_z_plus_dz_shift_temp
                U1_structure_endface = U1_z_plus_dz
            if for_th == sheet_th_section_1:  # 如果 for_th 是 想要观察的 第一个面 前面那一层的 层序数，则 将储存之于 该层 前面那一层的 后端面（毕竟 算出来的是 z + dz） 分布中
                G1_section_1_shift = G1_z_plus_dz_shift_temp  # 对应的是 zj[sheets_num_section_1]
                U1_section_1 = U1_z_plus_dz
            if for_th == sheet_th_section_2:  # 如果 for_th 是 想要观察的 第二个面 前面那一层的 层序数，则 将储存之于 该层 前面那一层的 后端面（毕竟 算出来的是 z + dz） 分布中
                G1_section_2_shift = G1_z_plus_dz_shift_temp  # 对应的是 zj[sheets_num_section_2]
                U1_section_2 = U1_z_plus_dz

    my_thread(10, sheets_num,
              Cal_modulation_squared_z, Cal_G1_z_plus_dz_shift, After_G1_z_plus_dz_shift_temp,
              is_ordered=1, is_print=is_print, )

    # %%

    zj[sheets_num] = Iz * size_PerPixel
    # print(zj)
    # print(z_stored)

    G1_z0_SSI_shift = G1_z_plus_dz_shift

    folder_address = ''

    if is_save == 1:
        folder_address = U_dir(U1_name, "G1_z0_SSI_shift", 1, z0)

    # %%
    # 绘图：G1_z0_SSI_shift_amp

    U_amp_plot_address, U_phase_plot_address = U_plot(U1_name, folder_address, 1,
                                                      G1_z0_SSI_shift, "G1_z0_SSI_shift", "NLA",
                                                      img_name_extension,
                                                      # %%
                                                      1, size_PerPixel,
                                                      is_save, dpi, size_fig,
                                                      cmap_2d, ticks_num, is_contourf,
                                                      is_title_on, is_axes_on, is_mm,
                                                      fontsize, font,
                                                      is_colorbar_on, is_energy,
                                                      # %%
                                                      z0, )

    # %%
    # 储存 G1_z0_SSI_shift 到 txt 文件

    if is_save == 1:
        U_address = U_save(U1_name, folder_address, 1,
                           G1_z0_SSI_shift, "G1_z0_SSI_shift", "NLA",
                           is_save_txt, z0, )

    # %%
    # 绘制 G1_z_shift_energy 随 z 演化的 曲线

    if is_energy_evolution_on == 1:
        U_energy_plot_address = U_energy_plot(U1_name, folder_address, 1,
                                              G1_z_shift_energy, "G1_z_SSI_shift_energy", "NLA",
                                              img_name_extension,
                                              # %%
                                              zj, sample, size_PerPixel,
                                              is_save, dpi, size_fig * 10, size_fig,
                                              color_1d, ticks_num,
                                              is_title_on, is_axes_on, is_mm,
                                              fontsize, font,
                                              # %%
                                              z0, )

    # %%
    # % H1_z0

    H1_z0_SSI_shift = G1_z0_SSI_shift / np.max(np.abs(G1_z0_SSI_shift)) / \
                      (g1_shift / np.max(np.abs(g1_shift)))
    # 扔掉 amp 偏离 amp 均值 3 倍于 总体 标准差 以外 的 数据，保留 剩下的 3 倍 以内的 数据。
    H1_z0_SSI_shift = U_Drop_n_sigma(H1_z0_SSI_shift, 3, is_energy)

    if is_save == 1:
        folder_address = U_dir(U1_name, "H1_z0_SSI_shift", 1, z0, )

    # %%
    # % H1_z0_SSI_shift

    U_amp_plot_address, U_phase_plot_address = U_plot(U1_name, folder_address, 1,
                                                      H1_z0_SSI_shift, "H1_z0_SSI_shift", "NLA",
                                                      img_name_extension,
                                                      # %%
                                                      1, size_PerPixel,
                                                      is_save, dpi, size_fig,
                                                      cmap_2d, ticks_num, is_contourf,
                                                      is_title_on, is_axes_on, is_mm,
                                                      fontsize, font,
                                                      is_colorbar_on, is_energy,
                                                      # %%
                                                      z0, )

    # %%
    # 储存 H1_z0_SSI_shift 到 txt 文件

    if is_save == 1:
        U_address = U_save(U1_name, folder_address, 1,
                           H1_z0_SSI_shift, "H1_z0_SSI_shift", "NLA",
                           is_save_txt, z0, )

    # %%
    # G1_z0_SSI = G1_z0_SSI(k1_x, k1_y) → IFFT2 → U1(x0, y0, z0) = U1_z0_SSI

    G1_z0_SSI = np.fft.ifftshift(G1_z0_SSI_shift)
    U1_z0_SSI = np.fft.ifft2(G1_z0_SSI)

    U_energy_print(U1_name, 1, 1,
                   U1_z0_SSI, "U1_z0_SSI", "NLA",
                   z0, )

    if is_save == 1:
        folder_address = U_dir(U1_z0_SSI, "U1_z0_SSI", 1, z0)

    # %%
    # 绘图：U1_z0_SSI

    U_amp_plot_address, U_phase_plot_address = U_plot(U1_name, folder_address, 1,
                                                      U1_z0_SSI, "U1_z0_SSI", "NLA",
                                                      img_name_extension,
                                                      # %%
                                                      1, size_PerPixel,
                                                      is_save, dpi, size_fig,
                                                      cmap_2d, ticks_num, is_contourf,
                                                      is_title_on, is_axes_on, is_mm,
                                                      fontsize, font,
                                                      is_colorbar_on, is_energy,
                                                      # %%
                                                      z0, )

    # %%
    # 储存 U1_z0_SSI 到 txt 文件

    if is_save == 1:
        U_address = U_save(U1_name, folder_address, 1,
                           U1_z0_SSI, "U1_z0_SSI", "NLA",
                           is_save_txt, z0, )

    # %%
    # 绘制 U1_z_energy 随 z 演化的 曲线

    if is_energy_evolution_on == 1:
        U_energy_plot_address = U_energy_plot(U1_name, folder_address, 1,
                                              U1_z_energy, "U1_z_SSI_energy", "NLA",
                                              img_name_extension,
                                              # %%
                                              zj, sample, size_PerPixel,
                                              is_save, dpi, size_fig * 10, size_fig,
                                              color_1d, ticks_num,
                                              is_title_on, is_axes_on, is_mm,
                                              fontsize, font,
                                              # %%
                                              z0, )

    # %%

    if is_stored == 1:

        sheet_th_stored[sheets_stored_num] = sheets_num
        iz_stored[sheets_stored_num] = Iz
        z_stored[sheets_stored_num] = Iz * size_PerPixel
        G1_z_shift_stored[:, :, sheets_stored_num] = G1_z0_SSI_shift  # 储存的 第一层，实际上不是 G1_0，而是 G1_dz
        U1_z_stored[:, :, sheets_stored_num] = U1_z0_SSI  # 储存的 第一层，实际上不是 U1_0，而是 U1_dz

        # %%

        if is_save == 1:
            folder_address = U_dir(U1_name, "G1_z0_SSI_shift_sheets_stored", 1, z0, )

        # -------------------------

        U_amps_z_plot(U1_name, folder_address, 1,
                      G1_z_shift_stored, "G1_z_SSI_shift", "NLA",
                      img_name_extension,
                      # %%
                      sample, size_PerPixel,
                      is_save, dpi, size_fig,
                      # %%
                      cmap_2d, ticks_num, is_contourf,
                      is_title_on, is_axes_on, is_mm,
                      fontsize, font,
                      # %%
                      is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                      # %%                          何况 一般默认 is_self_colorbar = 1...
                      z_stored, )

        U_phases_z_plot(U1_name, folder_address, 1,
                        G1_z_shift_stored, "G1_z_SSI_shift", "NLA",
                        img_name_extension,
                        # %%
                        sample, size_PerPixel,
                        is_save, dpi, size_fig,
                        # %%
                        cmap_2d, ticks_num, is_contourf,
                        is_title_on, is_axes_on, is_mm,
                        fontsize, font,
                        # %%
                        is_colorbar_on,  # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
                        # %%
                        z_stored, )

        # -------------------------

        if is_save == 1:
            folder_address = U_dir(U1_name, "U1_z0_SSI_sheets_stored", 1, z0, )

        U_amps_z_plot(U1_name, folder_address, 1,
                      U1_z_stored, "U1_z_SSI", "NLA",
                      img_name_extension,
                      # %%
                      sample, size_PerPixel,
                      is_save, dpi, size_fig,
                      # %%
                      cmap_2d, ticks_num, is_contourf,
                      is_title_on, is_axes_on, is_mm,
                      fontsize, font,
                      # %%
                      is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                      # %%                          何况 一般默认 is_self_colorbar = 1...
                      z_stored, )

        U_phases_z_plot(U1_name, folder_address, 1,
                        U1_z_stored, "U1_z_SSI", "NLA",
                        img_name_extension,
                        # %%
                        sample, size_PerPixel,
                        is_save, dpi, size_fig,
                        # %%
                        cmap_2d, ticks_num, is_contourf,
                        is_title_on, is_axes_on, is_mm,
                        fontsize, font,
                        # %%
                        is_colorbar_on,  # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
                        # %%
                        z_stored, )

        # %%
        # 这 sheets_stored_num 层 也可以 画成 3D，就是太丑了，所以只 整个 U1_amp 示意一下即可

        U_amp_plot_address = U_amp_plot_3d(U1_name, folder_address, 1,
                                           U1_z_stored, "U1_z0_SSI_sheets_stored", "NLA",
                                           img_name_extension,
                                           # %%
                                           sample, size_PerPixel,
                                           is_save, dpi, size_fig,
                                           elev, azim, alpha,
                                           # %%
                                           cmap_3d, ticks_num,
                                           is_title_on, is_axes_on, is_mm,
                                           fontsize, font,
                                           # %%
                                           is_colorbar_on, is_energy,
                                           # %%
                                           zj, z_stored, )

        # %%

        # 小写的 x,y 表示 电脑中 矩阵坐标系，大写 X,Y 表示 笛卡尔坐标系
        # G1_shift_xz_stored[:, sheets_num] = G1_z0_SSI_shift[:, I1_y // 2 + int(X / size_PerPixel)]
        # G1_shift_yz_stored[:, sheets_num] = G1_z0_SSI_shift[I1_x // 2 - int(Y / size_PerPixel), :]
        # U1_xz_stored[:, sheets_num] = U1_z0_SSI[:, I1_y // 2 + int(X / size_PerPixel)]
        # U1_yz_stored[:, sheets_num] = U1_z0_SSI[I1_x // 2 - int(Y / size_PerPixel), :]
        G1_shift_YZ_stored[:, sheets_num] = G1_z0_SSI_shift[:, I1_y // 2 + int(X / size_PerPixel)]
        G1_shift_XZ_stored[:, sheets_num] = G1_z0_SSI_shift[I1_x // 2 - int(Y / size_PerPixel), :]
        U1_YZ_stored[:, sheets_num] = U1_z0_SSI[:, I1_y // 2 + int(X / size_PerPixel)]
        U1_XZ_stored[:, sheets_num] = U1_z0_SSI[I1_x // 2 - int(Y / size_PerPixel), :]

        # %%

        if is_save == 1:
            folder_address = U_dir(U1_name, "G1_z0_SSI_shift_YZ_XZ_stored", 1, z0, )

        # ========================= G1_shift_YZ_stored_amp、G1_shift_XZ_stored_amp
        # ------------------------- G1_shift_YZ_stored_phase、G1_shift_XZ_stored_phase

        U_slices_plot(U1_name, folder_address, 1,
                      G1_shift_YZ_stored, "G1_z0_SSI_shift_YZ", "NLA",
                      G1_shift_XZ_stored, "G1_z0_SSI_shift_XZ",
                      img_name_extension,
                      # %%
                      zj, sample, size_PerPixel,
                      is_save, dpi, size_fig,
                      # %%
                      cmap_2d, ticks_num, is_contourf,
                      is_title_on, is_axes_on, is_mm,
                      fontsize, font,
                      # %%
                      is_colorbar_on, is_energy,
                      # %%
                      X, Y, )

        # %%

        if is_save == 1:
            folder_address = U_dir(U1_name, "U1_z0_SSI_YZ_XZ_stored", 1, z0, )

        # ========================= U1_YZ_stored_amp、U1_XZ_stored_amp
        # ------------------------- U1_YZ_stored_phase、U1_XZ_stored_phase

        U_slices_plot(U1_name, folder_address, 1,
                      U1_YZ_stored, "U1_z0_SSI_YZ", "NLA",
                      U1_XZ_stored, "U1_z0_SSI_XZ",
                      img_name_extension,
                      # %%
                      zj, sample, size_PerPixel,
                      is_save, dpi, size_fig,
                      # %%
                      cmap_2d, ticks_num, is_contourf,
                      is_title_on, is_axes_on, is_mm,
                      fontsize, font,
                      # %%
                      is_colorbar_on, is_energy,
                      # %%
                      X, Y, )

        # %%

        if is_save == 1:
            folder_address = U_dir(U1_name, "G1_z0_SSI_shift_sheets_selective_stored", 1, z0, )

        # ------------------------- 储存 G1_section_1_shift_amp、G1_section_1_shift_amp、G1_structure_frontface_shift_amp、G1_structure_endface_shift_amp
        # ------------------------- 储存 G1_section_1_shift_phase、G1_section_1_shift_phase、G1_structure_frontface_shift_phase、G1_structure_endface_shift_phase

        U_selects_plot(U1_name, folder_address, 1,
                       G1_section_1_shift, "G1_z0_SSI_sec1_shift", "NLA",
                       G1_section_2_shift, "G1_z0_SSI_sec2_shift",
                       G1_structure_frontface_shift, "G1_z0_SSI_front_shift",
                       G1_structure_endface_shift, "G1_z0_SSI_end_shift",
                       img_name_extension,
                       # %%
                       sample, size_PerPixel,
                       is_save, dpi, size_fig,
                       # %%
                       cmap_2d, ticks_num, is_contourf,
                       is_title_on, is_axes_on, is_mm,
                       fontsize, font,
                       # %%
                       is_colorbar_on, is_energy, is_show_structure_face,
                       # %%
                       z0_1, z0_2, z0_structure_frontface, z0_structure_endface, )

        # %%

        if is_save == 1:
            folder_address = U_dir(U1_name, "U1_z0_SSI_sheets_selective_stored", 1, z0, )

        # ------------------------- 储存 U1_section_1_amp、U1_section_1_amp、U1_structure_frontface_amp、U1_structure_endface_amp
        # ------------------------- 储存 U1_section_1_phase、U1_section_1_phase、U1_structure_frontface_phase、U1_structure_endface_phase

        U_selects_plot(U1_name, folder_address, 1,
                       U1_section_1, "U1_z0_SSI_sec1", "NLA",
                       U1_section_2, "U1_z0_SSI_sec2",
                       U1_structure_frontface, "U1_z0_SSI_front",
                       U1_structure_endface, "U1_z0_SSI_end",
                       img_name_extension,
                       # %%
                       sample, size_PerPixel,
                       is_save, dpi, size_fig,
                       # %%
                       cmap_2d, ticks_num, is_contourf,
                       is_title_on, is_axes_on, is_mm,
                       fontsize, font,
                       # %%
                       is_colorbar_on, is_energy, is_show_structure_face,
                       # %%
                       z0_1, z0_2, z0_structure_frontface, z0_structure_endface, )

        # %%
        # 绘制 G1_amp 的 侧面 3D 分布图，以及 初始 和 末尾的 G1_amp（现在 可以 任选位置 了）

        # vmax_G1_amp = np.max([vmax_G1_shift_YZ_XZ_stored_amp, vmax_G1_section_1_2_front_end_shift_amp])
        # vmin_G1_amp = np.min([vmin_G1_shift_YZ_XZ_stored_amp, vmin_G1_section_1_2_front_end_shift_amp])

        # G1_shift_XYZ_stored_amp_address = location + "\\" + "5. G1_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored" + "\\" + "5.1. NLA - " + "G1_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_XYZ" + "_amp" + img_name_extension

        # plot_3d_XYZ(zj, sample, size_PerPixel, 
        #             np.abs(G1_shift_YZ_stored), np.abs(G1_shift_XZ_stored), np.abs(G1_section_1_shift), np.abs(G1_section_1_shift), 
        #             np.abs(G1_structure_frontface_shift), np.abs(G1_structure_endface_shift), is_show_structure_face, 
        #             G1_shift_XYZ_stored_amp_address, "G1_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_XYZ" + "_amp", 
        #             I1_y // 2 + int(X / size_PerPixel), I1_x // 2 + int(Y / size_PerPixel), sheet_th_section_1, sheet_th_section_2, 
        #             sheets_num_frontface, sheets_num_endface, 
        #             is_save, dpi, size_fig, 
        #             cmap_3d, elev, azim, alpha, 
        #             ticks_num, is_title_on, is_axes_on, is_mm,  
        #             fontsize, font, 
        #             is_self_colorbar, is_colorbar_on, is_energy, vmax_G1_amp, vmin_G1_amp)

        # %%
        # 绘制 G1_phase 的 侧面 3D 分布图，以及 初始 和 末尾的 G1_phase

        # vmax_G1_phase = np.max([vmax_G1_shift_YZ_XZ_stored_phase, vmax_G1_section_1_2_front_end_shift_phase])
        # vmin_G1_phase = np.min([vmin_G1_shift_YZ_XZ_stored_phase, vmin_G1_section_1_2_front_end_shift_phase])

        # G1_shift_XYZ_stored_phase_address = location + "\\" + "5. G1_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored" + "\\" + "5.2. NLA - " + "G1_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_XYZ" + "_phase" + img_name_extension

        # plot_3d_XYZ(zj, sample, size_PerPixel, 
        #             np.angle(G1_shift_YZ_stored), np.angle(G1_shift_XZ_stored), np.angle(G1_section_1_shift), np.angle(G1_section_1_shift), 
        #             np.angle(G1_structure_frontface_shift), np.angle(G1_structure_endface_shift), is_show_structure_face, 
        #             G1_shift_XYZ_stored_phase_address, "G1_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_XYZ" + "_phase", 
        #             I1_y // 2 + int(X / size_PerPixel), I1_x // 2 + int(Y / size_PerPixel), sheet_th_section_1, sheet_th_section_2, 
        #             sheets_num_frontface, sheets_num_endface, 
        #             is_save, dpi, size_fig, 
        #             cmap_3d, elev, azim, alpha, 
        #             ticks_num, is_title_on, is_axes_on, is_mm,  
        #             fontsize, font, 
        #             is_self_colorbar, is_colorbar_on, 0, vmax_G1_phase, vmin_G1_phase)

        # %%
        # 绘制 U1_amp 的 侧面 3D 分布图，以及 初始 和 末尾的 U1_amp

        # vmax_U1_amp = np.max([vmax_U1_YZ_XZ_stored_amp, vmax_U1_section_1_2_front_end_shift_amp])
        # vmin_U1_amp = np.min([vmin_U1_YZ_XZ_stored_amp, vmin_U1_section_1_2_front_end_shift_amp])

        # U1_XYZ_stored_amp_address = location + "\\" + "6. U1_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored" + "\\" + "6.1. NLA - " + "U1_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_XYZ" + "_amp" + img_name_extension

        # plot_3d_XYZ(zj, sample, size_PerPixel, 
        #             np.abs(U1_YZ_stored), np.abs(U1_XZ_stored), np.abs(U1_section_1), np.abs(U1_section_2), 
        #             np.abs(U1_structure_frontface), np.abs(U1_structure_endface), is_show_structure_face, 
        #             U1_XYZ_stored_amp_address, "U1_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_XYZ" + "_amp", 
        #             I1_y // 2 + int(X / size_PerPixel), I1_x // 2 + int(Y / size_PerPixel), sheet_th_section_1, sheet_th_section_2, 
        #             sheets_num_frontface, sheets_num_endface, 
        #             is_save, dpi, size_fig, 
        #             cmap_3d, elev, azim, alpha, 
        #             ticks_num, is_title_on, is_axes_on, is_mm,  
        #             fontsize, font, 
        #             is_self_colorbar, is_colorbar_on, is_energy, vmax_U1_amp, vmin_U1_amp)

        # %%
        # 绘制 U1_phase 的 侧面 3D 分布图，以及 初始 和 末尾的 U1_phase

        # vmax_U1_phase = np.max([vmax_U1_YZ_XZ_stored_phase, vmax_U1_section_1_2_front_end_shift_phase])
        # vmin_U1_phase = np.min([vmin_U1_YZ_XZ_stored_phase, vmin_U1_section_1_2_front_end_shift_phase])

        # U1_XYZ_stored_phase_address = location + "\\" + "6. U1_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored" + "\\" + "6.2. NLA - " + "U1_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_XYZ" + "_phase" + img_name_extension

        # plot_3d_XYZ(zj, sample, size_PerPixel, 
        #             np.angle(U1_YZ_stored), np.angle(U1_XZ_stored), np.angle(U1_section_1), np.angle(U1_section_2), 
        #             np.angle(U1_structure_frontface), np.angle(U1_structure_endface), is_show_structure_face, 
        #             U1_XYZ_stored_phase_address, "U1_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_XYZ" + "_phase", 
        #             I1_y // 2 + int(X / size_PerPixel), I1_x // 2 + int(Y / size_PerPixel), sheet_th_section_1, sheet_th_section_2, 
        #             sheets_num_frontface, sheets_num_endface, 
        #             is_save, dpi, size_fig, 
        #             cmap_3d, elev, azim, alpha, 
        #             ticks_num, is_title_on, is_axes_on, is_mm,  
        #             fontsize, font, 
        #             is_self_colorbar, is_colorbar_on, 0, vmax_U1_phase, vmin_U1_phase)

    return U1_z0_SSI, G1_z0_SSI_shift


if __name__ == '__main__':
    nLA_SSI(U1_name="",
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
            L0_Crystal=1, z0_structure_frontface_expect=0.5, deff_structure_length_expect=2,
            deff_structure_sheet_expect=1.8, sheets_stored_num=10,
            z0_section_1_expect=1, z0_section_2_expect=1,
            X=0, Y=0,
            # %%
            is_bulk=1,
            is_stored=1, is_show_structure_face=1, is_energy_evolution_on=1,
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
            is_print=1,
            # %%
            border_percentage=0.1, )
