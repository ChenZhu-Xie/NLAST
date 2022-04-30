# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 20:23:31 2022

@author: Xcz
"""

import math
import numpy as np
from fun_array_Transform import Roll_xy
from fun_linear import Cal_kz, fft2, ifft2, Uz_AST, Find_energy_Dropto_fraction
from fun_statistics import find_Kxyz
from fun_global_var import Get, init_accu, tree_print


# %%

def Sinc(x):
    return np.nan_to_num(np.sin(x) / x) + np.isnan(np.sin(x) / x).astype(np.int8)


# %%

def Cosc(x):
    return np.nan_to_num((np.cos(x) - 1) / x)


# return np.nan_to_num( (np.cos(x) - 1) / x ) * ( 1 - np.isnan( (np.cos(x) - 1) / x ).astype(np.int8) ) 不够聪明

# %%
# 定义 对于 kz 的 类似 Sinc 的 函数：( e^ikz - 1 ) / kz

def Eikz(x):
    return Cosc(x) + 1j * Sinc(x)


# %%
# 定义 m 级次 的 倒格波 系数 Cm

def C_m(m):
    if m == 0:
        return 1
    else:
        return Sinc(math.pi * m / 2) - Sinc(math.pi * m)


# %%

def Cal_lc_SHG(k1, k2, Tz, size_PerPixel,
               is_print=1, **kwargs):
    dk = 2 * k1 - k2  # Unit: 无量纲
    lc = math.pi / abs(dk) * size_PerPixel * 1000  # Unit: μm
    is_print and print(tree_print(kwargs.get("is_end", 0), kwargs.get("add_level", 0)) +
                       "lc = {} μm, Tc = {} μm".format(lc, lc * 2))

    # print(type(Tz) != np.float64)
    # print(type(Tz) != float) # float = np.float ≠ np.float64
    
    if (type(Tz) != float and type(Tz) != np.float64 
        and type(Tz) != int) or Tz <= 0:  # 如果 传进来的 Tz 既不是 float 也不是 int，或者 Tz <= 0，则给它 安排上 2*lc
        Tz = 2 * lc  # Unit: μm
        
    return dk, lc, Tz


# %%

def Cal_GxGyGz(mx, my, mz,
               Tx, Ty, Tz, size_PerPixel,
               is_print=1, **kwargs):
    Gx = 2 * math.pi * mx * size_PerPixel / (Tx / 1000)  # Tz / 1000 即以 mm 为单位
    Gy = 2 * math.pi * my * size_PerPixel / (Ty / 1000)  # Tz / 1000 即以 mm 为单位
    Gz = 2 * math.pi * mz * size_PerPixel / (Tz / 1000)  # Tz / 1000 即以 mm 为单位

    is_print and print(tree_print(add_level=-1) +
                       "mx = {} μm, my = {} μm, mz = {} μm".format(mx, my, mz))
    is_print and print(tree_print(kwargs.get("is_end", 0), kwargs.get("add_level", 0)) +
                       "Tx = {} μm, Ty = {} μm, Tz = {} μm".format(Tx, Ty, Tz))

    return Gx, Gy, Gz


# %%

def args_SHG(k1, k2, size_PerPixel,
             mx, my, mz,
             Tx, Ty, Tz,
             is_print, **kwargs):
    info = "args_SHG"
    is_first = int(init_accu(info, 1) == 1) # 若第一次调用 args_SHG，则 is_first 为 1，否则为 0
    is_Print = is_print * is_first # 两个 得都 非零，才 print

    info = "参数_SHG"
    is_Print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs["is_end"], kwargs["add_level"] = 0, 0  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    dk, lc, Tz = Cal_lc_SHG(k1, k2, Tz, size_PerPixel,
                            is_Print, )

    Gx, Gy, Gz = Cal_GxGyGz(mx, my, mz,
                            Tx, Ty, Tz, size_PerPixel,
                            is_Print, is_end=1)
    return dk, lc, Tz, \
           Gx, Gy, Gz


# %%

def Cal_dk_zQ_SHG(k1, k1_z, k2_z,
                 mesh_k1_x_k1_y, mesh_k2_x_k2_y,
                 n2_x, n2_y,
                 Gx, Gy, Gz, ):
    # n2_x_n2_y 的 mesh 才用 Gy / (2 * math.pi) * I2_y)，这里是 k2_x_k2_y 的 mesh，所以用 Gy 才对应
    dk_x = mesh_k2_x_k2_y[n2_x, n2_y, 0] - mesh_k1_x_k1_y[:, :, 0] - Gy
    # 其实 mesh_k2_x_k2_y[:, :, 0]、mesh_n2_x_n2_y[:, :, 0]、mesh_n2_x_n2_y[:, :, 0]、 n2_x 均只和 y，即 [:, :] 中的 第 2 个数字 有关，
    # 只由 列 y、ky 决定，与行 即 x、kx 无关
    # 而 Gy 得与 列 y、ky 发生关系,
    # 所以是 - Gy 而不是 Gx
    # 并且这里的 dk_x 应写为 dk_y
    dk_y = mesh_k2_x_k2_y[n2_x, n2_y, 1] - mesh_k1_x_k1_y[:, :, 1] - Gx
    k1_z_dk_x_dk_y = (k1 ** 2 - dk_x ** 2 - dk_y ** 2 + 0j) ** 0.5

    dk_z = k1_z + k1_z_dk_x_dk_y - k2_z[n2_x, n2_y]
    dk_zQ = dk_z + Gz

    return dk_zQ


# %%

def Cal_roll_xy(Gx, Gy,
                Ix, Iy,
                *args):
    if len(args) >= 2:
        nx, ny = args[0], args[1]
        roll_x = np.floor(Ix // 2 - (Ix - 1) + nx - Gy / (2 * math.pi) * Iy).astype(np.int64)
        roll_y = np.floor(Iy // 2 - (Iy - 1) + ny - Gx / (2 * math.pi) * Ix).astype(np.int64)
        # 之后要平移列，而 Gx 才与列有关...
    else:
        roll_x = np.floor(Gy / (2 * math.pi) * Iy).astype(np.int64)
        roll_y = np.floor(Gx / (2 * math.pi) * Ix).astype(np.int64)

    return roll_x, roll_y


def G2_z_modulation_3D_NLAST(k1, k2, Tz_unit,
                             modulation, U1_0, iz, const, ):
    k1_z, mesh_k1_x_k1_y = Cal_kz(U1_0.shape[0], U1_0.shape[1], k1)
    k2_z, mesh_k2_x_k2_y = Cal_kz(U1_0.shape[0], U1_0.shape[1], k2)

    Big_version = 0
    Cal_version = 1

    if Big_version == 0:
        kiiz = k1 + k1_z
        dkiiz = kiiz - k2
        dz = Tz_unit / 2
        J = int(iz / dz - 1)
        # J = iz / dz - 1
        # J = iz / dz

        # print(J, (-1)**J)

        if Cal_version == 1:
            # %% version 1（更自洽）

            G_U1_z_Squared_modulated_1 = fft2(
                ifft2(fft2(modulation) / (kiiz ** 2 - k2 ** 2) / (1 + math.e ** (- dkiiz * dz * 1j))) \
                * Uz_AST(U1_0, k1, iz) ** 2)

            G_U1_z_Squared_modulated_2 = fft2(
                ifft2(fft2(modulation) / (kiiz ** 2 - k2 ** 2) / (1 + math.e ** (kiiz * dz * 1j))) \
                * Uz_AST(U1_0, k1, iz) ** 2)

            g_U1_0_Squared_modulated = fft2(
                ifft2(fft2(modulation) / (kiiz ** 2 - k2 ** 2)
                      * (1 / (1 + math.e ** (- dkiiz * dz * 1j)) - 1 / (1 + math.e ** (kiiz * dz * 1j)))) \
                * U1_0 ** 2)

            G2_z = const * (G_U1_z_Squared_modulated_1 * (-1) ** J
                                    - G_U1_z_Squared_modulated_2 * (-1) ** J * math.e ** (k2_z * iz * 1j)
                                    + g_U1_0_Squared_modulated * math.e ** (k2_z * iz * 1j))

        elif Cal_version == 2:
            # %% version 2（少近似）

            G_U1_z_Squared_modulated_1 = fft2(
                ifft2(fft2(modulation) / (kiiz ** 2 - k2 ** 2) / (1 + math.e ** (- dkiiz * dz * 1j))) \
                * Uz_AST(U1_0, k1, iz) ** 2)

            G_U1_z_Squared_modulated_2 = fft2(
                ifft2(fft2(modulation) / (kiiz ** 2 - k2 ** 2)) \
                * Uz_AST(U1_0, k1, iz) ** 2 / (1 + math.e ** (kiiz * dz * 1j)))

            g_U1_0_Squared_modulated_1 = fft2(
                ifft2(fft2(modulation) / (kiiz ** 2 - k2 ** 2) / (1 + math.e ** (- dkiiz * dz * 1j))) \
                * U1_0 ** 2)

            g_U1_0_Squared_modulated_2 = fft2(
                ifft2(fft2(modulation) / (kiiz ** 2 - k2 ** 2)) \
                * U1_0 ** 2 / (1 + math.e ** (kiiz * dz * 1j)))

            G2_z = const * (G_U1_z_Squared_modulated_1 * (-1) ** J
                                    - G_U1_z_Squared_modulated_2 * (-1) ** J * math.e ** (k2_z * iz * 1j)
                                    + g_U1_0_Squared_modulated_1 * math.e ** (k2_z * iz * 1j)
                                    - g_U1_0_Squared_modulated_2 * math.e ** (k2_z * iz * 1j))

    return G2_z * Get("size_PerPixel")**2

# %%

def G2_z_modulation_NLAST(k1, k2, Gz,
                          modulation, U1_0, iz, const, ):
    k1_z, mesh_k1_x_k1_y = Cal_kz(U1_0.shape[0], U1_0.shape[1], k1)
    k2_z, mesh_k2_x_k2_y = Cal_kz(U1_0.shape[0], U1_0.shape[1], k2)

    Big_version = 1
    Cal_version = 1
    Res_version = 1

    # 3.4 > 3.2 > 1.3    match OK
    # 3.5 wrong
    # 3.6 = 1.1 > 2      dismatch OK
    # 4.0 e 指数太大，溢出
    # 5.0 = 3.4 + 1.1    dismatch + match OK

    if Big_version == 0 or Big_version == 2:

        if Big_version == 0:
            # kiiz = k1 + k2_z + Gz # 草，倍频是加 k1_z，和频才是加 k2_z（而非 k3_z）
            kiizQ = k1 + k1_z + Gz

            if Cal_version == 1:
                # %% == version 1（更自洽）
                G1_Uz_Squared_modulated = fft2(
                    ifft2(fft2(modulation) / (kiizQ ** 2 - k2 ** 2)) * Uz_AST(U1_0, k1, iz) ** 2)
                # print(np.min(np.abs((kiiz ** 2 - k2 ** 2))))
                g1_U0_Squared_modulated = fft2(
                    ifft2(fft2(modulation) / (kiizQ ** 2 - k2 ** 2)) * U1_0 ** 2)

            elif Cal_version == 2:
                # %% == version 1.1
                G1_Uz_Squared_modulated = fft2(
                    ifft2(fft2(modulation) / (kiizQ ** 2 - k2_z ** 2)) * Uz_AST(U1_0, k1, iz) ** 2)
                g1_U0_Squared_modulated = fft2(
                    ifft2(fft2(modulation) / (kiizQ ** 2 - k2_z ** 2)) * U1_0 ** 2)

            elif Cal_version == 3:
                # %% == version 2（少近似）
                G1_Uz_Squared_modulated = fft2(
                    modulation * ifft2(fft2(Uz_AST(U1_0, k1, iz) ** 2) / (kiizQ ** 2 - k2_z ** 2)))

                g1_U0_Squared_modulated = fft2(
                    modulation * ifft2(fft2(U1_0 ** 2) / (kiizQ ** 2 - k2_z ** 2)))

            elif Cal_version == 4:
                # %% == version 2.1
                G1_Uz_Squared_modulated = fft2(
                    modulation * ifft2(fft2(Uz_AST(U1_0, k1, iz) ** 2) / (kiizQ ** 2 - k2 ** 2)))

                g1_U0_Squared_modulated = fft2(
                    modulation * ifft2(fft2(U1_0 ** 2) / (kiizQ ** 2 - k2 ** 2)))

        elif Big_version == 2:
            K1_z, K1_xy = find_Kxyz(fft2(U1_0), k1)
            kiizQ = 2 * K1_z + Gz
            kii2z = (k2 ** 2 - (2 * K1_xy[0] + mesh_k2_x_k2_y[:, :, 0]) ** 2 - (
                    2 * K1_xy[1] + mesh_k2_x_k2_y[:, :, 1]) ** 2 + 0j) ** 0.5
            denominator = kiizQ ** 2 - kii2z ** 2

            G1_Uz_Squared_modulated = fft2(
                ifft2(fft2(modulation) / denominator) * Uz_AST(U1_0, k1, iz) ** 2)
            # print(np.min(np.abs((kiiz ** 2 - k2 ** 2))))
            g1_U0_Squared_modulated = fft2(
                ifft2(fft2(modulation) / denominator) * U1_0 ** 2)

        if Res_version == 1:
            # 不加 负号，U 的相位 会差个 π，我也不知道 为什么
            G2_z = const * (G1_Uz_Squared_modulated * math.e ** (Gz * iz * 1j)
                                    - g1_U0_Squared_modulated * math.e ** (k2_z * iz * 1j))
        elif Res_version == 2:
            G2_z = const * G1_Uz_Squared_modulated * math.e ** (Gz * iz * 1j)

        elif Res_version == 3:
            G2_z = - const * g1_U0_Squared_modulated * math.e ** (k2_z * iz * 1j)

    elif Big_version == 1:
        G1_Uz_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, iz) ** 2) * math.e ** (Gz * iz * 1j)
        g1_U0_Squared_modulated = fft2(modulation * U1_0 ** 2) * math.e ** (k2_z * iz * 1j)
        molecule = G1_Uz_Squared_modulated - g1_U0_Squared_modulated


        if Cal_version != 2:
            K1_z, K1_xy = find_Kxyz(fft2(U1_0), k1)
            kiizQ = 2 * K1_z + Gz
        if Cal_version >= 3:
            dkiizQ = kiizQ - k2_z
            inside_sinc = dkiizQ * iz / 2
            dkiizQ = 1 / (np.sinc(inside_sinc / np.pi) * iz / 2)

        if Cal_version == 1:
            denominator = kiizQ ** 2 - k2_z ** 2
        elif Cal_version == 2:
            kiizQ = 2 * k1_z + Gz
            denominator = kiizQ ** 2 - k2_z ** 2
        elif Cal_version == 3:
            denominator = (kiizQ + k2_z) * dkiizQ
        elif Cal_version == 4:
            denominator = (dkiizQ + 2 * k2_z) * dkiizQ
        elif Cal_version == 5:
            denominator = (dkiizQ + 2 * k2_z) * (kiizQ - k2_z)

        # print(np.max(np.abs(dkiizQ)), np.max(np.abs(kiizQ - k2_z)))
        # print(np.min(np.abs(dkiizQ)), np.min(np.abs(kiizQ - k2_z)))

        if Res_version == 1:
            G2_z = const * molecule / denominator
        elif Res_version == 2:
            G2_z = const * G1_Uz_Squared_modulated / denominator
        elif Res_version == 3:
            G2_z = const * g1_U0_Squared_modulated / denominator

    elif Big_version == 3:
        G1_U_half_z_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, iz/2) ** 2)
        if Cal_version == 1:
            molecule = G1_U_half_z_Squared_modulated \
                       * math.e ** (Gz * iz * 1j) \
                       * math.e ** (k2_z * iz/2 * 1j) \
                       * (1j * iz)

            denominator = k2_z
        elif Cal_version >= 2:
            if Cal_version != 5:
                K1_z, K1_xy = find_Kxyz(fft2(U1_0), k1)
                kiizQ = 2 * K1_z + Gz
                if Cal_version < 4:
                    kii2z = (k2 ** 2 - (2 * K1_xy[0] + mesh_k2_x_k2_y[:, :, 0]) ** 2 - (
                            2 * K1_xy[1] + mesh_k2_x_k2_y[:, :, 1]) ** 2 + 0j) ** 0.5
                    dkiizQ = kiizQ - kii2z
                else:
                    dkiizQ = kiizQ - k2_z
                inside_sinc = dkiizQ / 2 * iz
            elif Cal_version == 5:
                G1_Uz_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, iz) ** 2)
                g1_U0_Squared_modulated = fft2(modulation * U1_0 ** 2)
                exp_dk_factor = math.e ** (1j * (Gz - k2_z) / 2 * iz)
                i_dkiizQ_z_1 = np.log(G1_Uz_Squared_modulated / G1_U_half_z_Squared_modulated * exp_dk_factor)
                i_dkiizQ_z_2 = np.log(g1_U0_Squared_modulated / G1_U_half_z_Squared_modulated / exp_dk_factor)
                i_dkiizQ_z = i_dkiizQ_z_1 - i_dkiizQ_z_2
                dkiizQ = i_dkiizQ_z / (1j * iz)
                # print(np.max(np.abs(dkiizQ)))
                kiizQ = dkiizQ + k2_z
                # print(np.max(np.abs(kiizQ)))
                inside_sinc = dkiizQ / 2 * iz
            if Cal_version == 6:
                G1_Uz_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, iz) ** 2)
                g1_U0_Squared_modulated = fft2(modulation * U1_0 ** 2)
                exp_dk_factor = math.e ** (1j * (Gz - k2_z) / 2 * iz)
                sin_molecule_1 = G1_Uz_Squared_modulated * exp_dk_factor
                sin_molecule_2 = g1_U0_Squared_modulated / exp_dk_factor
                sin_molecule = sin_molecule_1 - sin_molecule_2
                sin_denominator = 2*1j*G1_U_half_z_Squared_modulated

                sin_real = np.real(sin_molecule / sin_denominator)
                # sin_im = np.imag(sin_molecule / sin_denominator)
                # sin_amp = np.abs(sin_molecule / sin_denominator)
                print(np.max(np.abs(sin_real)))
                # print(np.max(np.abs(sin_im)))
                # print(np.max(sin_amp))

                # print(np.sin(inside_sinc))
                correction_factor = (sin_molecule / sin_denominator) / np.sin(inside_sinc)
                # print(correction_factor)

            if Cal_version == 2:
                sinc_denominator = (kiizQ + kii2z) / 2
                modulation_modified = ifft2(fft2(modulation) * np.sinc(inside_sinc / np.pi) / sinc_denominator)
                G1_U_half_z_Squared_modulated = fft2(modulation_modified * Uz_AST(U1_0, k1, iz / 2) ** 2)
                molecule = G1_U_half_z_Squared_modulated \
                           * math.e ** (Gz * iz * 1j) \
                           * math.e ** (k2_z * iz / 2 * 1j) \
                           * (1j * iz)

                denominator = 1
            elif Cal_version == 3:
                modulation_modified = ifft2(fft2(modulation) * np.sinc(inside_sinc / np.pi))
                G1_U_half_z_Squared_modulated = fft2(modulation_modified * Uz_AST(U1_0, k1, iz / 2) ** 2)
                molecule = G1_U_half_z_Squared_modulated \
                           * math.e ** (Gz * iz * 1j) \
                           * math.e ** (k2_z * iz / 2 * 1j) \
                           * (1j * iz)
                denominator = k2_z
            elif Cal_version >= 4:
                sinc_denominator = (kiizQ + k2_z) / 2
                molecule = G1_U_half_z_Squared_modulated \
                           * np.sinc(inside_sinc / np.pi) / sinc_denominator \
                           * math.e ** (Gz * iz * 1j) \
                           * math.e ** (k2_z * iz / 2 * 1j) \
                           * (1j * iz)
                denominator = 1
            if Cal_version == 6:
                molecule *= correction_factor
        G2_z = const * molecule / denominator

    elif Big_version == 4:
        K1_z, K1_xy = find_Kxyz(fft2(U1_0), k1)
        kiizQ = 2 * K1_z + Gz
        dkiizQ = kiizQ - k2_z
        denominator = (kiizQ + k2_z) / 2
        e_index = -(k1_z**2 + 2*k1_z*Gz) * iz**2 / 24
        # print(np.max(np.abs(e_index)))
        G1_U_half_z_Squared_modulated = fft2(modulation * ifft2(
            fft2(Uz_AST(U1_0, k1, iz/2)) * math.e ** e_index) ** 2)
        # print(np.max(np.abs(-(K1_z**2 - 2*K1_z*k2_z) * iz**2 / 12)))
        # G2_z = const * G1_U_half_z_Squared_modulated \
        #        * math.e ** (-(k2_z-Gz)**2 * iz**2 / 24) / denominator \
        #        * math.e ** (Gz * iz * 1j) \
        #        * math.e ** (k2_z * iz / 2 * 1j) \
        #        * math.e ** (-(K1_z**2 - 2*K1_z*k2_z) * iz**2 / 12) \
        #        * (1j * iz)
        A = -(k2_z - Gz) ** 2 * iz ** 2 / 24
        B = Gz * iz * 1j
        C = k2_z * iz / 2 * 1j
        D = -(K1_z ** 2 - 2 * K1_z * k2_z) * iz ** 2 / 12
        E = A+B+C+D
        # print(np.max(np.abs(e_index)) * 2, np.max(np.abs(E)))
        # print(np.max(np.abs(e_index)) * 2, np.max(np.abs(E)), np.max(np.abs(A)), np.max(np.abs(D)))
        # print(np.max(np.real(e_index)) * 2, np.max(np.real(E)), np.max(np.real(A)), np.max(np.abs(D)))
        # print(np.max(np.abs(dkiizQ))**2 * iz**2)
        G2_z = const * G1_U_half_z_Squared_modulated \
               * math.e ** E / denominator \
               * (1j * iz)

    elif Big_version == 5:
        G1_Uz_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, iz) ** 2)
        g1_U0_Squared_modulated = fft2(modulation * U1_0 ** 2)
        molecule = G1_Uz_Squared_modulated * math.e ** (Gz * iz * 1j)\
                   - g1_U0_Squared_modulated * math.e ** (k2_z * iz * 1j)
        K1_z, K1_xy = find_Kxyz(fft2(U1_0), k1)
        kiizQ = 2 * K1_z + Gz
        denominator = kiizQ ** 2 - k2_z ** 2
        dismatch = const * molecule / denominator


        G1_U_half_z_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, iz / 2) ** 2)
        dkiizQ = kiizQ - k2_z
        inside_sinc = dkiizQ / 2 * iz
        sinc_denominator = (kiizQ + k2_z) / 2
        match = const * G1_U_half_z_Squared_modulated \
               * np.sinc(inside_sinc / np.pi) / sinc_denominator \
               * math.e ** (Gz * iz * 1j) \
               * math.e ** (k2_z * iz / 2 * 1j) \
               * (1j * iz)

        match_factor = math.e ** (- inside_sinc ** 2 / 6)
        dismatch_factor = 1 - match_factor

        G2_z = dismatch_factor * dismatch + match_factor * match

    return G2_z * Get("size_PerPixel")**2


def G2_z_NLAST(k1, k2, Gx, Gy, Gz,
               U1_0, iz, const,
               is_linear_convolution, ):
    Ix, Iy = U1_0.shape[0], U1_0.shape[1]
    k2_z, mesh_k2_x_k2_y = Cal_kz(Ix, Iy, k2)

    Big_version = 3
    Cal_version = 3
    Res_version = 1

    if Big_version != 3:

        G_U1_z_Squared = fft2(Uz_AST(U1_0, k1, iz) ** 2)
        g_U1_0_Squared = fft2(U1_0 ** 2)

        roll_x, roll_y = Cal_roll_xy(Gx, Gy,
                                     Ix, Iy, )

        G_U1_z_Squared_Q = Roll_xy(G_U1_z_Squared,
                                      roll_x, roll_y,
                                      is_linear_convolution, )
        g_U1_0_Squared_Q = Roll_xy(g_U1_0_Squared,
                                     roll_x, roll_y,
                                     is_linear_convolution, )

        if Res_version == 1:
            molecule = G_U1_z_Squared_Q * math.e ** (Gz * iz * 1j) \
                       - g_U1_0_Squared_Q * math.e ** (k2_z * iz * 1j)

        elif Res_version == 2:
            molecule = G_U1_z_Squared_Q * math.e ** (Gz * iz * 1j)

        elif Res_version == 3:
            molecule = - g_U1_0_Squared_Q * math.e ** (k2_z * iz * 1j)

    K1_z, K1_xy = find_Kxyz(fft2(U1_0), k1)
    kiizQ = 2 * K1_z + Gz
    kii2z = (k2 ** 2 - (2 * K1_xy[0] + mesh_k2_x_k2_y[:, :, 0]) ** 2 - (
            2 * K1_xy[1] + mesh_k2_x_k2_y[:, :, 1]) ** 2 + 0j) ** 0.5
    if Big_version == 0:

        # %% denominator: dk_Squared

        # n2_x_n2_y 的 mesh 才用 Gy / (2 * math.pi) * I2_y)，这里是 k2_x_k2_y 的 mesh，所以用 Gy 才对应
        k1izQ = (k1 ** 2 - (mesh_k2_x_k2_y[:, :, 0] - Gy) ** 2 - (
                mesh_k2_x_k2_y[:, :, 1] - Gx) ** 2 + 0j) ** 0.5

        kizQ = k1 + k1izQ + Gz
        # kizQ = k1 + k2_z + Gz
        denominator = kizQ ** 2 - k2_z ** 2

        kizQ = k1 + (k1 ** 2 - Gx ** 2 - Gy ** 2) ** 0.5 + Gz
        denominator = kizQ ** 2 - k2 ** 2

    elif Big_version == 1:
        denominator = kiizQ ** 2 - k2_z ** 2

    elif Big_version == 2:
        denominator = kiizQ ** 2 - kii2z ** 2

    elif Big_version == 3:
        G_U1_z_Squared = fft2(Uz_AST(U1_0, k1, iz/2) ** 2)

        G_U1_z_Squared_Q = Roll_xy(G_U1_z_Squared,
                                   roll_x, roll_y,
                                   is_linear_convolution, )
        if Cal_version == 1:
            molecule = G_U1_z_Squared_Q \
                       * math.e ** (Gz * iz * 1j) \
                       * math.e ** (k2_z * iz/2 * 1j) * (1j * iz)

            denominator = k2_z
        elif Cal_version >= 2:
            if Cal_version < 4:
                dkiizQ = kiizQ - kii2z
            else:
                dkiizQ = kiizQ - k2_z
            inside_sinc = dkiizQ / 2 * iz
            if Cal_version == 2:
                modulation_denominator = (kiizQ + kii2z) / 2

                molecule = G_U1_z_Squared_Q \
                           * np.sinc(inside_sinc / np.pi) / modulation_denominator \
                           * math.e ** (Gz * iz * 1j) \
                           * math.e ** (k2_z * iz / 2 * 1j) \
                           * (1j * iz)

                denominator = 1
            elif Cal_version == 3:
                molecule = G_U1_z_Squared_Q \
                           * np.sinc(inside_sinc / np.pi) \
                           * math.e ** (Gz * iz * 1j) \
                           * math.e ** (k2_z * iz / 2 * 1j) \
                           * (1j * iz)

                denominator = k2_z
            elif Cal_version >= 4:
                sinc_denominator = (kiizQ + k2_z) / 2
                molecule = G_U1_z_Squared_Q \
                           * np.sinc(inside_sinc / np.pi) / sinc_denominator \
                           * math.e ** (Gz * iz * 1j) \
                           * math.e ** (k2_z * iz / 2 * 1j) \
                           * (1j * iz)
                denominator = 1

    # %%
    G2_z = 2 * const * molecule / denominator

    return G2_z * Get("size_PerPixel")**2


# %%

def G2_z_NLAST_false(k1, k2, Gx, Gy, Gz,
                     U1_0, iz, const,
                     is_linear_convolution, ):
    Ix, Iy = U1_0.shape[0], U1_0.shape[1]
    k1_z, mesh_k1_x_k1_y = Cal_kz(Ix, Iy, k1)
    k2_z, mesh_k2_x_k2_y = Cal_kz(Ix, Iy, k2)

    G_U1_z_Squared = fft2(Uz_AST(U1_0, k1, iz) ** 2)
    g_U1_0_Squared = fft2(U1_0 ** 2)

    dG_Squared = G_U1_z_Squared \
                       - g_U1_0_Squared * math.e ** (k2_z * iz * 1j)

    # %% denominator: dk_Squared

    kiizQ = k1 + k1_z + Gz

    dk_Squared = kiizQ ** 2 - k2_z ** 2

    # %% fractional

    fractional = dG_Squared / dk_Squared

    roll_x, roll_y = Cal_roll_xy(Gx, Gy,
                                 Ix, Iy, )

    fractional_Q = Roll_xy(fractional,
                           roll_x, roll_y,
                           is_linear_convolution, )

    # %% G2_z0

    G2_z = 2 * const * fractional_Q * math.e ** (Gz * iz * 1j)

    return G2_z * Get("size_PerPixel")**2


# %%
# 提供 查找 边缘的，参数的 提示 or 帮助信息 msg

def Info_find_contours_SHG(g1, k1_z, k2_z, Tz, mz,
                           z0, size_PerPixel, deff_structure_length_expect,
                           is_print=1, is_contours=1, n_TzQ=1,
                           Gz_max_Enhance=1, match_mode=1, **kwargs):
    # %%
    # 描边
    key = "Info_find_contours_SHG"
    is_first = int(init_accu(key, 1) == 1)  # 若第一次调用 Info_find_contours_SHG，则 is_first 为 1，否则为 0
    is_Print = is_print * is_first  # 两个 得都 非零，才 print

    is_contours != -1 and is_Print and print(tree_print(kwargs.get("is_end", 0), kwargs.get("add_level", 0)) + "info_描边")
    kwargs["is_end"], kwargs["add_level"] = 0, 0  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    if is_contours != -1 and is_contours != 0: # 等于 0 或 -1 则让该 子程序 完全不行使 contours 功能，甚至不提示...
        # 但 0 但会 约束 deff_structure_length_expect， -1 则彻底 啥也不干

        dk = 2 * np.max(np.abs(k1_z)) - np.max(np.abs(k2_z))
        # print(k2_z[0,0])
        is_Print and print(tree_print() + "dk = {} / μm, {}".format(dk / size_PerPixel / 1000, dk))
        lc = math.pi / abs(dk) * size_PerPixel * 1000  # Unit: um
        # print("相干长度 = {} μm".format(lc))
        # print("Tz_max = {} μm <= 畴宽 = {} μm ".format(lc*2, Tz))
        # print("畴宽_max = 相干长度 = {} μm <= 畴宽 = {} μm ".format(lc, Tz/2))
        if (type(Tz) != float and type(Tz) != np.float64 
            and type(Tz) != int) or Tz <= 0:  # 如果 传进来的 Tz 既不是 float 也不是 int，或者 Tz <= 0，则给它 安排上 2*lc
            Tz = 2 * lc  # Unit: um

        Gz = 2 * math.pi * mz * size_PerPixel / (Tz / 1000)  # Tz / 1000 即以 mm 为单位

        dkQ = dk + Gz
        lcQ = math.pi / abs(dkQ) * size_PerPixel  # Unit: mm
        # print("相干长度_Q = {} mm".format(lcQ))
        TzQ = 2 * lcQ

        # %%

        # print("k2_z_min = {} / μm, k1_z_min = {} / μm".format(np.min(np.abs(k2_z))/size_PerPixel/1000, np.min(np.abs(k1_z))/size_PerPixel/1000))
        # print(np.abs(k2_z))
        if match_mode == 1:
            ix, iy, scale, energy_fraction = Find_energy_Dropto_fraction(g1, 2 / 3, 0.1)
            Gz_max = np.abs(k2_z[ix, 0]) - 2 * np.abs(k1_z[ix, 0])
            is_Print and print(tree_print() + "scale = {}, energy_fraction = {}".format(scale, energy_fraction))
        else:
            Gz_max = np.min(np.abs(k2_z)) - 2 * np.min(np.abs(k1_z))

        Gz_max = Gz_max * Gz_max_Enhance
        is_Print and print(tree_print() + "Gz_max = {} / μm, {}".format(Gz_max / size_PerPixel / 1000, Gz_max))
        Tz_min = 2 * math.pi * mz * size_PerPixel / (
                    abs(Gz_max) / 1000)  # 以使 lcQ >= lcQ_exp = (wc**2 + z0**2)**0.5 - z0
        # print("Tz_min = {} μm".format(Tz_min))

        dkQ_max = dk + Gz_max
        lcQ_min = math.pi / abs(dkQ_max) * size_PerPixel
        # print("lcQ_min = {} mm".format(lcQ_min))
        TzQ_min = 2 * lcQ_min

        z0_min = TzQ_min

        # %%
        if is_contours != 2:
            is_Print and print(tree_print(add_level=1) + "info_描边 1：若无额外要求")  # 波长定，Tz 定 (lcQ 定)，z0 不定

            is_Print and print(tree_print() + "z0_exp = {} mm".format(z0_min * n_TzQ))
            is_Print and print(tree_print(-1) + "Tz_exp = {} μm".format(Tz_min))

        # %%

        if is_contours != 1:
            # 如果 is_contours 是 0，不不要 = 则不 print 描边信息，且不设置；但 deff_structure_length_expect = z0_recommend
            # 如果 is_contours 是 1，不要要 = 则不 print 描边信息，因为直接 就设置成 min 了；同时 deff_structure_length_expect = z0_recommend
            # 如果 is_contours 是 2，要要要 = 则要 print 描边信息，因为只是 设置成 exp；同时 deff_structure_length_expect = z0_recommend
            # 如果 is_contours 是 -1，不不不 = 则不 print 描边信息，且不设置；且不 deff_structure_length_expect = z0_recommend
            # 如果 is_contours 是 other，要不要 = 则要 print 描边信息，且不设置；但 deff_structure_length_expect = z0_recommend
            # 真值表中，还缺少 “不要不”，以及 “要要不”、“不不不”，才能使得 AAA + AAB + BBA+ BBB = 1 + 3 + 3 + 1 = 8
            # “不要不” 不存在，因为第 2 个设置之后，意味着要 进行描边，所以 第 3 个必须 = 要；同理 “要要不” 也不存在
            # “要不不” 也不存在，因为 提供信息 但啥也不干，有什么用？提供信息就是为了做事，提供了就要做。所以 若第 1 个为 要，则第 2 个也得为 要。
            # “不不不” 也就是 意味着 这个函数 没用。。。那拿你来干啥，提供点信息，总是好的吧。。。算了，也安排上 “不不不选项”
            is_Print and print(tree_print(add_level=1) + "info_描边 2：若希望 mod( 现 z0, TzQ_exp ) = 0")  # 波长定，z0 定，Tz 不定 (lcQ 不定)

            is_Print and print(tree_print() + "lcQ_min = {} mm".format(lcQ_min))
            TzQ_exp = z0 / (z0 // TzQ_min)  # 满足 Tz_min <= · <= Tz_max = 原 Tz， 且 能使 z0 整除 TzQ 中，最小（最接近 TzQ_min）的 TzQ
            lcQ_exp = TzQ_exp / 2
            is_Print and print(tree_print() + "lcQ_exp = {} mm".format(lcQ_exp))
            is_Print and print(tree_print() + "lcQ     = {} mm".format(lcQ))
            is_Print and print(tree_print() + "lc = {} μm".format(lc))
            # print("TzQ_min = {} mm".format(TzQ_min))
            # print("TzQ_exp = {} mm".format(TzQ_exp))
            # print("TzQ     = {} mm".format(TzQ))

            is_Print and print(tree_print() + "z0_min = {} mm # ==> 1.先调 z0 >= z0_min".format(
                z0_min))  # 先使 TzQ_exp 不遇分母 为零的错误，以 正确预测 lcQ_exp，以及后续的 Tz_exp
            z0_exp = TzQ_exp  # 满足 >= TzQ_min， 且 能整除 TzQ_exp 中，最小的 z0
            # z0_exp = TzQ # 满足 >= TzQ_min， 且 能整除 TzQ_exp 中，最小的 z0
            is_Print and print(tree_print() + "z0_exp = {} mm # ==> 2.再调 z0 = z0_exp".format(z0_exp * n_TzQ))
            # print("z0_exp = {} * n mm # ==> 3.最后调 z0 = z0_exp".format(z0_exp))
            is_Print and print(tree_print() + "z0     = {} mm".format(z0))

            dkQ_exp = math.pi / lcQ_exp * size_PerPixel
            Gz_exp = dkQ_exp - dk
            Tz_exp = 2 * math.pi * mz * size_PerPixel / (
                        abs(Gz_exp) / 1000)  # 以使 lcQ >= lcQ_exp = (wc**2 + z0**2)**0.5 - z0
            is_Print and print(tree_print() + "Tz_min = {} μm".format(Tz_min))
            is_Print and print(tree_print() + "Tz_exp = {} μm # ==> 2.同时 Tz = Tz_exp".format(Tz_exp))
            is_Print and print(tree_print() + "Tz     = {} μm".format(Tz))
            is_Print and print(tree_print() + "Tz_max = {} μm".format(lc * 2))

            domain_min = Tz_min / 2
            is_Print and print(tree_print() + "畴宽_min = {} μm".format(domain_min))
            domain_exp = Tz_exp / 2
            is_Print and print(tree_print() + "畴宽_exp = {} μm".format(domain_exp))
            is_Print and print(tree_print() + "畴宽     = {} μm".format(Tz / 2))
            is_Print and print(tree_print(-1) + "畴宽_max = {} μm".format(lc))

        # %%

    if is_contours == 1:
        z0_recommend = z0_min * n_TzQ
        Tz_recommend = Tz_min if Tz_min != 0 else Tz
    elif is_contours == 2:
        z0_recommend = z0_exp * n_TzQ
        Tz_recommend = Tz_exp if Tz_exp != 0 else Tz
    else:
        z0_recommend = z0
        Tz_recommend = Tz

    if is_contours != -1: # 等于 -1 则 不额外覆盖 deff_structure_sheet_expect 的值
        # if deff_structure_length_expect <= z0_recommend + deff_structure_sheet_expect / 1000:
        #     deff_structure_length_expect = z0_recommend + deff_structure_sheet_expect / 1000
        #     is_Print and print("deff_structure_length_expect = {} mm".format(deff_structure_length_expect))
        if deff_structure_length_expect < z0_recommend:
            deff_structure_length_expect = z0_recommend
    is_Print and print(tree_print(1) + "deff_structure_length_expect = {} mm".format(deff_structure_length_expect))
    # 无论 deff_structure_sheet_expect 的值 被覆盖 与否，都需要 print，为的是 加个 is_end=1 在这。

    return z0_recommend, Tz_recommend, deff_structure_length_expect


# %%
# 提供 查找 边缘的，参数的 提示 or 帮助信息 msg
# 注：旧版本，已经过时，当时并 未想清楚。

def Info_find_contours(dk, Tz, mz,
                       U_NonZero_size, w0, z0, size_PerPixel,
                       is_print=1):
    # %%
    # 描边
    if is_print == 1:  # 由于这个 函数不 return，只提供信息；因此 如果不 print，相当于什么都没做

        lc = math.pi / abs(dk) * size_PerPixel * 1000  # Unit: um
        # print("相干长度 = {} μm".format(lc))
        # print("Tz_max = {} μm <= 畴宽 = {} μm ".format(lc*2, Tz))
        # print("畴宽_max = 相干长度 = {} μm <= 畴宽 = {} μm ".format(lc, Tz/2))
        if (type(Tz) != float and type(Tz) != int) or Tz <= 0:  # 如果 传进来的 Tz 既不是 float 也不是 int，或者 Tz <= 0，则给它 安排上 2*lc
            Tz = 2 * lc  # Unit: um

        Gz = 2 * math.pi * mz * size_PerPixel / (Tz / 1000)  # Tz / 1000 即以 mm 为单位

        dkQ = dk + Gz
        lcQ = math.pi / abs(dkQ) * size_PerPixel  # Unit: mm
        # print("相干长度_Q = {} mm".format(lcQ))
        TzQ = 2 * lcQ

        if (type(w0) == float or type(w0) == int) and w0 > 0:  # 如果引入了 高斯限制
            wc = w0
        else:
            wc = U_NonZero_size / 2

        # %%

        print("        =·=·=·=·=·=·=·=·=·= 描边必需 1 =·=·=·=·=·=·=·=·=·=")  # 波长定，z0 定，Tz 不定 (lcQ 不定)

        lcQ_min = (wc ** 2 + z0 ** 2) ** 0.5 - z0
        print("相干长度_Q_min = {} mm".format(lcQ_min))
        TzQ_min = 2 * lcQ_min
        TzQ_exp = z0 / (z0 // TzQ_min)  # 满足 Tz_min <= · <= Tz_max = 原 Tz， 且 能使 z0 整除 TzQ 中，最小（最接近 TzQ_min）的 TzQ
        lcQ_exp = TzQ_exp / 2
        print("相干长度_Q_exp = {} mm".format(lcQ_exp))
        print("相干长度_Q     = {} mm".format(lcQ))

        dkQ_max_abs = math.pi / lcQ_min * size_PerPixel
        Gz_max = dkQ_max_abs - dk
        Tz_min = 2 * math.pi * mz * size_PerPixel / (
                    abs(Gz_max) / 1000)  # 以使 lcQ >= lcQ_exp = (wc**2 + z0**2)**0.5 - z0
        print("Tz_min = {} μm".format(Tz_min))

        dkQ_exp_abs = math.pi / lcQ_exp * size_PerPixel
        Gz_exp = dkQ_exp_abs - dk
        Tz_exp = 2 * math.pi * mz * size_PerPixel / (
                    abs(Gz_exp) / 1000)  # 以使 lcQ >= lcQ_exp = (wc**2 + z0**2)**0.5 - z0
        print("Tz_exp = {} μm # ==> 3.最后调 Tz = Tz_exp ".format(Tz_exp))
        print("Tz     = {} μm # ==> 1.先调 Tz < Tz_max".format(Tz))
        print("Tz_max = {} μm".format(lc * 2))

        domain_min = Tz_min / 2
        print("畴宽_min = {} μm".format(domain_min))
        domain_exp = Tz_exp / 2
        print("畴宽_exp = {} μm".format(domain_exp))
        print("畴宽     = {} μm".format(Tz / 2))
        print("畴宽_max = {} μm".format(lc))
        print("相干 长度 = {} μm".format(lc))

        # %%

        print("        =·=·=·=·=·=·=·=·=·= 描边必需 2 =·=·=·=·=·=·=·=·=·=")  # 波长定，Tz 定 (lcQ 定)，z0 不定

        z0_min = (wc ** 2 - lcQ ** 2) / (2 * lcQ)  # 以使 (wc**2 + z0**2)**0.5 - z0 = lcQ_exp <= lcQ
        # 这个玩意其实还得保证 >= TzQ_min，以先使 TzQ_exp 不遇分母 为零的错误，以 正确预测 lcQ_exp，以及后续的 Tz_exp
        print("z0_min = {} mm".format(z0_min))
        z0_exp = z0_min - np.mod(z0_min, TzQ) + TzQ  # 满足  >= TzQ_min， 且 能整除 TzQ_exp 中，最小的 z0
        print("z0_exp = {} mm".format(z0_exp))
        print("z0     = {} mm # ==> 2.接着调 z0 = z0_exp".format(z0))

        print("        =·=·=·=·=·=·=·=·=·= 描边 end =·=·=·=·=·=·=·=·=·=")
