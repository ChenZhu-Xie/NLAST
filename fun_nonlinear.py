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

def Cal_lc_SHG(k1_inc, k3_inc, Tz, size_PerPixel,
               is_print=1, **kwargs):
    k2_inc = kwargs.get("k2_inc", k1_inc)

    dk = k1_inc + k2_inc - k3_inc  # Unit: 无量纲
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

    from fun_global_var import init_Set
    init_Set("Gx", Gx)  # structure_nonrect_chi2_Generate_2D 会用到
    init_Set("Gy", Gy)

    return Gx, Gy, Gz


# %%

def args_SFG(k1_inc, k3_inc, size_PerPixel,
             mx, my, mz,
             Tx, Ty, Tz,
             is_print, **kwargs):
    info = "args_SFG"
    is_first = int(init_accu(info, 1) == 1)  # 若第一次调用 args_SFG，则 is_first 为 1，否则为 0
    is_Print = is_print * is_first  # 两个 得都 非零，才 print

    info = "参数_SHG"
    is_Print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    dk, lc, Tz = Cal_lc_SHG(k1_inc, k3_inc, Tz, size_PerPixel,
                            is_Print, **kwargs)

    Gx, Gy, Gz = Cal_GxGyGz(mx, my, mz,
                            Tx, Ty, Tz, size_PerPixel,
                            is_Print, is_end=1)
    return dk, lc, Tz, \
           Gx, Gy, Gz


# %%

def Cal_dk_zQ_SFG(k2, k1_z, k3_z,
                  mesh_k1_x_k1_y, mesh_k3_x_k3_y,
                  n3_x, n3_y,
                  Gx, Gy, Gz, ):
    # n3_x_n3_y 的 mesh 才用 Gy / (2 * math.pi) * I2_y)，这里是 k3_x_k3_y 的 mesh，所以用 Gy 才对应
    # dk_x = mesh_k3_x_k3_y[n3_x, n3_y, 0] - mesh_k1_x_k1_y[:, :, 0] - Gy  #  old
    dk_x = mesh_k3_x_k3_y[n3_x, n3_y, 0] - mesh_k1_x_k1_y[:, :, 0] - Gx  # 应该减 Gx 而非 Gy：因为 mesh 本身是 对的，只有 Iy 单独出现 才不对。
    # 其实 mesh_k3_x_k3_y[:, :, 0]、mesh_n3_x_n3_y[:, :, 0]、mesh_n3_x_n3_y[:, :, 0]、 n3_x 均只和 y，即 [:, :] 中的 第 2 个数字 有关，
    # 只由 列 y、ky 决定，与行 即 x、kx 无关
    # 而 Gy 得与 列 y、ky 发生关系,
    # 所以是 - Gy 而不是 Gx
    # 并且这里的 dk_x 应写为 dk_y
    # dk_y = mesh_k3_x_k3_y[n3_x, n3_y, 1] - mesh_k1_x_k1_y[:, :, 1] - Gx  #  old
    dk_y = mesh_k3_x_k3_y[n3_x, n3_y, 1] - mesh_k1_x_k1_y[:, :, 1] - Gy  # 应该减 Gy 而非 Gx：因为 mesh 本身是 对的，只有 Ix 单独出现 才不对。
    # print(dk_y[0], dk_y[1])  # 每行都是一样的，说明 dk_y 确实变的 只有 列，而 列 == y
    # 所以 mesh 生成的 确实是 标准的 笛卡尔坐标系，所以该 Gy 就 Gy
    k2_z_dk_x_dk_y = (k2 ** 2 - dk_x ** 2 - dk_y ** 2 + 0j) ** 0.5

    dk_z = k1_z + k2_z_dk_x_dk_y - k3_z[n3_x, n3_y]
    dk_zQ = dk_z + Gz

    return dk_zQ


# %%

def Cal_roll_xy(Gx, Gy,
                Ix, Iy,
                *args):
    if len(args) >= 2:
        nx, ny = args[0], args[1]
        # roll_x = np.floor(Ix // 2 - (Ix - 1) + nx - Gy / (2 * math.pi) * Iy).astype(np.int64)  # y 向（行）移动
        # # roll_y = np.floor(Iy // 2 - (Iy - 1) + ny - Gx / (2 * math.pi) * Ix).astype(np.int64)  # x 向（列）移动
        roll_x = np.floor(Ix // 2 - (Ix - 1) + nx - Gy / (2 * math.pi) * Ix).astype(np.int64)  # y 向（行）移动
        roll_y = np.floor(Iy // 2 - (Iy - 1) + ny - Gx / (2 * math.pi) * Iy).astype(np.int64)  # x 向（列）移动
        # 之后要平移列，而 Gx 才与列有关...（同时，Iy 才是 列数，所以 Gx 总与 Iy 搭配）
    else:
        # roll_x = np.floor(Gy / (2 * math.pi) * Iy).astype(np.int64)
        # roll_y = np.floor(Gx / (2 * math.pi) * Ix).astype(np.int64)
        roll_x = np.floor(Gy / (2 * math.pi) * Ix).astype(np.int64)
        roll_y = np.floor(Gx / (2 * math.pi) * Iy).astype(np.int64)  # Gx 才与列有关...同时，Iy 才是 列数

    return roll_x, roll_y


def G3_z_modulation_3D_NLAST(k1, k2, k3,
                             modulation, U1_0, U2_0, iz, const,
                             Tz=10, is_customized=0, ):
    k1_z, mesh_k1_x_k1_y = Cal_kz(U1_0.shape[0], U1_0.shape[1], k1)
    k2_z, mesh_k2_x_k2_y = Cal_kz(U2_0.shape[0], U2_0.shape[1], k2)
    k3_z, mesh_k3_x_k3_y = Cal_kz(U1_0.shape[0], U1_0.shape[1], k3)

    Big_version = 1 if is_customized == 1 else 2
    Cal_version = 1 if is_customized == 1 else 1

    # 1 对应 1.1 - 不匹配
    # 2 对应 第一性原理：非线性卷积
    # 3.1 对应 3.4 - 匹配
    # 3.2 对应 bulk 且 匹配

    # print(Tz)
    Tz_unit = (Tz / 1000) / Get("size_PerPixel")
    dz = Tz_unit / 2
    # print(dz, iz)
    J = int(iz / dz - 1)
    # J = iz / dz - 1
    # J = iz / dz

    if Big_version == 0:
        kiiz = k1 + k2_z
        dkiiz = kiiz - k3

        # print(J, (-1)**J)

        if Cal_version == 1:
            # %% version 1（更自洽）

            G_U1_z_Squared_modulated_1 = fft2(
                ifft2(fft2(modulation) / (kiiz ** 2 - k3 ** 2) / (1 + math.e ** (- dkiiz * dz * 1j))) \
                * Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2, iz))

            G_U1_z_Squared_modulated_2 = fft2(
                ifft2(fft2(modulation) / (kiiz ** 2 - k3 ** 2) / (1 + math.e ** (kiiz * dz * 1j))) \
                * Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2, iz))

            g_U1_0_Squared_modulated = fft2(
                ifft2(fft2(modulation) / (kiiz ** 2 - k3 ** 2)
                      * (1 / (1 + math.e ** (- dkiiz * dz * 1j)) - 1 / (1 + math.e ** (kiiz * dz * 1j)))) \
                * U1_0 ** 2)

            G3_z = const * (G_U1_z_Squared_modulated_1 * (-1) ** J
                            - G_U1_z_Squared_modulated_2 * (-1) ** J * math.e ** (k3_z * iz * 1j)
                            + g_U1_0_Squared_modulated * math.e ** (k3_z * iz * 1j))

        elif Cal_version == 2:
            # %% version 2（少近似）

            G_U1_z_Squared_modulated_1 = fft2(
                ifft2(fft2(modulation) / (kiiz ** 2 - k3 ** 2) / (1 + math.e ** (- dkiiz * dz * 1j))) \
                * Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2, iz))

            G_U1_z_Squared_modulated_2 = fft2(
                ifft2(fft2(modulation) / (kiiz ** 2 - k3 ** 2)) \
                * Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2, iz) / (1 + math.e ** (kiiz * dz * 1j)))

            g_U1_0_Squared_modulated_1 = fft2(
                ifft2(fft2(modulation) / (kiiz ** 2 - k3 ** 2) / (1 + math.e ** (- dkiiz * dz * 1j))) \
                * U1_0 ** 2)

            g_U1_0_Squared_modulated_2 = fft2(
                ifft2(fft2(modulation) / (kiiz ** 2 - k3 ** 2)) \
                * U1_0 ** 2 / (1 + math.e ** (kiiz * dz * 1j)))

            G3_z = const * (G_U1_z_Squared_modulated_1 * (-1) ** J
                            - G_U1_z_Squared_modulated_2 * (-1) ** J * math.e ** (k3_z * iz * 1j)
                            + g_U1_0_Squared_modulated_1 * math.e ** (k3_z * iz * 1j)
                            - g_U1_0_Squared_modulated_2 * math.e ** (k3_z * iz * 1j))

    else:
        g1_U0_Squared_modulated = fft2(modulation * U1_0 ** 2)
        G1_Uz_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2, iz))

        K1_z, K1_xy = find_Kxyz(fft2(U1_0), k1)
        K2_z, K2_xy = find_Kxyz(fft2(U2_0), k2)
        kiiz = K1_z + K2_z
        dkiiz = kiiz - k3_z
        if Big_version < 3:
            denominator = kiiz ** 2 - k3_z ** 2
        else:
            denominator = kiiz + k3_z
            inside_sinc = dkiiz * dz / 2
            # print(np.max(np.abs(inside_sinc)))

        if Big_version == 1:
            factor = 1 / (1 + math.e ** (- 1j * dkiiz * dz)) - \
                     1 / (1 + math.e ** (1j * dkiiz * dz))
            G3_z = const / denominator * factor * \
                   (g1_U0_Squared_modulated * math.e ** (k3_z * iz * 1j) +
                    G1_Uz_Squared_modulated * (-1) ** J)
        elif Big_version == 2:
            factor_1 = 1 / (1 + math.e ** (- 1j * kiiz * dz)) - \
                       1 / (1 + math.e ** (1j * dkiiz * dz))
            factor_2 = 1 / (1 + math.e ** (- 1j * kiiz * dz)) * math.e ** (1j * k3_z * iz) - \
                       1 / (1 + math.e ** (1j * dkiiz * dz))
            G3_z = const / denominator * \
                   (factor_1 * g1_U0_Squared_modulated * math.e ** (k3_z * iz * 1j) +
                    factor_2 * G1_Uz_Squared_modulated * (-1) ** J)
        elif Big_version == 3:
            factor = np.sinc(inside_sinc / np.pi) * 1j * dz / \
                     np.cos(inside_sinc)
            # print(np.max(np.abs(np.cos(inside_sinc))))
            if Cal_version == 1:
                G3_z = const / denominator * factor * \
                       (g1_U0_Squared_modulated * math.e ** (k3_z * iz * 1j) +
                        G1_Uz_Squared_modulated * (-1) ** J)
            elif Cal_version == 2:
                G3_z = const / denominator * factor * \
                       (G1_Uz_Squared_modulated -
                        g1_U0_Squared_modulated * math.e ** (k3_z * iz * 1j))

    return G3_z * Get("size_PerPixel") ** 2


# %%
def gan_factor_out(cos_num_expect, omite_level=1e-4):  # 1e-4 以下 的 系数 忽略掉
    factor_out = []
    factor_out.append([1, ])  # 1 个 cos
    factor_out.append([0.32, 0.445, ])
    # factor_out.append([0.301003, 0.455351, ])
    factor_out.append([0.169451, 0.298362, 0.350258, ])
    factor_out.append([0.102129, 0.204757, 0.260585, 0.285946, ])
    factor_out.append([-6.76961E-05, 0.130024, 0.217244, 0.251919, 0.265981, ])
    factor_out.append([-4.96711E-06, 0.0866104, 0.160762, 0.197904, 0.216228, 0.224824, ])
    factor_out.append([-7.37E-07, 0.0648475, 0.124482, 0.157835, 0.176501, 0.186978, 0.192352, ])
    factor_out.append([-2.59835E-05, -2.61578E-05, 0.0764049, 0.135342, 0.160369, 0.172968, 0.179733, 0.183142, ])
    factor_out.append([1.27E-08, -9.97E-07, 0.0571655, 0.105849, 0.131789, 0.146336, 0.154869, 0.159855, 0.162484, ])
    factor_out.append(
        [-1.56855E-06, -1.18615E-05, 2.40601E-05, 0.0549615, 0.110708, 0.136124, 0.147892, 0.154058, 0.15737,
         0.159078, ])
    factor_out.append(
        [8.15E-09, -9.56E-09, -1.38E-06, 0.050591189, 0.092271169, 0.113490491, 0.125190731, 0.132148097, 0.136429679,
         0.139023225, 0.140423452, ])
    factor_out.append(
        [1.38E-08, 9.54E-07, -1.45E-06, 5.93199E-05, 0.0588856, 0.0990237, 0.116212, 0.124933, 0.129881, 0.132858,
         0.13463, 0.135579, ])
    factor_out.append(
        [-6.04E-11, 7.92E-09, -1.22E-07, -3.55E-06, 0.0470445, 0.0821403, 0.0993555, 0.109026, 0.114907, 0.118651,
         0.121065, 0.122574, 0.123405, ])
    factor_out.append(
        [-6.95E-12, 9.73E-11, 1.17E-08, 6.00E-08, 0.0329041, 0.0653294, 0.0847515, 0.096197, 0.103247, 0.107785,
         0.110783, 0.112767, 0.114028, 0.11473, ])
    factor_out.append(
        [6.24E-13, -1.94E-11, -2.24E-09, 2.09E-08, 0.0241148, 0.052351, 0.072173, 0.0846568, 0.0925488, 0.0977093,
         0.101186, 0.103563, 0.105176, 0.106219, 0.106805, ])
    factor_out.append(
        [-4.77E-13, 7.77E-13, 8.90E-10, -1.97E-08, -4.70E-08, 0.0300253, 0.0591841, 0.0762949, 0.0862538, 0.0923673,
         0.0963222, 0.098976, 0.100788, 0.102017, 0.102811, 0.103259, ])
    factor_out.append(
        [3.32E-12, -1.11E-10, 1.86E-09, -2.87E-09, 5.21E-08, -4.92E-06, 0.0360376, 0.0652576, 0.0797354, 0.0874927,
         0.0920891, 0.0950126, 0.0969603, 0.0982858, 0.0991826, 0.099762, 0.100088, ])
    factor_out.append(
        [1.40E-11, -9.51E-11, -2.50E-10, -5.87E-10, 5.84E-08, 5.07E-06, 9.81E-06, 0.0432267, 0.0713149, 0.0827259,
         0.0883321, 0.0915149, 0.0935178, 0.0948661, 0.0957744, 0.0963876, 0.0967835, 0.0970025, ])
    factor_out.append(
        [1.25E-12, 2.77E-12, 2.30E-11, 6.10E-11, 9.73E-09, -6.34E-08, -1.52E-07, 0.0338622, 0.0603613, 0.0729303,
         0.079547, 0.0834764, 0.0860011, 0.0877104, 0.0889032, 0.0897435, 0.0903267, 0.0907102, 0.0909284, ])
    factor_out.append(
        [2.41E-14, 1.14E-12, -1.57E-12, 4.24E-12, -4.04E-10, 2.91E-09, -8.55E-09, 0.0259968, 0.0502469, 0.0638248,
         0.0715418, 0.0762544, 0.0793267, 0.0814285, 0.0829129, 0.0839793, 0.0847462, 0.0852864, 0.0856454, 0.085851, ])
    aj = factor_out[cos_num_expect - 1]
    aj_left = [a for a in aj if a > omite_level]  # 忽略了 1e-4 以下 的 系数后，剩下的 系数们
    cos_num = len(aj_left)  # 实际 的 cos 的 个数； 理应 len(aj) == cos_num_expect
    nums_to_omite = cos_num_expect - cos_num  # 一共忽略了 多少个系数
    aj_left += [1 - sum(aj_left)]  # 在末尾 添加 1 个 a[-1] 常系数
    region = (cos_num_expect + 1) * math.pi if cos_num_expect > 1 else math.pi
    return aj_left, cos_num, nums_to_omite, region


def gan_factor_in(cos_num_expect, nums_to_omite):
    factor_in = []
    factor_in.append([1.732050808, ])  # 1 个 cos
    factor_in.append([1.17, 2.18, ])
    # factor_in.append([1.14855, 2.09636, ])
    factor_in.append([1.07449, 1.45342, 2.78214, ])
    factor_in.append([1.04223, 1.24701, 1.76697, 3.4388, ])
    factor_in.append([0.608268, 1.05724, 1.30614, 1.89195, 3.72379, ])
    factor_in.append([0.574694, 1.03632, 1.19432, 1.52489, 2.2339, 4.41463, ])
    factor_in.append([0.359817, 1.02674, 1.14081, 1.3628, 1.76762, 2.60773, 5.169, ])
    factor_in.append([0.199089, 0.740792, 1.0315, 1.16434, 1.40941, 1.84433, 2.73596, 5.43949, ])
    factor_in.append([0.124246, 0.317121, 1.02373, 1.1201, 1.29423, 1.57993, 2.07484, 3.08289, 6.13366, ])
    factor_in.append([0.0299174, 0.304177, 0.596033, 1.02169, 1.12006, 1.30273, 1.60011, 2.11118, 3.14693, 6.27256, ])
    factor_in.append(
        [0.028990341, 0.23743036, 0.348599912, 1.020983903, 1.104175219, 1.247413522, 1.466735366, 1.80886966,
         2.390222581, 3.564685372, 7.106325288, ])
    factor_in.append(
        [0.00170016, 0.173797, 0.278128, 0.872531, 1.02505, 1.11907, 1.27405, 1.50629, 1.86477, 2.47055, 3.6908,
         7.36489, ])
    factor_in.append(
        [9.99999E-05, 0.0995394, 0.224026, 0.342159, 1.01977, 1.09433, 1.21628, 1.39361, 1.65189, 2.04728, 2.71366,
         4.05479, 8.0918, ])
    factor_in.append(
        [4.99922E-05, 0.00169576, 0.132058, 0.496849, 1.0133, 1.06779, 1.16197, 1.29929, 1.4933, 1.77311, 2.19976,
         2.91757, 4.36107, 8.70472, ])
    factor_in.append(
        [5.00014E-05, 4.00E-04, 0.0811973, 0.576743, 1.00954, 1.05071, 1.12518, 1.2348, 1.38705, 1.59834, 1.90081,
         2.36049, 3.13266, 4.6843, 9.35174, ])
    factor_in.append(
        [0.000040001, 5.00E-05, 8.92E-03, 0.401678, 0.480197, 1.01214, 1.06131, 1.14455, 1.26262, 1.42357, 1.64479,
         1.95984, 2.4372, 3.23768, 4.84457, 9.67538, ])
    factor_in.append(
        [5.00E-06, 3.99999E-05, 0.000900016, 0.00322863, 0.303287, 0.499231, 1.01491, 1.07183, 1.16322, 1.28915, 1.4583,
         1.68897, 2.01602, 2.51028, 3.33781, 4.99745, 9.98423, ])
    factor_in.append(
        [5.00E-07, 4.00E-06, 9.00E-06, 3.00E-05, 0.00300311, 0.254407, 0.387579, 1.01839, 1.08405, 1.18381, 1.31769,
         1.49511, 1.73534, 2.07462, 2.58621, 3.44157, 5.15562, 10.3033, ])
    factor_in.append(
        [5.00E-07, 4.00E-06, 9.00E-06, 3.00E-05, 0.00299912, 0.0950811, 0.134361, 1.01403, 1.06677, 1.14938, 1.26026,
         1.40485, 1.59491, 1.85156, 2.21365, 2.75942, 3.67186, 5.50031, 10.9919, ])
    factor_in.append(
        [5.00E-07, 4.00E-06, 9.00E-06, 3.00E-05, 0.00299078, 0.0485485, 0.168009, 1.01053, 1.05222, 1.12023, 1.21264,
         1.33231, 1.48659, 1.6885, 1.96068, 2.34438, 2.92253, 3.889, 5.8256, 11.642, ])
    if cos_num_expect < 1:
        cos_num_expect = 1
    bj = factor_in[cos_num_expect - 1]
    bj_left = bj[nums_to_omite:]  # 理应 len(aj) == len(bj) == cos_num_expect
    # bj_left = [bj[j] for j in range(len(bj)) if j > nums_to_omite - 1]  # 忽略了 1e-4 以下 的 系数后，剩下的 系数们
    return bj_left


# %%

def G3_z_modulation_NLAST(k1, k2, k3,
                          modulation, U1_0, U2_0, iz, const,
                          Gz=1, is_customized=0,
                          **kwargs, ):
    # from fun_os import try_to_call_me
    # print(try_to_call_me())
    if kwargs.get("Tz", None) != None:
        return G3_z_modulation_3D_NLAST(k1, k2, k3,
                                        modulation, U1_0, U2_0, iz, const,
                                        is_customized=is_customized,
                                        **kwargs, )
    else:
        k1_z, mesh_k1_x_k1_y = Cal_kz(U1_0.shape[0], U1_0.shape[1], k1)
        k2_z, mesh_k2_x_k1_y = Cal_kz(U2_0.shape[0], U2_0.shape[1], k2)
        k3_z, mesh_k3_x_k3_y = Cal_kz(U1_0.shape[0], U1_0.shape[1], k3)

        Big_version = 5 if is_customized == 1 else 3
        Cal_version = 1 if is_customized == 1 else 4
        Res_version = 1 if is_customized == 1 else 1
        # print(str(Big_version) + '.' + str(Cal_version) + "\n")
        cos_num_expect = 20 if is_customized == 1 else 20

        # 3.4 > 3.2 > 1.3    match OK
        # 3.5 wrong
        # 3.6 = 1.1 > 2      dismatch OK
        # 4.0 e 指数太大，溢出
        # 4.2                match super + OK
        # 5.0 = 1.1 + 3.4    dismatch + match OK
        # 5.1 = 1.1 + 4.2    dismatch + match 分区

        K1_z, K1_xy = find_Kxyz(fft2(U1_0), k1)
        K2_z, K2_xy = find_Kxyz(fft2(U2_0), k2)
        kiizQ = K1_z + K2_z + Gz
        dkiizQ = kiizQ - k3_z
        denominator = kiizQ + k3_z
        inside_sinc = dkiizQ / 2 * iz

        def gan_G3_z_dismatch():  # 1.1
            denominator = kiizQ ** 2 - k3_z ** 2
            G1_Uz_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2, iz))
            g1_U0_Squared_modulated = fft2(modulation * U1_0 * U2_0)
            molecule = G1_Uz_Squared_modulated * math.e ** (Gz * iz * 1j) \
                       - g1_U0_Squared_modulated * math.e ** (k3_z * iz * 1j)
            dismatch = const * molecule / denominator

            return dismatch

        def gan_G3_z_sinc():  # 3.4
            sinc_denominator = (kiizQ + k3_z) / 2
            G1_U_half_z_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, iz / 2) * Uz_AST(U2_0, k2, iz / 2))
            sinc = const * G1_U_half_z_Squared_modulated \
                   * np.sinc(inside_sinc / np.pi) / sinc_denominator \
                   * math.e ** (Gz * iz * 1j) \
                   * math.e ** (k3_z * iz / 2 * 1j) \
                   * (1j * iz)

            return sinc

        def gan_G3_z_1():  # 4.2 +
            denominator = kiizQ + k3_z
            G1_U_half_z_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, iz / 2) * Uz_AST(U2_0, k2, iz / 2))
            cos_1 = const * G1_U_half_z_Squared_modulated \
                    * math.e ** (Gz * iz / 2 * 1j) \
                    * math.e ** (k3_z * iz / 2 * 1j) \
                    / denominator \
                    * (1j * iz)
            return cos_1

        def gan_G3_z_cos(factor):  # 4.2
            denominator = (kiizQ + k3_z) * 2
            z_plus = (1 + 1 / factor) / 2
            z_minus = (1 - 1 / factor) / 2
            G_U_z_plus_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, z_plus * iz) *
                                                Uz_AST(U2_0, k2, z_plus * iz)) * \
                                           math.e ** (Gz * z_plus * iz * 1j) * \
                                           math.e ** (k3_z * z_minus * iz * 1j)
            G_U_z_minus_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, z_minus * iz) *
                                                 Uz_AST(U2_0, k2, z_minus * iz)) * \
                                            math.e ** (Gz * z_minus * iz * 1j) * \
                                            math.e ** (k3_z * z_plus * iz * 1j)
            cos = const * (G_U_z_plus_Squared_modulated + G_U_z_minus_Squared_modulated) \
                  / denominator \
                  * (1j * iz)
            return cos

        def gan_G3_z_cos_seq(cos_num_expect):
            aj, cos_num, nums_to_omite, region = gan_factor_out(cos_num_expect, )
            bj = gan_factor_in(cos_num_expect, nums_to_omite, )
            # print(cos_num, nums_to_omite, region, "\n")
            # print(aj, "\n")
            # print(bj, "\n")

            if cos_num == 1:
                cos_seq = gan_G3_z_cos(bj[0])
            else:
                cos_seq = aj[-1] * gan_G3_z_1()
                for j in range(cos_num):
                    cos_seq += aj[j] * gan_G3_z_cos(bj[j])
            return cos_seq, region

        if Big_version == 0 or Big_version == 2:

            if Big_version == 0:
                kiizQ = k1 + k2_z + Gz

                if Cal_version == 1:
                    # %% == version 1（更自洽）
                    G1_Uz_Squared_modulated = fft2(
                        ifft2(fft2(modulation) / (kiizQ ** 2 - k3 ** 2)) * Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2, iz))
                    # print(np.min(np.abs((kiiz ** 2 - k3 ** 2))))
                    g1_U0_Squared_modulated = fft2(
                        ifft2(fft2(modulation) / (kiizQ ** 2 - k3 ** 2)) * U1_0 ** 2)

                elif Cal_version == 2:
                    # %% == version 1.1
                    G1_Uz_Squared_modulated = fft2(
                        ifft2(fft2(modulation) / (kiizQ ** 2 - k3_z ** 2)) * Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2,
                                                                                                           iz))
                    g1_U0_Squared_modulated = fft2(
                        ifft2(fft2(modulation) / (kiizQ ** 2 - k3_z ** 2)) * U1_0 ** 2)

                elif Cal_version == 3:
                    # %% == version 2（少近似）
                    G1_Uz_Squared_modulated = fft2(
                        modulation * ifft2(
                            fft2(Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2, iz)) / (kiizQ ** 2 - k3_z ** 2)))

                    g1_U0_Squared_modulated = fft2(
                        modulation * ifft2(fft2(U1_0 ** 2) / (kiizQ ** 2 - k3_z ** 2)))

                elif Cal_version == 4:
                    # %% == version 2.1
                    G1_Uz_Squared_modulated = fft2(
                        modulation * ifft2(fft2(Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2, iz)) / (kiizQ ** 2 - k3 ** 2)))

                    g1_U0_Squared_modulated = fft2(
                        modulation * ifft2(fft2(U1_0 ** 2) / (kiizQ ** 2 - k3 ** 2)))

            elif Big_version == 2:
                K1_z, K1_xy = find_Kxyz(fft2(U1_0), k1)
                K2_z, K2_xy = find_Kxyz(fft2(U2_0), k2)
                kiizQ = K1_z + K2_z + Gz
                kii2z = (k3 ** 2 - (K1_xy[0] + K2_xy[0] + mesh_k3_x_k3_y[:, :, 0]) ** 2 - (
                        K1_xy[1] + K2_xy[1] + mesh_k3_x_k3_y[:, :, 1]) ** 2 + 0j) ** 0.5
                denominator = kiizQ ** 2 - kii2z ** 2

                G1_Uz_Squared_modulated = fft2(
                    ifft2(fft2(modulation) / denominator) * Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2, iz))
                # print(np.min(np.abs((kiiz ** 2 - k3 ** 2))))
                g1_U0_Squared_modulated = fft2(
                    ifft2(fft2(modulation) / denominator) * U1_0 ** 2)

            if Res_version == 1:
                # 不加 负号，U 的相位 会差个 π，我也不知道 为什么
                G3_z = const * (G1_Uz_Squared_modulated * math.e ** (Gz * iz * 1j)
                                - g1_U0_Squared_modulated * math.e ** (k3_z * iz * 1j))
            elif Res_version == 2:
                G3_z = const * G1_Uz_Squared_modulated * math.e ** (Gz * iz * 1j)

            elif Res_version == 3:
                G3_z = - const * g1_U0_Squared_modulated * math.e ** (k3_z * iz * 1j)

        elif Big_version == 1:
            G1_Uz_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2, iz)) * \
                                      math.e ** (Gz * iz * 1j)
            g1_U0_Squared_modulated = fft2(modulation * U1_0 ** 2) * math.e ** (k3_z * iz * 1j)
            molecule = G1_Uz_Squared_modulated - g1_U0_Squared_modulated

            if Cal_version >= 3:
                dkiizQ = 1 / (np.sinc(inside_sinc / np.pi) * iz / 2)

            if Cal_version == 1:
                denominator = kiizQ ** 2 - k3_z ** 2
            elif Cal_version == 2:
                kiizQ = k1_z + k2_z + Gz
                denominator = kiizQ ** 2 - k3_z ** 2
            elif Cal_version == 3:
                denominator = (kiizQ + k3_z) * dkiizQ
            elif Cal_version == 4:
                denominator = (dkiizQ + 2 * k3_z) * dkiizQ
            elif Cal_version == 5:
                denominator = (dkiizQ + 2 * k3_z) * (kiizQ - k3_z)

            # print(np.max(np.abs(dkiizQ)), np.max(np.abs(kiizQ - k3_z)))
            # print(np.min(np.abs(dkiizQ)), np.min(np.abs(kiizQ - k3_z)))

            if Res_version == 1:
                G3_z = const * molecule / denominator
            elif Res_version == 2:
                G3_z = const * G1_Uz_Squared_modulated / denominator
            elif Res_version == 3:
                G3_z = const * g1_U0_Squared_modulated / denominator

        elif Big_version == 3:
            G1_U_half_z_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, iz / 2) * Uz_AST(U2_0, k2, iz / 2))
            if Cal_version == 1:
                molecule = G1_U_half_z_Squared_modulated \
                           * math.e ** (Gz * iz * 1j) \
                           * math.e ** (k3_z * iz / 2 * 1j) \
                           * (1j * iz)

                denominator = k3_z
            elif Cal_version >= 2:
                if Cal_version != 5:
                    if Cal_version < 4:
                        kii2z = (k3 ** 2 - (K1_xy[0] + K2_xy[0] + mesh_k3_x_k3_y[:, :, 0]) ** 2 - (
                                K1_xy[1] + K2_xy[1] + mesh_k3_x_k3_y[:, :, 1]) ** 2 + 0j) ** 0.5
                        dkiizQ = kiizQ - kii2z
                    else:
                        dkiizQ = kiizQ - k3_z
                    inside_sinc = dkiizQ / 2 * iz
                elif Cal_version == 5:
                    G1_Uz_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2, iz))
                    g1_U0_Squared_modulated = fft2(modulation * U1_0 ** 2)
                    exp_dk_factor = math.e ** (1j * (Gz - k3_z) / 2 * iz)
                    i_dkiizQ_z_1 = np.log(G1_Uz_Squared_modulated / G1_U_half_z_Squared_modulated * exp_dk_factor)
                    i_dkiizQ_z_2 = np.log(g1_U0_Squared_modulated / G1_U_half_z_Squared_modulated / exp_dk_factor)
                    i_dkiizQ_z = i_dkiizQ_z_1 - i_dkiizQ_z_2
                    dkiizQ = i_dkiizQ_z / (1j * iz)
                    # print(np.max(np.abs(dkiizQ)))
                    kiizQ = dkiizQ + k3_z
                    # print(np.max(np.abs(kiizQ)))
                    inside_sinc = dkiizQ / 2 * iz
                if Cal_version == 6:
                    G1_Uz_Squared_modulated = fft2(modulation * Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2, iz))
                    g1_U0_Squared_modulated = fft2(modulation * U1_0 ** 2)
                    exp_dk_factor = math.e ** (1j * (Gz - k3_z) / 2 * iz)
                    sin_molecule_1 = G1_Uz_Squared_modulated * exp_dk_factor
                    sin_molecule_2 = g1_U0_Squared_modulated / exp_dk_factor
                    sin_molecule = sin_molecule_1 - sin_molecule_2
                    sin_denominator = 2 * 1j * G1_U_half_z_Squared_modulated

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
                    G1_U_half_z_Squared_modulated = fft2(modulation_modified * Uz_AST(U1_0, k1, iz / 2)
                                                         * Uz_AST(U2_0, k2, iz / 2))
                    molecule = G1_U_half_z_Squared_modulated \
                               * math.e ** (Gz * iz * 1j) \
                               * math.e ** (k3_z * iz / 2 * 1j) \
                               * (1j * iz)

                    denominator = 1
                elif Cal_version == 3:
                    modulation_modified = ifft2(fft2(modulation) * np.sinc(inside_sinc / np.pi))
                    G1_U_half_z_Squared_modulated = fft2(modulation_modified * Uz_AST(U1_0, k1, iz / 2)
                                                         * Uz_AST(U2_0, k2, iz / 2))
                    molecule = G1_U_half_z_Squared_modulated \
                               * math.e ** (Gz * iz * 1j) \
                               * math.e ** (k3_z * iz / 2 * 1j) \
                               * (1j * iz)
                    denominator = k3_z
                elif Cal_version >= 4:
                    sinc_denominator = (kiizQ + k3_z) / 2
                    molecule = G1_U_half_z_Squared_modulated \
                               * np.sinc(inside_sinc / np.pi) / sinc_denominator \
                               * math.e ** (Gz * iz * 1j) \
                               * math.e ** (k3_z * iz / 2 * 1j) \
                               * (1j * iz)
                    denominator = 1
                if Cal_version == 6:
                    molecule *= correction_factor
            G3_z = const * molecule / denominator

        elif Big_version == 4:
            if Cal_version == 0:
                denominator = (kiizQ + k3_z) / 2
                e_index_1 = -(k1_z ** 2 + 2 * k1_z * Gz) * iz ** 2 / 24
                e_index_2 = -(k2_z ** 2 + 2 * k2_z * Gz) * iz ** 2 / 24
                # print(np.max(np.abs(e_index)))
                G1_U_half_z_Squared_modulated = fft2(modulation * ifft2(
                    fft2(Uz_AST(U1_0, k1, iz / 2)) * math.e ** e_index_1) * ifft2(
                    fft2(Uz_AST(U2_0, k2, iz / 2)) * math.e ** e_index_2))
                # print(np.max(np.abs(-(K1_z**2 - 2*K1_z*k3_z) * iz**2 / 12)))
                # G3_z = const * G1_U_half_z_Squared_modulated \
                #        * math.e ** (-(k3_z-Gz)**2 * iz**2 / 24) / denominator \
                #        * math.e ** (Gz * iz * 1j) \
                #        * math.e ** (k3_z * iz / 2 * 1j) \
                #        * math.e ** (-(K1_z**2 - 2*K1_z*k3_z) * iz**2 / 12) \
                #        * (1j * iz)
                A = -(k3_z - Gz) ** 2 * iz ** 2 / 24
                B = Gz * iz * 1j
                C = k3_z * iz / 2 * 1j
                D = -(K1_z * K2_z - (K1_z + K2_z) * k3_z) * iz ** 2 / 12
                E = A + B + C + D
                # print(np.max(np.abs(e_index)) * 2, np.max(np.abs(E)))
                # print(np.max(np.abs(e_index)) * 2, np.max(np.abs(E)), np.max(np.abs(A)), np.max(np.abs(D)))
                # print(np.max(np.real(e_index)) * 2, np.max(np.real(E)), np.max(np.real(A)), np.max(np.abs(D)))
                # print(np.max(np.abs(dkiizQ))**2 * iz**2)
                G3_z = const * G1_U_half_z_Squared_modulated \
                       * math.e ** E / denominator \
                       * (1j * iz)
            elif Cal_version == 2:
                # print("--", np.max(np.abs(kiizQ)), np.max(np.abs(inside_sinc)), iz * Get("size_PerPixel"), "\n") # 老是 π/2 而与 iz 无关，是为啥？
                # G3_z = gan_G3_z_cos(3**0.5)
                # G3_z, region = gan_G3_z_cos_seq(1)
                G3_z, region = gan_G3_z_cos_seq(cos_num_expect)

        elif Big_version == 5:  # 匹配解 与 不匹配解 的 线性组合
            dismatch = gan_G3_z_dismatch()

            if Cal_version == 0:  # 整体 线性叠加
                match = gan_G3_z_sinc()

                match_factor = math.e ** (- inside_sinc ** 2 / 6)
                dismatch_factor = 1 - match_factor

                match = match_factor * match
                dismatch = dismatch_factor * dismatch
            elif Cal_version == 1:  # 分区 线性叠加
                match, region = gan_G3_z_cos_seq(cos_num_expect)
                match = np.where(np.abs(inside_sinc) <= region, match, 0)
                dismatch = np.where(np.abs(inside_sinc) > region, dismatch, 0)

            G3_z = match + dismatch

        return G3_z * Get("size_PerPixel") ** 2


def G3_z_NLAST(k1, k2, k3, Gx, Gy, Gz,
               U1_0, U2_0, iz, const,
               is_linear_convolution, ):
    Ix, Iy = U1_0.shape[0], U1_0.shape[1]
    k3_z, mesh_k3_x_k3_y = Cal_kz(Ix, Iy, k3)

    Big_version = 5
    Cal_version = 1
    Res_version = 1
    # print(str(Big_version) + '.' + str(Cal_version) + "\n")
    cos_num_expect = 10

    # 3.4 > 3.2 > 1.3    match OK
    # 3.6 = 1.1 > 2      dismatch OK
    # 4.2                match super+ OK
    # 5.1 = 1.1 + 4.2    dismatch + match 分区

    K1_z, K1_xy = find_Kxyz(fft2(U1_0), k1)
    K2_z, K2_xy = find_Kxyz(fft2(U2_0), k2)
    kiizQ = K1_z + K2_z + Gz
    dkiizQ = kiizQ - k3_z
    # denominator = kiizQ + k3_z
    inside_sinc = dkiizQ / 2 * iz

    roll_x, roll_y = Cal_roll_xy(Gx, Gy,
                                 Ix, Iy, )

    def gan_Gg_dismatch():  # 1.1
        G_U1_z_Squared = fft2(Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2, iz))
        g_U1_0_Squared = fft2(U1_0 * U2_0)
        G_U1_z_Squared_Q = Roll_xy(G_U1_z_Squared,
                                   roll_x, roll_y,
                                   is_linear_convolution, )
        g_U1_0_Squared_Q = Roll_xy(g_U1_0_Squared,
                                   roll_x, roll_y,
                                   is_linear_convolution, )
        return G_U1_z_Squared_Q, g_U1_0_Squared_Q

    if Big_version < 3:
        G_U1_z_Squared_Q, g_U1_0_Squared_Q = gan_Gg_dismatch()

        if Res_version == 1:
            molecule = G_U1_z_Squared_Q * math.e ** (Gz * iz * 1j) \
                       - g_U1_0_Squared_Q * math.e ** (k3_z * iz * 1j)

        elif Res_version == 2:
            molecule = G_U1_z_Squared_Q * math.e ** (Gz * iz * 1j)

        elif Res_version == 3:
            molecule = - g_U1_0_Squared_Q * math.e ** (k3_z * iz * 1j)

    elif Big_version == 3:
        G_U1_z_Squared = fft2(Uz_AST(U1_0, k1, iz / 2) * Uz_AST(U2_0, k2, iz / 2))
        G_U1_z_Squared_Q = Roll_xy(G_U1_z_Squared,
                                   roll_x, roll_y,
                                   is_linear_convolution, )
    elif Big_version > 3:
        def gan_G3_z_dismatch():  # 1.1
            G_U1_z_Squared_Q, g_U1_0_Squared_Q = gan_Gg_dismatch()

            molecule = G_U1_z_Squared_Q * math.e ** (Gz * iz * 1j) \
                       - g_U1_0_Squared_Q * math.e ** (k3_z * iz * 1j)
            denominator = kiizQ ** 2 - k3_z ** 2
            dismatch = const * molecule / denominator

            return dismatch

        def gan_G3_z_cos(factor):  # 4.2
            denominator = (kiizQ + k3_z) * 2
            z_plus = (1 + 1 / factor) / 2
            z_minus = (1 - 1 / factor) / 2
            G_U_z_plus_Squared = fft2(Uz_AST(U1_0, k1, z_plus * iz) * Uz_AST(U2_0, k2, z_plus * iz))
            G_U_z_minus_Squared = fft2(Uz_AST(U1_0, k1, z_minus * iz) * Uz_AST(U2_0, k2, z_minus * iz))
            G_U_z_plus_Squared_Q = Roll_xy(G_U_z_plus_Squared,
                                           roll_x, roll_y,
                                           is_linear_convolution, )
            G_U_z_minus_Squared_Q = Roll_xy(G_U_z_minus_Squared,
                                            roll_x, roll_y,
                                            is_linear_convolution, )
            G_U_z_plus_Squared_Q_total = G_U_z_plus_Squared_Q * \
                                         math.e ** (Gz * z_plus * iz * 1j) * \
                                         math.e ** (k3_z * z_minus * iz * 1j)
            G_U_z_minus_Squared_Q_total = G_U_z_minus_Squared_Q * \
                                          math.e ** (Gz * z_minus * iz * 1j) * \
                                          math.e ** (k3_z * z_plus * iz * 1j)
            cos = const * (G_U_z_plus_Squared_Q_total + G_U_z_minus_Squared_Q_total) \
                  / denominator \
                  * (1j * iz)
            return cos

        def gan_G3_z_1():  # 4.2 +
            denominator = kiizQ + k3_z
            G1_U_half_z_Squared = fft2(Uz_AST(U1_0, k1, iz / 2) * Uz_AST(U2_0, k2, iz / 2))
            G1_U_half_z_Squared_Q = Roll_xy(G1_U_half_z_Squared,
                                            roll_x, roll_y,
                                            is_linear_convolution, )
            G1_U_half_z_Squared_Q_total = G1_U_half_z_Squared_Q * \
                                          math.e ** (Gz * iz / 2 * 1j) * \
                                          math.e ** (k3_z * iz / 2 * 1j)
            cos_1 = const * G1_U_half_z_Squared_Q_total \
                    / denominator \
                    * (1j * iz)
            return cos_1

        def gan_G3_z_cos_seq(cos_num_expect):
            aj, cos_num, nums_to_omite, region = gan_factor_out(cos_num_expect, )
            bj = gan_factor_in(cos_num_expect, nums_to_omite, )
            # print(cos_num, nums_to_omite, region, "\n")
            # print(aj, "\n")
            # print(bj, "\n")

            if cos_num == 1:
                cos_seq = gan_G3_z_cos(bj[0])
            else:
                cos_seq = aj[-1] * gan_G3_z_1()
                for j in range(cos_num):
                    cos_seq += aj[j] * gan_G3_z_cos(bj[j])
            return cos_seq, region

    if Big_version <= 3:
        kii2z = (k3 ** 2 - (K1_xy[0] + K2_xy[0] + mesh_k3_x_k3_y[:, :, 0]) ** 2 - (
                K1_xy[1] + K2_xy[1] + mesh_k3_x_k3_y[:, :, 1]) ** 2 + 0j) ** 0.5
        if Big_version == 0:

            # %% denominator: dk_Squared

            # n3_x_n3_y 的 mesh 才用 Gy / (2 * math.pi) * I2_y)，这里是 k3_x_k3_y 的 mesh，所以用 Gy 才对应
            k2izQ = (k2 ** 2 - (mesh_k3_x_k3_y[:, :, 0] - Gy) ** 2 - (
                    mesh_k3_x_k3_y[:, :, 1] - Gx) ** 2 + 0j) ** 0.5

            kizQ = k1 + k2izQ + Gz
            # kizQ = k1 + k3_z + Gz
            denominator = kizQ ** 2 - k3_z ** 2

            kizQ = k1 + (k2 ** 2 - Gx ** 2 - Gy ** 2) ** 0.5 + Gz
            denominator = kizQ ** 2 - k3 ** 2

        elif Big_version == 1:
            denominator = kiizQ ** 2 - k3_z ** 2

        elif Big_version == 2:
            denominator = kiizQ ** 2 - kii2z ** 2

        elif Big_version == 3:
            if Cal_version == 1:
                molecule = G_U1_z_Squared_Q \
                           * math.e ** (Gz * iz * 1j) \
                           * math.e ** (k3_z * iz / 2 * 1j) * (1j * iz)

                denominator = k3_z
            elif Cal_version >= 2:
                if Cal_version < 4:
                    dkiizQ = kiizQ - kii2z
                else:
                    dkiizQ = kiizQ - k3_z
                inside_sinc = dkiizQ / 2 * iz
                if Cal_version == 2:
                    modulation_denominator = (kiizQ + kii2z) / 2

                    molecule = G_U1_z_Squared_Q \
                               * np.sinc(inside_sinc / np.pi) / modulation_denominator \
                               * math.e ** (Gz * iz * 1j) \
                               * math.e ** (k3_z * iz / 2 * 1j) \
                               * (1j * iz)

                    denominator = 1
                elif Cal_version == 3:
                    molecule = G_U1_z_Squared_Q \
                               * np.sinc(inside_sinc / np.pi) \
                               * math.e ** (Gz * iz * 1j) \
                               * math.e ** (k3_z * iz / 2 * 1j) \
                               * (1j * iz)

                    denominator = k3_z
                elif Cal_version >= 4:
                    sinc_denominator = (kiizQ + k3_z) / 2
                    molecule = G_U1_z_Squared_Q \
                               * np.sinc(inside_sinc / np.pi) / sinc_denominator \
                               * math.e ** (Gz * iz * 1j) \
                               * math.e ** (k3_z * iz / 2 * 1j) \
                               * (1j * iz)
                    denominator = 1

        # %%
        G3_z = 2 * const * molecule / denominator

    elif Big_version == 4:  # 4.2
        # if Cal_version == 2:
        G3_z, region = gan_G3_z_cos_seq(cos_num_expect)

    elif Big_version == 5:  # 匹配解 与 不匹配解 的 线性组合, 5.1
        dismatch = gan_G3_z_dismatch()
        # if Cal_version == 1:  # 分区 线性叠加

        match, region = gan_G3_z_cos_seq(cos_num_expect)
        match = np.where(np.abs(inside_sinc) <= region, match, 0)
        dismatch = np.where(np.abs(inside_sinc) > region, dismatch, 0)

        G3_z = match + dismatch

    return G3_z * Get("size_PerPixel") ** 2


# %%

def G3_z_NLAST_false(k1, k2, k3, Gx, Gy, Gz,
                     U1_0, U2_0, iz, const,
                     is_linear_convolution, ):
    Ix, Iy = U1_0.shape[0], U1_0.shape[1]
    k1_z, mesh_k1_x_k1_y = Cal_kz(Ix, Iy, k1)
    k2_z, mesh_k2_x_k2_y = Cal_kz(U2_0.shape[0], U2_0.shape[1], k2)
    k3_z, mesh_k3_x_k3_y = Cal_kz(Ix, Iy, k3)

    G_U1_z_Squared = fft2(Uz_AST(U1_0, k1, iz) * Uz_AST(U2_0, k2, iz))
    g_U1_0_Squared = fft2(U1_0 * U2_0)

    dG_Squared = G_U1_z_Squared \
                 - g_U1_0_Squared * math.e ** (k3_z * iz * 1j)

    # %% denominator: dk_Squared

    kiizQ = k1 + k2_z + Gz

    dk_Squared = kiizQ ** 2 - k3_z ** 2

    # %% fractional

    fractional = dG_Squared / dk_Squared

    roll_x, roll_y = Cal_roll_xy(Gx, Gy,
                                 Ix, Iy, )

    fractional_Q = Roll_xy(fractional,
                           roll_x, roll_y,
                           is_linear_convolution, )

    # %% G3_z0

    G3_z = 2 * const * fractional_Q * math.e ** (Gz * iz * 1j)

    return G3_z * Get("size_PerPixel") ** 2


# %%
# 提供 查找 边缘的，参数的 提示 or 帮助信息 msg

def Info_find_contours_SHG(g1, k1_z, k3_z, dk, Tz, mz,
                           z0, size_PerPixel, deff_structure_length_expect,
                           is_print=1, is_contours=1, n_TzQ=1,
                           Gz_max_Enhance=1, match_mode=1, **kwargs):
    # %%
    # 描边 （目前 暂时只 对倍频 有效，和频描 谁的边 都不知道...，更别说 怎么描了，搞屁）
    key = "Info_find_contours_SHG"
    is_first = int(init_accu(key, 1) == 1)  # 若第一次调用 Info_find_contours_SHG，则 is_first 为 1，否则为 0
    is_Print = is_print * is_first  # 两个 得都 非零，才 print

    is_contours != -1 and is_Print and print(
        tree_print(kwargs.get("is_end", 0), kwargs.get("add_level", 0)) + "info_描边")
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    if is_contours != -1 and is_contours != 0:  # 等于 0 或 -1 则让该 子程序 完全不行使 contours 功能，甚至不提示...
        # 但 0 但会 约束 deff_structure_length_expect， -1 则彻底 啥也不干

        # dk = 2 * np.max(np.abs(k1_z)) - np.max(np.abs(k3_z))
        # print(k3_z[0,0])
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

        # print("k3_z_min = {} / μm, k1_z_min = {} / μm".format(np.min(np.abs(k3_z))/size_PerPixel/1000, np.min(np.abs(k1_z))/size_PerPixel/1000))
        # print(np.abs(k3_z))
        if match_mode == 1:
            ix, iy, scale, energy_fraction = Find_energy_Dropto_fraction(g1, 2 / 3, 0.1)
            Gz_max = np.abs(k3_z[ix, 0]) - 2 * np.abs(k1_z[ix, 0])
            is_Print and print(tree_print() + "scale = {}, energy_fraction = {}".format(scale, energy_fraction))
        else:
            Gz_max = np.min(np.abs(k3_z)) - 2 * np.min(np.abs(k1_z))

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
            is_Print and print(
                tree_print(add_level=1) + "info_描边 2：若希望 mod( 现 z0, TzQ_exp ) = 0")  # 波长定，z0 定，Tz 不定 (lcQ 不定)

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

    if is_contours != -1:  # 等于 -1 则 不额外覆盖 deff_structure_sheet_expect 的值
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
