# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 21:37:19 2021

@author: Xcz
"""

# %%

import math

import numpy as np

from fun_array_Generate import mesh_shift


# %%

def LN_n(lam, T, p="e"):
    if p == "z" or p == "e" or p == "c":
        a = [0, 5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32e-2]
        b = [0, 2.860e-6, 4.700e-8, 6.113e-8, 1.516e-4]
    # elif p == "y" or p == "o" or p == "b":
    else:
        a = [0, 5.653, 0.1185, 0.2091, 89.61, 10.85, 1.97e-2]
        b = [0, 7.941e-7, 3.134e-8, -4.641e-9, 2.188e-6]
    F = (T - 24.5) * (T + 570.82)
    n = math.sqrt(a[1] + b[1] * F + (a[2] + b[2] * F) / (lam ** 2 - (a[3] + b[3] * F) ** 2) + (a[4] + b[4] * F) / (
            lam ** 2 - a[5] ** 2) - a[6] * lam ** 2)
    return n


# %%

def KTP_n_old(lam, T, p="z"):
    if p == "z" or p == "e" or p == "c":
        a = [0, 3.3134, 0.05694, 0.05657, 0, 0, 0.01682]
        b = [0, -1.1327e-7, 1.673e-7, -1.601e-8, 5.2833e-8]
    elif p == "y" or p == "o" or p == "b":
        a = [0, 3.0333, 0.04154, 0.04547, 0, 0, 0.01408]
        b = [0, -2.7261e-7, 1.7896e-7, 5.3168e-7, -3.4988e-7]
    # elif p == "x" or p == "a":
    else:
        a = [0, 3.0065, 0.03901, 0.04251, 0, 0, 0.01327]
        b = [0, -5.3580e-7, 2.8330e-7, 7.5693e-7, -3.9820e-7]
    F = T ** 2 - 400
    n = math.sqrt(a[1] + b[1] * F + (a[2] + b[2] * F) / (lam ** 2 - a[3] + b[3] * F) - (a[6] + b[4] * F) * lam ** 2)
    return n


def KTP_n(lam, T, p="z"):
    if p == "z" or p == "e" or p == "c":
        a = [0, 4.59423, 0.06206, 0.04763, 110.80672, 86.12171]
        if lam < 1.57:
            b = [0, 0.9221, -2.9220, 3.6677, -0.1897]  # lam = 0.53 ~ 1.57
        else:
            b = [0, -0.5523, 3.3920, -1.7101, 0.3424]  # lam = 1.32 ~ 3.53
    elif p == "y" or p == "o" or p == "b":
        a = [0, 3.45018, 0.04341, 0.04597, 16.98825, 39.43799]
        b = [0, 0.1997, -0.4063, 0.5154, 0.5425]  # lam = 0.43 ~ 1.58
    # elif p == "x" or p == "a":
    else:
        a = [0, 3.29100, 0.04140, 0.03978, 9.35522, 31.45571]
        b = [0, 0.1717, -0.5353, 0.8416, 0.1627]  # lam = 0.43 ~ 1.58
    n_T20 = math.sqrt(a[1] + a[2] / (lam ** 2 - a[3]) + a[4] / (lam ** 2 - a[5]))
    if lam < 1.57:
        dn_dT = (b[1] / lam ** 3 + b[2] / lam ** 2 + b[3] / lam + b[4]) * 1e-5
    elif p == "z" or p == "e" or p == "c":
        dn_dT = (b[1] / lam + b[2] + b[3] * lam + b[4] * lam ** 2) * 1e-5
    n = n_T20 + dn_dT * (T - 20)
    return n


# lam = 1.064 / 2
# T = 25

# print("LN_ne = {}".format(LN_n(lam, T, "e")))
# print("LN_no = {}".format(LN_n(lam, T, "o")))
# print("KTP_nz = {}".format(KTP_n(lam, T, "z")))
# print("KTP_ny = {}".format(KTP_n(lam, T, "y")))
# print("KTP_nx = {}".format(KTP_n(lam, T, "x")))
# print("KTP_ne = {}".format(KTP_n(lam, T, "e")))
# print("KTP_no = {}".format(KTP_n(lam, T, "o")))

# %%

def get_n(is_air, lam, T, p):
    if is_air == 1:
        n = 1
    elif is_air == 0:
        n = LN_n(lam, T, p)
    else:
        n = KTP_n(lam, T, p)
    return n


# %%
# 计算 折射率、波矢

def Cal_n(size_PerPixel,
          is_air,
          lam, T, p="e", **kwargs):
    if is_air != 1 and type(kwargs.get("phi_z", 0)) != str:

        # %%
        # 基波 与 倍频 都同享 同一个 theta_x：二者 的 中心波矢 k 差不多 共线，尽管 二次谐波 的 中心 k 还与 结构关系很大，甚至没有 中心 k 一说
        # （旧）"gamma_x" 为 晶体 c 轴 偏离 传播方向 的 夹角 θ<c,propa>，与 "theta_x" 共享 同一个 实验室 坐标系：x 朝右为正
        # （旧）有 "gamma_x" 关键字，则晶体 c 轴 躺在 垂直于 y 轴 的面内，则无 "phi_z" 关键字 可言
        # （旧）"gamma_y" 与 "theta_y" 共享 同一个 实验室 坐标系，也 y 朝上为正（实验室 坐标系，同时 也是 电脑坐标系），所以也得 取负

        # print(kwargs)
        theta_x = kwargs["theta_x"] / 180 * math.pi if "theta_x" in kwargs else 0
        theta_y = - kwargs["theta_y"] / 180 * math.pi if "theta_y" in kwargs else 0
        #  初始时，晶体的 a,b,c 轴，分别与 -x, y, k 重合
        theta_z_c = kwargs["theta_z"] / 180 * math.pi if "theta_z" in kwargs else 0  # 晶轴 c 对 实验室坐标系 方向 z 的 极角
        # （新）"theta_z_c" 为 晶体 c 轴 绕 传播方向 k，从 电脑坐标系 的 -x 轴（实验室 坐标系 的 x 轴）开始，
        #  朝 y 轴 正向，顺时针 旋转（记为 0），的 夹角
        #  即以 k 为 z 轴正向的 右手系 下的值
        phi_z_c = kwargs["phi_z"] / 180 * math.pi if "phi_z" in kwargs else 0  # 晶轴 c 对 实验室坐标系 方向 z 的 方位角
        # （新）"phi_z_c" 晶体 c 轴 与 传播方向 k 轴 的夹角，朝四周 都为正，不一定朝上为正。
        # print(phi_z_c)
        phi_c_c = kwargs["phi_c"] / 180 * math.pi if "phi_c" in kwargs else 0  # 晶体坐标系' 对 晶轴 c（初始晶体坐标系） 的 方位角
        # （新）"phi_c_c" 晶体 绕 自身 c 轴， 自旋 方位角，朝 a → b 为正。
        # print(phi_c_c)

        # %%  生成 折射率 椭球的 3 个主轴
        nz = get_n(is_air, lam, T, "z")  # n_c, n_e
        ny = get_n(is_air, lam, T, "y")  # n_b, n_o
        nx = get_n(is_air, lam, T, "x")  # n_a
        # print(nx, ny, nz)
        # %%
        phi_c_def = math.pi
        theta_c_z, phi_c_z = theta_z_c, phi_c_def - phi_c_c  # 算 实验室坐标系 方向 z 相对 晶轴 c 的 方位角 和 极角
        # 因为 初始时，晶体的 a,b,c 轴，分别与 -x, y, k 重合；且 极角 只沿 z - x 面内 方向 倾倒 折射率椭球；
        # 但按理说 KTP 还能 绕着 自己的 c 轴，自右手系的 a 向 b 旋，多这一个自由度：
        # 事后 在其坐标系下 反向旋转 其他 参照物 即可。
        # print(theta_z_inc, phi_z_inc)
        delta = Cal_delta(nx, ny, nz, theta_c_z, phi_c_z, )
        n_e, n_o = Cal_n_e(nx, ny, nz, theta_c_z, phi_c_z, delta, )
        # print(n_e, n_o)

        n_z = n_e if p == "z" or p == "e" or p == "c" else n_o  # 实验室坐标系 方向 z 上 的 折射率
        k_z = 2 * math.pi * size_PerPixel / (lam / 1000 / n_z)  # 后得到 中心波矢（实验室坐标系 方向 z） 大小

        # %% 生成 mesh
        from fun_global_var import Get
        Ix = kwargs["Ix_structure"] if "Ix_structure" in kwargs else Get("Ix")  # 可能会有 Ix = Ix_structure  从 kwargs 里传进来
        Iy = kwargs["Iy_structure"] if "Iy_structure" in kwargs else Get("Iy")  # 可能会有 Iy = Iy_structure 从 kwargs 里传进来

        mesh_nx_ny_shift = mesh_shift(Ix, Iy)
        mesh_kx_ky_shift = np.dstack(
            (2 * math.pi * mesh_nx_ny_shift[:, :, 0] / Iy, 2 * math.pi * mesh_nx_ny_shift[:, :, 1] / Ix))
        # Iy 才是 笛卡尔坐标系中 x 方向 的 像素数...

        # %%
        sin_theta_z_inc_nxny = (mesh_kx_ky_shift[:, :, 0] ** 2 + mesh_kx_ky_shift[:, :, 1] ** 2) ** 0.5 / k_z
        # 注意 是 kx,ky 或 nx,ny 的 函数（这里 假设了 k 附近的 采样点 分布 是个球面？那这也不准：k_inc 从一开始，就不是个 标量）
        theta_z_inc_nxny = np.arcsin(sin_theta_z_inc_nxny)  # 类比 Cal_theta_phi_z_inc 中的 theta_z_inc = math.acos(kz)
        phi_z_inc_nxny = np.arctan2(- mesh_kx_ky_shift[:, :, 1], - mesh_kx_ky_shift[:, :, 0])
        # phi_z_inc_nxny = np.arctan((- mesh_kx_ky_shift[:, :, 1]) / (- mesh_kx_ky_shift[:, :, 0]))  # 需要 变换到 直角坐标系下
        # print(np.max(theta_z_inc_nxny), np.max(phi_z_inc_nxny), np.min(phi_z_inc_nxny))

        theta_c_inc_nxny, phi_c_inc_nxny = \
            Cal_theta_phi_c_inc(theta_z_c, phi_z_c, phi_c_c,
                                theta_z_inc_nxny, phi_z_inc_nxny, phi_c_inc=phi_c_def)
        # print(np.max(np.abs(theta_c_inc_nxny)), np.max(np.abs(phi_c_inc_nxny)))
        delta_nxny = Cal_delta(nx, ny, nz, theta_c_inc_nxny, phi_c_inc_nxny, )
        n_e_nxny, n_o_nxny = Cal_n_e(nx, ny, nz, theta_c_inc_nxny, phi_c_inc_nxny, delta_nxny, )
        # print(np.max(n_e_nxny), np.max(n_o_nxny))

        n_nxny = n_e_nxny if p == "z" or p == "e" or p == "c" else n_o_nxny
        k_nxny = 2 * math.pi * size_PerPixel / (lam / 1000 / n_nxny)  # 不仅 kz，连 k 现在 都是个 椭球面了
        # Set("k_" + str(k).split('.')[-1], k_nxny) # 用值 来做名字：k 的 值 的 小数点 后的 nums 做为 str ！

        theta_z_inc, phi_z_inc = Cal_theta_phi_z_inc(theta_x, theta_y, )  # %% 算 中心级 相对 实验室坐标系 方向 z 的 方位角 和 极角
        theta_c_inc, phi_c_inc = \
            Cal_theta_phi_c_inc(theta_z_c, phi_z_c, phi_c_c,
                                theta_z_inc, phi_z_inc, phi_c_inc=phi_c_def)

        delta = Cal_delta(nx, ny, nz, theta_c_inc, phi_c_inc, )
        n_e, n_o = Cal_n_e(nx, ny, nz, theta_c_inc, phi_c_inc, delta, )
        # print(np.max(n_e), np.max(n_o))

        n_inc = n_e if p == "z" or p == "e" or p == "c" else n_o  # 基波 传播方向 上 的 折射率
        k_inc = 2 * math.pi * size_PerPixel / (lam / 1000 / n_inc)  # 后得到 中心级 大小

        # print(np.max(np.abs(k_nxny)), k_inc)
    else:  # KTP 有所谓 的 o 光么？
        n_inc = n_nxny = get_n(is_air, lam, T, p)
        k_inc = k_nxny = 2 * math.pi * size_PerPixel / (lam / 1000 / n_inc)  # lam / 1000 即以 mm 为单位

    # if inspect.stack()[1][3] == "pump_pic_or_U" or inspect.stack()[1][3] == "pump_pic_or_U2":
    # print(n_inc, n_nxny)
    return n_inc, n_nxny, k_inc, k_nxny


# %%

def Cal_theta_phi_z_inc(theta_x, theta_y, ):
    from fun_pump import Cal_Unit_kxkykz_based_on_theta_xy
    kx, ky, kz = Cal_Unit_kxkykz_based_on_theta_xy(theta_x, theta_y, )
    kx = - kx  # 即以 k 为 z 轴正向的 右手系 下的值（且已经 归一化 or 单位化）

    theta_z_inc = math.acos(kz)
    phi_z_inc = math.atan2(ky, kx)

    return theta_z_inc, phi_z_inc


def Cal_theta_phi_c_inc(theta_z_c, phi_z_c, phi_c_c,
                        theta_z_inc, phi_z_inc, **kwargs):
    # theta_c_inc = theta_z_inc - theta_z_c  #  没那么简单！
    # phi_c_inc = phi_z_inc - phi_z_c
    # print(phi_z_c, phi_c_c)

    # 边的五元素公式
    cos_theta_c_inc = np.cos(theta_z_c) * np.cos(theta_z_inc) + \
                      np.sin(theta_z_c) * np.sin(theta_z_inc) * np.cos(phi_z_c - phi_z_inc)
    cos_theta_c_inc = np.where(np.abs(cos_theta_c_inc) <= 1, cos_theta_c_inc, np.sign(cos_theta_c_inc))
    theta_c_inc = np.arccos(cos_theta_c_inc)
    # 角的五元素公式
    cos_phi_c_inc = (np.sin(theta_z_c) * np.cos(theta_z_inc) -
                     np.cos(theta_z_c) * np.sin(theta_z_inc) * np.cos(phi_z_c - phi_z_inc)) / \
                    np.sin(theta_c_inc)
    cos_phi_c_inc = np.where(np.sin(theta_c_inc) != 0, cos_phi_c_inc,
                             np.cos(kwargs.get("phi_c_inc", math.pi)))  # 分母（极角 为 0 时，无法定义 相位奇点 phi）
    # print(np.max(np.abs(cos_phi_c_inc)))
    cos_phi_c_inc = np.where(np.abs(cos_phi_c_inc) <= 1, cos_phi_c_inc,
                             np.sign(cos_phi_c_inc))  # 计算的 精度误差 可能导致 cos_phi_c_inc > 1
    # print(np.max(np.abs(cos_theta_c_inc)), np.max(np.abs(cos_phi_c_inc)))
    # print(np.max(np.abs(np.max(np.abs(cos_theta_c_inc)))), np.sign(np.max(np.abs(cos_phi_c_inc))))
    phi_c_inc = np.arccos(cos_phi_c_inc) - phi_c_c
    # print(np.max(np.abs(theta_c_inc)), np.max(np.abs(phi_c_inc)))
    return theta_c_inc, phi_c_inc


# %% biaxis 双轴晶体 折射率曲面 计算

def Cal_cot_Omega(nx, ny, nz, ):
    cot_Omega = (nz / nx) * ((ny ** 2 - nx ** 2) / (nz ** 2 - ny ** 2)) ** 0.5
    return cot_Omega


def Cal_delta1(nx, ny, nz, theta, phi, ):
    cot_Omega = Cal_cot_Omega(nx, ny, nz, )
    # print(cot_Omega)
    # print(np.max(theta), np.max(phi))
    numerator = np.cos(theta) * np.sin(2 * phi)
    denominator = (cot_Omega ** 2) * np.sin(theta) ** 2 - np.cos(theta) ** 2 * np.cos(phi) ** 2 \
                  + np.sin(phi) ** 2
    # cot_2_delta = numerator / denominator
    # tan_2_delta = 1 / cot_2_delta
    tan_2_delta = numerator / denominator if cot_Omega != 0 else 0 / (denominator + 100)
    return tan_2_delta


def Cal_delta2(nx, ny, nz, theta, phi, ):
    numerator = (1 / nx - 1 / ny) * np.cos(theta) * np.sin(2 * phi)
    denominator = (np.sin(phi) ** 2 / nx ** 2 + np.cos(phi) ** 2 / ny ** 2) - \
                  (np.cos(phi) ** 2 / nx ** 2 + np.sin(phi) ** 2 / ny ** 2) * np.cos(theta) ** 2 - \
                  np.sin(theta) ** 2 / nz ** 2
    # print(denominator)
    tan_2_delta = numerator / denominator if nx != ny else 0 / (denominator + 100)
    return tan_2_delta


def Cal_delta(nx, ny, nz, theta, phi, ):
    # tan_2_delta = Cal_delta1(nx, ny, nz, theta, phi, )
    tan_2_delta = Cal_delta2(nx, ny, nz, theta, phi, )

    # print(np.min(tan_2_delta))
    # print(np.min(numerator))
    # print(np.min(denominator))
    delta = np.arctan2(tan_2_delta, 1) / 2
    # delta = np.arctan(tan_2_delta) / 2
    # print(np.max(delta))
    return delta


def Cal_n_e(nx, ny, nz, theta, phi, delta, ):
    # print(np.max(theta) / math.pi * 180, np.max(phi) / math.pi * 180, np.max(delta) / math.pi * 180)

    factor_1 = np.cos(phi) ** 2 / nx ** 2 + np.sin(phi) ** 2 / ny ** 2
    Factor_1 = factor_1 * np.cos(theta) ** 2 + np.sin(theta) ** 2 / nz ** 2
    factor_2 = np.sin(phi) ** 2 / nx ** 2 + np.cos(phi) ** 2 / ny ** 2
    factor_3 = 1 / 2 * (1 / nx ** 2 - 1 / ny ** 2) * np.sin(2 * phi) * np.cos(theta)

    n_e_Squared_devided_by_1 = Factor_1 * np.cos(delta) ** 2 + factor_2 * np.sin(delta) ** 2 \
                               - factor_3 * np.sin(2 * delta)
    n_o_Squared_devided_by_1 = Factor_1 * np.sin(delta) ** 2 + factor_2 * np.cos(delta) ** 2 \
                               + factor_3 * np.sin(2 * delta)

    n_e = 1 / n_e_Squared_devided_by_1 ** 0.5
    n_o = 1 / n_o_Squared_devided_by_1 ** 0.5

    # print(np.max(n_e), np.max(n_o))

    return n_e, n_o


# %% 生成 kz 网格

def Cal_kz(Ix, Iy, k):  # 不仅 kz，连 k 现在 都是个 椭球面了
    mesh_nx_ny_shift = mesh_shift(Ix, Iy)
    mesh_kx_ky_shift = np.dstack(
        (2 * math.pi * mesh_nx_ny_shift[:, :, 0] / Iy, 2 * math.pi * mesh_nx_ny_shift[:, :, 1] / Ix))
    # Iy 才是 笛卡尔坐标系中 x 方向 的 像素数...

    # print(k.shape, mesh_kx_ky_shift.shape)
    kz_shift = (k ** 2 - mesh_kx_ky_shift[:, :, 0] ** 2 - mesh_kx_ky_shift[:, :, 1] ** 2 + 0j) ** 0.5

    return kz_shift, mesh_kx_ky_shift


# %% 透镜 传递函数

def Cal_H_lens(Ix, Iy, size_PerPixel, k, f, Cal_mode=1):
    mesh_ix_iy_shift = mesh_shift(Ix, Iy)
    f /= size_PerPixel
    r_shift = (mesh_ix_iy_shift[:, :, 0] ** 2 + mesh_ix_iy_shift[:, :, 1] ** 2 +
               f ** 2 + 0j) ** 0.5
    H_lens = math.e ** (- np.sign(f) * 1j * k * r_shift)

    rho_shift = (mesh_ix_iy_shift[:, :, 0] ** 2 + mesh_ix_iy_shift[:, :, 1] ** 2 + 0j) ** 0.5
    # H_lens = math.e ** (- np.sign(f) * 1j * k * f) * \
    #          math.e ** (- np.sign(f) * 1j * k * rho_shift ** 2 / (2 * f))
    if Cal_mode == 2:
        H_lens /= np.cos(np.arcsin(rho_shift / abs(f))) ** 3

        # Ix_max, Iy_max = int(np.cos(np.arctan(Ix / abs(f))) * Ix), int(np.cos(np.arctan(Iy / abs(f))) * Iy)
        # if np.mod(Ix - Ix_max, 2) != 0:
        #     Ix_max += 1
        # if np.mod(Iy - Iy_max, 2) != 0:
        #     Iy_max += 1
        # import cv2
        # H_lens = cv2.resize(np.real(H_lens), (Iy_max, Ix_max), interpolation=cv2.INTER_AREA) + \
        #          cv2.resize(np.imag(H_lens), (Iy_max, Ix_max), interpolation=cv2.INTER_AREA) * 1j
        # # 使用cv2.imread()读取图片之后,数据的形状和维度布局是(H,W,C),但是使用函数cv2.resize()进行缩放时候,传入的目标形状是(W,H)
        # border_width_x = (Ix - Ix_max) // 2
        # border_width_y = (Iy - Iy_max) // 2
        # H_lens = np.pad(H_lens, ((border_width_x, border_width_y), (border_width_x, border_width_y)), 'constant',
        #                 constant_values=(0, 0))
    return H_lens


# %%

def fft2(U):  # 返回 g_shift
    return np.fft.fftshift(np.fft.fft2(U))


def ifft2(G_shift):  # 返回 Uz
    return np.fft.ifft2(np.fft.ifftshift(G_shift))


# %%

def Uz_AST(U, k, iz):
    kz_shift, mesh_kx_ky_shift = Cal_kz(U.shape[0], U.shape[1], k)
    H = math.e ** (kz_shift * iz * 1j)
    g_shift = fft2(U)
    Uz = ifft2(g_shift * H)
    return Uz


# %%

def init_AST(Ix, Iy, size_PerPixel,
             lam1, is_air, T,
             theta_x, theta_y,
             **kwargs):
    p = kwargs["polar2"] if "polar2" in kwargs else kwargs.get("polar", "e")

    n_inc, n, k_inc, k = Cal_n(size_PerPixel,
                               is_air,
                               lam1, T, p=p,
                               theta_x=theta_x,
                               theta_y=theta_y, **kwargs)

    k_z, k_xy = Cal_kz(Ix, Iy, k)

    return n_inc, n, k_inc, k, k_z, k_xy


# %%
def Find_energy_Dropto_fraction(U, energy_fraction, relative_error):  # 类似 牛顿迭代法 的 思想

    # print(U)
    U_max_energy = np.max(np.abs(U) ** 2)
    # print(U_max_energy)
    U_total_energy = np.sum(np.abs(U) ** 2)
    # print(U_total_energy)
    U_slice_total_energy_record = 0

    Ix, Iy = U.shape

    scale_up = 1
    scale_down = 0
    scale = 1 / 64  # 默认的 起始 搜寻点 是 1/2 的 图片尺寸

    while (True):

        # print(scale)

        scale_1side = (1 - scale) / 2
        ix = int(Ix * scale_1side)
        iy = int(Iy * scale_1side)

        U_slice = U[ix:-ix, iy:-iy]
        # print(U_slice)
        U_slice_total_energy = np.sum(np.abs(U_slice) ** 2)
        # print(U_slice_total_energy)
        # time.sleep(1)

        if U_slice_total_energy < (
                1 - relative_error) * energy_fraction * U_total_energy:  # 比 设定范围的 下限 还低，则 通量过于低了，应该 扩大视场范围，且 scale 下限设置为该 scale
            if U_slice_total_energy == U_slice_total_energy_record:
                return ix, iy, scale, U_slice_total_energy / U_total_energy
            scale_down = scale
            scale = (scale + scale_up) / 2
            U_slice_total_energy_record = U_slice_total_energy
        elif U_slice_total_energy > (
                1 + relative_error) * energy_fraction * U_total_energy:  # 比 设定范围的 上限 还高，则 通量过于高了，应该 缩小视场范围，且 scale 上限设置为该 scale
            if U_slice_total_energy == U_slice_total_energy_record:
                return ix, iy, scale, U_slice_total_energy / U_total_energy
            scale_up = scale
            scale = (scale_down + scale) / 2
            U_slice_total_energy_record = U_slice_total_energy
        else:
            return ix, iy, scale, U_slice_total_energy / U_total_energy
