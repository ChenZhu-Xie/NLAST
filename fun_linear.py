# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 21:37:19 2021

@author: Xcz
"""

# %%

import math
import numpy as np
from fun_array_Generate import mesh_shift
import time


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

def KTP_n(lam, T, p="z"):
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


# lam = 0.8
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
    # if inspect.stack()[1][3] == "pump_pic_or_U" or inspect.stack()[1][3] == "pump_pic_or_U2":
    from fun_global_var import Get
    if is_air != 1 and (p == "z" or p == "e" or p == "c") and ("gamma_x" in kwargs or "gamma_y" in kwargs):
        n_c = get_n(is_air, lam, T, "c")  # n_e, n_p
        # n_b = get_n(is_air, lam, T, "b")  # n_o, n_s
        n_a = get_n(is_air, lam, T, "a")  # n_o
        Ix = kwargs["Ix_structure"] if "Ix_structure" in kwargs else Get("Ix")  # 可能会有 Ix =   从 kwargs 里传进来
        Iy = kwargs["Iy_structure"] if "Iy_structure" in kwargs else Get("Iy")  # 可能会有 Iy = Iy_structure 从 kwargs 里传进来
        mesh_nx_ny_shift = mesh_shift(Ix, Iy)
        mesh_kx_ky_shift = np.dstack(
            (2 * math.pi * mesh_nx_ny_shift[:, :, 0] / Iy, 2 * math.pi * mesh_nx_ny_shift[:, :, 1] / Ix))
        # Iy 才是 笛卡尔坐标系中 x 方向 的 像素数...

        # 基波 与 倍频 都同享 同一个 theta_x：二者 的 中心波矢 k 差不多 共线，尽管 二次谐波 的 中心 k 还与 结构关系很大，甚至没有 中心 k 一说
        if "gamma_x" in kwargs:  # "gamma_x" 为 晶体 c 轴 偏离 传播方向 的 夹角 θ<c,propa>，与 "theta_x" 共享 同一个 实验室 坐标系：x 朝右为正
            # 有该关键字，则晶体 c 轴 躺在 垂直于 y 轴 的面内，则无 "gamma_y" 关键字 可言
            alpha = - kwargs["gamma_x"] / 180 * math.pi  # 不是 倾斜相位 所对应的：最大光强 作为中心级，而是 图正中 作为 中心级
            alpha_inc = (kwargs["theta_x"] - kwargs["gamma_x"]) / 180 * math.pi  # 基波 传播方向 与 晶轴 c 的夹角
            # alpha = (kwargs["theta_x"] - kwargs["gamma_x"]) / 180 * math.pi  # 得到 光束 / 图 的 中心波矢 相对于 晶体坐标系 的 θ<k,c>
            # θ<k,c> = θ<k,propa> - θ<c,propa> ：中心波矢 相对于 实验室 坐标系 的 传播方向 夹角为 θ<k,propa>
            mesh = mesh_kx_ky_shift[:, :, 0]

        elif "gamma_y" in kwargs:  # "gamma_y" 也 y 朝上为正（实验室 坐标系，同时 也是 电脑坐标系）
            alpha = - kwargs["gamma_y"] / 180 * math.pi  # 光强最大 与 中心级 无关，而 图正中 = 中心级
            alpha_inc = (kwargs["theta_y"] - kwargs["gamma_y"]) / 180 * math.pi  # 基波 传播方向 与 晶轴 c 的夹角
            # alpha = (kwargs["theta_y"] - kwargs["gamma_y"]) / 180 * math.pi  # 转换为 弧度
            # θ<k,c> = θ<k,propa> - θ<c,propa>
            mesh = mesh_kx_ky_shift[:, :, 1]

        n = 1 / (np.sin(alpha) ** 2 / n_c ** 2 + np.cos(alpha) ** 2 / n_a ** 2) ** 0.5
        k = 2 * math.pi * size_PerPixel / (lam / 1000 / n)  # 先得到 中心波矢（图正中 方向） 大小

        sin_Alpha_nxny = mesh / k  # 注意 是 kx,ky 或 nx,ny 的 函数
        # print(mesh[0], mesh[1])  # 如果 是 一样的，说明是 每行一样，则 每列不同，则 自变量 = 列 = y
        Alpha_nxny = np.arcsin(sin_Alpha_nxny)  # θ<k_small,k>
        # print(Alpha_nxny[0], Alpha_nxny[1])
        alpha_nxny = Alpha_nxny + alpha  # θ<k_small,c> = θ<k_small,k> + θ<k,c>
        n_nxny = 1 / (np.sin(alpha_nxny) ** 2 / n_c ** 2 + np.cos(alpha_nxny) ** 2 / n_a ** 2) ** 0.5
        k_nxny = 2 * math.pi * size_PerPixel / (lam / 1000 / n_nxny)  # 不仅 kz，连 k 现在 都是个 椭球面了
        # Set("k_" + str(k).split('.')[-1], k_nxny) # 用值 来做名字：k 的 值 的 小数点 后的 nums 做为 str ！

        n_inc = 1 / (np.sin(alpha_inc) ** 2 / n_c ** 2 + np.cos(alpha_inc) ** 2 / n_a ** 2) ** 0.5
        # 基波 传播方向 上 的 折射率
        k_inc = 2 * math.pi * size_PerPixel / (lam / 1000 / n_inc)  # 后得到 中心波矢（基波 传播方向 上） 大小
    else:
        n = get_n(is_air, lam, T, p)
        n_inc = n_nxny = n
        k = 2 * math.pi * size_PerPixel / (lam / 1000 / n)  # lam / 1000 即以 mm 为单位
        k_inc = k_nxny = k
    return n_inc, n_nxny, k_inc, k_nxny


# %%

# 生成 kz 网格

def Cal_kz(Ix, Iy, k):  # 不仅 kz，连 k 现在 都是个 椭球面了
    mesh_nx_ny_shift = mesh_shift(Ix, Iy)
    mesh_kx_ky_shift = np.dstack(
        (2 * math.pi * mesh_nx_ny_shift[:, :, 0] / Iy, 2 * math.pi * mesh_nx_ny_shift[:, :, 1] / Ix))
    # Iy 才是 笛卡尔坐标系中 x 方向 的 像素数...

    # print(k.shape, mesh_kx_ky_shift.shape)
    kz_shift = (k ** 2 - mesh_kx_ky_shift[:, :, 0] ** 2 - mesh_kx_ky_shift[:, :, 1] ** 2 + 0j) ** 0.5

    return kz_shift, mesh_kx_ky_shift


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

def init_SFG(Ix, Iy, size_PerPixel,
             lam1, is_air, T,
             theta_x, theta_y,
             **kwargs):
    lam2 = kwargs.get("lam2", lam1)
    lam3 = 1 / (1 / lam1 + 1 / lam2)  # 能量守恒

    n3_inc, n3, k3_inc, k3 = Cal_n(size_PerPixel,
                                   is_air,
                                   lam3, T, p=kwargs.get("polar3", "e"),
                                   theta_x=theta_x,
                                   theta_y=theta_y, **kwargs)

    k3_z, k3_xy = Cal_kz(Ix, Iy, k3)

    return lam3, n3_inc, n3, k3_inc, k3, k3_z, k3_xy


# %%

def gan_k_vector(k_inc, theta_x, theta_y, ):
    theta_x = theta_x / 180 * math.pi
    theta_y = theta_y / 180 * math.pi
    kz = k_inc * math.cos(theta_x) * math.cos(theta_y)  # 通光方向 的 分量大小
    kx = k_inc * math.sin(theta_x) * math.cos(theta_y)
    ky = k_inc * math.cos(theta_x) * math.sin(theta_y)
    return kx, ky, kz


def gan_k3_vector(k1_inc, theta1_x, theta1_y,
                  k2_inc, theta2_x, theta2_y, ):
    k1_x, k1_y, k1_z = gan_k_vector(k1_inc, theta1_x, theta1_y, )
    k2_x, k2_y, k2_z = gan_k_vector(k2_inc, theta2_x, theta2_y, )
    k3_x, k3_y, k3_z = k1_x + k2_x, k1_y + k2_y, k1_z + k2_z  # 动量守恒
    k3_inc = k3_x ** 2 + k3_y ** 2 + k3_z ** 2
    return k3_x, k3_y, k3_z, k3_inc


def cal_theta3_xy(k1_inc, theta1_x, theta1_y,
                  k2_inc, theta2_x, theta2_y, ):
    k3_x, k3_y, k3_z, k3_inc = gan_k3_vector(k1_inc, theta1_x, theta1_y,
                                             k2_inc, theta2_x, theta2_y, )
    # sin_theta3_x = k3_x / k3_inc
    # sin_theta3_y = k3_y / k3_inc
    # theta3_x = math.arcsin(tan_theta3_x)
    # theta3_y = math.arcsin(tan_theta3_y)
    tan_theta3_x = k3_x / k3_z
    tan_theta3_y = k3_y / k3_z
    theta3_x = math.atan(tan_theta3_x)
    theta3_y = math.atan(tan_theta3_y)
    return theta3_x, theta3_y

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
