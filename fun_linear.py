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

def Cal_based_on_g(nx, ny, nz, lam, p, size_PerPixel,
                   theta_z_c, phi_z_c, phi_c_c, phi_c_def,
                   k_nxny, g, ):
    from fun_statistics import find_Kxyz
    K_z, K_xy = find_Kxyz(g, k_nxny)
    kx, ky = K_xy[0], K_xy[1]
    theta_z_inc = 0  # 初值 认为 是 0（其实有 K_z 的话，可以 用 atan 等算出来 更准的初值）
    # phi_z_inc = atan2(-kx, -ky)  # -x, y, z 实验室 / 旋转前 晶体 c 坐标系 下 的 方位角
    phi_z_inc = np.arctan2(-ky, -kx)
    # print(theta_z_inc * 180 / math.pi, phi_z_inc * 180 / math.pi)
    theta_z_inc, = solve_refraction_inc_Kx(theta_z_inc, phi_z_inc, kx, ky,
                                           nx, ny, nz, lam, p, size_PerPixel,
                                           theta_z_c, phi_z_c, phi_c_c, phi_c_def)
    return theta_z_inc, phi_z_inc


def step(X):
    return (X >= 0).astype(np.int8())


def atan2(x, y):  # 与 np.arctan2 输入相反，且返回 0~2π
    X = np.arctan2(y, x)
    return (1 - step(X)) * 2 * np.pi + X


def theta_phi_z_inc_to_theta_xy_inc(theta_z_inc, phi_z_inc):  # 有问题
    kx, ky, kz = Cal_Unit_kxkykz_based_on_theta_xy2(theta_z_inc, phi_z_inc)
    theta_x_inc, theta_y_inc = xyz_to_theta_xy_inc(kx, ky, kz)
    # theta_x_inc = - np.sin(theta_z_inc) * np.cos(phi_z_inc)  # 晶体 坐标系 c 转到 x右y下 图片 坐标系 或 x右y上 左手 坐标系
    # theta_y_inc = np.sin(theta_z_inc) * np.sin(phi_z_inc)  # 晶体 坐标系 c 转到 x右y上 左手 坐标系，
    # 因为 之后会进入 gan_k_vector 中的 Cal_Unit_kxkykz_based_on_theta_xy 中去，所以到 左手系
    # 其实 所有要用到 theta_x, theta_y 的，在 2 者定义之初，就是 左手系。所以 理所应当 原样输出。
    return theta_x_inc, theta_y_inc


def Gan_refractive_index_ellipsoid(is_air, lam, T):
    # %%  生成 折射率 椭球的 3 个主轴
    nz = get_n(is_air, lam, T, "z")  # n_c, n_e
    ny = get_n(is_air, lam, T, "y")  # n_b, n_o
    nx = get_n(is_air, lam, T, "x")  # n_a
    return nx, ny, nz


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
        # theta_y = - kwargs["theta_y"] / 180 * math.pi if "theta_y" in kwargs else 0
        theta_y = kwargs["theta_y"] / 180 * math.pi if "theta_y" in kwargs else 0
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
        nx, ny, nz = Gan_refractive_index_ellipsoid(is_air, lam, T)
        # print(nx, ny, nz)
        # %%  计算 实验室坐标系的 z 方向 的 折射率 n_z 和 k_z，作为 kx, ky 网格 所对应的 n_nxny, k_nxny 的 中心、参考、基准
        phi_c_def = math.pi

        n_z, k_z = cal_nk(nx, ny, nz, lam, p, size_PerPixel,
                          theta_z_c, phi_z_c, phi_c_c,
                          0, 0, phi_c_def,
                          record_delta_name="delta_z")
        from fun_global_var import Set
        Set("n_z", n_z)
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
        # 注意 是 kx,ky 或 nx,ny 的 函数（这里 假设了 k 附近的 采样点 分布 是个球面，半径为 k_z。那这也不准：k_inc 从一开始，就不是个 标量）
        theta_z_inc_nxny = np.arcsin(sin_theta_z_inc_nxny)  # 类比 Cal_theta_phi_z_inc 中的 theta_z_inc = math.acos(kz)
        # 这里用 sin 不用 cos，一方面是 不知道 kz 没法用 arccos；另一方面 约束了 只能算 正向 而非 反向传播 的波 的 极角；
        # 但极角本身的取值 0~π，约束我们最好用 arccos 的。只是这里 不需要用、且没法用，罢了。
        # print(np.min(theta_z_inc_nxny) / math.pi * 180, np.max(theta_z_inc_nxny) / math.pi * 180)
        # phi_z_inc_nxny = atan2(- mesh_kx_ky_shift[:, :, 0], - mesh_kx_ky_shift[:, :, 1])
        phi_z_inc_nxny = np.arctan2(- mesh_kx_ky_shift[:, :, 1], - mesh_kx_ky_shift[:, :, 0])
        # phi_z_inc_nxny = np.arctan((- mesh_kx_ky_shift[:, :, 1]) / (- mesh_kx_ky_shift[:, :, 0]))  # 需要 变换到 直角坐标系下
        # print(np.max(phi_z_inc_nxny) / math.pi * 180, np.min(phi_z_inc_nxny) / math.pi * 180)
        # print(phi_z_inc_nxny / math.pi * 180)
        # print(phi_z_inc_nxny[0] / math.pi * 180)
        # print(phi_z_inc_nxny[Ix//2] / math.pi * 180)
        # print(phi_z_inc_nxny[Ix-1] / math.pi * 180)

        # # 用椭球 精确计算 theta_z_inc_nxny、n_nxny, k_nxny
        # def fun1(for_th, fors_num, *args, **kwargs, ):
        #     for iy in range(Ix):
        #         theta_z_inc_nxny[for_th, iy], = \
        #             solve_refraction_inc_nxny_kx(theta_z_inc_nxny[for_th, iy],
        #                                          phi_z_inc_nxny[for_th, iy],
        #                                          mesh_kx_ky_shift[for_th, iy, 0],
        #                                          nx, ny, nz, lam, p, size_PerPixel,
        #                                          theta_z_c, phi_z_c, phi_c_c, phi_c_def)
        # from fun_thread import noop, my_thread
        # my_thread(10, Iy,
        #           fun1, noop, noop,
        #           is_ordered=1, is_print=0, is_end=1)

        # 没法 一次性 解方程组，内存不够
        # theta_z_inc_nxny = solve_refraction_inc_nxny_Kx(theta_z_inc_nxny, phi_z_inc_nxny, mesh_kx_ky_shift[:, :, 0],
        #                                                 nx, ny, nz, lam, p, size_PerPixel,
        #                                                 theta_z_c, phi_z_c, phi_c_c, phi_c_def)

        n_nxny, k_nxny = cal_nk(nx, ny, nz, lam, p, size_PerPixel,
                                theta_z_c, phi_z_c, phi_c_c,
                                theta_z_inc_nxny, phi_z_inc_nxny, phi_c_def,
                                record_delta_name="delta_nxny")

        # %% 算 中心级 相对 实验室坐标系 方向 z 的 方位角 和 极角

        if "k3_inc_x" in kwargs and "k3_inc_y" in kwargs:  # 如果算的是 k3_inc，则需要采取特殊的策略：
            # 已知的不是 theta3_x 和 theta3_y，而是 k3_inc_x, k3_inc_y，则不能用 Cal_theta_phi_z_inc
            # 这个不受 is_air_pump 的影响，因为 k3 一直都在晶体内，不是泵浦，所以 与折射定律无关
            kx, ky = kwargs["k3_inc_x"], kwargs["k3_inc_y"]  # 传入 图片 坐标系 下的 k3x, k3y
            theta_z_inc = 0  # 初值 认为 是 0
            # phi_z_inc = atan2(-kx, -ky)  # -x, y, z 实验室 坐标系 下 的 方位角
            phi_z_inc = np.arctan2(-ky, -kx)
            # print(theta_z_inc * 180 / math.pi, phi_z_inc * 180 / math.pi)
            theta_z_inc, = solve_refraction_inc_Kx(theta_z_inc, phi_z_inc, kx, ky,
                                                   nx, ny, nz, lam, p, size_PerPixel,
                                                   theta_z_c, phi_z_c, phi_c_c, phi_c_def)
        else:  # 否则 直接沿用 晶体内的 theta_x, theta_y
            if "gp" in kwargs:
                theta_z_inc, phi_z_inc = Cal_based_on_g(nx, ny, nz, lam, p, size_PerPixel,
                                                        theta_z_c, phi_z_c, phi_c_c, phi_c_def,
                                                        k_nxny, kwargs["gp"], )
            else:
                theta_z_inc, phi_z_inc = Cal_theta_phi_z_inc(theta_x, theta_y, )  # 初值 沿用 空气中的 极角 和 方位角
                # print(theta_z_inc * 180 / math.pi, phi_z_inc * 180 / math.pi)
                if kwargs.get("is_air_pump", 1) == 1:  # 如果 在空气中 泵浦，则有折射，则 折射定律 算晶体内的 中心级
                    # 否则 直接沿用 晶体内的 theta_x, theta_y
                    theta_z_inc, = solve_refraction_inc_kx(theta_z_inc, phi_z_inc, theta_x, theta_y,
                                                           nx, ny, nz, lam, p, size_PerPixel,
                                                           theta_z_c, phi_z_c, phi_c_c, phi_c_def)

        # print(theta_z_inc * 180 / math.pi, phi_z_inc * 180 / math.pi)
        n_inc, k_inc = cal_nk(nx, ny, nz, lam, p, size_PerPixel,
                              theta_z_c, phi_z_c, phi_c_c,
                              theta_z_inc, phi_z_inc, phi_c_def,
                              record_delta_name="delta_inc")

        # print(np.max(np.abs(k_nxny)), k_inc)

        if "set_theta_tag" in kwargs:
            from fun_global_var import Set
            theta_x, theta_y = theta_phi_z_inc_to_theta_xy_inc(theta_z_inc, phi_z_inc)
            # print(theta_x, theta_y)
            if kwargs["set_theta_tag"] == 1:
                Set("theta_x", theta_x)
                Set("theta_y", theta_y)
            elif kwargs["set_theta_tag"] == 2:
                Set("theta2_x", theta_x)
                Set("theta2_y", theta_y)
            elif kwargs["set_theta_tag"] == 3:
                Set("theta3_x", theta_x)
                Set("theta3_y", theta_y)

    else:  # KTP 有所谓 的 o 光么？
        n_inc = n_nxny = get_n(is_air, lam, T, p)
        k_inc = k_nxny = 2 * math.pi * size_PerPixel / (lam / 1000 / n_inc)  # lam / 1000 即以 mm 为单位

    # if inspect.stack()[1][3] == "pump_pic_or_U" or inspect.stack()[1][3] == "pump_pic_or_U2":
    # print(n_inc, n_nxny)
    return n_inc, n_nxny, k_inc, k_nxny


# %%

def Cal_Unit_kxkykz_based_on_theta_xy2(theta_z_c, phi_z_c, mode=0):  # 旋转前的 晶体 abc 坐标系 -x y z 下的 球坐标系，到其下的 直角坐标系
    kx = np.sin(theta_z_c) * np.cos(phi_z_c)
    ky = np.sin(theta_z_c) * np.sin(phi_z_c)
    kz = np.cos(theta_z_c)
    import inspect
    if inspect.stack()[1][3] == "Gan_E_vector" and mode == 3:
        from fun_global_var import init_accu
        if init_accu("test", 1) <= 1:
            # print(kx)
            print(phi_z_c)
            # print(theta_z_c)
    return kx, ky, kz


def Gan_k_z_inc_u(is_air, lam, T,
                  p, size_PerPixel,
                  theta_x, theta_y,
                  theta_z_c, phi_z_c,
                  k_z_inc, k_z_inc_z, k_z_inc_xy,
                  mode, **kwargs, ):
    # %% 晶体内 入射场 的 k 的 单位矢量，但参照 旋转前的 晶体坐标系 c，而非 图片坐标系： kx 向 左 为正，ky 向 上 为正
    if mode < 3:
        if "gp" in kwargs and mode == 2:
            phi_c_c = kwargs["phi_c"] / 180 * np.pi if "phi_c" in kwargs else 0  # 晶体坐标系' 对 晶轴 c（初始晶体坐标系） 的 方位角
            phi_c_def = math.pi
            # %%  生成 折射率 椭球的 3 个主轴
            nz = get_n(is_air, lam, T, "z")  # n_c, n_e
            ny = get_n(is_air, lam, T, "y")  # n_b, n_o
            nx = get_n(is_air, lam, T, "x")  # n_a
            theta_z_inc, phi_z_inc = Cal_based_on_g(nx, ny, nz, lam, p, size_PerPixel,
                                                    theta_z_c, phi_z_c, phi_c_c, phi_c_def,
                                                    k_z_inc, kwargs["gp"], )
            k_z_inc_ux, k_z_inc_uy, k_z_inc_uz = Cal_Unit_kxkykz_based_on_theta_xy2(theta_z_inc, phi_z_inc)
            k_z_inc_u = np.array([k_z_inc_ux, k_z_inc_uy, k_z_inc_uz])
        else:
            if mode == 1:
                theta_x, theta_y = 0, 0
            from fun_pump import Cal_Unit_kxkykz_based_on_theta_xy
            k_z_inc_ux, k_z_inc_uy, k_z_inc_uz = Cal_Unit_kxkykz_based_on_theta_xy(theta_x, theta_y, )  # 产生的是 左手系
            k_z_inc_ux *= -1  # 现在就换到了 右手 c 系 了
            k_z_inc_u = np.array([k_z_inc_ux, k_z_inc_uy, k_z_inc_uz])
            # print(k_z_inc_u)
    elif mode == 3:  # 从 图片坐标系，换到 c 系：x,y 取负，或 旋转 180 度（颠倒）
        # from fun_array_Transform import Rotate_180  # 方法一：旋转 180 度（颠倒）
        # k_z_inc = Rotate_180(k_z_inc)
        # k_z_inc_z = Rotate_180(k_z_inc_z)
        # k_z_inc_xy = Rotate_180(k_z_inc_xy)
        # k_z_inc_ux, k_z_inc_uy, k_z_inc_uz = \
        #     k_z_inc_xy[:, :, 0] / k_z_inc, k_z_inc_xy[:, :, 1] / k_z_inc, k_z_inc_z / k_z_inc
        # 方法二：x,y 取负
        k_z_inc_ux, k_z_inc_uy, k_z_inc_uz = \
            - k_z_inc_xy[:, :, 0] / k_z_inc, - k_z_inc_xy[:, :, 1] / k_z_inc, k_z_inc_z / k_z_inc  # 单位 unit 矢量 的 3 分量
        # print(k_z_inc_ux[0])
        # print(k_z_inc_uy[:, 0])
        k_z_inc_u = np.dstack((k_z_inc_ux, k_z_inc_uy, k_z_inc_uz))  # 是个 2 维 矢量场，先是 2 个 空间维度，后是 3 个 矢量 分量维度。
    return k_z_inc_u, k_z_inc_ux, k_z_inc_uy, k_z_inc_uz


def split_Array_to_xyz(array):
    if array.shape == (3,):
        array_x, array_y, array_z = array[0], array[1], array[2]
    else:
        array_x, array_y, array_z = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    return array_x, array_y, array_z


def scale_vector(scale_factor, *args, v=1):
    if len(args) != 3:
        vx, vy, vz = split_Array_to_xyz(v)
    else:
        vx, vy, vz = args[0], args[1], args[2]
    v_newx, v_newy, v_newz = vx * scale_factor, vy * scale_factor, vz * scale_factor  # 归一化 X'' 轴 的 单位矢量
    v_new = merge_array_xyz(v_newx, v_newy, v_newz)
    return v_new, v_newx, v_newy, v_newz


def normalize_vector(*args, v=1):
    if len(args) != 3:
        vx, vy, vz = split_Array_to_xyz(v)
    else:
        vx, vy, vz = args[0], args[1], args[2]
    v_amp = (vx ** 2 + vy ** 2 + vz ** 2) ** 0.5  # 2 维 矢量场 每个场点 的 场的模大小
    v_ux, v_uy, v_uz = vx / v_amp, vy / v_amp, vz / v_amp  # 归一化 X'' 轴 的 单位矢量
    v_ux, v_uy, v_uz = np.real(v_ux), np.real(v_uy), np.real(v_uz)
    # print((v_ux ** 2 + v_uy ** 2 + v_uz ** 2) ** 0.5)
    return v_ux, v_uy, v_uz


def normalize_merge_vector(*args, v=1):
    v_ux, v_uy, v_uz = normalize_vector(*args, v=v)
    v_u = merge_array_xyz(v_ux, v_uy, v_uz)  # 归一化后，还要赋值给 v_U，v_U 才也是归一化的
    return v_u, v_ux, v_uy, v_uz


def Apply_T_matrix(delta, axis_x, axis_y, axis_z,
                   obj_x, obj_y, obj_z, ):
    # %% 旋转矩阵 系数
    a1 = np.cos(delta)
    a2 = 1 - a1
    b1 = np.sin(delta)
    # %% 旋转矩阵 主项
    T_11 = axis_x ** 2 * a2 + a1
    T_22 = axis_y ** 2 * a2 + a1
    T_33 = axis_z ** 2 * a2 + a1
    # %% 旋转矩阵 交叉项
    T_12 = axis_x * axis_y * a2 - axis_z * b1
    T_21 = axis_x * axis_y * a2 + axis_z * b1

    T_13 = axis_x * axis_z * a2 + axis_y * b1
    T_31 = axis_x * axis_z * a2 - axis_y * b1

    T_23 = axis_y * axis_z * a2 - axis_x * b1
    T_32 = axis_y * axis_z * a2 + axis_x * b1
    # %% 找到 椭圆截痕 的 其中一个主轴 的 方向，即 差不多是 旋转后的 晶体坐标系 c' 的 c' 轴方向（如果 k ⊥ c' 即 n_zz 入射的话）
    obj_X = T_11 * obj_x + T_12 * obj_y + T_13 * obj_z  # 下标 z，表示 D 是 c 系下的
    obj_Y = T_21 * obj_x + T_22 * obj_y + T_23 * obj_z
    obj_Z = T_31 * obj_x + T_32 * obj_y + T_33 * obj_z
    return obj_X, obj_Y, obj_Z


def get_p_theta_xy_from_kwargs(mode, **kwargs):
    theta_z_c = kwargs["theta_z"] / 180 * np.pi if "theta_z" in kwargs else 0  # 晶轴 c 对 实验室坐标系 方向 z 的 极角
    phi_z_c = kwargs["phi_z"] / 180 * np.pi if "phi_z" in kwargs else 0  # 晶轴 c 对 实验室坐标系 方向 z 的 方位角
    # %%
    from fun_global_var import Get
    if "polar2" in kwargs:
        p = kwargs["polar2"]
        theta_x = Get("theta2_x")  # 注意这里的 theta_x 和 theta_y 得是 （已经算好的）晶体内的，也就是 得通过 Get 得到
        theta_y = Get("theta2_y")
    else:
        p = kwargs.get("polar", "e")
        theta_x = Get("theta_x")
        theta_y = Get("theta_y")
    # %%
    if mode == 1:
        delta = Get("delta_z")
        # print(delta)
    elif mode == 2:
        delta = Get("delta_inc")
    elif mode == 3:
        delta = Get("delta_nxny")
    return p, theta_x, theta_y, \
           delta, theta_z_c, phi_z_c


def merge_array_xyz(array_x, array_y, array_z):
    if array_x.shape == ():
        Array = np.array([array_x, array_y, array_z])
    else:
        Array = np.dstack([array_x, array_y, array_z])
    return Array


def format_f_scalar(x):
    from fun_global_var import Get
    if x.shape == ():
        # x = format(x, Get("F_f"))
        # x = np.sign(x) * float(Get('f_f') % abs(x))  # 为了能 把 e-34 这种都 格式化
        # x = 0 if abs(x) < 1e-5 else x
        x = float(Get('f_f') % x)
    return x


def format_e_scalar(x):
    from fun_global_var import Get
    if x.shape == ():
        # x = format(x, Get("F_f"))
        # x = np.sign(x) * float(Get('f_f') % abs(x))  # 为了能 把 e-34 这种都 格式化
        # x = 0 if abs(x) < 1e-5 else x
        x = float(format(x, Get("F_E")))  # ".1e", Get("F_E")
    return x


def Gan_D_vector(is_air, lam, T,
                 size_PerPixel,
                 k_z_inc, k_z_inc_z, k_z_inc_xy,
                 mode=1, **kwargs):  # i 表示 inc 或 inside 或 p：pump，但不是标量 inc，而是 2 维 数组
    p, theta_x, theta_y, \
    delta, theta_z_c, phi_z_c = get_p_theta_xy_from_kwargs(mode, **kwargs)
    # %% 算 c 系下的 D
    # phi_c_c = kwargs["phi_c"] / 180 * np.pi if "phi_c" in kwargs else 0  # 晶体坐标系' 对 晶轴 c（初始晶体坐标系） 的 方位角
    # D 的方向 与 晶体如何绕 折射率 最长轴 c 轴 自转 无关？ 有关，特别是 k // c 且 双轴晶体时，D 的方向 会随 自转 而转；但 k ⊥ c 时，关系不大。
    # 这种关系，其实已经 体现在 之前对 k_z_inc, k_z_inc_z, k_z_inc_xy 和 对 delta 的 求解 中了：因为 比如 delta 就与 phi_c_c 有关
    # %% 旋转前的 晶体 abc 坐标系 -x y z 下的 晶轴 c 方向 的 单位矢量： kx 向 左 为正，ky 向 上 为正
    k_z_c_ux, k_z_c_uy, k_z_c_uz = Cal_Unit_kxkykz_based_on_theta_xy2(theta_z_c, phi_z_c)
    # k_z_c 和 k_z_inc 都省略了 对 z 轴而言：都是 晶体坐标系 c 下的。
    # print(kx_c, ky_c, kz_c)
    k_z_c_u = np.array([k_z_c_ux, k_z_c_uy, k_z_c_uz])
    # %% 晶体内 入射场 的 k 的 单位矢量，但参照 旋转前的 晶体坐标系 c，而非 图片坐标系： kx 向 左 为正，ky 向 上 为正
    k_z_inc_u, k_z_inc_ux, k_z_inc_uy, k_z_inc_uz = Gan_k_z_inc_u(is_air, lam, T,
                                                                  p, size_PerPixel,
                                                                  theta_x, theta_y,
                                                                  theta_z_c, phi_z_c,
                                                                  k_z_inc, k_z_inc_z, k_z_inc_xy,
                                                                  mode, **kwargs, )
    # %% X'' 轴
    X2 = np.cross(np.cross(k_z_inc_u, k_z_c_u),
                  k_z_inc_u)  # k_z_ince_inc 叉 k_z_c_u，所得的 垂直于 k_z_ince_u 和 k_z_c_u 的 法向量，再 叉乘 k_z_ince_u
    # 是个 2 维 矢量场，先是 2 个 空间维度，后是 3 个 矢量 分量维度。
    # 得到 k_z_ince_u 与 k_z_c_u 所形成的 平面 与 ⊥ k_z_ince_u 的 平面 的 交线，即尚未主轴化 的，旋转 2 次使 Z'' 同向于 k_z_ince_u 的 坐标系 X'' 轴
    X2, X2_ux, X2_uy, X2_uz = normalize_merge_vector(v=X2)
    # %%  # 将 X2e_u 绕 k_z_ince_u 轴 逆时针 旋转 δ，得到 D 的方向（单位矢量）
    # %% 旋转矩阵 系数
    Dz_ux, Dz_uy, Dz_uz = Apply_T_matrix(delta, k_z_inc_ux, k_z_inc_uy, k_z_inc_uz,
                                         X2_ux, X2_uy, X2_uz, )
    # if mode == 1:
    #     print(np.real(Dz_ux), np.real(Dz_uy), np.real(Dz_uz))
    # %% 选择 椭圆截痕 两个主轴 中的某一个 作为 极化方向
    if p == "z" or p == "e" or p == "c":
        D_ux, D_uy, D_uz = Dz_ux, Dz_uy, Dz_uz
    else:
        # %% 将 D_u 再绕 k_z_inc_u 轴（轴不变，则 旋转矩阵 中的 k_z_inc 等分量 保持 不变） 逆时针 旋转 90 度，得到 另一个 极化 D 的方向（单位矢量）
        D_ux, D_uy, D_uz = Apply_T_matrix(math.pi / 2, k_z_inc_ux, k_z_inc_uy, k_z_inc_uz,
                                          Dz_ux, Dz_uy, Dz_uz, )
    # %%
    D_ux, D_uy, D_uz = normalize_vector(D_ux, D_uy, D_uz)  # 再次归一化，防止 T 旋转时 有误差
    # D_ux, D_uy, D_uz = np.real(D_ux), np.real(D_uy), np.real(D_uz)  # 无下标，表示 D 是 c 系下的
    # theta_x_D_u, theta_y_D_u = xyz_to_theta_xy_inc(D_ux, D_uy, D_uz)
    theta_D_u, phi_D_u = xyz_to_theta_phi_vertical(D_ux, D_uy, D_uz)

    theta_D_u = format_f_scalar(theta_D_u)
    phi_D_u = format_f_scalar(phi_D_u)

    # if mode == 1:
    #     print(D_ux, D_uy, D_uz)
    # D_uz = - D_uz  # 旋转前的 晶体坐标系 c，变到 笛卡尔 右手 坐标系（+z 反平行于 k 即 传播 的方向）
    D_ux = - D_ux  # 旋转前的 晶体坐标系 c，变到 笛卡尔 左/右手 坐标系（不变 z 的话，就是左手系）
    D_u = merge_array_xyz(D_ux, D_uy, D_uz)
    return D_u, theta_D_u, phi_D_u


def xyz_to_theta_phi(kx, ky, kz):  # kx, ky, kz, theta_z, phi_z 都是相对 未旋转前 的 c 系的
    theta_z = np.arccos(kz / (kx ** 2 + ky ** 2 + kz ** 2) ** 0.5)  # Dz_u
    # arccos 的 值域 才是 0~π，并且 极化 D 的方向很可能超出 π/2，所以必须用 arccos，不能用 arctan 或 arcsin
    # phi_z = atan2(ky, kx)
    phi_z = np.arctan2(ky, kx)  # 不能接收含 1j 等的 复数
    return theta_z, phi_z


def xyz_to_theta_phi_vertical(kx, ky, kz):  # 未旋转前 的 c 系下的 kx,ky,kz，
    theta_z, phi_z = xyz_to_theta_phi(kx, ky, kz)
    # phi_z -= math.pi / 2  # 变到 绕 c 轴 顺时针 旋转 90 度 的 偏振片 右手系 下 的 theta_z, phi_z
    phi_z = math.pi - phi_z  # 变到 左手系 下 的 theta_z, phi_z
    theta_z *= 180 / np.pi
    phi_z *= 180 / np.pi
    return theta_z, phi_z


def xyz_to_theta_xy_inc(kx, ky, kz):  # 晶体 c 系下的 kx, ky, kz，到 x右y上 的 左手系，
    kx = -kx  # 将 kx, ky, kz 转换到 左手系，方便生成 同一 左手系 下的 theta_x, theta_y
    theta_x_inc = np.arctan(kx / kz)
    theta_y_inc = np.arctan(ky / kz)
    theta_x_inc *= 180 / np.pi
    theta_y_inc *= 180 / np.pi
    return theta_x_inc, theta_y_inc


def Gan_E_vector(is_air, lam, T,
                 D_u, **kwargs, ):  # 算 主轴 c' 系 下的 E，然后 算 c 系 下的 E
    theta_z_c = kwargs["theta_z"] / 180 * np.pi if "theta_z" in kwargs else 0  # 晶轴 c 对 实验室坐标系 方向 z 的 极角
    phi_z_c = kwargs["phi_z"] / 180 * np.pi if "phi_z" in kwargs else 0  # 晶轴 c 对 实验室坐标系 方向 z 的 方位角
    # %% 将 D_u 从直角坐标系 写成 旋转前 晶体坐标系 c 下的 极坐标形式 theta_z_D, phi_z_D
    D_ux, D_uy, D_uz = split_Array_to_xyz(D_u)
    D_ux *= -1  # 注意这里的 D_u 仍是 左手系下的（因为 Gan_D_vector 输出的是 左手系），需要变换
    theta_z_D, phi_z_D = xyz_to_theta_phi(D_ux, D_uy, D_uz)  # Dz_u
    # print(theta_z_D, phi_z_D)
    # %% 将 旋转前 晶体坐标系 c 下的 D_u，变换到 旋转后的 晶体坐标系 c' 下（主轴化 D_u）
    phi_c_c = kwargs["phi_c"] / 180 * np.pi if "phi_c" in kwargs else 0  # 晶体坐标系' 对 晶轴 c（初始晶体坐标系） 的 方位角
    theta_c_D, phi_c_D = Cal_theta_phi_c_inc(theta_z_c, phi_z_c, phi_c_c,
                                             theta_z_D, phi_z_D, **kwargs)
    Dc_ux, Dc_uy, Dc_uz = Cal_Unit_kxkykz_based_on_theta_xy2(theta_c_D, phi_c_D)
    if D_ux.shape != ():
        mode = 3
        from fun_global_var import init_accu
        if init_accu("test2", 1) <= 1:
            # print(phi_z_D)
            # print(phi_c_D)
            # print(phi_z_D)
            pass
    else:
        # print(D_ux, D_uy, D_uz)
        # print(Dc_ux, Dc_uy, Dc_uz)
        # print(theta_z_D, phi_z_D)
        mode = 0
    # print(theta_c_D, phi_c_D)
    # 检验 Inverse_Transform_theta_phi_c_inc_to_z_inc 的 正确性 # 如果下面 检验部分 在 Cal_Unit_kxkykz_based_on_theta_xy2 上面，
    # 并且 取消注释它，则其 返回值 总会传入 Cal_Unit_kxkykz_based_on_theta_xy2，使得 Dc_ux, Dc_uy, Dc_uz 被改变，
    # 但又不是变成 phi_z_D ... 服了。还 tm 能级联。有鬼，真他妈邪门。
    # Inverse_Transform_theta_phi_c_inc_to_z_inc(theta_z_c, phi_z_c, phi_c_c,
    #                                            theta_c_D, phi_c_D, **kwargs)
    # Inverse_Transform_theta_phi_c_inc_to_z_inc(theta_z_c, phi_z_c, phi_c_c,
    #                                            theta_c_D, phi_c_D, **kwargs)
    theta_z_D_back, phi_z_D_back = Inverse_Transform_theta_phi_c_inc_to_z_inc(theta_z_c, phi_z_c, phi_c_c,
                                                                              theta_c_D, phi_c_D, **kwargs)
    # if mode == 0:
    #     print(theta_z_D_back, phi_z_D_back, phi_z_D - phi_z_D_back)
    # Dc_ux, Dc_uy, Dc_uz = Cal_Unit_kxkykz_based_on_theta_xy2(theta_c_D, phi_c_D, mode)
    # %%  生成 折射率 椭球的 3 个主轴
    nx, ny, nz = Gan_refractive_index_ellipsoid(is_air, lam, T)
    # %% 得到 主轴化 的 晶体系 c 下的 E 分量，并归一化为 E_u
    Ec_x, Ec_y, Ec_z = Dc_ux / nx ** 2, Dc_uy / ny ** 2, Dc_uz / nz ** 2  # 模长上 仍不是 E 的大小，只有方向是对的。
    Ec_ux, Ec_uy, Ec_uz = normalize_vector(Ec_x, Ec_y, Ec_z)
    # print(Ec_ux, Ec_uy, Ec_uz)
    # %% 将 Ec_u 写成 旋转后 晶体坐标系 c' 下的 极坐标形式 theta_c_E, phi_c_E
    theta_c_E, phi_c_E = xyz_to_theta_phi(Ec_ux, Ec_uy, Ec_uz)
    # print(theta_c_E, phi_c_E)
    # %% 将 主轴化 的 晶体系 c 下的 E_u，逆变换 回 旋转前 晶体坐标系 c 下
    theta_z_E, phi_z_E = Inverse_Transform_theta_phi_c_inc_to_z_inc(theta_z_c, phi_z_c, phi_c_c,
                                                                    theta_c_E, phi_c_E, **kwargs)
    phi_z_E += phi_z_D - phi_z_D_back
    # print(theta_z_E, phi_z_E)
    E_ux, E_uy, E_uz = Cal_Unit_kxkykz_based_on_theta_xy2(theta_z_E, phi_z_E)  # Ez_u
    E_ux, E_uy, E_uz = normalize_vector(E_ux, E_uy, E_uz)  # 再次归一化，防止 theta_z_E, phi_z_E 计算时 有误差
    # %%
    # E_ux, E_uy, E_uz = np.real(E_ux), np.real(E_uy), np.real(E_uz)  # 无下标，表示 E 是 c 系下的
    # theta_x_E_u, theta_y_E_u = xyz_to_theta_xy_inc(E_ux, E_uy, E_uz)
    theta_E_u, phi_E_u = xyz_to_theta_phi_vertical(E_ux, E_uy, E_uz)

    theta_E_u = format_f_scalar(theta_E_u)
    phi_E_u = format_f_scalar(phi_E_u)

    E_ux = - E_ux  # 旋转前的 晶体坐标系 c，变到 笛卡尔 左手 坐标系（x 右，y 上，z 里）
    E_u = merge_array_xyz(E_ux, E_uy, E_uz)

    return E_u, theta_E_u, phi_E_u


def vector_amp(v):
    return np.sum(v ** 2, -1)


def Gan_S_vector(is_air, lam, T,
                 size_PerPixel,
                 k_z_inc, k_z_inc_z, k_z_inc_xy,
                 D_u, E_u, g_p=0, mode=1, scale_factor=0.05, **kwargs, ):
    # KTP 的 o 光 走离：scale_factor=0.05
    # %% 获取 kwargs 参数
    p, theta_x, theta_y, \
    delta, theta_z_c, phi_z_c = get_p_theta_xy_from_kwargs(mode, **kwargs)
    # %% 旋转前的 晶体 abc 坐标系 -x y z 下的 晶轴 c 方向 的 单位矢量： kx 向 左 为正，ky 向 上 为正
    k_z_inc_u, k_z_inc_ux, k_z_inc_uy, k_z_inc_uz = Gan_k_z_inc_u(is_air, lam, T,
                                                                  p, size_PerPixel,
                                                                  theta_x, theta_y,
                                                                  theta_z_c, phi_z_c,
                                                                  k_z_inc, k_z_inc_z, k_z_inc_xy,
                                                                  mode, **kwargs, )
    # %% 将 D_u 从直角坐标系 写成 旋转前 晶体坐标系 c 下的 极坐标形式 theta_z_D, phi_z_D
    D_ux, D_uy, D_uz = split_Array_to_xyz(D_u)
    E_ux, E_uy, E_uz = split_Array_to_xyz(E_u)
    D_ux *= -1  # 注意这里的 D_u 仍是 左手系下的（因为 Gan_D_vector 输出的是 左手系），需要变换
    E_ux *= -1
    D_u = merge_array_xyz(D_ux, D_uy, D_uz)
    E_u = merge_array_xyz(E_ux, E_uy, E_uz)
    # %%
    H = np.cross(k_z_inc_u, D_u)
    H_u, H_ux, H_uy, H_uz = normalize_merge_vector(v=H)
    S_u = np.cross(E_u, H_u)
    S_u, S_ux, S_uy, S_uz = normalize_merge_vector(v=S_u)  # 归一化后，还要赋值给 S_U，S_U 才也是归一化的
    # S_ux, S_uy, S_uz = split_Array_to_xyz(S_u)  # 我 tm 是真的醉了，归一化后的 E 与 归一化后的 H 叉乘，不是归一化后的 S
    theta_x_S_u, theta_y_S_u = xyz_to_theta_xy_inc(S_ux, S_uy, S_uz)

    if mode < 3:
        # print(S_u)
        # cos_walk_off_angle = np.dot(D_u, E_u)
        cos_walk_off_angle = np.dot(S_u, k_z_inc_u)
        # theta_z_D, phi_z_D = xyz_to_theta_phi(S_ux, S_uy, S_uz)
        # print(theta_z_D / np.pi * 180)
        # print(theta_x_S_u, theta_y_S_u)
        # print(S_u)
        # print(k_z_inc_u)
        # print(D_u)
        # print(E_u)
        # print(cos_walk_off_angle)
        # np.abs(np.dot(S_u, k_z_inc_u)) 保证 cos > 0，但已经不需要，因为 已使 S_uz > 0 为 左手系 下的了
    else:
        # cos_walk_off_angle = np.sum(D_u * E_u, -1)
        cos_walk_off_angle = np.sum(S_u * k_z_inc_u, -1)  # np.abs(np.sum(S_u * k_z_inc_u, -1)) 保证 cos > 0
        # %%  法一：产生 s_z_inc, s_z_inc_z, s_z_inc_xy：s 方向的 等效 k, kz, kxy 的大小
        # %%
        from fun_array_Transform import Rotate_180
        s_z_inc = Rotate_180(k_z_inc) * cos_walk_off_angle  # 得到 S 方向 等效 k (即 s) 的 大小（之前 只得到了 S 的方向 S_u）
        # 这里不能写成 s_z_inc = k_z_inc * cos_walk_off_angle，
        # 因为 cos_walk_off_angle 仍在 c 系下，而非 图片坐标系，需要变成 图片坐标系；
        # 或将 k_z_inc 变到 c 系，这里选择了后者，为的是先与 S_u 同参考系，得到 s_z_inc_z 和 s_z_inc_xy 后，再变换回 图片坐标系
        # 晶体内的 光线速度 v'_s' 一般比 晶体内的 光波面速度 v'_k' 更偏离 入射方向 v_k
        # 因此 晶体内 s' 方向的 群速度 v'_group， 一般比 k' 方向的 相速度 更大（快），但走的路程长，所以用的时间差不多，积累的相位 也差不多
        # 以至于 晶体内 s' 方向的 波矢 s'，一般比 k' 方向的 波矢 更小，以至于 传到晶体 后端面 后，尽管 经历的 路程长，积累的相位 也差不多
        # print(S_ux[0])
        # print(S_ux[:, 0])
        # s_z_inc_x, s_z_inc_y, s_z_inc_z = s_z_inc * k_z_inc_ux, s_z_inc * k_z_inc_uy, s_z_inc * k_z_inc_uz
        s_z_inc_x, s_z_inc_y, s_z_inc_z = s_z_inc * S_ux, s_z_inc * S_uy, s_z_inc * S_uz
        s_z_inc = Rotate_180(s_z_inc)
        s_z_inc_x = Rotate_180(s_z_inc_x)
        s_z_inc_y = Rotate_180(s_z_inc_y)
        s_z_inc_z = Rotate_180(s_z_inc_z)
        s_z_inc_xy = np.dstack((s_z_inc_x, s_z_inc_y))
        # # print(s_z_inc[0])
        # # print(k_z_inc[0])
        # # print(s_z_inc[:, 0])
        # # print(k_z_inc[:, 0])
    # %%  法二：同一 z = z0 面处，s 方向 与 z = z0 面 的 交点 的 相位， 相比 k 方向 与 z = z0 面 的 交点 的 相位，超前的 倍率
    # %%
    # 错误的方法：是平面的，但实际是立体的
    z = np.array([0, 0, 1])
    # # sin_sz = vector_amp(np.cross(S_u, z))  # python 的 叉乘 在幅值上 这么不靠谱？
    # cos_sz = np.dot(S_u, z)  # np.abs(np.dot(S_u, z))
    # sin_sz = (1 - cos_sz ** 2) ** 0.5
    # tan_sz = sin_sz / cos_sz
    # # sin_kz = vector_amp(np.cross(k_z_inc_u, z))
    cos_kz = np.dot(k_z_inc_u, z)  # np.abs(np.dot(k_z_inc_u, z))
    # sin_kz = (1 - cos_kz ** 2) ** 0.5
    # tan_kz = sin_kz / cos_kz
    # delta_sk = (tan_sz - tan_kz) * tan_kz
    # %%
    # 正确的方法： 先 2 个矢量 z 分量 相等（归一化），相减后就是 垂直于 z 轴 的 矢量了，然后再 往 k 矢量上 投影，这样方向 和 大小都有
    # vector_k_to_s_vertical_to_z = scale_vector(1 / S_uz, v=S_u)[0] - scale_vector(1 / k_z_inc_uz, v=k_z_inc_u)[0]
    vector_k_to_s_vertical_to_z = scale_vector(k_z_inc_uz / S_uz, v=S_u)[0] - k_z_inc_u  # 共享 z 向长度 k_z_inc_uz 或 1
    vector_k_to_s_vertical_to_z *= scale_factor
    delta_sk_parallel_to_k = np.sum(vector_k_to_s_vertical_to_z * k_z_inc_u, -1)
    # print(delta_sk_parallel_to_k)
    delta_sk_parallel_to_z = delta_sk_parallel_to_k / cos_kz
    if mode == 1:
        pass
        # print(S_u)
        # print(np.cross(S_u, z))
        # print(np.arcsin(sin_sz) / math.pi * 180)  # python 的 叉乘 在幅值上 这么不靠谱？
        # print(np.arcsin(sin_kz) / math.pi * 180)  # 要是在方向上还不靠谱，那就炸了。
        # print(np.arccos(cos_sz) / math.pi * 180)
        # print(np.arccos(cos_kz) / math.pi * 180)
        # print(np.arctan(tan_sz) / math.pi * 180)
        # print(np.arctan(tan_kz) / math.pi * 180)
    if mode == 3:
        # print(np.arcsin(sin_sz) / math.pi * 180)
        from fun_array_Transform import Rotate_180
        delta_sk_parallel_to_z = Rotate_180(delta_sk_parallel_to_z)
        # print(delta_sk[0])
        # print(delta_sk[:, 0])
        delta_sk_parallel_to_z += 1
    # %%  法三：修改 g_p：双折射 S 与 k 的分离，所导致 的 实空间 整体 位移，可以由 频谱 g 的 整体移动 再衍射，或 随 z 引入倾斜相位，得到。
    # %%
    # 方案一：除了 衍射之外，再引入 随 z 的 倾斜相位
    # 好处：1. 倾斜相位 需要随 z 相关，但不知该 怎么相关 2. 网格不需要 重新采样
    #      3. 图的 复振幅 绝对分布 整体移动，相对分布 不变 4. 晶体后端面 出射时，不需要更改 频谱，频谱 可以保持 连续性 不变
    #      5. 相比 法二，z 向 不再额外衍射，只横向位移，与现实吻合：oe 两光 轮廓 大小差不多
    # 坏处：1. U 倒是 因 倒空间 相位梯度 分离了，但 g 的 强度分布 还是重合 在一起的，导致 不知道 可否使 倍频 效率下降（是否是 其根因）
    #      2. 背后的机制也不清楚，导致 算 g 的 横向移动矢量 的 方法不唯一。
    v_x, v_y, v_z = split_Array_to_xyz(vector_k_to_s_vertical_to_z)
    if mode < 3:
        v_x = - v_x  # 旋转前的 晶体坐标系 c，变到 笛卡尔 左手 坐标系（x 右，y 上，z 里）
        delta_sk_vertical_to_z = np.array([format_e_scalar(v_x),
                                           format_e_scalar(v_y)])
        from fun_global_var import Set
        if mode == 1:  # 设置来 “标量 梯度” 处 使用
            Set("v_z_x", v_x)  # 注意 此处 设置的是 左手系下的
            Set("v_z_y", v_y)
        else:
            Set("v_inc_x", v_x)
            Set("v_inc_y", v_y)
    else:
        from fun_array_Generate import mesh_shift
        mesh_Ix0_Iy0_shift = mesh_shift(v_x.shape[0], v_y.shape[1])
        # 矢场 梯度
        # v_x = Rotate_180(v_x)
        # v_y = Rotate_180(v_y)
        # v_z = Rotate_180(v_z)
        # delta_sk_vertical_to_z = v_x * mesh_Ix0_Iy0_shift[:, :, 0] + \
        #                          v_y * mesh_Ix0_Iy0_shift[:, :, 1]
        # 本来应叫做 Phase_Gradient_vertical_to_z，但方便导出，用的名是 delta_sk_vertical_to_z
        # 标量 梯度
        from fun_global_var import Get
        # 左手系 变到 图片坐标系，x 保持不变，y 反
        v_x, v_y = Get("v_z_x"), - Get("v_z_y")
        delta_sk_vertical_to_z = v_x * mesh_Ix0_Iy0_shift[:, :, 0] + \
                                 v_y * mesh_Ix0_Iy0_shift[:, :, 1]

    # 但该方法 本质上 并不是修改 g_p，而是 像 s 比 k 相位超前倍率 那样，算出一个 衍射传递函数 的 修正因子，并且可直接 沿用那里的
    # 只不过 不再是 delta_sk = (tan_sz - tan_kz) * tan_kz，而是 delta_sk_vertical_to_z = (tan_sz - tan_kz) * z
    # 额不， 不再是 delta_sk = delta_sk_parallel_to_k / cos_kz，而是 delta_sk_vertical_to_z = vector_k_to_s_vertical_to_z 点积 kx,ky
    # 意味着 正入射 也会有 倾斜相位，并且 delta_sk 的 乘以对象 不再是 kz · z，而是 横向的 kx,ky
    # %%
    # 方案二：频谱 g 一次性 整体移动，后续 再衍射
    # 好处：1. 图的移动 直接 与 z 相关 2. 真正的 走离效应，可用于 仿真 倍频效率 （g 的 交叠区域 减小，交叠积分 降低）
    #      3. 图的 强度 绝对分布 整体移动，但 相对分布 不变（但空域 引入倾斜相位，这与 S 沿用 等相位面 的 物理图像 不兼容）
    #      4. 中心级 可看做 群速度 中心 5. 相比 法二，z 向 不再额外衍射，只横向位移，与现实吻合：oe 两光 轮廓 大小差不多
    # 坏处：1. 修改完 并计算到 晶体后端面后，得修改回来，否则 再折射到空气中时 不满足 保持 传播方向 不变（复杂）
    #      2. 怎么算 g 的 横向移动矢量？且 网格 每个格点 单独移动 会导致 网格不再 均匀。可能需要像 产生 s, s_z, s_xy 一样 重新采样。

    # %%

    walk_off_angle = np.arccos(cos_walk_off_angle)
    walk_off_angle = walk_off_angle / np.pi * 180

    theta_x_S_u = format_f_scalar(theta_x_S_u)
    theta_y_S_u = format_f_scalar(theta_y_S_u)
    walk_off_angle = format_f_scalar(walk_off_angle)
    delta_sk_parallel_to_z = format_e_scalar(delta_sk_parallel_to_z)

    if mode < 3:
        return S_u, theta_x_S_u, theta_y_S_u, \
               walk_off_angle, delta_sk_parallel_to_z, delta_sk_vertical_to_z
    else:
        return S_u, theta_x_S_u, theta_y_S_u, \
               walk_off_angle, delta_sk_parallel_to_z, delta_sk_vertical_to_z, \
               s_z_inc, s_z_inc_z, s_z_inc_xy, g_p
    # return S_u, theta_x_S_u, theta_y_S_u, \
    #        walk_off_angle, delta_sk_parallel_to_z


# %%

def solve_refraction_inc_nxny_kx(theta_z_inc_nxny, phi_z_inc_nxny, kx_nxny,
                                 nx, ny, nz, lam, p, size_PerPixel,
                                 theta_z_c, phi_z_c, phi_c_c, phi_c_def):  # phi_z_inc_nxny 当公有 常量，折射前后 必然相等

    def your_funcs(X):
        # from fun_pump import Cal_Unit_kxkykz_based_on_theta_xy
        # k_air = 2 * math.pi * size_PerPixel / (lam / 1000)
        theta_z_inc, = X
        # print(theta_z_inc)

        f = [cal_nk(nx, ny, nz, lam, p, size_PerPixel,
                    theta_z_c, phi_z_c, phi_c_c,
                    theta_z_inc, phi_z_inc_nxny, phi_c_def)[1] * np.sin(theta_z_inc) * np.cos(phi_z_inc_nxny) \
             - (- kx_nxny)]

        return f

    from scipy.optimize import root
    sol = root(your_funcs, [theta_z_inc_nxny])
    # print(sol)
    return sol.x


def solve_refraction_inc_nxny_Kx(theta_z_inc_nxny, phi_z_inc_nxny, kx_nxny,
                                 nx, ny, nz, lam, p, size_PerPixel,
                                 theta_z_c, phi_z_c, phi_c_c, phi_c_def):  # phi_z_inc_nxny 当公有 常量，折射前后 必然相等
    phi_z_inc_nxny = phi_z_inc_nxny.reshape(phi_z_inc_nxny.shape[0] * phi_z_inc_nxny.shape[1])
    kx_nxny = kx_nxny.reshape(kx_nxny.shape[0] * kx_nxny.shape[1])

    def your_funcs(X):
        # from fun_pump import Cal_Unit_kxkykz_based_on_theta_xy
        # k_air = 2 * math.pi * size_PerPixel / (lam / 1000)
        theta_z_inc = X
        # print(theta_z_inc)

        f = cal_nk(nx, ny, nz, lam, p, size_PerPixel,
                   theta_z_c, phi_z_c, phi_c_c,
                   theta_z_inc, phi_z_inc_nxny, phi_c_def)[1] * np.sin(theta_z_inc) * np.cos(phi_z_inc_nxny) \
            - (- kx_nxny)

        return f

    from scipy.optimize import root
    sol = root(your_funcs, theta_z_inc_nxny)
    # print(sol)
    return sol.x


def solve_refraction_inc_kx(theta_z_inc, phi_z_inc, theta_x, theta_y,
                            nx, ny, nz, lam, p, size_PerPixel,
                            theta_z_c, phi_z_c, phi_c_c, phi_c_def):  # phi_z_inc 当公有 常量，折射前后 必然相等
    def your_funcs(X):
        from fun_pump import Cal_Unit_kxkykz_based_on_theta_xy
        k_air = 2 * math.pi * size_PerPixel / (lam / 1000)
        theta_z_inc, = X

        f = [cal_nk(nx, ny, nz, lam, p, size_PerPixel,
                    theta_z_c, phi_z_c, phi_c_c,
                    theta_z_inc, phi_z_inc, phi_c_def)[1] * math.sin(theta_z_inc) * math.cos(phi_z_inc) \
             - k_air * (- Cal_Unit_kxkykz_based_on_theta_xy(theta_x, theta_y, )[0])]

        return f

    from scipy.optimize import root
    sol = root(your_funcs, [theta_z_inc])
    # print(sol)
    return sol.x


def solve_refraction_inc_Kx(theta_z_inc, phi_z_inc, kx, ky,
                            nx, ny, nz, lam, p, size_PerPixel,
                            theta_z_c, phi_z_c, phi_c_c, phi_c_def):  # phi_z_inc 当公有 常量，折射前后 必然相等
    def your_funcs(X):
        theta_z_inc, = X

        f = [cal_nk(nx, ny, nz, lam, p, size_PerPixel,
                    theta_z_c, phi_z_c, phi_c_c,
                    theta_z_inc, phi_z_inc, phi_c_def)[1] * math.sin(theta_z_inc) * math.cos(phi_z_inc) \
             - (- kx)]

        return f

    from scipy.optimize import root
    sol = root(your_funcs, [theta_z_inc])
    # print(sol)
    return sol.x


def solve_refraction_inc_ky(theta_z_inc, phi_z_inc, theta_x, theta_y,
                            nx, ny, nz, lam, p, size_PerPixel,
                            theta_z_c, phi_z_c, phi_c_c, phi_c_def):  # phi_z_inc 当公有 常量，折射前后 必然相等
    def your_funcs(X):
        from fun_pump import Cal_Unit_kxkykz_based_on_theta_xy
        k_air = 2 * math.pi * size_PerPixel / (lam / 1000)
        theta_z_inc, = X

        f = [cal_nk(nx, ny, nz, lam, p, size_PerPixel,
                    theta_z_c, phi_z_c, phi_c_c,
                    theta_z_inc, phi_z_inc, phi_c_def)[1] * math.sin(theta_z_inc) * math.sin(phi_z_inc) \
             - k_air * Cal_Unit_kxkykz_based_on_theta_xy(theta_x, theta_y, )[1]]

        return f

    from scipy.optimize import root
    sol = root(your_funcs, [theta_z_inc])
    # print(sol)
    return sol.x


def solve_refraction_inc_kxky(theta_z_inc, phi_z_inc, theta_x, theta_y,
                              nx, ny, nz, lam, p, size_PerPixel,
                              theta_z_c, phi_z_c, phi_c_c, phi_c_def):  # phi_z_inc 当未知量，这样计算量 会稍大，但更 general：可验证 横向动量守恒
    def your_funcs(X):
        from fun_pump import Cal_Unit_kxkykz_based_on_theta_xy
        k_air = 2 * math.pi * size_PerPixel / (lam / 1000)
        theta_z_inc, phi_z_inc = X

        f = [cal_nk(nx, ny, nz, lam, p, size_PerPixel,
                    theta_z_c, phi_z_c, phi_c_c,
                    theta_z_inc, phi_z_inc, phi_c_def)[1] * math.sin(theta_z_inc) * math.cos(phi_z_inc) \
             - k_air * (- Cal_Unit_kxkykz_based_on_theta_xy(theta_x, theta_y, )[0]),
             cal_nk(nx, ny, nz, lam, p, size_PerPixel,
                    theta_z_c, phi_z_c, phi_c_c,
                    theta_z_inc, phi_z_inc, phi_c_def)[1] * math.sin(theta_z_inc) * math.sin(phi_z_inc) \
             - k_air * Cal_Unit_kxkykz_based_on_theta_xy(theta_x, theta_y, )[1]]

        return f

    from scipy.optimize import root
    sol = root(your_funcs, [theta_z_inc, phi_z_inc])
    # print(sol)
    return sol.x


# %%

def cal_nk(nx, ny, nz, lam, p, size_PerPixel,
           theta_z_c, phi_z_c, phi_c_c,
           theta_z_inc, phi_z_inc, phi_c_def, record_delta_name=""):  # 传一个 非空名 进来，就以之为名地记录 delta
    # theta_c_z, phi_c_z = theta_z_c, phi_c_def - phi_c_c  # 算 实验室坐标系 方向 z 相对 晶轴 c 的 方位角 和 极角
    # # 因为 初始时，晶体的 a,b,c 轴，分别与 -x, y, k 重合；且 极角 只沿 z - (-x) 面内 方向 倾倒 折射率椭球；
    # # 但按理说 KTP 还能 绕着 自己的 c 轴，自右手系的 a 向 b 旋，多这一个自由度：
    # # 事后 在其坐标系下 反向旋转 其他 参照物 即可。
    # # print(theta_z_inc, phi_z_inc)
    # delta = Cal_delta(nx, ny, nz, theta_c_z, phi_c_z, )
    # n_e, n_o = Cal_n_eo(nx, ny, nz, theta_c_z, phi_c_z, delta, )
    # # print(n_e, n_o)
    #
    # n_z = n_e if p == "z" or p == "e" or p == "c" else n_o  # 实验室坐标系 方向 z 上 的 折射率
    # k_z = 2 * math.pi * size_PerPixel / (lam / 1000 / n_z)  # 后得到 中心波矢（实验室坐标系 方向 z） 大小

    theta_c_inc, phi_c_inc = \
        Cal_theta_phi_c_inc(theta_z_c, phi_z_c, phi_c_c,
                            theta_z_inc, phi_z_inc, phi_c_inc=phi_c_def)

    delta = Cal_delta(nx, ny, nz, theta_c_inc, phi_c_inc, )
    if record_delta_name != '':
        from fun_global_var import Set
        Set(record_delta_name, delta)
    n_e, n_o = Cal_n_eo(nx, ny, nz, theta_c_inc, phi_c_inc, delta, )
    # print(np.max(n_e), np.max(n_o))

    n_inc = n_e if p == "z" or p == "e" or p == "c" else n_o  # 基波 传播方向 上 的 折射率
    k_inc = 2 * math.pi * size_PerPixel / (lam / 1000 / n_inc)  # 后得到 中心级 大小

    # theta_c_inc_nxny, phi_c_inc_nxny = \
    #     Cal_theta_phi_c_inc(theta_z_c, phi_z_c, phi_c_c,
    #                         theta_z_inc_nxny, phi_z_inc_nxny, phi_c_inc=phi_c_def)
    # # print(np.max(np.abs(theta_c_inc_nxny)), np.max(np.abs(phi_c_inc_nxny)))
    # delta_nxny = Cal_delta(nx, ny, nz, theta_c_inc_nxny, phi_c_inc_nxny, )
    # n_e_nxny, n_o_nxny = Cal_n_eo(nx, ny, nz, theta_c_inc_nxny, phi_c_inc_nxny, delta_nxny, )
    # # print(np.max(n_e_nxny), np.max(n_o_nxny))
    #
    # n_nxny = n_e_nxny if p == "z" or p == "e" or p == "c" else n_o_nxny
    # k_nxny = 2 * math.pi * size_PerPixel / (lam / 1000 / n_nxny)  # 不仅 kz，连 k 现在 都是个 椭球面了
    # # Set("k_" + str(k).split('.')[-1], k_nxny) # 用值 来做名字：k 的 值 的 小数点 后的 nums 做为 str ！

    return n_inc, k_inc


# %%

def Cal_theta_phi_z_inc(theta_x, theta_y, ):
    from fun_pump import Cal_Unit_kxkykz_based_on_theta_xy
    kx, ky, kz = Cal_Unit_kxkykz_based_on_theta_xy(theta_x, theta_y, )
    kx = - kx  # 即以 k 为 z 轴正向的 右手系 下的值（且已经 归一化 or 单位化）
    # ky = - ky  # 之前 theta_y = - kwargs["theta_y"] 给 搞成电脑坐标系下的了，所以 ky 也得取负 变成 y 轴向上...

    theta_z_inc = math.acos(kz)
    # phi_z_inc = atan2(kx, ky)  # -x, y, z 实验室 坐标系 下 的 方位角
    phi_z_inc = np.arctan2(ky, kx)

    return theta_z_inc, phi_z_inc


def Cal_theta_phi_c_inc(theta_z_c, phi_z_c, phi_c_c,
                        theta_z_inc, phi_z_inc, **kwargs):
    # theta_c_inc = theta_z_inc - theta_z_c  #  没那么简单！
    # phi_c_inc = phi_z_inc - phi_z_c
    # print(phi_z_c, phi_c_c)

    # 边的余弦定理
    cos_theta_c_inc = np.cos(theta_z_c) * np.cos(theta_z_inc) + \
                      np.sin(theta_z_c) * np.sin(theta_z_inc) * np.cos(phi_z_inc - phi_z_c)
    # inc 对 z 减 c 对 z，才是 inc 对 c
    cos_theta_c_inc = np.where(np.abs(cos_theta_c_inc) <= 1, cos_theta_c_inc, np.sign(cos_theta_c_inc))
    theta_c_inc = np.arccos(cos_theta_c_inc)

    # # 边的五元素公式
    # cos_phi_c_inc = - (np.sin(theta_z_c) * np.cos(theta_z_inc) -
    #                  np.cos(theta_z_c) * np.sin(theta_z_inc) * np.cos(phi_z_inc - phi_z_c)) / \
    #                 np.sin(theta_c_inc)
    # cos_phi_c_inc = np.where(np.sin(theta_c_inc) != 0, cos_phi_c_inc,
    #                          np.cos(kwargs.get("phi_c_inc", math.pi)))  # 分母（极角 为 0 时，无法定义 相位奇点 phi）
    # # print(np.max(np.abs(cos_phi_c_inc)))
    # cos_phi_c_inc = np.where(np.abs(cos_phi_c_inc) <= 1, cos_phi_c_inc,
    #                          np.sign(cos_phi_c_inc))  # 计算的 精度误差 可能导致 cos_phi_c_inc > 1
    # # print(cos_phi_c_inc)
    # # print(np.max(np.abs(cos_theta_c_inc)), np.max(np.abs(cos_phi_c_inc)))
    # # print(np.max(np.abs(np.max(np.abs(cos_theta_c_inc)))), np.sign(np.max(np.abs(cos_phi_c_inc))))
    # phi_c_inc = np.arccos(cos_phi_c_inc)
    # phi_c_inc -= phi_c_c

    # 正弦定理 （才能区分 phi_c_inc ~ pi 附近，sin(phi_c_inc) 的正负，对应 kx, ky 等的正负）
    sin_phi_c_inc = - np.sin(theta_z_inc) / np.sin(theta_c_inc) * np.sin(phi_z_inc - phi_z_c)
    phi_c_inc = np.arcsin(sin_phi_c_inc)
    phi_c_inc = np.nan_to_num(phi_c_inc)
    # print(phi_c_inc)
    phi_c_inc -= phi_c_c
    # if phi_c_inc >= 0:
    #     phi_c_inc -= phi_c_c  # 这要求 atan2 出来的 phi_c_inc 必须取值 0~2π，所以不能用 np.arctan2 ?
    #     # 不不不，其他地方最好还是用 -π~π 的取值，否则 似乎 phi_z_inc - phi_z_c 又容易出错
    #     # 只需要 这里改改就行
    # else:
    #     phi_c_inc += phi_c_c

    # 如果 算出来 inc 沿 旋转后的 晶体坐标系 c' 的 光轴（即 theta_c_inc == 0），则无法定义 c' 系下的 phi，则主动去 定义之 为 0

    # print(np.arccos(cos_phi_c_inc) / math.pi * 180)
    # print(phi_c_inc / math.pi * 180)
    # print(type(phi_c_inc))
    # if type(phi_c_inc) == np.ndarray:
    #     print(phi_c_inc[0] / math.pi * 180)
    #     # print(phi_c_inc[:, 0] / math.pi * 180)
    # print(np.max(np.abs(theta_c_inc)), np.max(np.abs(phi_c_inc)))
    return theta_c_inc, phi_c_inc


def Inverse_Transform_theta_phi_c_inc_to_z_inc(theta_z_c, phi_z_c, phi_c_c,
                                               theta_c_inc, phi_c_inc, **kwargs):
    # print(phi_c_inc, phi_c_c)
    phi_c_inc += phi_c_c
    # if phi_c_inc >= 0:
    #     phi_c_inc += phi_c_c  # 先退环境到 绕 c' 轴 自转之前（包括 inc 对 c 以及 z 对 c）
    # else:
    #     phi_c_inc -= phi_c_c
    # np.where(phi_c_inc >= 0, phi_c_inc + phi_c_c, phi_c_inc - phi_c_c)
    # print(phi_c_inc)
    theta_c_z = theta_z_c
    # phi_c_z = math.pi - phi_z_c  # 此时的 z 对 c 的 phi 还算可知
    phi_c_z = math.pi  # z 对 c 的 phi 始终是 π
    phi_z_z = math.pi - phi_z_c  # 要直接沿用 Cal_theta_phi_c_inc 这同一个模型的话，还得自转 π 减去 c 对 c 的 逆时针 自转方位角
    theta_z_inc, phi_z_inc = Cal_theta_phi_c_inc(theta_c_z, phi_c_z, phi_z_z,
                                                 theta_c_inc, phi_c_inc, **kwargs)
    return theta_z_inc, phi_z_inc


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
    # delta = atan2(1, tan_2_delta) / 2
    delta = np.arctan2(tan_2_delta, 1) / 2
    # delta = np.arctan(tan_2_delta) / 2
    # print(np.max(delta))
    return delta


def Cal_n_eo(nx, ny, nz, theta, phi, delta, ):
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
    # print(np.min(n_e), np.min(n_o))

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
    if "polar2" in kwargs:
        p = kwargs["polar2"]
        set_theta_tag = 2
    else:
        p = kwargs.get("polar", "e")
        set_theta_tag = 1

    n_inc, n, k_inc, k = Cal_n(size_PerPixel,
                               is_air,
                               lam1, T, p=p,
                               theta_x=theta_x,
                               theta_y=theta_y,
                               set_theta_tag=set_theta_tag, **kwargs)

    k_z, k_xy = Cal_kz(Ix, Iy, k)

    return n_inc, n, k_inc, k, k_z, k_xy


# %%

def init_AST_pro(Ix, Iy, size_PerPixel,
                 lam1, is_air, T,
                 theta_x, theta_y,
                 g_p, p_p, is_print, **kwargs):
    is_end = kwargs.get("is_end", 0)
    is_end2 = kwargs.get("is_end2", 0)
    add_level = kwargs.get("add_level", 0)
    kwargs.pop("is_end", None)
    kwargs.pop("is_end2", None)
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    # %%
    n1_inc, n1, k1_inc, k1, k1_z, k1_xy = init_AST(Ix, Iy, size_PerPixel,
                                                   lam1, is_air, T,
                                                   theta_x, theta_y, **kwargs)

    # %%
    from fun_global_var import Get
    if "polar2" in kwargs:
        p = kwargs["polar2"]
        theta_x = Get("theta2_x")
        theta_y = Get("theta2_y")
    else:
        p = kwargs.get("polar", "e")
        theta_x = Get("theta_x")
        theta_y = Get("theta_y")
    theta_x = float(Get('f_f') % theta_x)
    theta_y = float(Get('f_f') % theta_y)
    # %%

    args_Gan_E_vector = \
        [is_air, lam1, T, ]
    args_Gan_D_vector = \
        [is_air, lam1, T,
         size_PerPixel,
         k1, k1_z, k1_xy, ]

    # %%

    D_u_0kx0ky, theta_D_u_0kx0ky, phi_D_u_0kx0ky = Gan_D_vector(*args_Gan_D_vector,  # 左手系
                                                                mode=1, **kwargs)
    E_u_0kx0ky, theta_E_u_0kx0ky, phi_E_u_0kx0ky = Gan_E_vector(*args_Gan_E_vector,
                                                                D_u_0kx0ky, **kwargs, )
    S_u, theta_x_S_0kx0ky, theta_y_S_0kx0ky, \
    walk_off_angle_0kx0ky, delta_sk_pz_0kx0ky, delta_sk_vz_0kx0ky = Gan_S_vector(*args_Gan_D_vector,
                                                                                 D_u_0kx0ky, E_u_0kx0ky, mode=1,
                                                                                 **kwargs, )
    D_u_inc, theta_D_u_inc, phi_D_u_inc = Gan_D_vector(*args_Gan_D_vector,
                                                       mode=2, **kwargs)
    # print(D_u_inc)
    E_u_inc, theta_E_u_inc, phi_E_u_inc = Gan_E_vector(*args_Gan_E_vector,
                                                       D_u_inc, **kwargs, )
    # print(E_u_inc)
    S_u, theta_x_S_u_inc, theta_y_S_u_inc, \
    walk_off_angle_inc, delta_sk_pz_inc, delta_sk_vz_inc = Gan_S_vector(*args_Gan_D_vector,
                                                                        D_u_inc, E_u_inc, mode=2, **kwargs, )
    D_u, theta_D_u, phi_D_u = Gan_D_vector(*args_Gan_D_vector,
                                           mode=3, **kwargs)
    E_u, theta_E_u, phi_E_u = Gan_E_vector(*args_Gan_E_vector,
                                           D_u, **kwargs, )
    S_u, theta_x_S_u, theta_y_S_u, \
    walk_off_angle, delta_sk_pz, PG_vz, \
    s, s_z, s_xy, g_p = Gan_S_vector(*args_Gan_D_vector,
                                     D_u, E_u, g_p, mode=3, **kwargs, )

    # %%

    info = "n_" + p + " 的大小、" + "k_" + p + ", S_" + p + "; D_" + p + ", E_" + p + " 的 方向 与 夹角"
    from fun_os import tree_print
    is_print and print(tree_print(is_end, add_level=add_level) + info)
    # %% inc
    is_print and print(tree_print() + "———————————————— inc, {} ————————————————".format(p))
    is_print and print(tree_print() + "n_{} - inc: n_{}(k_{}_inc) = {}".format(p, p, p, n1_inc))
    is_print and print(
        tree_print() + "{} 光 离散角 - inc：<k_{}_inc, S_{}_inc> = {} °".format(p, p, p, walk_off_angle_inc))
    is_print and print(
        tree_print() + "纵向 S_{}_inc 对 k_{}_inc 传播相位 超前系数 - z：δ_p<S_{}_inc, k_{}_inc> = {}".format(p, p, p, p,
                                                                                                  delta_sk_pz_inc))
    is_print and print(
        tree_print() + "横向 S_{}_inc 对 k_{}_inc 梯度相位 矢量 - z：δ_v<S_{}_inc, k_{}_inc> = {}".format(p, p, p, p,
                                                                                                delta_sk_vz_inc))
    is_print and print(tree_print() + "————————————————————————————————".format())
    # %%
    is_print and print(
        tree_print() + "k_{}_inc - θ_x,y: θ_x(k_{}_inc) = {} °, θ_y(k_{}_inc) = {} °".format(p, p, theta_x, p, theta_y))
    is_print and print(
        tree_print() + "S_{}_inc - θ_x,y: θ_x(S_{}_inc) = {} °, θ_y(S_{}_inc) = {} °".format(p, p, theta_x_S_u_inc, p,
                                                                                             theta_y_S_u_inc))
    # %%
    is_print and print(tree_print() + "————————————————————————————————".format())
    is_print and print(
        tree_print() + "D_{}_inc - θ,φ: θ(D_{}_inc) = {} °, φ(D_{}_inc) = {} °".format(p, p, theta_D_u_inc, p,
                                                                                       phi_D_u_inc))
    is_print and print(
        tree_print() + "E_{}_inc - θ,φ: θ(E_{}_inc) = {} °, φ(E_{}_inc) = {} °".format(p, p, theta_E_u_inc, p,
                                                                                       phi_E_u_inc))
    # %% z
    is_print and print(tree_print() + "———————————————— z, {} ————————————————".format(p))
    is_print and print(tree_print() + "n_{} - z: n_{}(k_{}_z) = {}".format(p, p, p, Get("n_z")))
    is_print and print(
        tree_print() + "{} 光 离散角 - z：<k_{}_z, S_{}_z> = {} °".format(p, p, p, walk_off_angle_0kx0ky))
    is_print and print(
        tree_print() + "纵向 S_{}_z 对 k_{}_z 传播相位 超前系数 - z：δ_p<S_{}_z, k_{}_z> = {}".format(p, p, p, p,
                                                                                          delta_sk_pz_0kx0ky))
    is_print and print(
        tree_print() + "横向 S_{}_z 对 k_{}_z 梯度相位 矢量 - z：δ_v<S_{}_z, k_{}_z> = {}".format(p, p, p, p,
                                                                                        delta_sk_vz_0kx0ky))
    # %%
    is_print and print(tree_print() + "————————————————————————————————".format())
    is_print and print(tree_print() + "k_{}_z - θ_x,y: θ_x(k_{}_z) = {} °, θ_y(k_{}_z) = {} °".format(p, p, 0, p, 0))
    is_print and print(
        tree_print() + "S_{}_z - θ_x,y: θ_x(S_{}_z) = {} °, θ_y(S_{}_z) = {} °".format(p, p, theta_x_S_0kx0ky, p,
                                                                                       theta_y_S_0kx0ky))
    # %%
    is_print and print(tree_print() + "————————————————————————————————".format())
    is_print and print(
        tree_print() + "D_{}_z - θ,φ: θ(D_{}_z) = {} °, φ(D_{}_z) = {} °".format(p, p, theta_D_u_0kx0ky, p,
                                                                                 phi_D_u_0kx0ky))
    is_print and print(
        tree_print() + "E_{}_z - θ,φ: θ(E_{}_z) = {} °, φ(E_{}_z) = {} °".format(p, p, theta_E_u_0kx0ky, p,
                                                                                 phi_E_u_0kx0ky))
    # %%
    is_print and print(tree_print(is_end2) + "————————————————————————————————".format())

    # print(D_u[0])
    # print(D_u[:,0])
    # print(D_u[0,0])
    g_oe = g_p * np.dot(E_u, p_p)  # 不能是 p_p * D_u，得是 D_u * p_p，因为 D_u 的 最末维度 是 2，而 p_p 的 第一个维度 也是 2
    # return n1_inc, n1, k1_inc, k1, k1_z, k1_xy, g_oe, E_u
    # return n1_inc, n1, k1_inc, k1, k1_z * delta_sk_pz, k1_xy, g_oe, E_u
    # return n1_inc, n1, k1_inc, s, s_z, s_xy, g_oe, E_u
    # return n1_inc, n1, k1_inc, s, s_z * delta_sk_pz, s_xy, g_oe, E_u
    return n1_inc, n1, k1_inc, k1, k1_z, k1_xy, g_oe, E_u, PG_vz


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
