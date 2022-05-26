# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 00:42:22 2022

@author: Xcz
"""

import cv2
import math
import numpy as np
import scipy.stats
import inspect
from fun_os import img_squared_bordered_Read, U_Read, U_read_only, U_dir, U_plot_save
from fun_global_var import Get, init_accu, tree_print
from fun_img_Resize import img_squared_Resize
from fun_array_Generate import mesh_shift, Generate_r_shift, random_phase
from fun_linear import Cal_n, Cal_kz, fft2, ifft2
from fun_SSI import Cal_IxIy


# %%
# 生成 束腰 z = 0 处的 HG 光束（但不包括 复振幅的 高斯分布）

def HG_without_Gauss_profile(Ix=0, Iy=0, size_PerPixel=0.77,
                             w0=0,
                             m=1, n=0,
                             theta_x=1, theta_y=0, ):
    mesh_Ix0_Iy0_shift = mesh_shift(Ix, Iy,
                                    )

    C_HG_mn = 1
    x, y = 2 ** 0.5 * mesh_Ix0_Iy0_shift[:, :, 0] * size_PerPixel / w0, \
           2 ** 0.5 * mesh_Ix0_Iy0_shift[:, :, 1] * size_PerPixel / w0
    M, N = [0] * m, [0] * n
    M.append(m), N.append(n)
    HG_mn = C_HG_mn * np.polynomial.hermite.hermval(x, M) * np.polynomial.hermite.hermval(y, N)

    return HG_mn


# %%
# 生成 束腰 z = 0 处的 LG 光束（但不包括 复振幅的 高斯分布）

def LG_without_Gauss_profile(Ix=0, Iy=0, size_PerPixel=0.77,
                             w0=0,
                             l=1, p=0,
                             theta_x=1, theta_y=0, ):
    r_shift = Generate_r_shift(Ix, Iy, size_PerPixel,
                               theta_x, theta_y, )

    C_LG_pl = (2 / math.pi * math.factorial(p) / math.factorial(p + abs(l))) ** 0.5
    x = 2 ** 0.5 * r_shift / w0
    LG_pl = C_LG_pl / w0 * x ** abs(l) * scipy.special.genlaguerre(p, abs(l), True)(x ** 2)

    return LG_pl


# %%
# 生成 束腰 z = 0 处的 高斯光束

def Gauss(Ix=0, Iy=0, size_PerPixel=0.77,
          w0=0,
          theta_x=1, theta_y=0, ):
    r_shift = Generate_r_shift(Ix, Iy, size_PerPixel,
                               theta_x, theta_y, )

    if (type(w0) == float or type(w0) == np.float64 or type(
            w0) == int) and w0 > 0:  # 如果 传进来的 w0 既不是 float 也不是 int，或者 w0 <= 0，则 图片为 1
        U = np.power(math.e, - r_shift ** 2 / w0 ** 2)
    else:
        U = np.ones((Ix, Iy), dtype=np.complex128)

    return U


# %%
# 对 输入场 引入 高斯限制

def Gauss_profile(Ix=0, Iy=0, size_PerPixel=0.77,
                  U=0, w0=0,
                  theta_x=1, theta_y=0, ):
    if (type(w0) == float or type(w0) == np.float64 or type(
            w0) == int) and w0 > 0:  # 如果 传进来的 w0 既不是 float 也不是 int，或者 w0 <= 0，则表示 不对原图 引入 高斯限制

        r_shift = Generate_r_shift(Ix, Iy, size_PerPixel,
                                   theta_x, theta_y, )

        U = U * np.power(math.e, - r_shift ** 2 / w0 ** 2)

    return U


# %%
# 生成 纯相位 OAM

def OAM(Ix=0, Iy=0,
        l=1,
        theta_x=1, theta_y=0, ):
    mesh_Ix0_Iy0_shift = mesh_shift(Ix, Iy,
                                    theta_x, theta_y)
    U = np.power(math.e, l * np.arctan2(mesh_Ix0_Iy0_shift[:, :, 0], mesh_Ix0_Iy0_shift[:, :, 1]) * 1j)

    return U


# %%
# 对 输入场 引入额外螺旋相位

def OAM_profile(Ix=0, Iy=0,
                U=0,
                l=1,
                theta_x=0, theta_y=0, ):
    mesh_Ix0_Iy0_shift = mesh_shift(Ix, Iy,
                                    theta_x, theta_y)
    # print(U.shape, mesh_Ix0_Iy0_shift[:, :, 0].shape,mesh_Ix0_Iy0_shift[:, :, 1].shape)
    U = U * np.power(math.e, l * np.arctan2(mesh_Ix0_Iy0_shift[:, :, 0], mesh_Ix0_Iy0_shift[:, :, 1]) * 1j)

    return U


# θx 增大时，y = x 这个 45 度 的 线，会 越来越 偏向 x 轴 正向。

# %%
# 对 输入场 的 频谱 引入额外螺旋相位

def OAM_profile_G(Ix=0, Iy=0,
                  U=0,
                  l=1,
                  theta_x=1, theta_y=0, ):
    g_shift = fft2(U)

    g_shift = OAM_profile(Ix, Iy,
                          g_shift,
                          l,
                          theta_x, theta_y, )

    U = ifft2(g_shift)

    return U, g_shift


# %%
# 对 输入场 引入 额外的 倾斜相位

def incline_profile(Ix=0, Iy=0,
                    U=0, k_inc=0,
                    theta_x=1, theta_y=0, ):
    theta_y = - theta_y  # 笛卡尔 坐标系 转 图片 / 电脑 坐标系
    # 在空气中 倾斜，还是 在晶体中 倾斜，取决于 k 中的 n 是空气 还是 晶体的 折射率：
    # 同样的 倾角，n 不同，则积累的 空域 倾斜相位 梯度 不同
    # 其中 k 是 k 或 k_nxny，其正中 是 倒空间 正中
    # 各向异性晶体 中 积累的 倾斜相位 其实 也应是 各向异性的，这里做不到；似乎实际上 也做不到在晶体从倾斜：溯源都可归结到 在空气中倾斜
    # if type(k) != float and type(k) != np.float64 and type(k) != int:  # 如果 是 array，则 只取中心级 的 k
    #     k = k[Iy // 2, Ix // 2]  # 取中心级 的 k
    # %%
    theta_x = theta_x / 180 * math.pi
    theta_y = theta_y / 180 * math.pi  # 笛卡尔 坐标系 转 图片 / 电脑 坐标系
    # # %%  现实：无论先转 theta_x 还是先转 theta_y
    # kz = k_inc * math.cos(theta_x) * math.cos(theta_y)  # 通光方向 的 分量大小
    # # %%  现实：先转 theta_x 再转 theta_y
    # ky = k_inc * math.cos(theta_x) * math.sin(theta_y)
    # kx = k_inc * math.sin(theta_x)
    # # %%  现实：先转 theta_y 再转 theta_x
    # ky = k_inc * math.sin(theta_y)
    # kx = k_inc * math.sin(theta_x) * math.cos(theta_y)
    # return kx, ky, kz
    # %%  非现实
    Kx, Ky = k_inc * np.sin(theta_x), k_inc * np.sin(theta_y)
    # %%
    # 但这个 k 其实 只是 中心 k；或者说，上述 隐含了 球面 折射率 方程...
    # 椭球的话，kx,ky 的关系似乎得用 tan，但 tan 只在小角有效；45 度 就 1:1 了，也不对
    # Kx, Ky = k * np.tan(theta_x / 180 * math.pi), k * np.tan(theta_y / 180 * math.pi)

    mesh_Ix0_Iy0_shift = mesh_shift(Ix, Iy)
    # k_shift = (k**2 - Kx**2 - Ky**2 + 0j )**0.5
    U = U * np.power(math.e, (Kx * mesh_Ix0_Iy0_shift[:, :, 0] + Ky * mesh_Ix0_Iy0_shift[:, :, 1]) * 1j)

    return U


# mesh_Ix0_Iy0_shift[:, :, 0] 只与 第 2 个参数 有关，
# 则 对于 第 1 个参数 而言，对于 不同的 第 1 个参数，都 引入了 相同的 倾斜相位。
# 也就是 对于 同一列 的 不同的行，其 倾斜相位 是相同的
# 因此 倾斜相位 也 只与 列 相关，也就是 只与 第 2 个参数 有关，所以 与 x 有关。

# %%
# 对输入场 频域 引入 额外倾斜相位

def incline_profile_G(Ix=0, Iy=0,
                      U=0, k_inc=0,
                      theta_x=1, theta_y=0, ):
    g_shift = fft2(U)

    g_shift = incline_profile(Ix, Iy,
                              g_shift, k_inc,
                              theta_x, theta_y)
    # 本该 H_shift 的 e 指数 的 相位部分，还要 加上 k_shift * i1_z0 的，不过这里 i1_z0 = i1_0 = 0，所以加了 等于没加

    U = ifft2(g_shift)

    return U, g_shift


# %%
# 对输入场 频域 引入 传播相位

def propagation_profile_G(Ix=0, Iy=0, size_PerPixel=0.77,
                          U=0, k=0, z=0, ):
    g_shift = fft2(U)

    kz_shift, mesh_kx_ky_shift = Cal_kz(Ix, Iy, k)
    i_z0 = z / size_PerPixel
    H_z0_shift = np.power(math.e, kz_shift * i_z0 * 1j)

    G_z0_shift = g_shift * H_z0_shift
    U_z0 = ifft2(G_z0_shift)

    return U_z0, G_z0_shift


# %%
# 对输入场 空域 引入 传播相位

def propagation_profile_U(Ix=0, Iy=0, size_PerPixel=0.77,
                          U=0, k=0, z=0, ):
    kz_shift, mesh_kx_ky_shift = Cal_kz(Ix, Iy, k)
    i_z0 = z / size_PerPixel
    H_z0_shift = np.power(math.e, kz_shift * i_z0 * 1j)

    U_z0 = U * H_z0_shift
    G_z0_shift = ifft2(U_z0)

    return U_z0, G_z0_shift


# %%

def pump(Ix=0, Iy=0, size_PerPixel=0.77,
         Up=0, w0=0, k_inc=0, k=0, z=0,
         # %%
         is_LG=0, is_Gauss=1, is_OAM=1,
         l=1, p=0,
         theta_x=1, theta_y=0,
         is_random_phase=0,
         is_H_l=0, is_H_theta=0, is_H_random_phase=0,
         # %%
         is_save=0, is_save_txt=0, dpi=100,
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
         **kwargs, ):
    # %%
    # file_name = os.path.splitext(file_full_name)[0]
    # img_name_extension = os.path.splitext(file_full_name)[1]  # 都能获取了
    # size_fig = Ix / dpi  # 都能获取了
    # %%
    # 将输入场 改为 LG 光束

    if is_LG == 1:
        # 将 实空间 输入场 变为 束腰 z = 0 处的 LG 光束

        Up = LG_without_Gauss_profile(Ix, Iy, size_PerPixel,
                                      w0,
                                      l, p,
                                      theta_x, theta_y, )
    elif is_LG == 2:
        # 将 实空间 输入场 变为 束腰 z = 0 处的 HG 光束

        Up = HG_without_Gauss_profile(Ix, Iy, size_PerPixel,
                                      w0,
                                      l, p,
                                      theta_x, theta_y, )

    # %%
    # 对输入场 引入 高斯限制

    if is_Gauss == 1 and is_LG == 0:
        # 将 实空间 输入场 变为 束腰 z = 0 处的 高斯光束

        Up = Gauss(Ix, Iy, size_PerPixel,
                   w0,
                   theta_x, theta_y, )

    else:
        # 对 实空间 输入场 引入 高斯限制

        Up = Gauss_profile(Ix, Iy, size_PerPixel,
                           Up, w0,
                           theta_x, theta_y, )

    # %%
    # 对输入场 引入 额外的 螺旋相位

    if is_OAM == 1 and is_Gauss == 0:
        # 高斯则 乘以 额外螺旋相位，非高斯 才直接 更改原场：高斯 已经 更改原场 了
        # 将输入场 在实空间 改为 纯相位 的 OAM

        Up = OAM(Ix, Iy,
                 l,
                 theta_x, theta_y, )

    elif is_LG != 2:  # 只有 非厄米高斯时，l ≠ 0 时 才加 螺旋相位
        # 对输入场 引入 额外的 螺旋相位

        if is_H_l == 1:
            # 对 频谱空间 引入额外螺旋相位

            Up, G_z0_shift = OAM_profile_G(Ix, Iy,
                                           Up,
                                           l,
                                           theta_x, theta_y, )

        else:
            # 对 实空间 引入额外螺旋相位

            Up = OAM_profile(Ix, Iy,
                             Up,
                             l,
                             theta_x, theta_y, )

    # %%
    # 对输入场 引入 额外的 倾斜相位

    if is_H_theta == 1:
        # 对 频谱空间 引入额外倾斜相位

        Up, G_z0_shift = incline_profile_G(Ix, Iy,
                                           Up, k_inc,
                                           theta_x, theta_y, )

    else:
        # 对 实空间 引入额外倾斜相位

        Up = incline_profile(Ix, Iy,
                             Up, k_inc,
                             theta_x, theta_y)

    # Up = Up**2
    # %%
    # 对输入场 引入 传播相位

    if is_H_l == 1:

        # 对输入场 的 频谱 引入 额外的 螺旋相位（纯相位）， 并在 频域传播 一定距离（纯相位），之后 返回空域（其实就是 在空域传播 / 乘以 传递函数）
        # 由于是两次 纯相位操作，不会 改变 频域 或 空域 的 总能量
        # 其他 总能量守恒，但改变 频谱能量分布 的 操作，如 希尔伯特变换，可能也行（不一定 加 螺旋相位 后，再 频域传播）
        # 或者 先后进行多次 不同的 能量守恒 操作 也行

        U_z0, G_z0_shift = propagation_profile_U(Ix, Iy, size_PerPixel,
                                                 Up, k, z, )

    else:

        U_z0, G_z0_shift = propagation_profile_G(Ix, Iy, size_PerPixel,
                                                 Up, k, z, )

    # %%
    # 对输入场 引入 随机相位

    if is_random_phase == 1:

        if is_H_random_phase == 1:

            # G_z0_shift *= random_phase(Ix, Iy)
            G_z0_shift = G_z0_shift * random_phase(Ix, Iy)
            U_z0 = ifft2(G_z0_shift)

        else:

            # U_z0 *= random_phase(Ix, Iy)
            U_z0 = U_z0 * random_phase(Ix, Iy)
            G_z0_shift = fft2(U_z0)

    # %%
    # 绘图：g_0_amp

    method = "PUMP"
    cmap_2d = "inferno"  # "plasma", "magma", "inferno", "cividis",
    # "Reds", "BuPu"
    # "pink", "gist_heat",
    # "Spectral_r", "coolwarm", "seismic", "PuOr_r"
    # "gnuplot", "rainbow", "nipy_spectral", "gist_earth"

    # ray = kwargs['ray'] + "0" if "ray" in kwargs else "0"
    if inspect.stack()[1][3] == "pump_pic_or_U" or inspect.stack()[1][3] == "pump_pic_or_U2":
        ray = kwargs['ray_pump'] + "p" if "ray_pump" in kwargs else "p"
    elif inspect.stack()[1][3] == "pump_pic_or_U_structure":  # 如果 调用该 pump 的 函数，名为 这个
        ray = kwargs['ray_structure'] + "p" if "ray_structure" in kwargs else "p"
    else:
        ray = "p"
    # ray = kwargs['ray'] + "_p" if "ray" in kwargs else "_p"

    name = "G" + ray
    title = method + " - " + name

    # print(kwargs)
    kwargs.pop('U', None)  # 要想把 kwargs 传入 U_plot_save，kwargs 里不能含 'U'
    U_plot_save(G_z0_shift, title, 0,
                Get("img_name_extension"),
                # %%
                size_PerPixel,
                is_save, is_save_txt, dpi, Get("size_fig"),
                # %%
                cmap_2d, ticks_num, is_contourf,
                is_title_on, is_axes_on, is_mm,
                fontsize, font,
                # %%
                is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                # %%                          何况 一般默认 is_self_colorbar = 1...
                z=z, **kwargs, )

    # %%

    name = "U" + ray
    title = method + " - " + name

    U_plot_save(U_z0, title, 1,
                Get("img_name_extension"),
                # %%
                size_PerPixel,
                is_save, is_save_txt, dpi, Get("size_fig"),
                # %%
                cmap_2d, ticks_num, is_contourf,
                is_title_on, is_axes_on, is_mm,
                fontsize, font,
                # %%
                is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                # %%                          何况 一般默认 is_self_colorbar = 1...
                z=z, **kwargs, )

    folder_address = U_dir(title, is_save,
                           z=z, **kwargs, )

    return U_z0, G_z0_shift


def pump_pic_or_U(U_name="",
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
                  U_NonZero_size=1, w0=0.3,
                  # %%
                  lam1=0.8, is_air_pump=0, T=25,
                  # %%
                  is_save=0, is_save_txt=0, dpi=100,
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
                  is_print=1,
                  # %%
                  **kwargs, ):
    # %%
    kwargs['p_dir'] = 'PUMP'
    # %%
    if (type(U_name) != str) or U_name == "":

        # %%
        # 导入 方形，以及 加边框 的 图片

        img_name, img_name_extension, img_squared, \
        size_PerPixel, size_fig, Ix, Iy, U = \
            img_squared_bordered_Read(img_full_name,
                                      U_NonZero_size, dpi,
                                      is_phase_only)

        if "U" in kwargs:
            U = kwargs["U"]
        elif "U1" in kwargs:
            U = kwargs["U1"]
        else:
            # %%
            # 预处理 输入场

            n_inc, n, k_inc, k = Cal_n(size_PerPixel,
                                       is_air_pump,
                                       lam1, T, p=kwargs.get("polar", "e"),
                                       theta_x=theta_x,
                                       theta_y=theta_y, **kwargs)

            U, g_shift = pump(Ix, Iy, size_PerPixel,
                              U, w0, k_inc, k, z_pump,
                              is_LG, is_Gauss, is_OAM,
                              l, p,
                              theta_x, theta_y,
                              is_random_phase,
                              is_H_l, is_H_theta, is_H_random_phase,
                              is_save, is_save_txt, dpi,
                              ticks_num, is_contourf,
                              is_title_on, is_axes_on, is_mm,
                              fontsize, font,
                              is_colorbar_on, is_energy,
                              **kwargs, )

    else:

        # %%
        # 导入 方形 的 图片，以及 U

        img_name, img_name_extension, img_squared, \
        size_PerPixel, size_fig, Ix, Iy, U = U_Read(U_name,
                                                    img_full_name,
                                                    U_NonZero_size,
                                                    dpi,
                                                    is_save_txt, )
    g_shift = fft2(U)

    return img_name, img_name_extension, img_squared, \
           size_PerPixel, size_fig, Ix, Iy, \
           U, g_shift


def pump_pic_or_U2(U2_name="",
                   img2_full_name="Grating.png",
                   is_phase_only_2=0,
                   # %%
                   z_pump2=0,
                   is_LG_2=0, is_Gauss_2=0, is_OAM_2=0,
                   l2=0, p2=0,
                   theta2_x=0, theta2_y=0,
                   # %%
                   is_random_phase_2=0,
                   is_H_l2=0, is_H_theta2=0, is_H_random_phase_2=0,
                   # %%
                   U_NonZero_size=1, w0_2=0.3,
                   # %%
                   lam2=0.8, is_air_pump2=0, T2=25,
                   polar2='e',
                   # %%
                   is_save=0, is_save_txt=0, dpi=100,
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
                   is_print=1,
                   # %%
                   **kwargs, ):
    # %%
    kwargs['p_dir'] = 'PUMP - 2'
    # %%
    info = "pump_pic_or_U2"
    is_first = int(init_accu(info, 1) == 1)  # 若第一次调用 pump_pic_or_U_structure，则 is_first 为 1，否则为 0
    is_Print = is_print * is_first  # 两个 得都 非零，才 print

    info = "泵浦_2"
    is_Print and print(tree_print(kwargs.get("is_end", 0), kwargs.get("add_level", 0)) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    # %%
    # 导入 方形，以及 加边框 的 图片

    if (type(U2_name) != str) or U2_name == "":
        img2_name, img2_name_extension, img2_squared, \
        size_PerPixel_2, size_fig_2, I2x, I2y, U2 = img_squared_bordered_Read(img2_full_name,
                                                                              U_NonZero_size, dpi,
                                                                              is_phase_only_2)
    else:
        img2_name, img2_name_extension, img2_squared, \
        size_PerPixel_2, size_fig_2, I2x, I2y, U2 = U_Read(U2_name,  # 需要 用 U_Read 覆盖 size_PerPixel 等 Set 的 值
                                                           img2_full_name,
                                                           U_NonZero_size,
                                                           dpi,
                                                           is_save_txt, )

    # %%
    # 定义 调制区域 的 横向实际像素、调制区域 的 实际横向尺寸

    U2_NonZero_size_Enlarge = 0
    U2_NonZero_size = U_NonZero_size * (1 + U2_NonZero_size_Enlarge)
    is_Print and print(tree_print() + "U2_NonZero_size = {} mm".format(U2_NonZero_size))

    I2x_NonZero, I2y_NonZero, deff_structure_size = Cal_IxIy(Get("Ix"), Get("Iy"),
                                                             U2_NonZero_size, Get("size_PerPixel"),
                                                             is_Print)

    # %%
    # 需要先将 目标 U2_NonZero = img2_squared 给 放大 或 缩小 到 与 I2x_NonZero, I2y_NonZero 相同，才能开始 之后的工作
    # 最终结果就是 img2_squared_resize 与 img_squared 尺寸相同，也就是 I2x_NonZero, I2y_NonZero = Ix_NonZero, Iy_NonZero

    border_width, img2_squared_resize_full_name, img2_squared_resize = \
        img_squared_Resize(img2_full_name, img2_squared,
                           I2x_NonZero, I2y_NonZero, Get("Ix"),
                           is_Print, )

    # %% 补零后 再拿进去 Pump

    img2_squared_resize_bordered = np.pad(img2_squared_resize, ((border_width, border_width),
                                                                (border_width, border_width)),
                                          'constant', constant_values=(0, 0))

    # %%

    # %%
    if (type(U2_name) != str) or U2_name == "":

        if "U2" in kwargs:
            U2 = kwargs["U2"]
        else:
            if is_phase_only_2 == 1:
                U2 = np.power(math.e, (img2_squared_resize_bordered.astype(np.complex128())
                                       / 255 * 2 * math.pi - math.pi) * 1j)  # 变成相位图
            else:
                U2 = img2_squared_resize_bordered.astype(np.complex128)
            # %%
            # 预处理 输入场

            n2_inc, n2, k2_inc, k2 = Cal_n(Get("size_PerPixel"),
                                           is_air_pump2,
                                           lam2, T2, p=polar2,
                                           theta_x=theta2_x,
                                           theta_y=theta2_y, **kwargs)

            kwargs["is_end"] = 1
            U2, g2_shift = pump(Get("Ix"), Get("Iy"), Get("size_PerPixel"),
                                U2, w0_2, k2_inc, k2, z_pump2,
                                is_LG_2, is_Gauss_2, is_OAM_2,
                                l2, p2,
                                theta2_x, theta2_y,
                                is_random_phase_2,
                                is_H_l2, is_H_theta2, is_H_random_phase_2,
                                is_save, is_save_txt, dpi,
                                ticks_num, is_contourf,
                                is_title_on, is_axes_on, is_mm,
                                fontsize, font,
                                is_colorbar_on, is_energy,
                                **kwargs, )

    if ((type(U2_name) == str) and U2_name != "") or "U2" in kwargs:
        U2 = cv2.resize(np.real(U2), (Get("Ix"), Get("Iy")), interpolation=cv2.INTER_AREA) + \
             cv2.resize(np.imag(U2), (Get("Ix"), Get("Iy")), interpolation=cv2.INTER_AREA) * 1j
        # U2 必须 resize 为 Ix, Iy 大小；
        # 但 cv2 、 skimage.transform 中 resize 都能处理 图片 和 float64，
        # 但似乎 没有东西 能直接 处理 complex128，但可 分别处理 实部和虚部，再合并为 complex128
    g2_shift = fft2(U2)

    return U2, g2_shift


def pump_pic_or_U_structure(U_structure_name="",
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
                            # %%
                            lam1=0.8, is_air_pump=0, T=25,
                            # %%
                            is_save=0, is_save_txt=0, dpi=100,
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
                            is_print=1,
                            # %%
                            **kwargs, ):
    # %%
    kwargs['p_dir'] = 'PUMP - for_modulation'
    # %%
    info = "pump_pic_or_U_structure"
    is_first = int(init_accu(info, 1) == 1)  # 若第一次调用 pump_pic_or_U_structure，则 is_first 为 1，否则为 0
    is_Print = is_print * is_first  # 两个 得都 非零，才 print

    info = "泵浦_for_结构"
    is_Print and print(tree_print(kwargs.get("is_end", 0), kwargs.get("add_level", 0)) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # %%
    # 导入 方形，以及 加边框 的 图片

    if (type(U_structure_name) != str) or U_structure_name == "":
        img_name, img_name_extension, img_squared, \
        size_PerPixel, size_fig, Ix, Iy, U = img_squared_bordered_Read(img_full_name,
                                                                       U_NonZero_size, dpi,
                                                                       is_phase_only)
    else:
        img_name, img_name_extension, img_squared, \
        size_PerPixel, size_fig, Ix, Iy, U = U_Read(U_structure_name,  # 需要 用 U_Read 覆盖 size_PerPixel 等 Set 的 值
                                                    img_full_name,
                                                    U_NonZero_size,
                                                    dpi,
                                                    is_save_txt, )

    # %%
    # 定义 调制区域 的 横向实际像素、调制区域 的 实际横向尺寸

    deff_structure_size_expect = U_NonZero_size * (1 + structure_size_Enlarge)
    is_Print and print(tree_print() + "deff_structure_size_expect = {} mm".format(deff_structure_size_expect))

    Ix_structure, Iy_structure, deff_structure_size = Cal_IxIy(Ix, Iy,
                                                               deff_structure_size_expect, size_PerPixel,
                                                               is_Print)

    # %%
    # 需要先将 目标 U_NonZero = img_squared 给 放大 或 缩小 到 与 全息图（结构） 横向尺寸 Ix_structure, Iy_structure 相同，才能开始 之后的工作

    border_width, img_squared_resize_full_name, img_squared_resize = \
        img_squared_Resize(img_full_name, img_squared,
                           Ix_structure, Iy_structure, Ix,
                           is_Print, )

    if (type(U_structure_name) != str) or U_structure_name == "":
        # %%
        # U = U(x, y, 0) = img_squared_resize

        if "U_structure" in kwargs:
            U = kwargs["U_structure"]
        else:
            if is_phase_only == 1:
                U_structure = np.power(math.e,
                                       (img_squared_resize.astype(
                                           np.complex128()) / 255 * 2 * math.pi - math.pi) * 1j)  # 变成相位图
            else:
                U_structure = img_squared_resize.astype(np.complex128)

            # %%
            # 预处理 输入场

            n_inc, n, k_inc, k = Cal_n(size_PerPixel,
                                       is_air_pump,
                                       lam1, T, p=kwargs.get("polar_structure", "e"),
                                       theta_x=theta_x,
                                       theta_y=theta_y,
                                       Ix_structure=Ix_structure,
                                       Iy_structure=Iy_structure, **kwargs)

            kwargs["is_end"] = 1
            U_structure, g_shift_structure = pump(Ix_structure, Iy_structure, size_PerPixel,
                                                  U_structure, w0, k_inc, k, z_pump,
                                                  is_LG, is_Gauss, is_OAM,
                                                  l, p,
                                                  theta_x, theta_y,
                                                  is_random_phase,
                                                  is_H_l, is_H_theta, is_H_random_phase,
                                                  is_save, is_save_txt, dpi,
                                                  ticks_num, is_contourf,
                                                  is_title_on, is_axes_on, is_mm,
                                                  fontsize, font,
                                                  is_colorbar_on, is_energy,
                                                  **kwargs, )

    if ((type(U_structure_name) == str) and U_structure_name != "") or "U_structure" in kwargs:
        U_structure = cv2.resize(np.real(U), (Ix_structure, Iy_structure), interpolation=cv2.INTER_AREA) + \
                      cv2.resize(np.imag(U), (Ix_structure, Iy_structure), interpolation=cv2.INTER_AREA) * 1j
        # U 必须 resize 为 Ix_structure, Iy_structure 大小；
        # 但 cv2 、 skimage.transform 中 resize 都能处理 图片 和 float64，
        # 但似乎 没有东西 能直接 处理 complex128，但可 分别处理 实部和虚部，再合并为 complex128
    g_shift_structure = fft2(U_structure)

    return img_name, img_name_extension, img_squared, \
           size_PerPixel, size_fig, Ix, Iy, \
           Ix_structure, Iy_structure, deff_structure_size, \
           border_width, img_squared_resize_full_name, img_squared_resize, \
           U_structure, g_shift_structure
