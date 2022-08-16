# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

import math
import numpy as np


# %%

def mesh_shift(Ix, Iy, *args):
    nx, ny = np.meshgrid(range(Iy), range(Ix))
    # ny = ny[::-1]  # y 序号 上下颠倒 一下，转成 平面 笛卡尔 坐标系
    mesh_nx_ny = np.dstack((nx, ny))
    mesh_nx_ny_shift = mesh_nx_ny - (Iy // 2, Ix // 2)

    # print(len(arg))
    if len(args) >= 2:  # 确保 额外 传入了 theta_x, theta_y 两个参数

        theta_x = args[0] / 180 * math.pi
        theta_y = - args[1] / 180 * math.pi  # 笛卡尔 坐标系 转 图片 / 电脑 坐标系

        # print(mesh_nx_ny_shift[:, :, 0])
        # print(type(mesh_nx_ny_shift[0,0,0]))

        # mesh_nx_ny_shift[:, :, 0] = mesh_nx_ny_shift[:, :, 0] * np.cos(theta_x / 180 * math.pi) # python 像 go 一样，切片 只是引用，无法改变 被引用的 数组的 相应切片 的值？
        # mesh_nx_ny_shift[:, :, 1] = mesh_nx_ny_shift[:, :, 1] * np.cos(theta_y / 180 * math.pi) # 并不，确实是 改变了的，赋值是 深拷贝；引用是 浅拷贝

        # print(mesh_nx_ny_shift[:, :, 0])
        # print(type(mesh_nx_ny_shift[0,0,0])) # 破案了，tmd 是 int 的锅：右边 非 int 赋值给 左边 的 时候，左边的 数据类型 仍是 int，导致 右边的 计算结果 会强制转换为 int。
        # 额，不对，type(mesh_nx_ny_shift[0,0,0]) 总是 int，不是 数据类型 强制转换 成右侧的 int 的 缘故，
        # 而是 切片 并未引用 原来的，而是创建了 一块新的 内存区域，这样 并没有 改变 所引用的 原来的 区域的 数据
        # 而且 这里是 对数组 array 的 切片，不是 对 list 的；对 list 的 切片还是 list，然而 list 相当于 labview 的 簇，不能乘以 非整数，
        # 但 list 乘以 整数 也不是 每个元素 都乘以 整数，而是 将其 复制 几份后 并加入原来的 List

        mesh_nx_ny_shift = np.dstack(
            (mesh_nx_ny_shift[:, :, 0] * np.cos(theta_x), mesh_nx_ny_shift[:, :, 1] * np.cos(theta_y)))
    #  nx[y, x] 和 mesh_nx_ny_shift[:, :, 0][y, x] 均只与 x（列） 有关，但向右是增
    #  ny[y, x] 和 mesh_nx_ny_shift[:, :, 1][y, x] 均只与 y（行） 有关，但向下是增
    #  所以 生成的 mesh 与 图片 or 电脑 坐标系 是 共用同一个 坐标系的
    #  因此 该 mesh 可直接与 g 相乘，并帮 g 规定了，对于像素点的索引，横往右 x 是加， 纵往下 y 是加
    #  但这 只涉及 像素点的 索引 坐标系，不涉及 每个像素点 的 矢量，所在的 坐标系 ———— 后者所在的 3 个 维度，与前者所在的 2 个维度，坐标系 是独立的
    #  因此 只作用于 2 维索引的 Rotate 坐标系，不能 作用于 3 维的 矢量场 分布 的 x,y 分量上

    # %%  plot 一下，看看分布 是否是 笛卡尔坐标系
    # from fun_os import U_test_plot
    # method = "MESH"
    # U_test_plot(method, "ix", mesh_nx_ny_shift[:, :, 0])
    # U_test_plot(method, "iy", mesh_nx_ny_shift[:, :, 1])

    return mesh_nx_ny_shift


# %%
# 生成 rho_shift

def Generate_rho_shift(Ix=0, Iy=0, size_PerPixel=0.77,
                       theta_x=0, theta_y=0, ):
    # print(Ix, Iy)
    mesh_nx_ny_shift = mesh_shift(Ix, Iy,
                                  theta_x, theta_y)
    rho_shift = (mesh_nx_ny_shift[:, :, 0] ** 2 + mesh_nx_ny_shift[:, :, 1] ** 2 + 0j) ** 0.5 * size_PerPixel

    # 倾斜的 傍轴但不沿轴的 高斯光束，除了 乘以倾斜相位之外，（垂直于 系统 主光轴 的）横截面 还会变椭（这是 倾斜相位 没有的效应 ？）
    # 本质上 这是 矢量场 的问题，需要 矢量角谱解，且需要在 倒空间 解决问题

    # 先 标量或矢量场 FFT 到 倒空间 g
    # 再想办法 得到 倒空间的 g_xyz 三分量
    # 再执行 rotate 旋转操作，得到 旋转后的 g_xyz 三分量
    # 再对 g_xyz 三分量 傅立叶 逆变换

    # 才能得到 正确的 矢量 傍轴 但不沿轴的 倾斜 高斯解 —— 倾斜相位，和 这里的 实空间 横向 或 纵向扩展，只是 唯象的近似

    return rho_shift


def random_phase(Ix, Iy):
    return math.e ** ((np.random.rand(Ix, Iy) * 2 * math.pi - math.pi) * 1j)
