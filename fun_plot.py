# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 00:42:22 2022

@author: Xcz
"""

import os
import math
import numpy as np

# import matplotlib.ticker as mticker
import matplotlib as mpl
# mpl.use('Agg') # 该句 需要在 夹在中间使用。
import matplotlib.pyplot as plt
# from mpl_toolkits import ax1_grid1
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline, interp1d, interp2d, griddata
from fun_algorithm import find_nearest, remove_elements
from fun_statistics import find_data_1d_level


# plt.rcParams['xtick.direction'] = 'out' # 设置刻度线在坐标轴内
# plt.rcParams['ytick.direction'] = 'out' # 一次设置，全局生效，所以：记得关闭

def lormat(ticks, Str, Max, **kwargs):
    if "%" in Str:  # 返回 float list
        if kwargs.get("reverse", 0) == 1:
            z_ticks = [float(Str % z) for z in ticks]
            return [float(Str % (Max - z)) for z in z_ticks], z_ticks
        else:
            return [float(Str % z) for z in ticks], [float(Str % z) for z in ticks]
        # 当 '%.3f' = '%.3f' 时， 这玩意 等价于 Get('.1e') = '.2e' 的 下述
        # 但只当 个位数 不是零的时候：.2f 是保留 2 位小数，不计入 个位数（小数点后 的 位数一致，比较整齐）
        # .2e 是保留 3 位有效数字，如果个位数是零，则保留 3 位小数
        # [float(format(z, Get('.1e'))) for z in ticks]
    else:  # 返回 float list 作为 ticks，同时返回 字符串 list 作为 tickslabels
        # return [float(format(z, '.1e')) for z in ticks], [format(z, Get('.1e')) for z in ticks]
        if kwargs.get("reverse", 0) == 1:
            z_ticks = [float(format(z, Str)) for z in ticks]
            return [float(format(Max - z, Str)) for z in z_ticks], [format(z, Str) for z in ticks]
        else:
            return [float(format(z, Str)) for z in ticks], [format(z, Str) for z in ticks]


def gan_ticks(Max, ticks_num, Min=0, is_centered=0, **kwargs):
    Str = format((Max - Min) / ticks_num, '.0e')
    # print(Str, Max, Min)
    if Str != 'nan':
        if int(Str[0]) == 3:  # step 默认为 更小的 step，比如 3 则 2，9 则 8，7 则 6，
            Str = "2" + Str[1:]
        elif int(Str[0]) == 7:  # = 8, 6, 5, 4, 2, 1 则不理睬
            Str = "6" + Str[1:]
        elif int(Str[0]) == 9:
            Str = "8" + Str[1:]
        step = float(Str) if float(Str) != 0 else 1  # 保留 1 位有效数字
        ticks_num_real = (Max - Min) // step if step != 0 else 6
        ticks = np.arange(0, ticks_num_real + 1, 1)
        gan_tickslabels = ticks * step
        if is_centered == 1:
            Average = (Max + Min) / 2
            Center_divisible = Average // step * step
            # Center_divisible = (-Average) // step * step * (-1)
            gan_tickslabels -= Center_divisible
            gan_ticks = gan_tickslabels + Average  # 把 0 放中间
            if abs(Max) < 10:
                gan_tickslabels = lormat(gan_tickslabels, '%.3f', Max, **kwargs)[0]
            else:
                gan_tickslabels = lormat(gan_tickslabels, '%.1f', Max, **kwargs)[0]
        else:
            Min_divisible = (-Min) // step * step * (-1)
            # Min_divisible = Min // step * step
            # print(Min_divisible)
            # 连 (Max - Min) 除以 step 都有余， 更何况 Min
            # _Min = np.sign(Min) * (abs(Min) // step + int(np.sign(Min)==-1)) * step
            # 额，我发现 np.sign(Min) * (abs(Min) // step + int(np.sign(Min)==-1)) = Min // step
            # _Min = Min // step * step # 负 得更多，正 得更少，保证 _Min < Min，这样才能 在图上显示 Min ？
            # 不，恰恰相反，图上的 Min 比 轴上的 _Min 更靠左，才能把 轴上的 _Min 显示出来（但得 xlim 和 ylim 比 xyticks 的 左右边界 稍大）
            gan_tickslabels += Min_divisible  # 注意 * 的 优先级比 // 高
            # if abs(Max) >= 1e3 or abs(Min) >= 1e3 or abs(Max) <= 1e-2 or abs(Min) <= 1e-2:
            if abs(Max) >= 1e3 or abs(Max) < 1e-2:
                gan_ticks, gan_tickslabels = lormat(gan_tickslabels, '.1e', Max, **kwargs)
            elif abs(Max) < 10:
                gan_ticks, gan_tickslabels = lormat(gan_tickslabels, '%.3f', Max, **kwargs)
            else:
                gan_ticks, gan_tickslabels = lormat(gan_tickslabels, '%.1f', Max, **kwargs)
        gan_ticks = [z / Max * kwargs.get("I", Max) for z in gan_ticks]
        # size_PerPixel 基本只适用于 x 轴 居中，且 is_centered=1 的情况，z 轴 不适用？
    else:
        gan_ticks = gan_tickslabels = np.arange(0, ticks_num + 1, 1)
    return gan_ticks, gan_tickslabels


def mjrFormatter_sci(x, pos):
    x_str = format(x, '.1e')
    # print(x_str)
    e_before = x_str.split('e')[0]
    e_after = x_str.split('e')[1]
    # sci = e_before + " × " + "$10^{{{0}}}$".format("%.f" % int(e_after))  # "%.f" 是 四舍五入 取整
    sci = e_before + " · " + "$10^{{{0}}}$".format("%.f" % int(e_after))  # "%.f" 是 四舍五入 取整
    return sci


# %%

def mjrFormatter_log(x, pos):
    return "$10^{{{0}}}$".format("%.1f" % x)  # 奇了怪了， x 本身已经是 格式化过了的，咋还得格式化一次...


def convert_inf_to_min(array):  # 防止 绘图 纵坐标 遇 inf 无法解析，并 正确生成 tickslabel
    array_min = min(remove_elements(array, -float('inf')))
    list = [array_min if array[i] == -float('inf') else array[i] for i in range(len(array))]
    return np.array(list)  # 转成数组


def convert_inf_to_min_new(array):  # 支持 2D array 了
    Max = np.max(array)
    if Max == -float('inf'):  # 如果最大值都是 log(0) = -无穷，则全场为 0 ，则 设为 - 50 吧。
        convert_inf_to_min = np.where(array == -float('inf'), -50, array)
    else:
        convert_inf_to_maxer = np.where(array == -float('inf'), Max + 6, array)
        Min = np.min(convert_inf_to_maxer)
        convert_inf_to_min = np.where(convert_inf_to_maxer == Max + 6, Min, array)
    return convert_inf_to_min


def log10_include_0(array):
    array = np.log10(array)
    array = convert_inf_to_min_new(array)
    return array


def log10_include_0_ax_yscale(array1D_new, **kwargs, ):
    if kwargs.get("ax_yscale", "linear") != 'linear':  # 无论如何， 第 2 个 坐标系 上的 第 3 条 误差曲线，都默认取 log
        array1D_new = log10_include_0(array1D_new)
    return array1D_new


def log10_include_0_ax1_xticklabel(array1D_new, **kwargs, ):
    if 'ax1_xticklabel' in kwargs:  # 如果 强迫 ax1 的 x 轴标签 保持原样（则看的是 随 dk 的 演化，则 能量 也得 log）
        array1D_new = log10_include_0_ax_yscale(array1D_new, **kwargs, )
    return array1D_new


# %%

def UnivariateSpline_to(array1D, sample,
                        ix, ix_new, ):
    if sample > 1:  # 我发现 哪怕 sample == 1，也会导致 被 插值作用，导致 原始值 被改变（不是说好了过每个点么...）
        f = UnivariateSpline(ix, array1D, s=0)  # ix 必须是 严格递增的，若 ix 是 zj 的话，zj 也必须是
        array1D_new = f(ix_new)
    else:
        array1D_new = array1D
    return array1D_new


def energy_log10_ax_yscale(array1D_new, is_energy, **kwargs, ):
    array1D_new = array1D_new if is_energy != 1 else np.abs(array1D_new) ** 2
    array1D_new = log10_include_0_ax_yscale(array1D_new, **kwargs, )
    return array1D_new


def energy_log10_ax1_xticklabel(array1D_new, is_energy, **kwargs, ):
    array1D_new = array1D_new if is_energy != 1 else np.abs(array1D_new) ** 2
    array1D_new = log10_include_0_ax1_xticklabel(array1D_new, **kwargs, )
    return array1D_new


def gan_l2_new_error(l2_new, array1D_new,
                     ix_new, ix2_new, **kwargs):
    index = [find_nearest(ix_new, goal)[0] for goal in ix2_new]
    # print(index)
    l2_new_error = np.abs(l2_new - array1D_new[index])  # 花式索引，可以用 list 或 array 作为一个 array 的下标
    l2_new_error = log10_include_0_ax1_xticklabel(l2_new_error, **kwargs, )
    return l2_new_error


# %%

def gan_len_zj_resampled(zj, sample):
    Iz = len(zj)
    Iz_new = (Iz - 1) * sample + 1  # zj 区间范围 保持不变，分段数 乘以 sample 后，新划分出的 刻度的个数
    return Iz, Iz_new


def gan_zj_resampled(zj, Iz_new):
    ix = zj
    ix_new = np.linspace(zj[0], zj[-1], Iz_new)
    return ix, ix_new


def gan_ix_non_resampled(Ix):
    ix = range(Ix)
    ix_new = ix
    # 要么都 range，要么都 np.linspace(0, Ix - 1, Ix)  # 非传播 则 不对某个方向，偏爱地 重/上采样
    return ix, ix_new


# %% 注意：所有带 ax 对象 作为参数 进去的 def 内，对其进行操作，没用

# def set_ax_xticks_xticklabels(ax, xticks, xticklabels,
#                               fontsize, font, ):
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
#
#
# def set_ax_yticks_yticklabels(ax, yticks, yticklabels,
#                               fontsize, font, ):
#     ax.set_xticks(yticks)
#     ax.set_xticklabels(yticklabels, fontsize=fontsize, fontdict=font)


# %% 注意：所有带 ax 对象 作为参数 进去的 def 内，对其进行操作，没用

def is_mjrFormatter_sci(ticklabels):
    return len(ticklabels) > 1 and (np.max(np.abs([float(str) for str in ticklabels])) >= 1e3 or np.max(
        np.abs([float(str) for str in ticklabels])) < 1e-2)


# def ax_xticklabels_mjrFormatter_sci(ax, xticklabels):
#     if is_mjrFormatter_sci(xticklabels):
#         ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))
#
#
# def ax_yticklabels_mjrFormatter_sci(ax, yticklabels):
#     if is_mjrFormatter_sci(yticklabels):
#         ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))

# %%

def format_dict(fontsize, font, size_enlarge=1.0):
    return {"fontsize": fontsize * size_enlarge, "fontdict": font}


# %%

def plot_1d(zj, sample=1, size_PerPixel=0.007,
            # %%
            array1D=0, array1D_address=os.path.dirname(os.path.abspath(__file__)), array1D_title='',
            # %%
            is_save=0, dpi=100, size_fig_x=3, size_fig_y=3,
            # %%
            color_1d='b', ticks_num=6, is_title_on=1, is_axes_on=1, is_mm=1, is_propagation=0,
            # %%
            fontsize=9,
            font={'family': 'Times New Roman',  # 'serif'
                  'style': 'normal',  # 'normal', 'italic', 'oblique'
                  'weight': 'normal',
                  'color': 'black',  # 'black','gray','darkred'
                  },
            # %%
            is_energy=0,
            # %% 可选 参数（可不传入）
            xlabel='', ylabel='', xlabel2='', ylabel2='', **kwargs, ):
    fontsize += 10  # plot_1d 的 图，稍微要大点，所以 字号 得大点
    # print(size_fig_x)
    # %%
    # fig, ax1 = plt.subplots(1, 1, figsize=(size_fig_x, size_fig_y), dpi=dpi)
    fig = plt.figure(figsize=(size_fig_x, size_fig_y), dpi=dpi)
    fontsize_set = size_fig_y * 4 if type(fontsize) != int else fontsize
    title_fontsize_enlarge = 1.3

    ax1 = fig.add_subplot(111, label="1")
    ax1.spines["left"].set_color(color_1d)  # 修改 左侧 坐标轴线 颜色
    # ax1.spines["bottom"].set_color(color_1d)  # 修改 下边 坐标轴线 颜色
    # ax1.tick_params(axis='x', colors=color_1d) # 刻度线 本身 的 颜色
    # ax1.tick_params(axis='y', colors=color_1d) # 刻度线 本身 的 颜色

    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    Ix = array1D.shape[0]
    Iz, Iz_new = gan_len_zj_resampled(zj, sample)

    # %% 插值 begin

    if is_propagation != 0:
        ix, ix_new = gan_zj_resampled(zj, Iz_new)
    else:
        ix, ix_new = gan_ix_non_resampled(Ix)  # 非传播 则 不对某个方向，偏爱地 重/上采样

    # kind = 'cubic' # kind = 0,1,2,3 nono，1 维才可以这么写，2 维只有 'linear', 'cubic', 'quintic'
    # f = interp1d(ix, array1D, kind = kind)

    # print(ix)
    # print(array1D)
    array1D_new = UnivariateSpline_to(array1D, sample, ix, ix_new, )
    array1D_new = energy_log10_ax1_xticklabel(array1D_new, is_energy, **kwargs, )

    if "l2" in kwargs:
        if "zj2" in kwargs:  # 如果 zj2 在，则以 zj2 为 xticks
            Iz2, Iz2_new = gan_len_zj_resampled(kwargs["zj2"], sample)
            ix2, ix2_new = gan_zj_resampled(kwargs["zj2"], Iz2_new)
        else:
            ix2_new = ix_new

        l2_new = UnivariateSpline_to(kwargs['l2'], sample, ix2, ix2_new, )
        l2_new = energy_log10_ax1_xticklabel(l2_new, is_energy, **kwargs, )
        l2_new_error = gan_l2_new_error(l2_new, array1D_new,
                                        ix_new, ix2_new, **kwargs)

        if 'l3' in kwargs:
            l3_new = UnivariateSpline_to(kwargs['l3'], sample, ix2, ix2_new, )
            l3_new = energy_log10_ax_yscale(l3_new, is_energy, **kwargs, )

    # %%

    plt.xticks(rotation=kwargs.get("xticklabels_rotate", 0))
    if is_axes_on == 0:
        ax1.axis('off')
    else:
        # if len(zj) != 0: xticks_z = np.linspace(zj[0], zj[-1], ticks_num + 1)
        # if Ix != 0: xticks_x = range(- Ix // 2, Ix - Ix // 2, Ix // ticks_num)
        # xticks = range(0, Iz, Iz // ticks_num)
        # if len(zj) != 0: xticks_z = gan_ticks(zj[-1], ticks_num, Min=zj[0], is_centered=0)
        # if Ix != 0: xticks_x = gan_ticks(Ix*size_PerPixel, ticks_num, Min=0, is_centered=1)
        if is_mm == 1:  # round(i * size_PerPixel,2) 保留 2 位小数，改为 保留 2 位 有效数字
            if is_propagation != 0:
                if "ax1_xticklabel" in kwargs:  # 如果传了 第2个x轴的 label 即 "ax2_xticklabel" 进来（通常是 非线性的），
                    xticks = kwargs["ax1_xticklabel"]  # 这个一般是 zj
                    xticklabels = [float('%.3f' % i) for i in xticks]
                    ax1.set_xticks(xticks)  # 则 第1个x轴的 label 需要与之 对齐，则保留原汁原味的 zj 作为 刻度 和 刻度的 label。
                    ax1.set_xticklabels(xticklabels, **format_dict(fontsize_set, font))
                else:
                    xticks, xticklabels = gan_ticks(ix_new[-1], ticks_num, Min=ix_new[0])
                    ax1.set_xticks(xticks)
                    ax1.set_xticklabels(xticklabels, **format_dict(fontsize_set, font))
            else:
                xticks, xticklabels = gan_ticks(Ix * size_PerPixel, ticks_num, is_centered=1)
                ax1.set_xticks(xticks)
                ax1.set_xticklabels(xticklabels, **format_dict(fontsize_set, font))
        else:
            xticks, xticklabels = gan_ticks(Iz, ticks_num)
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticklabels, **format_dict(fontsize_set, font))

        # if 'ax1_yscale' in kwargs or 'ax1_xticklabel' in kwargs:
        #     # ax1.set_yscale(kwargs.get('ax1_yscale', 'log'))
        #     # ax1.semilogy(x, np.log10(y))

        if "l2" in kwargs and 'l3' in kwargs:  # 如果 ax1 上要绘制 3 条曲线
            vmax = kwargs.get("vmax", max(np.max(array1D_new), np.max(l2_new), np.max(l2_new_error)))
            vmin = kwargs.get("vmin", min(np.min(array1D_new), np.min(l2_new), np.min(l2_new_error)))
        elif "l2" in kwargs and 'l3' not in kwargs:  # 如果 ax1 上 只画 2 条 能量曲线 时，需要 给 min 补个零，防止 ganticks 的时候，不从 0 开始
            vmax = kwargs.get("vmax", np.max(array1D_new))
            vmin = kwargs.get("vmin", np.min(array1D_new)) if kwargs.get("ax_yscale", "linear") != 'linear' else 0
        else:
            vmax = kwargs.get("vmax", np.max(array1D_new))
            vmin = kwargs.get("vmin", np.min(array1D_new))

        ax1_yticks, ax1_yticklabels = gan_ticks(vmax, ticks_num, Min=vmin)
        ax1.set_yticks(ax1_yticks)
        ax1.set_yticklabels(ax1_yticklabels, **format_dict(fontsize_set, font))

        if is_mjrFormatter_sci(xticklabels):
            ax1.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))
        if is_mjrFormatter_sci(ax1_yticklabels):
            ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))
        if 'ax1_xticklabel' in kwargs:  # 如果 强迫 ax1 的 x 轴标签 保持原样（则看的是 随 dk 的 演化，则 能量 也得 log）
            if kwargs.get("ax_yscale", "linear") != 'linear':
                # logfmt = mpl.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)
                # ax1.yaxis.set_major_formatter(logfmt)
                ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_log))

        ax1.set_xlabel(xlabel, **format_dict(fontsize_set, font))  # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        ax1.set_ylabel(ylabel, **format_dict(fontsize_set, font))  # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize

    # %% 画 第 1 条 曲线

    ax1_plot_dict = {"color": color_1d, "label": kwargs.get('label', None)}
    # 如果有第 2 个 ax 加进来，则提高 第 1 个 ax 的 透明度
    ax1_plot_dict.update({"alpha": kwargs.get("ax1_alpha", 0.5 if "l2" in kwargs else 1),
                          "linestyle": kwargs.get("ax1_linestyle", '-'),  # 线型
                          "linewidth": kwargs.get("ax1_linewidth", int(fontsize_set / 10) + 1)})  # 线宽
    ax1_plot_dict.update({"marker": kwargs.get("ax1_marker", ''),  # 标记点：'+' 'x' '.' '|' ''
                          "markeredgecolor": kwargs.get("ax1_markeredgecolor", color_1d),  # 标记点颜色 ‘green’
                          "markersize": kwargs.get("ax1_markersize", str(int(fontsize_set))),  # 标记点大小
                          "markeredgewidth": kwargs.get("ax1_markeredgewidth", int(fontsize_set / 10) + 1), })  # 标记点边宽

    # ax1.set_yscale(kwargs.get('ax1_yscale', 'linear')) # linear 会覆盖 之前的 set_yticks，如果该语句在 set_yticks 之后的话

    l1, = ax1.plot(ix_new, array1D_new, **ax1_plot_dict)
    ax1.grid()

    # %% 画 第 2 条 曲线

    legend_dict = {'loc': kwargs.get("loc", 0),  # 5: ‘right’ （右边中间），0: "best" 右上角（默认）
                   'fontsize': fontsize, }
    if "l2" in kwargs:
        ax2 = fig.add_subplot(111, label="2", frameon=False)  # 不覆盖 下面的 图层
        ax2.xaxis.tick_top()
        ax2.yaxis.tick_right()
        ax2.xaxis.set_label_position('top')
        ax2.yaxis.set_label_position('right')

        color_1d2 = kwargs.get("color_1d2", color_1d)
        ax1.spines["right"].set_color(color_1d2)  # 修改 右侧 坐标轴线 颜色
        # ax1.spines["top"].set_color(color_1d2)  # 修改 上边 坐标轴线 颜色
        # ax2.tick_params(axis='x', colors=color_1d2) # 刻度线 本身 的 颜色
        # ax2.tick_params(axis='y', colors=color_1d2) # 刻度线 本身 的 颜色

        plt.xticks(rotation=kwargs.get("xticklabels_rotate", 0))
        if is_axes_on == 0:
            ax2.axis('off')
        else:
            if "ax2_xticklabel" in kwargs:
                xticklabels = [float('%.3f' % i) for i in kwargs["ax2_xticklabel"]]
                if is_axes_on == 0:
                    ax2.axis('off')
                else:
                    #  这里的 xticks 自动是 zj、ix_new、Ix 等，is_mm、is_propagation 都包含进了
                    ax2.set_xticks(xticks)  # ax2 是 Tz，不像 dkzQ，是非线性变化的，所以不能人工 gan 其刻度，也不能有 ix2_new。
                    ax2.set_xticklabels(xticklabels, **format_dict(fontsize_set, font))
            else:
                ax2.set_xticks(())  # 否则 ax2 的 x 不设刻度

            # if 'ax2_yscale' in kwargs or 'ax1_xticklabel' in kwargs:
            #     # ax2.set_yscale(kwargs.get('ax2_yscale', 'log'))
            #     # ax2.semilogy(x, np.log10(y))

            if 'l3' in kwargs:
                vmax2 = kwargs.get("vmax2", np.max(l3_new))
                vmin2 = kwargs.get("vmin2", np.min(l3_new)) if kwargs.get("ax_yscale", "linear") != 'linear' else 0
                # print(vmax2, vmin2)
            else:
                if kwargs.get("is_energy_normalized", False) == 2:  # 如果要画 随 T 的 演化
                    # ax2.set_ylim(ax1.get_ylim())  # ax2 的 y 轴范围 不再自动，而是 强制 ax2 的 y 轴 范围 等于 ax1 的 y 轴范围
                    vmax2, vmin2 = vmax, vmin  # 与 ax2.set_ylim(ax1.get_ylim()) 配合，强制 ax2 的 y 轴 刻度线 等于 ax1 的 刻度线。
                else:  # 如果要画 随 T 的 演化
                    vmax2 = kwargs.get("vmax2", max(np.max(l2_new), np.max(l2_new_error)))
                    if kwargs.get("ax_yscale", "linear") != 'linear':
                        vmin2 = kwargs.get("vmin2", min(np.min(l2_new), np.min(l2_new_error)))
                    else:
                        vmin2 = 0  # 这样才能使 ticks 和 labels 的 第一个元素 是 0

            ax2_yticks, ax2_yticklabels = gan_ticks(vmax2, ticks_num, Min=vmin2)
            ax2.set_yticks(ax2_yticks)
            ax2.set_yticklabels(ax2_yticklabels, **format_dict(fontsize_set, font))

            if is_mjrFormatter_sci(xticklabels):
                ax2.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))
            if is_mjrFormatter_sci(ax2_yticklabels):
                ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))
            if 'l3' in kwargs:
                if kwargs.get("ax_yscale", "linear") != 'linear':
                    # logfmt = mpl.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)
                    # ax2.yaxis.set_major_formatter(logfmt)
                    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_log))

            ax2.set_xlabel(xlabel2, **format_dict(fontsize_set, font))
            ax2.set_ylabel(ylabel2, **format_dict(fontsize_set, font))

        ax2_plot_dict = {"color": color_1d2, "label": kwargs.get('label2', None)}
        ax2_plot_dict.update({"alpha": kwargs.get("ax2_alpha", 1),  # 1 即 不透明
                              "linestyle": kwargs.get("ax2_linestyle", '-'),  # 线型
                              "linewidth": kwargs.get("ax2_linewidth", int(fontsize_set / 10) + 1), })  # 线宽
        ax2_marker_dict = {"marker": kwargs.get("ax2_marker", '|'),  # 标记点
                           "markeredgecolor": kwargs.get("ax2_markeredgecolor", 'purple'),  # 标记点颜色
                           "markersize": kwargs.get("ax2_markersize", str(int(fontsize_set))),  # 标记点大小
                           "markeredgewidth": kwargs.get("ax2_markeredgewidth", int(fontsize_set / 10) + 1), }
        ax2_plot_dict.update(ax2_marker_dict)

        # %% ax1
        ax1_plot_dict.update({"label": kwargs.get('label2', None),
                              "linestyle": kwargs.get("ax2_linestyle", '--'), })
        ax1_plot_dict.update(ax2_marker_dict)
        l2, = ax1.plot(ix2_new, l2_new, **ax1_plot_dict, )
        if 'l3' in kwargs:
            # %%
            ax1_plot_dict.update({"label": "energy_error",
                                  "linestyle": kwargs.get("l2_error_linestyle", '-.'), })

            l2_error, = ax1.plot(ix2_new, l2_new_error, **ax1_plot_dict, )
            # %% ax2
            ax2_plot_dict.update({"label": kwargs.get('label3', None),
                                  "linestyle": kwargs.get("l2_error_linestyle", '-.'),
                                  "marker": kwargs.get("l2_error_marker", 'x'),
                                  "markeredgecolor": kwargs.get("l3_markeredgecolor", 'orange'), })
            l3, = ax2.plot(ix2_new, l3_new, **ax2_plot_dict, )
            # %%  80 % 误差线
            l3_new = 10 ** l3_new if kwargs.get("ax_yscale", "linear") != 'linear' else l3_new
            l3_level, real_level_percentage = find_data_1d_level(l3_new, kwargs.get("l3_level", 0.8))
            l3_y = np.log10(l3_level) if kwargs.get("ax_yscale", "linear") != 'linear' else l3_level
            l3_label = (' = ' + str(float('%.3f' % l3_level))) if l3_level > 0.001 else ""
            # l3_label = ' = ' + "{:.2}".format(l3_level)
            line_plot_dict = {"linestyle": kwargs.get("l2_error_linestyle", '-.'),  # 线型
                              "color": kwargs.get("l3_markeredgecolor", 'orange'),
                              "label": str(format(real_level_percentage * 100, '.2f')) +
                                       ' %' + ' data\'s_error < ' +
                                       str(mjrFormatter_sci(l3_level, 0)) + l3_label}
            l3_hline = ax2.axhline(y=l3_y, **line_plot_dict)
            # %%  100 % 误差线
            l3_level2, real_level_percentage2 = find_data_1d_level(l3_new, kwargs.get("l3_level2", 1))
            l3_y2 = np.log10(l3_level2) if kwargs.get("ax_yscale", "linear") != 'linear' else l3_level2
            l3_label2 = (' = ' + str(float('%.3f' % l3_level2))) if l3_level2 > 0.001 else ""
            line_plot_dict = {"linestyle": kwargs.get("l2_error_linestyle", '-.'),  # 线型
                              "color": kwargs.get("l3_markeredgecolor", 'green'),
                              "label": str(format(real_level_percentage2 * 100, '.2f')) +
                                       ' %' + ' data\'s_error < ' +
                                       str(mjrFormatter_sci(l3_level2, 0)) + l3_label2}
            l3_hline2 = ax2.axhline(y=l3_y2, **line_plot_dict)
        else:
            ax2_plot_dict.update({"label": "energy_error",
                                  "linestyle": kwargs.get("l2_error_linestyle", '--'), })
            l2_error, = ax2.plot(ix2_new, l2_new_error, **ax2_plot_dict, )
        # ax2.grid()

        # 要等 ax1 中 所有 曲线 plot 完事 之后，ax1.get_ylim() 获取到的 ax1 的 ylim 才是真实的
        # %% 获取 ax1 的 上下 lim 的 相对位置，和 相对 间隔 大小，为之后 设置 ax2 的 绝对 lim 范围（y 刻度 线性时）
        ax1_interval_relative = (ax1_yticks[1] - ax1_yticks[0]) / (ax1.get_ylim()[-1] - ax1.get_ylim()[0])
        # print(ax1_yticks[1] - ax1_yticks[0], ax1.get_ylim())
        ax1_down_lim_relative_location = (ax1_yticks[0] - ax1.get_ylim()[0]) / (
                ax1_yticks[1] - ax1_yticks[0])  # 下对齐
        # ax1_up_lim_relative_location = (ax1.get_ylim()[-1] - ax1_yticks[-1]) / (ax1_yticks[-1] - ax1_yticks[-2]) # 上对齐
        # --------- 搭配 start（y 刻度 线性时）

        # %% 设置 ax2 的 绝对 lim 范围，使其 刻度线 与 ax1 的 刻度线 对齐（y 刻度 线性时）
        ax2_lim_absolute = (ax2_yticks[1] - ax2_yticks[0]) / ax1_interval_relative
        # print(ax2_yticks[1] - ax2_yticks[0], ax1_interval_relative)
        ax2_down_lim = ax2_yticks[0] - (ax2_yticks[1] - ax2_yticks[0]) * ax1_down_lim_relative_location  # 下对齐
        ax2_up_lim = ax2_down_lim + ax2_lim_absolute  # 下对齐
        # ax2_up_lim = ax2_yticks[-1] + (ax2_yticks[-1] - ax2_yticks[-2]) * ax1_up_lim_relative_location # 上对齐
        # ax2_down_lim = ax2_up_lim - ax2_lim_absolute # 上对齐
        ax2.set_ylim(ax2_down_lim, ax2_up_lim)
        # --------- 搭配 end（y 刻度 线性时）

        if "label" in kwargs and "label2" in kwargs:
            if 'l3' in kwargs:
                handles = [l1, l2, l2_error, l3, l3_hline, l3_hline2, ]
            else:
                handles = [l1, l2, l2_error, ]
            plt.legend(handles=handles, **legend_dict, )
    else:
        if "label" in kwargs:
            plt.legend(**legend_dict, )

    array1D_title = array1D_title if is_energy != 1 else array1D_title + "_Squared"
    if is_title_on:
        # fig.suptitle(array1D_title, fontsize=fontsize+add_size, fontdict=font)
        # sgtitle 放置位置与 suptitle 相似，必须将其放在所有 subplot 的最后
        if "l2" in kwargs:
            ax2.set_title(array1D_title, pad=fontsize_set / 2,
                          **format_dict(fontsize_set, font, title_fontsize_enlarge))
        else:
            ax1.set_title(array1D_title, pad=fontsize_set / 2,
                          **format_dict(fontsize_set, font, title_fontsize_enlarge))

    plt.show()

    if is_title_on == 0 and is_axes_on == 0:
        ax1.margins(0, 0)
        if "l2" in kwargs:
            ax2.margins(0, 0)
        if is_save == 1:
            fig.savefig(array1D_address, transparent=True, pad_inches=0)  # 不包含图例等，且无白边
    else:
        if is_save == 1:  # bbox_inches='tight' 的缺点是会导致对输出图片的大小设置失效。
            fig.savefig(array1D_address, transparent=True, bbox_inches='tight')  # 包含图例等，但有白边
            # fig.savefig(array1D_address, transparent = True, bbox_inches='tight', pad_inches=0) # 包含图例，且无白边


# %%

def interp2d_to(array2D, sample, kind,
                ix, iy, ix_new, iy_new, ):
    if sample > 1:
        f = interp2d(ix, iy, array2D, kind=kind)
        array2D_new = f(ix_new, iy_new)
    else:
        array2D_new = array2D
    return array2D_new


def log10_include_0_colorbar(array2D_new, add_con=True, **kwargs, ):
    if kwargs.get("is_colorbar_log", 0) >= 1 and add_con:  # 无论如何， 第 2 个 坐标系 上的 第 3 条 误差曲线，都默认取 log
        array2D_new = log10_include_0(array2D_new)
    return array2D_new


def energy_log10_colorbar(array2D_new, is_energy,
                          add_con=True, **kwargs, ):
    array2D_new = array2D_new if is_energy != 1 else np.abs(array2D_new) ** 2
    array2D_new = log10_include_0_colorbar(array2D_new, add_con=add_con, **kwargs, )
    return array2D_new


# %%

def add_right_cax(ax, pad=0.05, width=0.05, height=0.0, ):
    '''
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距.
    width是cax的宽度.
    '''
    axpos = ax.get_position()
    caxpos = mpl.transforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0 + height,
        axpos.x1 + pad + width,
        axpos.y1 - height,
    )
    cax = ax.figure.add_axes(caxpos)

    return cax


# %%

def find_U_center(U):
    ix, iy = np.meshgrid(range(U.shape[1]), range(U.shape[0]))
    U_energy = np.sum(np.abs(U) ** 2)
    U_weight = np.abs(U) ** 2 / U_energy
    Uc_x = int(np.sum(U_weight * ix))  # U 点阵 的 坐标系 与 ix, iy 的 相同么 ？
    Uc_y = int(np.sum(U_weight * iy))
    return Uc_x, Uc_y


# %%

def plot_2d(zj, sample=1, size_PerPixel=0.007,
            # %%
            array2D=0, array2D_address=os.path.dirname(os.path.abspath(__file__)), array2D_title='',
            # %%
            is_save=0, dpi=100, size_fig=3,
            # %%
            cmap_2d='viridis', ticks_num=6, is_contourf=0, is_title_on=1, is_axes_on=1, is_mm=1, is_propagation=0,
            # %%
            fontsize=9,
            font={'family': 'Times New Roman',  # 'serif'
                  'style': 'normal',  # 'normal', 'italic', 'oblique'
                  'weight': 'normal',
                  'color': 'black',  # 'black','gray','darkred'
                  },
            # %%
            is_self_colorbar=1, is_colorbar_on=1, is_energy=1,
            # %% 可选 参数（可不传入）
            xlabel='', ylabel='', clabel='',
            # %%
            plot_center=0, **kwargs, ):
    # %%
    # fig, ax1 = plt.subplots(1, 1, figsize=(size_fig, size_fig), dpi=dpi)
    fig = plt.figure(figsize=(size_fig, size_fig), dpi=dpi)
    fontsize_set = size_fig * 4 if type(fontsize) != int else fontsize
    # print(fontsize_set)
    title_fontsize_enlarge = 1.3

    ax1 = fig.add_subplot(111, label="1")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # print(size_fig)
    # %% 插值 begin

    Ix, Iy = array2D.shape[1], array2D.shape[0]
    Iz, Iz_new = gan_len_zj_resampled(zj, sample)

    if is_propagation != 0:
        ix, ix_new = gan_zj_resampled(zj, Iz_new)
    else:
        ix, ix_new = gan_ix_non_resampled(Ix)
        # ix_new = np.linspace(0, Ix - 1, Ix*sample) # 非传播 则 不对某个方向，偏爱地 重/上采样
    iy, iy_new = gan_ix_non_resampled(Iy)
    # iy_new = np.linspace(0, Iy - 1, Iy*sample) # 除非将 另一个方向 也上采样 相同倍数

    # %%

    kind = 'cubic'  # kind = 0,1,2,3 nono，1 维才可以这么写，2 维只有 'linear', 'cubic', 'quintic'

    # ix_mesh, iy_mesh = np.meshgrid(ix, iy)
    # f = interp2d(ix_mesh,iy_mesh,array2D,kind=kind)
    array2D_new = interp2d_to(array2D, sample, kind,
                              ix, iy, ix_new, iy_new, )

    import inspect
    white_list = []
    white_list.append("U_amp_plot_save")
    add_con = inspect.stack()[1][3] in white_list

    if add_con and plot_center == 1:
        Uc_x, Uc_y = find_U_center(array2D_new)
    array2D_new = energy_log10_colorbar(array2D_new, is_energy,
                                        add_con=add_con, **kwargs, )
    # %% 插值 end

    if is_axes_on == 0:
        ax1.axis('off')
    else:
        # if len(zj) !=0: xticks_z = range(0, Iz_new, Iz_new // ticks_num)
        # xticks_x = range(- Ix // 2, Ix - Ix // 2, Ix // ticks_num)
        # xticks = range(0, Ix, Ix // ticks_num)
        # yticks_y = range(- Iy // 2, Iy - Iy // 2, Iy // ticks_num)
        # yticks = range(0, Iy, Iy // ticks_num)

        # if len(zj) !=0: xticks_z = gan_ticks(Iz_new, ticks_num, Min=0, is_centered=0)
        # if len(zj) != 0: xticks_z = gan_ticks(ix_new[-1], ticks_num, Min=ix_new[0],red=0) is_cente
        # array_x = np.arange(0, Ix*size_PerPixel, size_PerPixel)
        # xticks_x = gan_ticks(Ix*size_PerPixel, ticks_num, Min=0, is_centered=1)
        # xticks = gan_ticks(Ix, ticks_num, Min=0, is_centered=0)
        # array_y = np.arange(0, Ix*size_PerPixel, size_PerPixel)
        # yticks_y = gan_ticks(Iy*size_PerPixel, ticks_num, Min=0, is_centered=1)
        # yticks = gan_ticks(Iy, ticks_num, Min=0, is_centered=0)

        # plt.xticks(range(0, Ix, Ix // ticks_num), fontsize=fontsize) # Text 对象没有 fontdict 标签
        # plt.yticks(range(0, Iy, Iy // ticks_num), fontsize=fontsize) # Text 对象没有 fontdict 标签
        # ax1.set_xticks(xticks_z if is_propagation != 0 else xticks)
        # ax1.set_yticks(yticks)  # 按理 等价于 np.linspace(0, Iy, ticks_num + 1)，但并不
        if is_mm == 1:  # round(i * size_PerPixel,2) 保留 2 位小数，改为 保留 2 位 有效数字
            if is_propagation != 0 and kwargs.get("is_propa_ax_reverse", 0) == 0:
                xticks, xticklabels = gan_ticks(ix_new[-1], ticks_num, Min=ix_new[0], I=Iz_new)
                # xticks = [find_nearest(ix_new, z)[0] for z in xticks_z]
                ax1.set_xticks(xticks)
                # ax1.set_xticklabels([float('%.3f' % i) for i in ix_new[list(xticks_z)]], fontsize=fontsize, fontdict=font)
                ax1.set_xticklabels(xticklabels, **format_dict(fontsize_set, font))
            else:
                xticks, xticklabels = gan_ticks(Ix * size_PerPixel, ticks_num, is_centered=1, I=Ix)
                # array_x = np.arange(0, Ix*size_PerPixel, size_PerPixel)
                # xticks = [find_nearest(array_x, x)[0] for x in xticks_x]
                ax1.set_xticks(xticks)
                ax1.set_xticklabels(xticklabels, **format_dict(fontsize_set, font))

            if kwargs.get("is_propa_ax_reverse", 0) == 0:
                yticks, yticklabels = gan_ticks(Iy * size_PerPixel, ticks_num, is_centered=1, I=Iy)
                # array_y = np.arange(0, Ix*size_PerPixel, size_PerPixel)
                # yticks = [find_nearest(array_y, y)[0] for y in yticks_y]
                yticklabels = [-y for y in yticklabels]
                ax1.set_yticks(yticks)
                ax1.set_yticklabels(yticklabels, **format_dict(fontsize_set, font))
            else:
                yticks, yticklabels = gan_ticks(ix_new[-1], ticks_num, Min=ix_new[0], I=Iz_new, reverse=1)
                ax1.set_yticks(yticks)
                ax1.set_yticklabels(yticklabels, **format_dict(fontsize_set, font))
        else:
            xticks, xticklabels = gan_ticks(Ix, ticks_num)
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticklabels, **format_dict(fontsize_set, font))
            yticks, yticklabels = gan_ticks(Iy, ticks_num)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticklabels, **format_dict(fontsize_set, font))

        if is_mjrFormatter_sci(xticklabels):
            ax1.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))
        if is_mjrFormatter_sci(yticklabels):
            ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))

        ax1.set_xlabel(xlabel, **format_dict(fontsize_set, font))  # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        ax1.set_ylabel(ylabel, **format_dict(fontsize_set, font))  # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize

    vmax = kwargs.get("vmax", np.max(array2D_new))
    vmin = kwargs.get("vmin", np.min(array2D_new))
    # 尽管可以放在 is_self_colorbar == 0 的分支中，但 is_colorbar_on == 1 要用到...
    if "vmax" in kwargs:
        if kwargs.get("is_colorbar_log", 0) >= 1 and add_con:
            vmax = np.log10(vmax)
            if vmax == -float('inf'): vmax = -50
    # if "vmin" in kwargs:
    #     if kwargs.get("is_colorbar_log", 0) >= 1 and add_con:
    #         vmin = np.log10(vmin)
    #         if vmin == -float('inf'): vmin = -50
    if kwargs.get("is_colorbar_log", 0) >= 1 and add_con:
        vmin = vmax - kwargs["is_colorbar_log"]  # 调整 Min 到 Max 的 数量级（有点像 曝光程度）

    if is_self_colorbar == 1:
        if is_contourf == 1:
            img = ax1.contourf(array2D_new, cmap=cmap_2d, )
        else:
            img = ax1.imshow(array2D_new, cmap=cmap_2d, )
    else:
        if is_contourf == 1:
            img = ax1.contourf(array2D_new, cmap=cmap_2d, vmin=vmin, vmax=vmax, )
        else:
            img = ax1.imshow(array2D_new, cmap=cmap_2d, vmin=vmin, vmax=vmax, )
    if add_con and plot_center == 1:
        center_mode = 3
        if center_mode <= 1:
            # %%  头尾 标记点：相同形状
            ax1_plot_dict = {"color": "white", "label": kwargs.get('label', None)}
            ax1_plot_dict.update({"alpha": kwargs.get("ax1_alpha", 1),
                                  "linestyle": kwargs.get("ax1_linestyle", '-'),  # 线型
                                  "linewidth": kwargs.get("ax1_linewidth", 1)})  # 线宽（marker = 'o' 时生效）
            if center_mode == 1:
                ax1_plot_dict.update({"marker": kwargs.get("ax1_marker", 'x'),  # 标记点：'+' 'x' '.' '|' 'o'
                                      "markeredgecolor": kwargs.get("ax1_markeredgecolor", "white"),  # 标记点颜色 ‘green’
                                      "markersize": kwargs.get("ax1_markersize", str(int(fontsize_set/3)+1)),  # 标记点大小
                                      "markeredgewidth": kwargs.get("ax1_markeredgewidth", 1), })  # 标记点边宽
            ax1.plot([Iy // 2, Uc_x], [Ix // 2, Uc_y], **ax1_plot_dict, )
            # %%  头尾 标记点：不同形状 和 颜色
            if center_mode == 2:
                ax1.plot([Iy // 2, Uc_x], [Ix // 2, Uc_y], **ax1_plot_dict, )
                ax1_plot_dict.update({"marker": kwargs.get("ax1_marker", '1'),  # 标记点：'+' 'x' '.' '|' '_' 'o' ',' 's' '^'
                                      "markeredgecolor": kwargs.get("ax1_markeredgecolor", "springgreen"),  # 标记点颜色 ‘lime’
                                      "markersize": kwargs.get("ax1_markersize", str(int(fontsize_set/2)+1)),  # 标记点大小
                                      "markeredgewidth": kwargs.get("ax1_markeredgewidth", 1), })  # 标记点边宽
                ax1.plot([Iy // 2], [Ix // 2], **ax1_plot_dict, )
                ax1_plot_dict.update({"marker": kwargs.get("ax1_marker", '2'),  # 标记点：'+' 'x' '.' '|' '_' 'o' ',' '1' '2'
                                      "markeredgecolor": kwargs.get("ax1_markeredgecolor", "yellow"),  # 标记点颜色 ‘gold’
                                      "markersize": kwargs.get("ax1_markersize", str(int(fontsize_set/2)+1)), })  # 标记点大小
                ax1.plot([Uc_x], [Uc_y], **ax1_plot_dict, )
        elif center_mode <= 4:
            # %%  线条 与 标记点 均：颜色渐变
            def gan_points_dl(points_x, points_y):
                dx, dy = (points_x - Iy // 2) * size_PerPixel, (points_y - Ix // 2) * size_PerPixel
                dl = (dx ** 2 + dy ** 2) ** 0.5
                return dl
    
            points_num = int(2 * gan_points_dl(Uc_x, Uc_y) / size_PerPixel)
            points_x, points_y = np.linspace(Iy // 2, Uc_x, points_num), np.linspace(Ix // 2, Uc_y, points_num)
            ax1_plot_dict = {"cmap": 'Wistia_r', }  # 渐变 'cool' 'Wistia' 'spring' 'summer' 'autumn' 'winter'; 突变 'Pastel1' 'Pastel2' 'Paired' 'Set3'
            ax1_plot_dict.update({"alpha": kwargs.get("ax1_alpha", 1),
                                  # "linestyle": kwargs.get("ax1_linestyle", '-'),  # 线型
                                  "linewidth": kwargs.get("ax1_linewidth", 1),  # 边缘线宽
                                  "edgecolors": None, })  # edgecolors → 散点的 边缘颜色
            ax1_plot_dict.update({"marker": kwargs.get("ax1_marker", '.'),  # 标记点：'+' 'x' '.' '|' 'o' ','
                                  "s": 1, })  # 散点的面积
            ax1.scatter(points_x, points_y, c=gan_points_dl(points_x, points_y), **ax1_plot_dict, )
            if center_mode == 3:
                ax1_plot_dict.update({"marker": kwargs.get("ax1_marker", '.'),  # 标记点：'+' 'x' '.' '|' 'o' ','
                                      "s": (int(fontsize_set / 3) + 1) ** 2, })  # 散点的面积
                points_x, points_y = np.array([Iy // 2, Uc_x]), np.array([Ix // 2, Uc_y])
                ax1.scatter(points_x, points_y, c=gan_points_dl(points_x, points_y), **ax1_plot_dict, )
            elif center_mode == 4:
                # %%  线条：颜色渐变，头尾 标记点：不同形状（但相同颜色）
                ax1_plot_dict.update({"marker": kwargs.get("ax1_marker", '1'),  # 标记点：'+' 'x' '.' '|' 'o' ','
                                      "s": (int(fontsize_set / 3) + 1) ** 2, })  # 散点的面积
                point_x, point_y = np.array([Iy // 2]), np.array([Ix // 2])
                ax1.scatter(point_x, point_y, c=gan_points_dl(point_x, point_y), **ax1_plot_dict, )
                ax1_plot_dict.update({"marker": kwargs.get("ax1_marker", '2'), })  # 标记点：'+' 'x' '.' '|' 'o' ','
                point_x, point_y = np.array([Uc_x]), np.array([Uc_y])
                ax1.scatter(point_x, point_y, c=gan_points_dl(point_x, point_y), **ax1_plot_dict, )

    if is_colorbar_on == 1:
        cax = add_right_cax(ax1, pad=0.05, width=0.05)
        cb = fig.colorbar(img, cax=cax)
        # cb = fig.colorbar(img, cax=cax, extend='both')
        cb.ax.tick_params(labelsize=fontsize_set)  # 设置 colorbar 刻度字体；字体大小 labelsize=fontsize。 # Text 对象没有 fontdict 标签
        if is_self_colorbar != 1:
            cticks, cticklabels = gan_ticks(vmax, ticks_num, Min=vmin)
            cb.set_ticks(cticks)
            cb.set_ticklabels(cticklabels)
            if is_mjrFormatter_sci(cticklabels):
                # print(cticklabels)
                cb.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))
            if kwargs.get("is_colorbar_log", 0) >= 1 and add_con:
                cb.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_log))
        cb.set_label(clabel, **format_dict(fontsize_set, font))  # 设置 colorbar 的 标签名、标签字体；字体大小 fontsize=fontsize

    array2D_title = array2D_title if is_energy != 1 else array2D_title + "_Squared"
    if is_title_on:
        ax1.set_title(array2D_title, pad=fontsize_set / 2, **format_dict(fontsize_set, font, title_fontsize_enlarge))

    if is_title_on == 0 and is_axes_on == 0 and is_colorbar_on == 0:
        ax1.margins(0, 0)
        if is_save == 1:
            fig.savefig(array2D_address, transparent=True, pad_inches=0)  # 不包含图例等，且无白边
        elif is_save == -1:
            img = savefig_to_buffer(transparent=True, pad_inches=0)
    else:
        if is_save == 1:
            fig.savefig(array2D_address, transparent=True, bbox_inches='tight')  # 包含图例等，但有白边
            # fig.savefig(array2D_address, transparent = True, bbox_inches='tight', pad_inches=0) # 包含图例，且无白边
        elif is_save == -1:
            img = savefig_to_buffer(transparent=True, bbox_inches='tight')

    # img = AxesImage_to_Image(fig)  #  plt.close() 之前截获 fig
    # img = fig2data(fig)  # plt.close() 之前截获 fig
    # img = fig2array(fig)
    # %%
    plt.show()
    # plt.cla() # 清除所有 活动的 ax1，但其他不关
    # plt.clf() # 清除所有 ax1，但 fig 不关，可用同一个 fig 作图 新的 ax1（复用 设定好的 同一个 fig）
    # plt.close() # 关闭 fig（似乎 spyder 和 pycharm 的 scitific mode 自动就 close 了，内存本身就 不会上去）

    return img


# %% https://blog.csdn.net/aa846555831/article/details/52372884

def savefig_to_buffer(**kwargs):  # using buffer, great way!
    import PIL
    from io import BytesIO

    buffer_ = BytesIO()  # 申请缓冲地址
    plt.savefig(buffer_, **kwargs)  # 保存在内存中，而不是在本地磁盘，注意这个默认认为你要保存的就是plt中的内容
    buffer_.seek(0)
    dataPIL = PIL.Image.open(buffer_)  # 用PIL或CV2从内存中读取
    data = np.asarray(dataPIL)  # 转换为nparrary，PIL转换就非常快了,data 即为所需
    # import cv2
    # cv2.imshow('image', data)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    buffer_.close()  # 释放缓存
    return data


# %% https://blog.csdn.net/tequila53/article/details/123486737

def AxesImage_to_Image(fig):
    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # 需要用到matplotlib的后端
    canvas = FigureCanvasAgg(fig)  # 由当前图表获取整体画布
    canvas.draw()  # 绘制图像
    w, h = canvas.get_width_height()  # 获取图像尺寸
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    # 解码string 得到argb图像，原博主是用fromstring，目前的np推荐是frombuffer
    buf.shape = (w, h, 4)  # w,h,a*r*g*b
    buf = np.roll(buf, 3, axis=2)  # w,h,r*g*b*a
    img = Image.frombytes('RGBA', (w, h), buf.tobytes())  # 从axes转为PIL
    # 同样的，原博主使用tostring()，使用tobytes()更符合目前标准
    img = np.asarray(img)
    return img


# %% https://blog.csdn.net/guyuealian/article/details/104179723

def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


# %% https://blog.csdn.net/u012762410/article/details/119038022

def fig2array(fig):
    '''
    matplotlib.figure.Figure转为np.ndarray
    '''
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf_ndarray = np.frombuffer(fig.canvas.tostring_rgb(), dtype='u1')
    im = buf_ndarray.reshape(h, w, 3)
    return im


def fig2img(fig):
    '''
    matplotlib.figure.Figure转为PIL image
    '''
    from PIL import Image
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # 将Image.frombytes替换为Image.frombuffer,图像会倒置
    img = Image.frombytes('RGB', (w, h), fig.canvas.tostring_rgb())
    return img


# %%

def plot_3d_XYZ(zj, sample=1, size_PerPixel=0.007,
                # %%
                U_YZ=0, U_XZ=0, U_1=0, U_2=0,
                U_structure_front=0, U_structure_end=0, is_show_structure_face=1,
                # %%
                img_address=os.path.dirname(os.path.abspath(__file__)), img_title='',
                # %%
                iX=0, iY=0, iZ_1=0, iZ_2=0,
                iZ_structure_front=0, iZ_structure_end=0,
                # %%
                is_save=0, dpi=100, size_fig=3,
                # %%
                cmap_3d='viridis', elev=10, azim=-65, alpha=2,
                ticks_num=6, is_title_on=1, is_axes_on=1, is_mm=1,
                # %%
                fontsize=9,
                font={'family': 'Times New Roman',  # 'serif'
                      'style': 'normal',  # 'normal', 'italic', 'oblique'
                      'weight': 'normal',
                      'color': 'black',  # 'black','gray','darkred'
                      },
                # %%
                is_self_colorbar=1, is_colorbar_on=1, is_energy=1,
                # %% 可选 参数（可不传入）
                xlabel='Z', ylabel='X', zlabel='Y', clabel='', **kwargs, ):
    # %%

    size_fig_3D_x, size_fig_3D_y = size_fig * kwargs.get("size_fig_3D_x_scale", 10), \
                                   size_fig * kwargs.get("size_fig_3D_y_scale", 10)
    fig = plt.figure(figsize=(size_fig_3D_x, size_fig_3D_y), dpi=dpi)
    fontsize_set = size_fig_3D_y * 1.2 if type(fontsize) != int else fontsize
    title_fontsize_enlarge = 1.3

    ax1 = fig.add_subplot(111, projection='3d', label="1")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    # %% 插值 begin

    Ix, Iy = U_1.shape[1], U_1.shape[0]
    Iz, Iz_new = gan_len_zj_resampled(zj, sample)  # zj 区间范围 保持不变，分段数 乘以 sample 后，新划分出的 刻度的个数

    ix, ix_new = gan_zj_resampled(zj, Iz_new)
    iy, iy_new = gan_ix_non_resampled(Iy)

    # %%

    kind = 'cubic'  # kind = 0,1,2,3 nono，1 维才可以这么写，2 维只有 'linear', 'cubic', 'quintic'

    U_YZ_new = interp2d_to(U_YZ, sample, kind,
                           ix, iy, ix_new, iy_new, )
    U_XZ_new = interp2d_to(U_XZ, sample, kind,
                           ix, iy, ix_new, iy_new, )

    U_YZ_new = energy_log10_colorbar(U_YZ_new, is_energy, **kwargs, )
    U_XZ_new = energy_log10_colorbar(U_XZ_new, is_energy, **kwargs, )
    U_1 = energy_log10_colorbar(U_1, is_energy, **kwargs, )
    U_2 = energy_log10_colorbar(U_2, is_energy, **kwargs, )
    if is_show_structure_face == 1:
        U_structure_front = energy_log10_colorbar(U_structure_front, is_energy, **kwargs, )
        U_structure_end = energy_log10_colorbar(U_structure_end, is_energy, **kwargs, )
        UZ = np.dstack((U_1, U_2, U_structure_front, U_structure_end))
    else:
        UZ = np.dstack((U_1, U_2))

    # %% 插值 end

    if is_axes_on == 0:
        ax1.axis('off')
    else:
        # if len(zj) !=0: xticks_z = range(0, Iz_new, Iz_new // ticks_num) # ax1.set_xticks(range(0, Iz, Iz // ticks_num))  # 按理 等价于 np.linspace(0, Ix, ticks_num + 1)，但并不
        # xticks_x = range(- Ix // 2, Ix - Ix // 2, Ix // ticks_num)
        # xticks = range(0, Ix, Ix // ticks_num)
        # yticks_y = range(- Iy // 2, Iy - Iy // 2, Iy // ticks_num)
        # yticks = range(0, Iy, Iy // ticks_num)

        # if len(zj) !=0: xticks_z = gan_ticks(Iz_new, ticks_num, Min=0, is_centered=0)
        # if len(zj) != 0: xticks_z = gan_ticks(ix_new[-1], ticks_num, Min=ix_new[0], is_centered=0)
        # xticks_x = gan_ticks(Ix*size_PerPixel, ticks_num, Min=0, is_centered=1)
        # xticks = gan_ticks(Ix, ticks_num, Min=0, is_centered=0)
        # yticks_y = gan_ticks(Iy*size_PerPixel, ticks_num, Min=0, is_centered=1)
        # yticks = gan_ticks(Iy, ticks_num, Min=0, is_centered=0)

        # ax1.set_xticks(xticks_z) # Pair 1
        # ax1.set_xticks([find_nearest(ix_new, i)[0] for i in xticks_z])
        # ax1.set_yticks(xticks)  # 按理 等价于 np.linspace(0, Iy, ticks_num + 1)，但并不
        # ax1.set_zticks(yticks)
        # if is_mm == 1:  # round(i * size_PerPixel,2) 保留 2 位小数，改为 保留 2 位 有效数字
        #     # ax1.set_xticklabels([float('%.3f' % i) for i in ix_new[list(xticks_z)]], fontsize=fontsize, fontdict=font) # Pair 1
        #     ax1.set_xticklabels([float('%.3f' % i) for i in xticks_z], fontsize=fontsize, fontdict=font)
        #     # ax1.set_xticklabels([float('%.3f' % (k * diz * size_PerPixel)) for k in range(0, Iz, Iz // ticks_num)],
        #     #                      fontsize=fontsize, fontdict=font)
        #     ax1.set_yticklabels([float('%.3f' % i) for i in xticks_x], fontsize=fontsize, fontdict=font)
        #     ax1.set_zticklabels([float('%.3f' % j) for j in yticks_y], fontsize=fontsize, fontdict=font)
        # else:
        #     ax1.set_xticklabels(xticks_z, fontsize=fontsize, fontdict=font)
        #     ax1.set_yticklabels(xticks, fontsize=fontsize, fontdict=font)
        #     ax1.set_zticklabels(yticks, fontsize=fontsize, fontdict=font)

        if is_mm == 1:
            xticks, xticklabels = gan_ticks(ix_new[-1], ticks_num, Min=ix_new[0], I=Iz_new)
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticklabels, **format_dict(fontsize_set, font))

            yticks, yticklabels = gan_ticks(Ix * size_PerPixel, ticks_num, is_centered=1, I=Ix)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticklabels, **format_dict(fontsize_set, font))

            zticks, zticklabels = gan_ticks(Iy * size_PerPixel, ticks_num, is_centered=1, I=Iy)
            # zticklabels = [-z for z in zticklabels]
            ax1.set_zticks(zticks)
            ax1.set_zticklabels(zticklabels, **format_dict(fontsize_set, font))
        else:
            xticks, xticklabels = gan_ticks(Iz_new, ticks_num)
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticklabels, **format_dict(fontsize_set, font))
            yticks, yticklabels = gan_ticks(Ix, ticks_num)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticklabels, **format_dict(fontsize_set, font))
            zticks, zticklabels = gan_ticks(Iy, ticks_num)
            ax1.set_zticks(zticks)
            ax1.set_zticklabels(zticklabels, **format_dict(fontsize_set, font))

        if is_mjrFormatter_sci(xticklabels):
            ax1.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))
        if is_mjrFormatter_sci(yticklabels):
            ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))
        if is_mjrFormatter_sci(zticklabels):
            ax1.zaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))

        ax1.set_xlabel(xlabel, labelpad=fontsize_set, **format_dict(fontsize_set, font, title_fontsize_enlarge))
        ax1.set_ylabel(ylabel, labelpad=fontsize_set, **format_dict(fontsize_set, font, title_fontsize_enlarge))
        ax1.set_zlabel(zlabel, labelpad=fontsize_set, **format_dict(fontsize_set, font, title_fontsize_enlarge))

    ax1.view_init(elev=elev, azim=azim)  # 后一个为负 = 绕 z 轴逆时针

    vmax = kwargs.get("vmax", max(np.max(U_YZ_new), np.max(U_XZ_new), np.max(UZ)))
    vmin = kwargs.get("vmin", min(np.max(U_YZ_new), np.max(U_XZ_new), np.min(UZ)))
    # 尽管可以放在 is_self_colorbar == 0 的分支中，但 is_colorbar_on == 1 要用到...
    if "vmax" in kwargs:
        if kwargs.get("is_colorbar_log", 0) >= 1:
            vmax = np.log10(vmax)
            if vmax == -float('inf'): vmax = -50
    if kwargs.get("is_colorbar_log", 0) >= 1:
        vmin = vmax - kwargs["is_colorbar_log"]  # 调整 Min 到 Max 的 数量级（有点像 曝光程度）

    color_3d_dict = {"cmap": cmap_3d, "alpha": math.e ** (-1 * alpha)}

    if is_self_colorbar == 1:
        i_Z, i_Y = np.meshgrid(range(Iz_new), range(Iy))
        i_Y = i_Y[::-1]
        img = ax1.scatter3D(i_Z, iX, i_Y, c=U_YZ_new, **color_3d_dict)
        i_Z, i_X = np.meshgrid(range(Iz_new), range(Ix))
        # i_X = i_X[::-1]
        img = ax1.scatter3D(i_Z, i_X, iY, c=U_XZ_new, **color_3d_dict)

        i_X, i_Y = np.meshgrid(range(Ix), range(Iy))
        i_Y = i_Y[::-1]
        img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_1])[0], i_X, i_Y, c=U_1, **color_3d_dict)
        # ix_new.tolist().index(zj[iZ_1])
        img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_2])[0], i_X, i_Y, c=U_2, **color_3d_dict)
        # ix_new.tolist().index(zj[iZ_2])

        if is_show_structure_face == 1:
            img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_structure_front])[0], i_X, i_Y,
                                # ix_new.tolist().index(zj[iZ_structure_front])
                                c=U_structure_front, **color_3d_dict)

            img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_structure_end])[0], i_X, i_Y,
                                # ix_new.tolist().index(zj[iZ_structure_end])
                                c=U_structure_end, **color_3d_dict)

    else:
        i_Z, i_Y = np.meshgrid(range(Iz_new), range(Iy))
        i_Y = i_Y[::-1]
        img = ax1.scatter3D(i_Z, iX, i_Y, c=U_YZ_new, **color_3d_dict, vmin=vmin, vmax=vmax)
        i_Z, i_X = np.meshgrid(range(Iz_new), range(Ix))
        # i_X = i_X[::-1]
        img = ax1.scatter3D(i_Z, i_X, iY, c=U_XZ_new, **color_3d_dict, vmin=vmin, vmax=vmax)

        i_X, i_Y = np.meshgrid(range(Ix), range(Iy))
        i_Y = i_Y[::-1]
        img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_1])[0], i_X, i_Y, c=U_1, **color_3d_dict,
                            # ix_new.tolist().index(zj[iZ_1])
                            vmin=vmin, vmax=vmax)
        img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_2])[0], i_X, i_Y, c=U_2, **color_3d_dict,
                            # ix_new.tolist().index(zj[iZ_2])
                            vmin=vmin, vmax=vmax)

        if is_show_structure_face == 1:
            img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_structure_front])[0], i_X, i_Y,
                                # ix_new.tolist().index(zj[iZ_structure_front])
                                c=U_structure_front, **color_3d_dict, vmin=vmin, vmax=vmax)
            img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_structure_end])[0], i_X, i_Y,
                                # ix_new.tolist().index(zj[iZ_structure_end])
                                c=U_structure_end, **color_3d_dict, vmin=vmin, vmax=vmax)

    if is_colorbar_on == 1:
        cax = add_right_cax(ax1, pad=0.03, width=0.03, height=0.2, )
        cb = fig.colorbar(img, cax=cax)
        # cb = fig.colorbar(img, cax=cax, extend='both')
        cb.ax.tick_params(labelsize=fontsize_set)  # 设置 colorbar 刻度字体；字体大小 labelsize=fontsize。 # Text 对象没有 fontdict 标签
        if is_self_colorbar != 1:
            cticks, cticklabels = gan_ticks(vmax, ticks_num, Min=vmin)
            cb.set_ticks(cticks)
            cb.set_ticklabels(cticklabels)
            if is_mjrFormatter_sci(cticklabels):
                cb.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))
        cb.set_label(clabel, **format_dict(fontsize_set, font))  # 设置 colorbar 的 标签名、标签字体；字体大小 fontsize=fontsize

    img_title = img_title if is_energy != 1 else img_title + "_Squared"
    if is_title_on:
        ax1.set_title(img_title, pad=fontsize_set / 2, **format_dict(fontsize_set, font, title_fontsize_enlarge))

    plt.show()

    if is_title_on == 0 and is_axes_on == 0 and is_colorbar_on == 0:
        ax1.margins(0, 0, 0)
        if is_save == 1:
            fig.savefig(img_address, transparent=True, pad_inches=0)  # 不包含图例等，且无白边
    else:
        if is_save == 1:
            fig.savefig(img_address, transparent=True, bbox_inches='tight')  # 包含图例等，但有白边
            # fig.savefig(img_address, transparent = True, bbox_inches='tight', pad_inches=0) # 包含图例，且无白边


def plot_3d_XYz(zj, sample=1, size_PerPixel=0.007,
                U_z_stored=0, z_stored=[],
                # %%
                img_address=os.path.dirname(os.path.abspath(__file__)), img_title='',
                # %%
                is_save=0, dpi=100, size_fig=3,
                # %%
                cmap_3d='viridis', elev=10, azim=-65, alpha=2,
                ticks_num=6, is_title_on=1, is_axes_on=1, is_mm=1,
                # %%
                fontsize=9,
                font={'family': 'Times New Roman',  # 'serif'
                      'style': 'normal',  # 'normal', 'italic', 'oblique'
                      'weight': 'normal',
                      'color': 'black',  # 'black','gray','darkred'
                      },
                # %%
                is_self_colorbar=1, is_colorbar_on=1,
                is_energy=1,
                # %% 可选 参数（可不传入）
                xlabel='Z', ylabel='X', zlabel='Y', clabel='', **kwargs, ):
    # %%

    size_fig_3D_x, size_fig_3D_y = size_fig * kwargs.get("size_fig_3D_x_scale", 10), \
                                   size_fig * kwargs.get("size_fig_3D_y_scale", 10)
    fig = plt.figure(figsize=(size_fig_3D_x, size_fig_3D_y), dpi=dpi)
    fontsize_set = size_fig_3D_y * 1.2 if type(fontsize) != int else fontsize
    title_fontsize_enlarge = 1.3

    ax1 = fig.add_subplot(111, projection='3d', label="1")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    # %%

    Ix, Iy = U_z_stored[:, :, 0].shape[1], U_z_stored[:, :, 0].shape[0]
    Iz, Iz_new = gan_len_zj_resampled(zj, sample)  # zj 区间范围 保持不变，分段数 乘以 sample 后，新划分出的 刻度的个数

    ix, ix_new = gan_zj_resampled(zj, Iz_new)
    iy, iy_new = gan_ix_non_resampled(Iy)

    # %%
    if is_axes_on == 0:
        ax1.axis('off')
    else:
        # if len(zj) !=0: xticks_z = range(0, Iz_new,  Iz_new // ticks_num)
        # xticks_x = range(- Ix // 2, Ix - Ix // 2, Ix // ticks_num)
        # xticks = range(0, Ix, Ix // ticks_num)
        # yticks_y = range(- Iy // 2, Iy - Iy // 2, Iy // ticks_num)
        # yticks = range(0, Iy, Iy // ticks_num)

        # if len(zj) !=0: xticks_z = gan_ticks(Iz_new, ticks_num, Min=0, is_centered=0)
        # if len(zj) != 0: xticks_z = gan_ticks(ix_new[-1], ticks_num, Min=ix_new[0], is_centered=0)
        # xticks_x = gan_ticks(Ix*size_PerPixel, ticks_num, Min=0, is_centered=1)
        # xticks = gan_ticks(Ix, ticks_num, Min=0, is_centered=0)
        # yticks_y = gan_ticks(Iy*size_PerPixel, ticks_num, Min=0, is_centered=1)
        # yticks = gan_ticks(Iy, ticks_num, Min=0, is_centered=0)

        if is_mm == 1:
            xticks, xticklabels = gan_ticks(ix_new[-1], ticks_num, Min=ix_new[0], I=Iz_new)
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticklabels, **format_dict(fontsize_set, font))

            yticks, yticklabels = gan_ticks(Ix * size_PerPixel, ticks_num, is_centered=1, I=Ix)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticklabels, **format_dict(fontsize_set, font))

            zticks, zticklabels = gan_ticks(Iy * size_PerPixel, ticks_num, is_centered=1, I=Iy)
            # zticklabels = [-z for z in zticklabels]
            ax1.set_zticks(zticks)
            ax1.set_zticklabels(zticklabels, **format_dict(fontsize_set, font))
        else:
            xticks, xticklabels = gan_ticks(Iz_new, ticks_num)
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticklabels, **format_dict(fontsize_set, font))
            yticks, yticklabels = gan_ticks(Ix, ticks_num)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticklabels, **format_dict(fontsize_set, font))
            zticks, zticklabels = gan_ticks(Iy, ticks_num)
            ax1.set_zticks(zticks)
            ax1.set_zticklabels(zticklabels, **format_dict(fontsize_set, font))

        if is_mjrFormatter_sci(xticklabels):
            ax1.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))
        if is_mjrFormatter_sci(yticklabels):
            ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))
        if is_mjrFormatter_sci(zticklabels):
            ax1.zaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))

        ax1.set_xlabel(xlabel, labelpad=fontsize_set, **format_dict(fontsize_set, font, title_fontsize_enlarge))
        ax1.set_ylabel(ylabel, labelpad=fontsize_set, **format_dict(fontsize_set, font, title_fontsize_enlarge))
        ax1.set_zlabel(zlabel, labelpad=fontsize_set, **format_dict(fontsize_set, font, title_fontsize_enlarge))

    ax1.view_init(elev=elev, azim=azim);  # 后一个为负 = 绕 z 轴逆时针

    # %%

    U_z_stored = energy_log10_colorbar(U_z_stored, is_energy, **kwargs, )

    vmax = kwargs.get("vmax", np.max(U_z_stored))
    vmin = kwargs.get("vmin", np.min(U_z_stored))
    # 尽管可以放在 is_self_colorbar == 0 的分支中，但 is_colorbar_on == 1 要用到...
    if "vmax" in kwargs:
        if kwargs.get("is_colorbar_log", 0) >= 1:
            vmax = np.log10(vmax)
            if vmax == -float('inf'): vmax = -50
    if kwargs.get("is_colorbar_log", 0) >= 1:
        vmin = vmax - kwargs["is_colorbar_log"]  # 调整 Min 到 Max 的 数量级（有点像 曝光程度）

    sheets_stored_num = len(z_stored) - 1
    x_stretch_factor = sheets_stored_num ** 0.5 * 2
    # ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1 * x_stretch_factor, 1, 1, 1]))
    ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 1 / x_stretch_factor, 1 / x_stretch_factor, 1]))

    # ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 1/x_stretch_factor, 1/x_stretch_factor, 1/x_stretch_factor]))

    def color_3d_dict(sheet_stored_th):
        return {"cmap": cmap_3d,
                "alpha": math.e ** -3 * math.e ** (-1 * alpha * sheet_stored_th / sheets_stored_num)}

    if is_self_colorbar == 1:
        i_X, i_Y = np.meshgrid(range(Ix), range(Iy))
        i_Y = i_Y[::-1]
        for sheet_stored_th in range(sheets_stored_num + 1):
            img = ax1.scatter3D(find_nearest(ix_new, z_stored[sheet_stored_th])[0], i_X, i_Y,
                                # ix_new.tolist().index(z_stored[sheet_stored_th])
                                c=U_z_stored[:, :, sheet_stored_th], **color_3d_dict(sheet_stored_th))
    else:
        i_X, i_Y = np.meshgrid(range(Ix), range(Iy))
        i_Y = i_Y[::-1]
        for sheet_stored_th in range(sheets_stored_num + 1):
            img = ax1.scatter3D(find_nearest(ix_new, z_stored[sheet_stored_th])[0], i_X, i_Y,
                                # ix_new.tolist().index(z_stored[sheet_stored_th])
                                c=U_z_stored[:, :, sheet_stored_th], **color_3d_dict(sheet_stored_th),
                                vmin=vmin, vmax=vmax)

    if is_colorbar_on == 1:
        cax = add_right_cax(ax1, pad=0.03, width=0.03, height=0.03, )
        cb = fig.colorbar(img, cax=cax)
        # cb = fig.colorbar(img, cax=cax, extend='both')
        cb.ax.tick_params(labelsize=fontsize_set)  # 设置 colorbar 刻度字体；字体大小 labelsize=fontsize。 # Text 对象没有 fontdict 标签
        if is_self_colorbar != 1:
            cticks, cticklabels = gan_ticks(vmax, ticks_num, Min=vmin)
            cb.set_ticks(cticks)
            cb.set_ticklabels(cticklabels)
            if is_mjrFormatter_sci(cticklabels):
                cb.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter_sci))
        cb.set_label(clabel, **format_dict(fontsize_set, font))  # 设置 colorbar 的 标签名、标签字体；字体大小 fontsize=fontsize

    img_title = img_title if is_energy != 1 else img_title + "_Squared"
    if is_title_on:
        ax1.set_title(img_title, pad=fontsize_set / 2, **format_dict(fontsize_set, font, title_fontsize_enlarge))

    plt.show()

    if is_title_on == 0 and is_axes_on == 0 and is_colorbar_on == 0:
        ax1.margins(0, 0, 0)
        if is_save == 1:
            fig.savefig(img_address, transparent=True, pad_inches=0)  # 不包含图例等，且无白边
    else:
        if is_save == 1:
            fig.savefig(img_address, transparent=True, bbox_inches='tight')  # 包含图例等，但有白边
            # fig.savefig(img_address, transparent = True, bbox_inches='tight', pad_inches=0) # 包含图例，且无白边
