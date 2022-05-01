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
from fun_global_var import Get


# plt.rcParams['xtick.direction'] = 'out' # 设置刻度线在坐标轴内
# plt.rcParams['ytick.direction'] = 'out' # 一次设置，全局生效，所以：记得关闭

def lormat(ticks, Str):
    if "%" in Str:  # 返回 float list
        return [float(Str % z) for z in ticks], [float(Str % z) for z in ticks]
        # 当 '%.3f' = '%.3f' 时， 这玩意 等价于 Get('.1e') = '.2e' 的 下述
        # 但只当 个位数 不是零的时候：.2f 是保留 2 位小数，不计入 个位数（小数点后 的 位数一致，比较整齐）
        # .2e 是保留 3 位有效数字，如果个位数是零，则保留 3 位小数
        # [float(format(z, Get('.1e'))) for z in ticks]
    else:  # 返回 float list 作为 ticks，同时返回 字符串 list 作为 tickslabels
        # return [float(format(z, '.1e')) for z in ticks], [format(z, Get('.1e')) for z in ticks]
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
                gan_tickslabels = lormat(gan_tickslabels, '%.3f')[0]
            else:
                gan_tickslabels = lormat(gan_tickslabels, '%.1f')[0]
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
                gan_ticks, gan_tickslabels = lormat(gan_tickslabels, '.1e')
            elif abs(Max) < 10:
                gan_ticks, gan_tickslabels = lormat(gan_tickslabels, '%.3f')
            else:
                gan_ticks, gan_tickslabels = lormat(gan_tickslabels, '%.1f')
        gan_ticks = [z / Max * kwargs.get("I", Max) for z in gan_ticks]
        # size_PerPixel 基本只适用于 x 轴 居中，且 is_centered=1 的情况，z 轴 不适用？
    else:
        gan_ticks = gan_tickslabels = np.arange(0, ticks_num + 1, 1)
    return gan_ticks, gan_tickslabels

def mjrFormatter(x, pos):
    return "$10^{{{0}}}$".format("%.1f" % x)  # 奇了怪了， x 本身已经是 格式化过了的，咋还得格式化一次...

def plot_1d(zj, sample=2, size_PerPixel=0.007,
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
    # %%
    # fig, ax1 = plt.subplots(1, 1, figsize=(size_fig_x, size_fig_y), dpi=dpi)
    fig = plt.figure(figsize=(size_fig_x, size_fig_y), dpi=dpi)
    ax1 = fig.add_subplot(111, label="1")
    ax1.spines["left"].set_color(color_1d)  # 修改 左侧 坐标轴线 颜色
    # ax1.spines["bottom"].set_color(color_1d)  # 修改 下边 坐标轴线 颜色
    # ax1.tick_params(axis='x', colors=color_1d) # 刻度线 本身 的 颜色
    # ax1.tick_params(axis='y', colors=color_1d) # 刻度线 本身 的 颜色

    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    Ix = array1D.shape[0]
    Iz = len(zj)
    Iz_new = (Iz - 1) * sample + 1  # zj 区间范围 保持不变，分段数 乘以 sample 后，新划分出的 刻度的个数

    # %% 插值 begin

    if is_propagation != 0:
        ix = zj
        ix_new = np.linspace(zj[0], zj[-1], Iz_new)
    else:
        ix = [i for i in range(Ix)]
        ix_new = np.linspace(0, Ix - 1, Ix)  # 非传播 则 不对某个方向，偏爱地 重/上采样

    # kind = 'cubic' # kind = 0,1,2,3 nono，1 维才可以这么写，2 维只有 'linear', 'cubic', 'quintic'
    # f = interp1d(ix, array1D, kind = kind)

    # print(ix)
    # print(array1D)
    if sample > 1:
        f = UnivariateSpline(ix, array1D, s=0)  # ix 必须是 严格递增的，若 ix 是 zj 的话，zj 也必须是
        array1D_new = f(ix_new)
    else:
        array1D_new = array1D
    array1D_new = array1D_new if is_energy != 1 else np.abs(array1D_new) ** 2

    if 'ax1_xticklabel' in kwargs:
        array1D_new = np.log10(array1D_new)
        array1D_new_min = min(remove_elements(array1D_new, -float('inf')))  # 防止 绘图 纵坐标 遇 inf 无法解析，并 正确生成 tickslabel
        array1D_new = [array1D_new_min if array1D_new[i] == -float('inf') else array1D_new[i] for i in range(len(array1D_new))]
        array1D_new = np.array(array1D_new) # 转成数组

    if "l2" in kwargs:
        if "zj2" in kwargs:  # 如果 zj2 在，则以 zj2 为 xticks
            zj = kwargs["zj2"]
            Iz = len(zj)
            Iz_new = (Iz - 1) * sample + 1
            ix = zj
            ix2_new = np.linspace(zj[0], zj[-1], Iz_new)
        else:
            ix2_new = ix_new

        if sample > 1:
            f = UnivariateSpline(ix, kwargs['l2'], s=0)  # ix 必须是 严格递增的，若 ix 是 zj 的话，zj 也必须是
            l2_new = f(ix2_new)
        else:
            l2_new = kwargs['l2']
        l2_new = l2_new if is_energy != 1 else np.abs(l2_new) ** 2

        if 'ax1_xticklabel' in kwargs:
            l2_new = np.log10(l2_new)
            l2_new_min = min(remove_elements(l2_new, -float('inf'))) # 防止 绘图 纵坐标 遇 inf 无法解析，并 正确生成 tickslabel
            l2_new = [l2_new_min if l2_new[i] == -float('inf') else l2_new[i] for i in range(len(l2_new))]
            l2_new = np.array(l2_new)  # 转成数组
        # print(l2_new)

        if "zj2" in kwargs:
            index = [find_nearest(ix_new, goal)[0] for goal in ix2_new]
            # print(index)
            l3_new = np.abs(l2_new - array1D_new[index])  # 花式索引，可以用 list 或 array 作为一个 array 的下标

        if 'l3' in kwargs:
            if sample > 1: # 我发现 哪怕 sample == 1，也会导致 被 插值作用，导致 原始值 被改变（不是说好了过每个点么...）
                f = UnivariateSpline(ix, kwargs['l3'], s=0)  # ix 必须是 严格递增的，若 ix 是 zj 的话，zj 也必须是
                l4_new = f(ix2_new)
            else:
                l4_new = kwargs['l3']
            l4_new = l4_new if is_energy != 1 else np.abs(l4_new) ** 2

            l4_new = np.log10(l4_new)
            # print(l4_new)
            l4_new_min = min(remove_elements(l4_new, -float('inf')))
            l4_new = [l4_new_min if l4_new[i] == -float('inf') else l4_new[i] for i in range(len(l4_new))]
            l4_new = np.array(l4_new)  # 转成数组

    # %%

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
                    ax1_xticklabel = kwargs["ax1_xticklabel"]
                    ax1.set_xticks(ax1_xticklabel)  # 则 第1个x轴的 label 需要与之 对齐，则保留原汁原味的 zj 作为 刻度 和 刻度的 label。
                    ax1.set_xticklabels([float('%.3f' % i) for i in ax1_xticklabel], fontsize=fontsize, fontdict=font)
                else:
                    xticks, xticklabels = gan_ticks(ix_new[-1], ticks_num, Min=ix_new[0])
                    ax1.set_xticks(xticks)
                    ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
            else:
                xticks, xticklabels = gan_ticks(Ix * size_PerPixel, ticks_num, is_centered=1)
                ax1.set_xticks(xticks)
                ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
        else:
            xticks, xticklabels = gan_ticks(Iz, ticks_num)
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)

        # if 'ax1_yscale' in kwargs or 'ax1_xticklabel' in kwargs:
        #     # ax1.set_yscale(kwargs.get('ax1_yscale', 'log'))
        #     # ax1.semilogy(x, np.log10(y))

        if "l2" in kwargs and 'l3' in kwargs: # 如果 要绘制 4 条曲线
            vmax = kwargs.get("vmax", max(np.max(array1D_new), np.max(l2_new), np.max(l3_new)))
            vmin = kwargs.get("vmin", min(np.min(array1D_new), np.min(l2_new), np.min(l3_new)))
        elif "l2" in kwargs and 'l3' not in kwargs and "ax2_xticklabel" not in kwargs: # 需要 给 min 补个零，防止 ganticks 的时候，不从 0 开始
            vmax = kwargs.get("vmax", np.max(array1D_new))
            vmin = 0
        else:
            vmax = kwargs.get("vmax", np.max(array1D_new))
            vmin = kwargs.get("vmin", np.min(array1D_new))

        ax1_yticks, ax1_yticklabels = gan_ticks(vmax, ticks_num, Min=vmin)
        ax1.set_yticks(ax1_yticks)
        ax1.set_yticklabels(ax1_yticklabels, fontsize=fontsize, fontdict=font)

        if 'ax1_xticklabel' in kwargs:
            # logfmt = mpl.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)
            # ax1.yaxis.set_major_formatter(logfmt)
            ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))

        ax1.set_xlabel(xlabel, fontsize=fontsize, fontdict=font)  # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        ax1.set_ylabel(ylabel, fontsize=fontsize, fontdict=font)  # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize

    # %% 画 第 1 条 曲线

    ax1_plot_dict = {"color": color_1d, "label": kwargs.get('label', None)}
    ax1_plot_dict.update({"alpha": kwargs.get("ax1_alpha", 1),  # 1 即 不透明
                          "linestyle": kwargs.get("ax1_linestyle", '-'),  # 线型
                          "linewidth": kwargs.get("ax1_linewidth", 2), })  # 线宽
    ax1_plot_dict.update({"marker": kwargs.get("ax1_marker", ''),  # 标记点：'+' 'x' '.' '|' ''
                          "markeredgecolor": kwargs.get("ax1_markeredgecolor", color_1d),  # 标记点颜色 ‘green’
                          "markersize": kwargs.get("ax1_markersize", '5'),  # 标记点大小
                          "markeredgewidth": kwargs.get("ax1_markeredgewidth", 1), })  # 标记点边宽

    # ax1.set_yscale(kwargs.get('ax1_yscale', 'linear')) # linear 会覆盖 之前的 set_yticks，如果该语句在 set_yticks 之后的话

    l1, = ax1.plot(ix_new, array1D_new, **ax1_plot_dict)
    ax1.grid()

    # %% 画 第 2 条 曲线

    legend_dict = {'loc': kwargs.get("loc", 0)}  # 5: ‘right’ （右边中间），0: "best" 右上角（默认）
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

        if is_axes_on == 0:
            ax2.axis('off')
        else:
            if "ax2_xticklabel" in kwargs:
                ax2_xticklabel = kwargs["ax2_xticklabel"]
                if is_axes_on == 0:
                    ax2.axis('off')
                else:
                    if is_mm == 1:  # round(i * size_PerPixel,2) 保留 2 位小数，改为 保留 2 位 有效数字
                        if is_propagation != 0:
                            ax2.set_xticks(zj)  # ax2 是 Tz，不像 dkzQ，是非线性变化的，所以不能人工 gan 其刻度，也不能有 ix2_new。
                            ax2.set_xticklabels([float('%.3f' % i) for i in ax2_xticklabel], fontsize=fontsize,
                                                fontdict=font)
                        else:
                            ax2.set_xticks(xticks)
                            ax2.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
                    else:
                        ax2.set_xticks(xticks)
                        ax2.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
            else:
                ax2.set_xticks(())  # 否则 ax2 的 x 不设刻度

            # if 'ax2_yscale' in kwargs or 'ax1_xticklabel' in kwargs:
            #     # ax2.set_yscale(kwargs.get('ax2_yscale', 'log'))
            #     # ax2.semilogy(x, np.log10(y))

            if 'l3' in kwargs:
                vmax2 = kwargs.get("vmax2", np.max(l4_new))
                vmin2 = kwargs.get("vmin2", np.min(l4_new))
                # print(vmax2, vmin2)
            else:
                if kwargs.get("is_energy_normalized", False) == 2: # 如果要画 随 T 的 演化
                    # ax2.set_ylim(ax1.get_ylim())  # ax2 的 y 轴范围 不再自动，而是 强制 ax2 的 y 轴 范围 等于 ax1 的 y 轴范围
                    vmax2, vmin2 = vmax, vmin  # 与 ax2.set_ylim(ax1.get_ylim()) 配合，强制 ax2 的 y 轴 刻度线 等于 ax1 的 刻度线。
                elif "ax2_xticklabel" not in kwargs: # 如果要画 随 T 的 演化
                    vmax2 = kwargs.get("vmax2", max(np.max(l2_new), np.max(l3_new)))
                    # vmin2 = kwargs.get("vmin2", min(np.min(l2_new), np.min(l3_new)))
                    vmin2 = 0 # 这样才能使 ticks 和 labels 的 第一个元素 是 0
                else: # 如果 要画 随 dk 的 演化
                    vmax2 = kwargs.get("vmax2", np.max(l2_new))
                    vmin2 = kwargs.get("vmin2", np.min(l2_new))

            ax2_yticks, ax2_yticklabels = gan_ticks(vmax2, ticks_num, Min=vmin2)
            ax2.set_yticks(ax2_yticks)
            ax2.set_yticklabels(ax2_yticklabels, fontsize=fontsize, fontdict=font)

            if 'ax1_xticklabel' in kwargs or 'l3' in kwargs:
                # logfmt = mpl.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)
                # ax2.yaxis.set_major_formatter(logfmt)
                ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))

            ax2.set_xlabel(xlabel2, fontsize=fontsize, fontdict=font)
            ax2.set_ylabel(ylabel2, fontsize=fontsize, fontdict=font)

        ax2_plot_dict = {"color": color_1d2, "label": kwargs.get('label2', None)}
        ax2_plot_dict.update({"alpha": kwargs.get("ax2_alpha", 1),  # 1 即 不透明
                              "linestyle": kwargs.get("ax2_linestyle", '-'),  # 线型
                              "linewidth": kwargs.get("ax2_linewidth", 2), })  # 线宽
        ax2_marker_dict = {"marker": kwargs.get("ax2_marker", '|'),  # 标记点
                           "markeredgecolor": kwargs.get("ax2_markeredgecolor", 'green'),  # 标记点颜色
                           "markersize": kwargs.get("ax2_markersize", '20'),  # 标记点大小
                           "markeredgewidth": kwargs.get("ax2_markeredgewidth", 1), }
        ax2_plot_dict.update(ax2_marker_dict)

        if 'l3' in kwargs:
            ax1_plot_dict.update({"label": kwargs.get('label2', None),
                                  "linestyle": kwargs.get("ax2_linestyle", '--'), })
            ax1_plot_dict.update(ax2_marker_dict)
            l2, = ax1.plot(ix2_new, l2_new, **ax1_plot_dict, )

            ax2_plot_dict.update({"label": kwargs.get('label3', None),
                                  "linestyle": kwargs.get("l3_linestyle", '-.'),
                                  "marker": kwargs.get("l3_marker", 'x'),
                                  "markeredgecolor": kwargs.get("l4_markeredgecolor", 'gray'), })
            l4, = ax2.plot(ix2_new, l4_new, **ax2_plot_dict, )
        else:
            l2, = ax2.plot(ix2_new, l2_new, **ax2_plot_dict, )
        # ax2.grid()

        if "zj2" in kwargs:
            if 'l3' in kwargs:
                ax1_plot_dict.update({"label": "energy_error",
                                      "linestyle": kwargs.get("l3_linestyle", '-.'), })

                l3, = ax1.plot(ix2_new, l3_new, **ax1_plot_dict, )
            else:
                ax2_plot_dict.update({"label": "energy_error",
                                      "linestyle": kwargs.get("l3_linestyle", '--'), })
                l3, = ax2.plot(ix2_new, l3_new, **ax2_plot_dict, )

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
            if "zj2" in kwargs:
                if 'l3' in kwargs:
                    plt.legend(handles=[l1, l2, l3, l4], **legend_dict, )
                else:
                    plt.legend(handles=[l1, l2, l3], **legend_dict, )
            else:
                plt.legend(handles=[l1, l2], **legend_dict, )
    else:
        if "label" in kwargs:
            plt.legend(**legend_dict, )

    array1D_title = array1D_title if is_energy != 1 else array1D_title + "_Squared"
    add_size = kwargs.get("add_size", 3)
    if is_title_on:
        # fig.suptitle(array1D_title, fontsize=fontsize+add_size, fontdict=font)
        # sgtitle 放置位置与 suptitle 相似，必须将其放在所有 subplot 的最后
        if "l2" in kwargs:
            ax2.set_title(array1D_title, fontsize=fontsize + add_size, fontdict=font)
        else:
            ax1.set_title(array1D_title, fontsize=fontsize + add_size, fontdict=font)

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


def add_right_cax(ax, pad, width):
    '''
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距.
    width是cax的宽度.
    '''
    axpos = ax.get_position()
    caxpos = mpl.transforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax


def plot_2d(zj, sample=2, size_PerPixel=0.007,
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
            xlabel='', ylabel='', clabel='', **kwargs, ):
    # %%
    # fig, ax1 = plt.subplots(1, 1, figsize=(size_fig, size_fig), dpi=dpi)
    fig = plt.figure(figsize=(size_fig, size_fig), dpi=dpi)
    ax1 = fig.add_subplot(111, label="1")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    # %% 插值 begin

    Ix, Iy = array2D.shape[1], array2D.shape[0]
    Iz = len(zj)
    Iz_new = (Iz - 1) * sample + 1  # zj 区间范围 保持不变，分段数 乘以 sample 后，新划分出的 刻度的个数

    if is_propagation != 0:
        ix, iy = zj, [j for j in range(Iy)]
        ix_new, iy_new = np.linspace(zj[0], zj[-1], Iz_new), iy
    else:
        ix, iy = [i for i in range(Ix)], [j for j in range(Iy)]
        ix_new, iy_new = ix, iy  # 非传播 则 不重/上采样
        # ix_new = np.linspace(0, Ix - 1, Ix*sample) # 非传播 则 不对某个方向，偏爱地 重/上采样
        # iy_new = np.linspace(0, Iy - 1, Iy*sample) # 除非将 另一个方向 也上采样 相同倍数

    kind = 'cubic'  # kind = 0,1,2,3 nono，1 维才可以这么写，2 维只有 'linear', 'cubic', 'quintic'

    # ix_mesh, iy_mesh = np.meshgrid(ix, iy)
    # f = interp2d(ix_mesh,iy_mesh,array2D,kind=kind)
    if sample > 1:
        f = interp2d(ix, iy, array2D, kind=kind)
        array2D_new = f(ix_new, iy_new)
    else:
        array2D_new = array2D
    array2D_new = array2D_new if is_energy != 1 else np.abs(array2D_new) ** 2
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
            if is_propagation != 0:
                xticks, xticklabels = gan_ticks(ix_new[-1], ticks_num, Min=ix_new[0], I=Iz_new)
                # xticks = [find_nearest(ix_new, z)[0] for z in xticks_z]
                ax1.set_xticks(xticks)
                # ax1.set_xticklabels([float('%.3f' % i) for i in ix_new[list(xticks_z)]], fontsize=fontsize, fontdict=font)
                ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
            else:
                xticks, xticklabels = gan_ticks(Ix * size_PerPixel, ticks_num, is_centered=1, I=Ix)
                # array_x = np.arange(0, Ix*size_PerPixel, size_PerPixel)
                # xticks = [find_nearest(array_x, x)[0] for x in xticks_x]
                ax1.set_xticks(xticks)
                ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
            yticks, yticklabels = gan_ticks(Iy * size_PerPixel, ticks_num, is_centered=1, I=Iy)
            # array_y = np.arange(0, Ix*size_PerPixel, size_PerPixel)
            # yticks = [find_nearest(array_y, y)[0] for y in yticks_y]
            yticklabels = [-y for y in yticklabels]
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticklabels, fontsize=fontsize, fontdict=font)
        else:
            xticks, xticklabels = gan_ticks(Ix, ticks_num)
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
            yticks, yticklabels = gan_ticks(Iy, ticks_num)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticklabels, fontsize=fontsize, fontdict=font)
        ax1.set_xlabel(xlabel, fontsize=fontsize, fontdict=font)  # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        ax1.set_ylabel(ylabel, fontsize=fontsize, fontdict=font)  # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize

    vmax = kwargs.get("vmax", np.max(array2D_new))
    vmin = kwargs.get("vmin", np.min(array2D_new))
    # 尽管可以放在 is_self_colorbar == 0 的分支中，但 is_colorbar_on == 1 要用到...

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

    if is_colorbar_on == 1:
        cax = add_right_cax(ax1, pad=0.05, width=0.05)
        cb = fig.colorbar(img, cax=cax)
        # cb = fig.colorbar(img, cax=cax, extend='both')
        cb.ax.tick_params(labelsize=fontsize)  # 设置 colorbar 刻度字体；字体大小 labelsize=fontsize。 # Text 对象没有 fontdict 标签
        if is_self_colorbar != 1:
            cticks, cticklabels = gan_ticks(vmax, ticks_num, Min=vmin)
            cb.set_ticks(cticks)
            cb.set_ticklabels(cticklabels)
        cb.set_label(clabel, fontsize=fontsize, fontdict=font)  # 设置 colorbar 的 标签名、标签字体；字体大小 fontsize=fontsize

    array2D_title = array2D_title if is_energy != 1 else array2D_title + "_Squared"
    add_size = kwargs.get("add_size", 3)
    if is_title_on:
        ax1.set_title(array2D_title, fontsize=fontsize + add_size, fontdict=font)

    if is_title_on == 0 and is_axes_on == 0 and is_colorbar_on == 0:
        ax1.margins(0, 0)
        if is_save == 1:
            fig.savefig(array2D_address, transparent=True, pad_inches=0)  # 不包含图例等，且无白边
    else:
        if is_save == 1:
            fig.savefig(array2D_address, transparent=True, bbox_inches='tight')  # 包含图例等，但有白边
            # fig.savefig(array2D_address, transparent = True, bbox_inches='tight', pad_inches=0) # 包含图例，且无白边

    plt.show()
    # plt.cla() # 清除所有 活动的 ax1，但其他不关
    # plt.clf() # 清除所有 ax1，但 fig 不关，可用同一个 fig 作图 新的 ax1（复用 设定好的 同一个 fig）
    # plt.close() # 关闭 fig（似乎 spyder 和 pycharm 的 scitific mode 自动就 close 了，内存本身就 不会上去）


def plot_3d_XYZ(zj, sample=2, size_PerPixel=0.007,
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

    fig = plt.figure(figsize=(size_fig * 10, size_fig * 10), dpi=dpi)
    ax1 = fig.add_subplot(111, projection='3d', label="1")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    # %% 插值 begin

    Ix, Iy = U_1.shape[1], U_1.shape[0]
    Iz = len(zj)
    Iz_new = (Iz - 1) * sample + 1  # zj 区间范围 保持不变，分段数 乘以 sample 后，新划分出的 刻度的个数

    ix, iy = zj, [j for j in range(Iy)]
    ix_new, iy_new = np.linspace(zj[0], zj[-1], Iz_new), iy

    kind = 'cubic'  # kind = 0,1,2,3 nono，1 维才可以这么写，2 维只有 'linear', 'cubic', 'quintic'

    if sample > 1:
        f = interp2d(ix, iy, U_YZ, kind=kind)
        U_YZ_new = f(ix_new, iy_new)
        f = interp2d(ix, iy, U_XZ, kind=kind)
        U_XZ_new = f(ix_new, iy_new)
    else:
        U_YZ_new = U_YZ
        U_XZ_new = U_XZ

    U_YZ_new = U_YZ_new if is_energy != 1 else np.abs(U_YZ_new) ** 2
    U_XZ_new = U_XZ_new if is_energy != 1 else np.abs(U_XZ_new) ** 2
    U_1 = U_1 if is_energy != 1 else np.abs(U_1) ** 2
    U_2 = U_2 if is_energy != 1 else np.abs(U_2) ** 2
    if is_show_structure_face == 1:
        U_structure_front = U_structure_front if is_energy != 1 else np.abs(U_structure_front) ** 2
        U_structure_end = U_structure_end if is_energy != 1 else np.abs(U_structure_end) ** 2

    if is_show_structure_face == 1:
        UXY = np.dstack((U_YZ_new, U_XZ_new))
        UZ = np.dstack((U_1, U_2, U_structure_front, U_structure_end))
    else:
        UXY = np.dstack((U_YZ_new, U_XZ_new))
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
            ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)

            yticks, yticklabels = gan_ticks(Ix * size_PerPixel, ticks_num, is_centered=1, I=Ix)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticklabels, fontsize=fontsize, fontdict=font)

            zticks, zticklabels = gan_ticks(Iy * size_PerPixel, ticks_num, is_centered=1, I=Iy)
            # zticklabels = [-z for z in zticklabels]
            ax1.set_zticks(zticks)
            ax1.set_zticklabels(zticklabels, fontsize=fontsize, fontdict=font)
        else:
            xticks, xticklabels = gan_ticks(Iz_new, ticks_num)
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
            yticks, yticklabels = gan_ticks(Ix, ticks_num)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticklabels, fontsize=fontsize, fontdict=font)
            zticks, zticklabels = gan_ticks(Iy, ticks_num)
            ax1.set_zticks(zticks)
            ax1.set_zticklabels(zticklabels, fontsize=fontsize, fontdict=font)
        ax1.set_xlabel(xlabel, fontsize=fontsize, fontdict=font)  # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        ax1.set_ylabel(ylabel, fontsize=fontsize, fontdict=font)  # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        ax1.set_zlabel(zlabel, fontsize=fontsize, fontdict=font)  # 设置 z 轴的 标签名、标签字体；字体大小 fontsize=fontsize

    ax1.view_init(elev=elev, azim=azim);  # 后一个为负 = 绕 z 轴逆时针

    vmax = kwargs.get("vmax", max(np.max(UXY), np.max(UZ)))
    vmin = kwargs.get("vmin", min(np.min(UXY), np.min(UZ)))
    # 尽管可以放在 is_self_colorbar == 0 的分支中，但 is_colorbar_on == 1 要用到...
    Ixy = Iy
    if is_self_colorbar == 1:
        i_Z, i_XY = np.meshgrid([i for i in range(Iz_new)], [j for j in range(Ixy)])
        i_XY = i_XY[::-1]
        img = ax1.scatter3D(i_Z, iX, i_XY, c=U_YZ_new, cmap=cmap_3d, alpha=math.e ** (-1 * alpha))
        i_XY = i_XY[::-1]
        img = ax1.scatter3D(i_Z, i_XY, iY, c=U_XZ_new, cmap=cmap_3d, alpha=math.e ** (-1 * alpha))

        i_X, i_Y = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        i_Y = i_Y[::-1]
        img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_1])[0], i_X, i_Y, c=U_1, cmap=cmap_3d,
                            # ix_new.tolist().index(zj[iZ_1])
                            alpha=math.e ** (-1 * alpha))
        img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_2])[0], i_X, i_Y, c=U_2, cmap=cmap_3d,
                            # ix_new.tolist().index(zj[iZ_2])
                            alpha=math.e ** (-1 * alpha))

        if is_show_structure_face == 1:
            img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_structure_front])[0], i_X, i_Y,
                                # ix_new.tolist().index(zj[iZ_structure_front])
                                c=U_structure_front, cmap=cmap_3d, alpha=math.e ** (-1 * alpha))
            img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_structure_end])[0], i_X, i_Y,
                                # ix_new.tolist().index(zj[iZ_structure_end])
                                c=U_structure_end, cmap=cmap_3d, alpha=math.e ** (-1 * alpha))
    else:
        i_Z, i_XY = np.meshgrid([i for i in range(Iz_new)], [j for j in range(Ixy)])
        i_XY = i_XY[::-1]
        img = ax1.scatter3D(i_Z, iX, i_XY, c=U_YZ_new, cmap=cmap_3d, alpha=math.e ** (-1 * alpha), vmin=vmin, vmax=vmax)
        i_XY = i_XY[::-1]
        img = ax1.scatter3D(i_Z, i_XY, iY, c=U_XZ_new, cmap=cmap_3d, alpha=math.e ** (-1 * alpha), vmin=vmin, vmax=vmax)

        i_X, i_Y = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        i_Y = i_Y[::-1]
        img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_1])[0], i_X, i_Y, c=U_1, cmap=cmap_3d,
                            # ix_new.tolist().index(zj[iZ_1])
                            alpha=math.e ** (-1 * alpha), vmin=vmin, vmax=vmax)
        img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_2])[0], i_X, i_Y, c=U_2, cmap=cmap_3d,
                            # ix_new.tolist().index(zj[iZ_2])
                            alpha=math.e ** (-1 * alpha), vmin=vmin, vmax=vmax)

        if is_show_structure_face == 1:
            img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_structure_front])[0], i_X, i_Y,
                                # ix_new.tolist().index(zj[iZ_structure_front])
                                c=U_structure_front, cmap=cmap_3d, alpha=math.e ** (-1 * alpha), vmin=vmin, vmax=vmax)
            img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_structure_end])[0], i_X, i_Y,
                                # ix_new.tolist().index(zj[iZ_structure_end])
                                c=U_structure_end, cmap=cmap_3d, alpha=math.e ** (-1 * alpha), vmin=vmin, vmax=vmax)

    if is_colorbar_on == 1:
        cax = add_right_cax(ax1, pad=0.05, width=0.05)
        cb = fig.colorbar(img, cax=cax)
        # cb = fig.colorbar(img, cax=cax, extend='both')
        cb.ax.tick_params(labelsize=fontsize)  # 设置 colorbar 刻度字体；字体大小 labelsize=fontsize。 # Text 对象没有 fontdict 标签
        if is_self_colorbar != 1:
            cticks, cticklabels = gan_ticks(vmax, ticks_num, Min=vmin)
            cb.set_ticks(cticks)
            cb.set_ticklabels(cticklabels)
        cb.set_label(clabel, fontsize=fontsize, fontdict=font)  # 设置 colorbar 的 标签名、标签字体；字体大小 fontsize=fontsize

    img_title = img_title if is_energy != 1 else img_title + "_Squared"
    add_size = kwargs.get("add_size", 3)
    if is_title_on:
        ax1.set_title(img_title, fontsize=fontsize + add_size, fontdict=font)

    plt.show()

    if is_title_on == 0 and is_axes_on == 0 and is_colorbar_on == 0:
        ax1.margins(0, 0, 0)
        if is_save == 1:
            fig.savefig(img_address, transparent=True, pad_inches=0)  # 不包含图例等，且无白边
    else:
        if is_save == 1:
            fig.savefig(img_address, transparent=True, bbox_inches='tight')  # 包含图例等，但有白边
            # fig.savefig(img_address, transparent = True, bbox_inches='tight', pad_inches=0) # 包含图例，且无白边


def plot_3d_XYz(zj, sample=2, size_PerPixel=0.007,
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

    fig = plt.figure(figsize=(size_fig * 10, size_fig * 10), dpi=dpi)
    ax1 = fig.add_subplot(111, projection='3d', label="1")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    Ix, Iy = U_z_stored[:, :, 0].shape[1], U_z_stored[:, :, 0].shape[0]
    Iz = len(zj)
    Iz_new = (Iz - 1) * sample + 1  # zj 区间范围 保持不变，分段数 乘以 sample 后，新划分出的 刻度的个数

    ix, iy = zj, [j for j in range(Iy)]
    ix_new, iy_new = np.linspace(zj[0], zj[-1], Iz_new), iy

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
            ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)

            yticks, yticklabels = gan_ticks(Ix * size_PerPixel, ticks_num, is_centered=1, I=Ix)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticklabels, fontsize=fontsize, fontdict=font)

            zticks, zticklabels = gan_ticks(Iy * size_PerPixel, ticks_num, is_centered=1, I=Iy)
            # zticklabels = [-z for z in zticklabels]
            ax1.set_zticks(zticks)
            ax1.set_zticklabels(zticklabels, fontsize=fontsize, fontdict=font)
        else:
            xticks, xticklabels = gan_ticks(Iz_new, ticks_num)
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
            yticks, yticklabels = gan_ticks(Ix, ticks_num)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticklabels, fontsize=fontsize, fontdict=font)
            zticks, zticklabels = gan_ticks(Iy, ticks_num)
            ax1.set_zticks(zticks)
            ax1.set_zticklabels(zticklabels, fontsize=fontsize, fontdict=font)
        ax1.set_xlabel(xlabel, fontsize=fontsize, fontdict=font)  # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        ax1.set_ylabel(ylabel, fontsize=fontsize, fontdict=font)  # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        ax1.set_zlabel(zlabel, fontsize=fontsize, fontdict=font)  # 设置 z 轴的 标签名、标签字体；字体大小 fontsize=fontsize

    ax1.view_init(elev=elev, azim=azim);  # 后一个为负 = 绕 z 轴逆时针

    sheets_stored_num = len(z_stored) - 1
    x_stretch_factor = sheets_stored_num ** 0.5 * 2
    # ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1 * x_stretch_factor, 1, 1, 1]))
    ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 1 / x_stretch_factor, 1 / x_stretch_factor, 1]))
    # ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 1/x_stretch_factor, 1/x_stretch_factor, 1/x_stretch_factor]))

    U_z_stored = U_z_stored if is_energy != 1 else np.abs(U_z_stored) ** 2
    vmax = kwargs.get("vmax", np.max(U_z_stored))
    vmin = kwargs.get("vmin", np.min(U_z_stored))
    # 尽管可以放在 is_self_colorbar == 0 的分支中，但 is_colorbar_on == 1 要用到...

    if is_self_colorbar == 1:
        i_X, i_Y = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        i_Y = i_Y[::-1]
        for sheet_stored_th in range(sheets_stored_num + 1):
            img = ax1.scatter3D(find_nearest(ix_new, z_stored[sheet_stored_th])[0], i_X, i_Y,
                                # ix_new.tolist().index(z_stored[sheet_stored_th])
                                c=U_z_stored[:, :, sheet_stored_th], cmap=cmap_3d,
                                alpha=math.e ** -3 * math.e ** (-1 * alpha * sheet_stored_th / sheets_stored_num))
    else:
        i_X, i_Y = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        i_Y = i_Y[::-1]
        for sheet_stored_th in range(sheets_stored_num + 1):
            img = ax1.scatter3D(find_nearest(ix_new, z_stored[sheet_stored_th])[0], i_X, i_Y,
                                # ix_new.tolist().index(z_stored[sheet_stored_th])
                                c=U_z_stored[:, :, sheet_stored_th], cmap=cmap_3d,
                                alpha=math.e ** -3 * math.e ** (-1 * alpha * sheet_stored_th / sheets_stored_num),
                                vmin=vmin, vmax=vmax)

    if is_colorbar_on == 1:
        cax = add_right_cax(ax1, pad=0.05, width=0.05)
        cb = fig.colorbar(img, cax=cax)
        # cb = fig.colorbar(img, cax=cax, extend='both')
        cb.ax.tick_params(labelsize=fontsize)  # 设置 colorbar 刻度字体；字体大小 labelsize=fontsize。 # Text 对象没有 fontdict 标签
        if is_self_colorbar != 1:
            cticks, cticklabels = gan_ticks(vmax, ticks_num, Min=vmin)
            cb.set_ticks(cticks)
            cb.set_ticklabels(cticklabels)
        cb.set_label(clabel, fontsize=fontsize, fontdict=font)  # 设置 colorbar 的 标签名、标签字体；字体大小 fontsize=fontsize

    img_title = img_title if is_energy != 1 else img_title + "_Squared"
    add_size = kwargs.get("add_size", 3)
    if is_title_on:
        ax1.set_title(img_title, fontsize=fontsize + add_size, fontdict=font)

    plt.show()

    if is_title_on == 0 and is_axes_on == 0 and is_colorbar_on == 0:
        ax1.margins(0, 0, 0)
        if is_save == 1:
            fig.savefig(img_address, transparent=True, pad_inches=0)  # 不包含图例等，且无白边
    else:
        if is_save == 1:
            fig.savefig(img_address, transparent=True, bbox_inches='tight')  # 包含图例等，但有白边
            # fig.savefig(img_address, transparent = True, bbox_inches='tight', pad_inches=0) # 包含图例，且无白边
