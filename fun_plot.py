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
from fun_algorithm import find_nearest
from fun_global_var import Get

# plt.rcParams['xtick.direction'] = 'out' # 设置刻度线在坐标轴内
# plt.rcParams['ytick.direction'] = 'out' # 一次设置，全局生效，所以：记得关闭

def lormat(ticks, Str):
    if "%" in Str: # 返回 float list
        return [float(Str % z) for z in ticks], [float(Str % z) for z in ticks]
        # 当 '%.2f' = '%.2f' 时， 这玩意 等价于 Get('.1e') = '.2e' 的 下述
        # 但只当 个位数 不是零的时候：.2f 是保留 2 位小数，不计入 个位数（小数点后 的 位数一致，比较整齐）
        # .2e 是保留 3 位有效数字，如果个位数是零，则保留 3 位小数
        # [float(format(z, Get('.1e'))) for z in ticks]
    else: # 返回 float list 作为 ticks，同时返回 字符串 list 作为 tickslabels
        # return [float(format(z, '.1e')) for z in ticks], [format(z, Get('.1e')) for z in ticks]
        return [float(format(z, Str)) for z in ticks], [format(z, Str) for z in ticks]

def gan_ticks(Max, ticks_num, Min=0, is_centered=0, **kwargs):
    Str = format((Max - Min)/ticks_num,'.0e')
    if Str != 'nan':
        if int(Str[0]) > 2 and int(Str[0]) < 5: # step 默认为 更小的 step，比如 34 则 2，9876 则 5
            Str = "2" + Str[1:]
        elif int(Str[0]) > 5: # =5, =2, =1 则不理睬
            Str = "5" + Str[1:]
        step = float(Str) # 保留 1 位有效数字
        ticks_num_real = (Max - Min)//step
        ticks = np.arange(0, ticks_num_real + 1, 1)
        gan_tickslabels = ticks * step
        if is_centered == 1:
            Average = (Max + Min) / 2
            Center_divisible = Average // step * step
            # Center_divisible = (-Average) // step * step * (-1)
            gan_tickslabels -= Center_divisible
            gan_ticks = gan_tickslabels + Average # 把 0 放中间
            if abs(Max) < 10:
                gan_tickslabels = lormat(gan_tickslabels, '%.2f')[0]
            else:
                gan_tickslabels = lormat(gan_tickslabels, '%.1f')[0]
        else:
            Min_divisible = (-Min) // step * step * (-1)
            # 连 (Max - Min) 除以 step 都有余， 更何况 Min
            # _Min = np.sign(Min) * (abs(Min) // step + int(np.sign(Min)==-1)) * step
            # 额，我发现 np.sign(Min) * (abs(Min) // step + int(np.sign(Min)==-1)) = Min // step
            # _Min = Min // step * step # 负 得更多，正 得更少，保证 _Min < Min，这样才能 在图上显示 Min ？
            # 不，恰恰相反，图上的 Min 比 轴上的 _Min 更靠左，才能把 轴上的 _Min 显示出来
            gan_tickslabels += Min_divisible # 注意 * 的 优先级比 // 高
            # if abs(Max) >= 1e3 or abs(Min) >= 1e3 or abs(Max) <= 1e-2 or abs(Min) <= 1e-2:
            if abs(Max) >= 1e3 or abs(Max) < 1e-2 :
                gan_ticks, gan_tickslabels = lormat(gan_tickslabels, '.1e')
            elif abs(Max) < 10:
                gan_ticks, gan_tickslabels = lormat(gan_tickslabels, '%.2f')
            else:
                gan_ticks, gan_tickslabels = lormat(gan_tickslabels, '%.1f')
        gan_ticks = [z / Max * kwargs.get("I", Max) for z in gan_ticks]
        # size_PerPixel 基本只适用于 x 轴 居中，且 is_centered=1 的情况，z 轴 不适用？
    else:
        gan_ticks = gan_tickslabels = np.arange(0, ticks_num + 1, 1)
    return gan_ticks, gan_tickslabels



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
    Iz_new = (Iz-1)*sample+1 # zj 区间范围 保持不变，分段数 乘以 sample 后，新划分出的 刻度的个数

    #%% 插值 begin

    if is_propagation != 0:
        ix = zj
        ix_new = np.linspace(zj[0], zj[-1], Iz_new)
        # img = ax1.plot(zj, array1D if is_energy != 1 else array1D ** 2, color=color_1d)
    else:
        ix = [i for i in range(Ix)]
        ix_new = np.linspace(0, Ix - 1, Ix) # 非传播 则 不对某个方向，偏爱地 重/上采样
        # img = ax1.plot(range(0, Iz), array1D if is_energy != 1 else array1D ** 2, color=color_1d)
        
    # kind = 'cubic' # kind = 0,1,2,3 nono，1 维才可以这么写，2 维只有 'linear', 'cubic', 'quintic'
    # f = interp1d(ix, array1D, kind = kind)

    # print(ix)
    # print(array1D)
    f = UnivariateSpline(ix,array1D,s=0) # ix 必须是 严格递增的，若 ix 是 zj 的话，zj 也必须是
    array1D_new = f(ix_new)
    
    #%%
    
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
                xticks, xticklabels = gan_ticks(ix_new[-1], ticks_num, Min=ix_new[0])
                ax1.set_xticks(xticks)
                ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
            else:
                xticks, xticklabels = gan_ticks(Ix*size_PerPixel, ticks_num, is_centered=1)
                ax1.set_xticks(xticks)
                ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
        else:
            xticks, xticklabels = gan_ticks(Iz, ticks_num)
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
        
        vmax = kwargs.get("vmax", np.max(array1D) if is_energy != 1 else np.max(np.abs(array1D)) ** 2)
        vmin = kwargs.get("vmin", np.min(array1D) if is_energy != 1 else np.min(np.abs(array1D)) ** 2)
        # yticks = np.linspace(vmin, vmax, ticks_num + 1)
        yticks, yticklabels = gan_ticks(vmax, ticks_num, Min=vmin)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels(yticklabels, fontsize=fontsize, fontdict=font)

        ax1.set_xlabel(xlabel, fontsize=fontsize, fontdict=font)  # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        ax1.set_ylabel(ylabel, fontsize=fontsize, fontdict=font)  # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize

    #%% 画 第 1 条 曲线
    
    array1D_new = array1D_new if is_energy != 1 else np.abs(array1D_new) ** 2
    if "label" in kwargs:
        l1, = ax1.plot(ix_new, array1D_new, color=color_1d, label=kwargs['label'], )
    else:
        l1, = ax1.plot(ix_new, array1D_new, color=color_1d, )

    # %% 画 第 2 条 曲线

    loc = kwargs.get("loc", 5)
    if "l2" in kwargs:
        ax2 = fig.add_subplot(111, label="2", frameon=False) # 不覆盖 下面的 图层
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
            if "zj2" in kwargs:
                zj2 = kwargs["zj2"]
                if is_axes_on == 0:
                    ax2.axis('off')
                else:
                    if is_mm == 1:  # round(i * size_PerPixel,2) 保留 2 位小数，改为 保留 2 位 有效数字
                        if is_propagation != 0:
                            ax2.set_xticks(zj) # ax2 是 Tz，不像 dkzQ，是非线性变化的，所以不能人工 gan 其刻度，也不能有 ix_new。
                            ax2.set_xticklabels([float('%.2f' % i) for i in zj2], fontsize=fontsize, fontdict=font)
                        else:
                            ax2.set_xticks(xticks)
                            ax2.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
                    else:
                        ax2.set_xticks(xticks)
                        ax2.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
            else:
                ax2.xticks(()) # 否则 ax2 的 x 不设刻度
            
            vmax2 = kwargs.get("vmax2", np.max(kwargs['l2']) if is_energy != 1 else np.max(np.abs(kwargs['l2'])) ** 2)
            vmin2 = kwargs.get("vmin2", np.min(kwargs['l2']) if is_energy != 1 else np.max(np.abs(kwargs['l2'])) ** 2)
            yticks, yticklabels = gan_ticks(vmax2, ticks_num, Min=vmin2)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticklabels, fontsize=fontsize, fontdict=font)

            ax2.set_xlabel(xlabel2, fontsize=fontsize, fontdict=font)
            ax2.set_ylabel(ylabel2, fontsize=fontsize, fontdict=font)

        f = UnivariateSpline(ix, kwargs['l2'], s=0)  # ix 必须是 严格递增的，若 ix 是 zj 的话，zj 也必须是
        l2_new = f(ix_new)
        args = [ix_new, l2_new if is_energy != 1 else np.abs(l2_new) ** 2]
        if "label2" in kwargs:
            l2, = ax2.plot(*args, color=color_1d2, label=kwargs['label2'], )
        else:
            l2, = ax2.plot(*args, color=color_1d2, )

        if "label" in kwargs and "label2" in kwargs:
            plt.legend(handles=[l1, l2], loc=loc)
    else:
        if "label" in kwargs:
            plt.legend(loc=loc)

    array1D_title = array1D_title if is_energy != 1 else array1D_title + "_Squared"
    add_size = kwargs.get("add_size",3)
    if is_title_on:
        # fig.suptitle(array1D_title, fontsize=fontsize+add_size, fontdict=font)
        # sgtitle 放置位置与 suptitle 相似，必须将其放在所有 subplot 的最后
        ax1.set_title(array1D_title, fontsize=fontsize+2, fontdict=font)

    plt.show()

    if is_title_on == 0 and is_axes_on == 0:
        ax1.margins(0, 0)
        if "l2" in kwargs:
            ax2.margins(0, 0)
        if is_save == 1:
            fig.savefig(array1D_address, transparent=True, pad_inches=0)  # 不包含图例等，且无白边
    else:
        if is_save == 1: # bbox_inches='tight' 的缺点是会导致对输出图片的大小设置失效。
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

    #%% 插值 begin
    
    Ix, Iy = array2D.shape[1], array2D.shape[0]
    Iz = len(zj)
    Iz_new = (Iz-1)*sample+1 # zj 区间范围 保持不变，分段数 乘以 sample 后，新划分出的 刻度的个数

    if is_propagation != 0:
        ix, iy = zj, [j for j in range(Iy)]
        ix_new, iy_new = np.linspace(zj[0], zj[-1], Iz_new), iy
    else:
        ix, iy = [i for i in range(Ix)], [j for j in range(Iy)]
        ix_new, iy_new = ix, iy # 非传播 则 不重/上采样
        # ix_new = np.linspace(0, Ix - 1, Ix*sample) # 非传播 则 不对某个方向，偏爱地 重/上采样
        # iy_new = np.linspace(0, Iy - 1, Iy*sample) # 除非将 另一个方向 也上采样 相同倍数
    
    kind = 'cubic' # kind = 0,1,2,3 nono，1 维才可以这么写，2 维只有 'linear', 'cubic', 'quintic'

    # ix_mesh, iy_mesh = np.meshgrid(ix, iy)
    # f = interp2d(ix_mesh,iy_mesh,array2D,kind=kind)
    f = interp2d(ix, iy, array2D, kind=kind)
    array2D_new = f(ix_new, iy_new)
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
                # ax1.set_xticklabels([float('%.2f' % i) for i in ix_new[list(xticks_z)]], fontsize=fontsize, fontdict=font)
                ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
            else:
                xticks, xticklabels = gan_ticks(Ix*size_PerPixel, ticks_num, is_centered=1, I=Ix)
                # array_x = np.arange(0, Ix*size_PerPixel, size_PerPixel)
                # xticks = [find_nearest(array_x, x)[0] for x in xticks_x]
                ax1.set_xticks(xticks)
                ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
            yticks, yticklabels = gan_ticks(Iy*size_PerPixel, ticks_num, is_centered=1, I=Iy)
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

    vmax = kwargs.get("vmax", np.max(array2D) if is_energy != 1 else np.max(np.abs(array2D)) ** 2)
    vmin = kwargs.get("vmin", np.min(array2D) if is_energy != 1 else np.min(np.abs(array2D)) ** 2)
    # 尽管可以放在 is_self_colorbar == 0 的分支中，但 is_colorbar_on == 1 要用到...
    array2D_new = array2D_new if is_energy != 1 else np.abs(array2D_new) ** 2
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
        if is_self_colorbar != 1:  # np.round(np.linspace(vmin if is_energy != 1 else vmin**2, vmax if is_energy != 1 else vmax**2, ticks_num + 1), 2) 保留 2 位小数，改为 保留 2 位 有效数字
            cticks, cticklabels = gan_ticks(vmax, ticks_num, Min=vmin)
            cb.set_ticks(cticks)  # range(vmin if is_energy != 1 else vmin**2, vmax if is_energy != 1 else vmax**2, round((vmax-vmin) / ticks_num ,2)) 其中 range 步长不支持 非整数，只能用 np.arange 或 np.linspace
            cb.set_ticklabels(cticklabels)
        cb.set_label(clabel, fontsize=fontsize, fontdict=font)  # 设置 colorbar 的 标签名、标签字体；字体大小 fontsize=fontsize

    array2D_title = array2D_title if is_energy != 1 else array2D_title + "_Squared"
    add_size = kwargs.get("add_size",3)
    if is_title_on:
        ax1.set_title(array2D_title, fontsize=fontsize+add_size, fontdict=font)

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
    
    #%% 插值 begin
    
    Ix, Iy = U_1.shape[1], U_1.shape[0]
    Iz = len(zj)
    Iz_new = (Iz-1)*sample+1 # zj 区间范围 保持不变，分段数 乘以 sample 后，新划分出的 刻度的个数
    
    ix, iy = zj, [j for j in range(Iy)]
    ix_new, iy_new = np.linspace(zj[0], zj[-1], Iz_new), iy
    
    kind = 'cubic' # kind = 0,1,2,3 nono，1 维才可以这么写，2 维只有 'linear', 'cubic', 'quintic'
    
    f = interp2d(ix,iy,U_YZ,kind=kind)
    U_YZ_new = f(ix_new, iy_new)
    f = interp2d(ix,iy,U_XZ,kind=kind)
    U_XZ_new = f(ix_new, iy_new)
    #%% 插值 end

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
        #     # ax1.set_xticklabels([float('%.2f' % i) for i in ix_new[list(xticks_z)]], fontsize=fontsize, fontdict=font) # Pair 1
        #     ax1.set_xticklabels([float('%.2f' % i) for i in xticks_z], fontsize=fontsize, fontdict=font)
        #     # ax1.set_xticklabels([float('%.2f' % (k * diz * size_PerPixel)) for k in range(0, Iz, Iz // ticks_num)],
        #     #                      fontsize=fontsize, fontdict=font)
        #     ax1.set_yticklabels([float('%.2f' % i) for i in xticks_x], fontsize=fontsize, fontdict=font)
        #     ax1.set_zticklabels([float('%.2f' % j) for j in yticks_y], fontsize=fontsize, fontdict=font)
        # else:
        #     ax1.set_xticklabels(xticks_z, fontsize=fontsize, fontdict=font)
        #     ax1.set_yticklabels(xticks, fontsize=fontsize, fontdict=font)
        #     ax1.set_zticklabels(yticks, fontsize=fontsize, fontdict=font)
        
        if is_mm == 1:
            xticks, xticklabels = gan_ticks(ix_new[-1], ticks_num, Min=ix_new[0], I=Iz_new)
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticklabels, fontsize=fontsize, fontdict=font)
            
            yticks, yticklabels = gan_ticks(Ix*size_PerPixel, ticks_num, is_centered=1, I=Ix)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticklabels, fontsize=fontsize, fontdict=font)
            
            zticks, zticklabels = gan_ticks(Iy*size_PerPixel, ticks_num, is_centered=1, I=Iy)
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

    U_YZ_new = U_YZ_new if is_energy != 1 else np.abs(U_YZ_new) ** 2
    U_XZ_new = U_XZ_new if is_energy != 1 else np.abs(U_XZ_new) ** 2
    U_1 = U_1 if is_energy != 1 else np.abs(U_1) ** 2
    U_2 = U_2 if is_energy != 1 else np.abs(U_2) ** 2
    if is_show_structure_face == 1:
        U_structure_front = U_structure_front if is_energy != 1 else np.abs(U_structure_front) ** 2
        U_structure_end = U_structure_end if is_energy != 1 else np.abs(U_structure_end) ** 2

    if is_show_structure_face == 1:
        UXY = np.dstack((U_YZ, U_XZ))
        UZ = np.dstack((U_1, U_2, U_structure_front, U_structure_end))
    else:
        UXY = np.dstack((U_YZ, U_XZ))
        UZ = np.dstack((U_1, U_2))
    vmax = kwargs.get("vmax", max(np.max(UXY),np.max(UZ)) if is_energy != 1 else max(np.max(np.abs(UXY)),np.max(np.abs(UZ))) ** 2)
    vmin = kwargs.get("vmin", min(np.min(UXY),np.min(UZ)) if is_energy != 1 else min(np.min(np.abs(UXY)),np.min(np.abs(UZ))) ** 2)
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
        img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_1])[0], i_X, i_Y, c=U_1, cmap=cmap_3d, # ix_new.tolist().index(zj[iZ_1])
                             alpha=math.e ** (-1 * alpha))
        img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_2])[0], i_X, i_Y, c=U_2, cmap=cmap_3d, # ix_new.tolist().index(zj[iZ_2])
                             alpha=math.e ** (-1 * alpha))

        if is_show_structure_face == 1:
            img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_structure_front])[0], i_X, i_Y, # ix_new.tolist().index(zj[iZ_structure_front])
                                 c=U_structure_front, cmap=cmap_3d, alpha=math.e ** (-1 * alpha))
            img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_structure_end])[0], i_X, i_Y, # ix_new.tolist().index(zj[iZ_structure_end])
                                 c=U_structure_end, cmap=cmap_3d, alpha=math.e ** (-1 * alpha))
    else:
        i_Z, i_XY = np.meshgrid([i for i in range(Iz_new)], [j for j in range(Ixy)])
        i_XY = i_XY[::-1]
        img = ax1.scatter3D(i_Z, iX, i_XY, c=U_YZ_new, cmap=cmap_3d, alpha=math.e ** (-1 * alpha), vmin=vmin, vmax=vmax)
        i_XY = i_XY[::-1]
        img = ax1.scatter3D(i_Z, i_XY, iY, c=U_XZ_new, cmap=cmap_3d, alpha=math.e ** (-1 * alpha), vmin=vmin, vmax=vmax)

        i_X, i_Y = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        i_Y = i_Y[::-1]
        img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_1])[0], i_X, i_Y, c=U_1, cmap=cmap_3d, # ix_new.tolist().index(zj[iZ_1])
                             alpha=math.e ** (-1 * alpha), vmin=vmin, vmax=vmax)
        img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_2])[0], i_X, i_Y, c=U_2, cmap=cmap_3d, # ix_new.tolist().index(zj[iZ_2])
                             alpha=math.e ** (-1 * alpha), vmin=vmin, vmax=vmax)

        if is_show_structure_face == 1:
            img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_structure_front])[0], i_X, i_Y, # ix_new.tolist().index(zj[iZ_structure_front])
                                 c=U_structure_front, cmap=cmap_3d, alpha=math.e ** (-1 * alpha), vmin=vmin, vmax=vmax)
            img = ax1.scatter3D(find_nearest(ix_new, zj[iZ_structure_end])[0], i_X, i_Y, # ix_new.tolist().index(zj[iZ_structure_end])
                                 c=U_structure_end, cmap=cmap_3d, alpha=math.e ** (-1 * alpha), vmin=vmin, vmax=vmax)

    if is_colorbar_on == 1:
        cax = add_right_cax(ax1, pad=0.05, width=0.05)
        cb = fig.colorbar(img, cax=cax)
        # cb = fig.colorbar(img, cax=cax, extend='both')
        cb.ax.tick_params(labelsize=fontsize)  # 设置 colorbar 刻度字体；字体大小 labelsize=fontsize。 # Text 对象没有 fontdict 标签
        if is_self_colorbar != 1:  # np.round(np.linspace(vmin if is_energy != 1 else vmin**2, vmax if is_energy != 1 else vmax**2, ticks_num + 1), 2) 保留 2 位小数，改为 保留 2 位 有效数字
            cticks, cticklabels = gan_ticks(vmax, ticks_num, Min=vmin)
            cb.set_ticks(cticks)
            cb.set_ticklabels(cticklabels)
        cb.set_label(clabel, fontsize=fontsize, fontdict=font)  # 设置 colorbar 的 标签名、标签字体；字体大小 fontsize=fontsize

    img_title = img_title if is_energy != 1 else img_title + "_Squared"
    add_size = kwargs.get("add_size",3)
    if is_title_on:
        ax1.set_title(img_title, fontsize=fontsize+add_size, fontdict=font)

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
    Iz_new = (Iz-1)*sample+1 # zj 区间范围 保持不变，分段数 乘以 sample 后，新划分出的 刻度的个数
    
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
            
            yticks, yticklabels = gan_ticks(Ix*size_PerPixel, ticks_num, is_centered=1, I=Ix)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticklabels, fontsize=fontsize, fontdict=font)
            
            zticks, zticklabels = gan_ticks(Iy*size_PerPixel, ticks_num, is_centered=1, I=Iy)
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

    vmax = kwargs.get("vmax", np.max(U_z_stored) if is_energy != 1 else np.max(np.abs(U_z_stored)) ** 2)
    vmin = kwargs.get("vmin", np.min(U_z_stored) if is_energy != 1 else np.min(np.abs(U_z_stored)) ** 2)
    # 尽管可以放在 is_self_colorbar == 0 的分支中，但 is_colorbar_on == 1 要用到...
    if is_self_colorbar == 1:
        i_X, i_Y = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        i_Y = i_Y[::-1]
        for sheet_stored_th in range(sheets_stored_num + 1):
            img = ax1.scatter3D(find_nearest(ix_new, z_stored[sheet_stored_th])[0], i_X, i_Y, # ix_new.tolist().index(z_stored[sheet_stored_th])
                                 c=U_z_stored[:, :, sheet_stored_th] if is_energy != 1 else np.abs(
                                     U_z_stored[:, :, sheet_stored_th]) ** 2, cmap=cmap_3d,
                                 alpha=math.e ** -3 * math.e ** (-1 * alpha * sheet_stored_th / sheets_stored_num))
    else:
        i_X, i_Y = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        i_Y = i_Y[::-1]
        for sheet_stored_th in range(sheets_stored_num + 1):
            img = ax1.scatter3D(find_nearest(ix_new, z_stored[sheet_stored_th])[0], i_X, i_Y, # ix_new.tolist().index(z_stored[sheet_stored_th])
                                 c=U_z_stored[:, :, sheet_stored_th] if is_energy != 1 else np.abs(
                                     U_z_stored[:, :, sheet_stored_th]) ** 2, cmap=cmap_3d,
                                 alpha=math.e ** -3 * math.e ** (-1 * alpha * sheet_stored_th / sheets_stored_num),
                                 vmin=vmin, vmax=vmax)

    if is_colorbar_on == 1:
        cax = add_right_cax(ax1, pad=0.05, width=0.05)
        cb = fig.colorbar(img, cax=cax)
        # cb = fig.colorbar(img, cax=cax, extend='both')
        cb.ax.tick_params(labelsize=fontsize)  # 设置 colorbar 刻度字体；字体大小 labelsize=fontsize。 # Text 对象没有 fontdict 标签
        if is_self_colorbar != 1:  # np.round(np.linspace(vmin if is_energy != 1 else vmin**2, vmax if is_energy != 1 else vmax**2, ticks_num + 1), 2) 保留 2 位小数，改为 保留 2 位 有效数字
           cticks, cticklabels = gan_ticks(vmax, ticks_num, Min=vmin)
           cb.set_ticks(cticks)
           cb.set_ticklabels(cticklabels)
        cb.set_label(clabel, fontsize=fontsize, fontdict=font)  # 设置 colorbar 的 标签名、标签字体；字体大小 fontsize=fontsize

    img_title = img_title if is_energy != 1 else img_title + "_Squared"
    add_size = kwargs.get("add_size",3)
    if is_title_on:
        ax1.set_title(img_title, fontsize=fontsize+add_size, fontdict=font)

    plt.show()

    if is_title_on == 0 and is_axes_on == 0 and is_colorbar_on == 0:
        ax1.margins(0, 0, 0)
        if is_save == 1:
            fig.savefig(img_address, transparent=True, pad_inches=0)  # 不包含图例等，且无白边
    else:
        if is_save == 1:
            fig.savefig(img_address, transparent=True, bbox_inches='tight')  # 包含图例等，但有白边
            # fig.savefig(img_address, transparent = True, bbox_inches='tight', pad_inches=0) # 包含图例，且无白边