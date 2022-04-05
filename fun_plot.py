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
import matplotlib.pyplot as plt
# from mpl_toolkits import axes_grid1
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline, interp1d, interp2d, griddata
from fun_algorithm import find_nearest

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
            is_energy=1, vmax=1, vmin=0):

    # %%

    fig, axes = plt.subplots(1, 1, figsize=(size_fig_x, size_fig_y), dpi=dpi)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if is_title_on:
        axes.set_title(array1D_title if is_energy != 1 else array1D_title + "_Squared", fontsize=fontsize,
                       fontdict=font)
    
    Ix = array1D.shape[0]
    Iz = len(zj)
    Iz_new = (Iz-1)*sample+1 # zj 区间范围 保持不变，分段数 乘以 sample 后，新划分出的 刻度的个数
    
    if is_axes_on == 0:
        axes.axis('off')
    else:
        if is_propagation != 0:
            axes.set_xticks(np.linspace(0, zj[-1], ticks_num + 1))
            # axes.set_xticks(range(0, Iz, Iz // ticks_num))
        else:
            axes.set_xticks(range(0, Iz, Iz // ticks_num))  # 按理 等价于 np.linspace(0, Ix, ticks_num + 1)，但并不
        axes.set_yticks([float(format(y, '.2g')) for y in
                         np.linspace(vmin if is_energy != 1 else vmin ** 2, vmax if is_energy != 1 else vmax ** 2,
                                     ticks_num + 1)])  # 按理 等价于 np.linspace(0, Iy, ticks_num + 1)，但并不
        if is_mm == 1:  # round(i * size_PerPixel,2) 保留 2 位小数，改为 保留 2 位 有效数字
            if is_propagation != 0:
                axes.set_xticklabels([float('%.2g' % i) for i in np.linspace(0, zj[-1], ticks_num + 1)], 
                                     fontsize=fontsize, fontdict=font)
                # axes.set_xticklabels([float('%.2g' % (i * diz * size_PerPixel)) for i in range(0, Iz, Iz // ticks_num)],
                #                      fontsize=fontsize, fontdict=font)
            else:
                axes.set_xticklabels(
                    [float('%.2g' % (i * size_PerPixel)) for i in range(- Ix // 2, Ix - Ix // 2, Ix // ticks_num)],
                    fontsize=fontsize, fontdict=font)
            axes.set_yticklabels([float(format(y, '.2g')) for y in np.linspace(vmin if is_energy != 1 else vmin ** 2,
                                                                               vmax if is_energy != 1 else vmax ** 2,
                                                                               ticks_num + 1)], fontsize=fontsize,
                                 fontdict=font)
        else:
            axes.set_xticklabels(range(0, Iz, Iz // ticks_num), fontsize=fontsize, fontdict=font)
            axes.set_yticklabels([float(format(y, '.2g')) for y in np.linspace(vmin if is_energy != 1 else vmin ** 2,
                                                                               vmax if is_energy != 1 else vmax ** 2,
                                                                               ticks_num + 1)], fontsize=fontsize,
                                 fontdict=font)
        axes.set_xlabel('', fontsize=fontsize, fontdict=font)  # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        axes.set_ylabel('', fontsize=fontsize, fontdict=font)  # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize

    #%% 插值 begin
    
    if is_propagation != 0:
        ix = zj
        ix_new = np.linspace(0, zj[-1], Iz_new)
        # img = axes.plot(zj, array1D if is_energy != 1 else array1D ** 2, color=color_1d)
    else:
        ix = [i for i in range(Ix)]
        ix_new = np.linspace(0, Ix - 1, Ix) # 非传播 则 不对某个方向，偏爱地 重/上采样
        # img = axes.plot(range(0, Iz), array1D if is_energy != 1 else array1D ** 2, color=color_1d)
        
    # kind = 'cubic' # kind = 0,1,2,3 nono，1 维才可以这么写，2 维只有 'linear', 'cubic', 'quintic'
    # f = interp1d(ix, array1D, kind = kind)
    
    f = UnivariateSpline(ix,array1D,s=0)
    array1D_new = f(ix_new)
    
    #%% 插值 end
    
    img = axes.plot(ix_new, array1D_new if is_energy != 1 else array1D_new ** 2, color=color_1d)

    plt.show()

    if is_title_on == 0 and is_axes_on == 0:
        axes.margins(0, 0)
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
            is_self_colorbar=1, is_colorbar_on=1, is_energy=1, vmax=1, vmin=0,
            *args, ):


    # %%
    fig, axes = plt.subplots(1, 1, figsize=(size_fig, size_fig), dpi=dpi)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    if is_title_on:
        axes.set_title(array2D_title if is_energy != 1 else array2D_title + "_Squared", fontsize=fontsize,
                       fontdict=font)

    #%% 插值 begin
    
    Ix, Iy = array2D.shape[1], array2D.shape[0]
    Iz = len(zj)
    Iz_new = (Iz-1)*sample+1 # zj 区间范围 保持不变，分段数 乘以 sample 后，新划分出的 刻度的个数
    
    if is_propagation != 0:
        ix, iy = zj, [j for j in range(Iy)]
        ix_new, iy_new = np.linspace(0, zj[-1], Iz_new), iy
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
        axes.axis('off')
    else:
        # plt.xticks(range(0, Ix, Ix // ticks_num), fontsize=fontsize) # Text 对象没有 fontdict 标签
        # plt.yticks(range(0, Iy, Iy // ticks_num), fontsize=fontsize) # Text 对象没有 fontdict 标签
        if is_propagation != 0:
            axes.set_xticks(range(0, Iz_new, Iz_new // ticks_num))  # Pair 1
            # axes.set_xticks([i for i in np.linspace(0, Iz_new, ticks_num + 1)]) # Pair 2
            # axes.set_xticks(np.linspace(0, zj[-1], ticks_num + 1))
        else:
            axes.set_xticks(range(0, Ix, Ix // ticks_num))  # 按理 等价于 np.linspace(0, Ix, ticks_num + 1)，但并不
        axes.set_yticks(range(0, Iy, Iy // ticks_num))  # 按理 等价于 np.linspace(0, Iy, ticks_num + 1)，但并不
        if is_mm == 1:  # round(i * size_PerPixel,2) 保留 2 位小数，改为 保留 2 位 有效数字
            if is_propagation != 0:
                axes.set_xticklabels([float('%.2g' % i) for i in ix_new[list(range(0, Iz_new, Iz_new // ticks_num))]],
                                     fontsize=fontsize, fontdict=font)  # Pair 1 # 用之前整数决定的步长，进行花式索引（传个 list 进去）
                # axes.set_xticklabels([float('%.2g' % i) for i in np.linspace(0, zj[-1], ticks_num + 1)], fontsize=fontsize, fontdict=font) # Pair 2
                # axes.set_xticklabels([float('%.2g' % i) for i in np.arrange(0, zj[-1], zj[-1] // ticks_num)], fontsize=fontsize, fontdict=font)
                # axes.set_xticklabels([float('%.2g' % (i * diz * size_PerPixel)) for i in range(0, Ix, Ix // ticks_num)],
                #                       fontsize=fontsize, fontdict=font)
            else:
                axes.set_xticklabels(
                    [float('%.2g' % (i * size_PerPixel)) for i in range(- Ix // 2, Ix - Ix // 2, Ix // ticks_num)],
                    fontsize=fontsize, fontdict=font)
            axes.set_yticklabels(
                [- float('%.2g' % (j * size_PerPixel)) for j in range(- Iy // 2, Iy - Iy // 2, Iy // ticks_num)],
                fontsize=fontsize, fontdict=font)
        else:
            axes.set_xticklabels(range(0, Ix, Ix // ticks_num), fontsize=fontsize, fontdict=font)
            axes.set_yticklabels(range(0, Iy, Iy // ticks_num), fontsize=fontsize, fontdict=font)
        axes.set_xlabel('', fontsize=fontsize, fontdict=font)  # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        axes.set_ylabel('', fontsize=fontsize, fontdict=font)  # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        # plt.xlabel('', fontsize=fontsize, fontdict=font) # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        # plt.ylabel('', fontsize=fontsize, fontdict=font) # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize

    if is_contourf == 1:
        if is_self_colorbar == 1:
            img = axes.contourf(array2D_new if is_energy != 1 else array2D_new ** 2, cmap=cmap_2d, )
        else:
            img = axes.contourf(array2D_new if is_energy != 1 else array2D_new ** 2, cmap=cmap_2d,
                                vmin=vmin if is_energy != 1 else vmin ** 2,
                                vmax=vmax if is_energy != 1 else vmax ** 2, )
    else:
        if is_self_colorbar == 1:
            img = axes.imshow(array2D_new if is_energy != 1 else array2D_new ** 2, cmap=cmap_2d, )
        else:
            img = axes.imshow(array2D_new if is_energy != 1 else array2D_new ** 2, cmap=cmap_2d,
                              vmin=vmin if is_energy != 1 else vmin ** 2,
                              vmax=vmax if is_energy != 1 else vmax ** 2, )

    if is_colorbar_on == 1:
        cax = add_right_cax(axes, pad=0.05, width=0.05)
        cb = fig.colorbar(img, cax=cax)
        # cb = fig.colorbar(img, cax=cax, extend='both')
        cb.ax.tick_params(labelsize=fontsize)  # 设置 colorbar 刻度字体；字体大小 labelsize=fontsize。 # Text 对象没有 fontdict 标签
        if is_self_colorbar != 1:  # np.round(np.linspace(vmin if is_energy != 1 else vmin**2, vmax if is_energy != 1 else vmax**2, ticks_num + 1), 2) 保留 2 位小数，改为 保留 2 位 有效数字
            cb.set_ticks([float(format(x, '.2g')) for x in
                          np.linspace(vmin if is_energy != 1 else vmin ** 2, vmax if is_energy != 1 else vmax ** 2,
                                      ticks_num + 1)])  # range(vmin if is_energy != 1 else vmin**2, vmax if is_energy != 1 else vmax**2, round((vmax-vmin) / ticks_num ,2)) 其中 range 步长不支持 非整数，只能用 np.arange 或 np.linspace
            cb.set_ticklabels([float(format(x, '.2g')) for x in
                               np.linspace(vmin if is_energy != 1 else vmin ** 2, vmax if is_energy != 1 else vmax ** 2,
                                           ticks_num + 1)])
        cb.set_label('', fontsize=fontsize, fontdict=font)  # 设置 colorbar 的 标签名、标签字体；字体大小 fontsize=fontsize

    if is_title_on == 0 and is_axes_on == 0 and is_colorbar_on == 0:
        axes.margins(0, 0)
        if is_save == 1:
            fig.savefig(array2D_address, transparent=True, pad_inches=0)  # 不包含图例等，且无白边
    else:
        if is_save == 1:
            fig.savefig(array2D_address, transparent=True, bbox_inches='tight')  # 包含图例等，但有白边
            # fig.savefig(array2D_address, transparent = True, bbox_inches='tight', pad_inches=0) # 包含图例，且无白边

    plt.show()
    # plt.cla() # 清除所有 活动的 axes，但其他不关
    # plt.clf() # 清除所有 axes，但 fig 不关，可用同一个 fig 作图 新的 axes（复用 设定好的 同一个 fig）
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
                is_self_colorbar=1, is_colorbar_on=1, is_energy=1, vmax=1, vmin=0):

    # %%

    fig = plt.figure(figsize=(size_fig * 10, size_fig * 10), dpi=dpi)
    axes = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if is_title_on:
        axes.set_title(img_title if is_energy != 1 else img_title + "_Squared", fontsize=fontsize, fontdict=font)
    
    #%% 插值 begin
    
    Ix, Iy = U_1.shape[1], U_1.shape[0]
    Iz = len(zj)
    Iz_new = (Iz-1)*sample+1 # zj 区间范围 保持不变，分段数 乘以 sample 后，新划分出的 刻度的个数
    
    ix, iy = zj, [j for j in range(Iy)]
    ix_new, iy_new = np.linspace(0, zj[-1], Iz_new), iy
    
    kind = 'cubic' # kind = 0,1,2,3 nono，1 维才可以这么写，2 维只有 'linear', 'cubic', 'quintic'
    
    f = interp2d(ix,iy,U_YZ,kind=kind)
    U_YZ_new = f(ix_new, iy_new)
    f = interp2d(ix,iy,U_XZ,kind=kind)
    U_XZ_new = f(ix_new, iy_new)
    #%% 插值 end

    if is_axes_on == 0:
        axes.axis('off')
    else:
        axes.set_xticks(range(0, Iz_new, Iz_new // ticks_num)) # Pair 1
        # axes.set_xticks(range(0, Iz, Iz // ticks_num))  # 按理 等价于 np.linspace(0, Ix, ticks_num + 1)，但并不
        axes.set_yticks(range(0, Ix, Ix // ticks_num))  # 按理 等价于 np.linspace(0, Iy, ticks_num + 1)，但并不
        axes.set_zticks(range(0, Iy, Iy // ticks_num))
        if is_mm == 1:  # round(i * size_PerPixel,2) 保留 2 位小数，改为 保留 2 位 有效数字
            axes.set_xticklabels([float('%.2g' % i) for i in ix_new[list(range(0, Iz_new, Iz_new // ticks_num))]], 
                                 fontsize=fontsize, fontdict=font) # Pair 1
            # axes.set_xticklabels([float('%.2g' % (k * diz * size_PerPixel)) for k in range(0, Iz, Iz // ticks_num)],
            #                      fontsize=fontsize, fontdict=font)
            axes.set_yticklabels(
                [float('%.2g' % (i * size_PerPixel)) for i in range(- Ix // 2, Ix - Ix // 2, Ix // ticks_num)],
                fontsize=fontsize, fontdict=font)
            axes.set_zticklabels(
                [float('%.2g' % (j * size_PerPixel)) for j in range(- Iy // 2, Iy - Iy // 2, Iy // ticks_num)],
                fontsize=fontsize, fontdict=font)
        else:
            axes.set_xticklabels(range(0, Iz_new, Iz_new // ticks_num), fontsize=fontsize, fontdict=font)
            axes.set_yticklabels(range(0, Ix, Ix // ticks_num), fontsize=fontsize, fontdict=font)
            axes.set_zticklabels(range(0, Iy, Iy // ticks_num), fontsize=fontsize, fontdict=font)
        axes.set_xlabel('Z', fontsize=fontsize, fontdict=font)  # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        axes.set_ylabel('X', fontsize=fontsize, fontdict=font)  # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        axes.set_zlabel('Y', fontsize=fontsize, fontdict=font)  # 设置 z 轴的 标签名、标签字体；字体大小 fontsize=fontsize

    axes.view_init(elev=elev, azim=azim);  # 后一个为负 = 绕 z 轴逆时针
    
    Ixy = Iy
    if is_self_colorbar == 1:
        i_Z, i_XY = np.meshgrid([i for i in range(Iz_new)], [j for j in range(Ixy)])
        i_XY = i_XY[::-1]
        img = axes.scatter3D(i_Z, iX, i_XY, c=U_YZ_new if is_energy != 1 else U_YZ_new ** 2, cmap=cmap_3d,
                             alpha=math.e ** (-1 * alpha))
        i_XY = i_XY[::-1]
        img = axes.scatter3D(i_Z, i_XY, iY, c=U_XZ_new if is_energy != 1 else U_XZ_new ** 2, cmap=cmap_3d,
                             alpha=math.e ** (-1 * alpha))

        i_X, i_Y = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        i_Y = i_Y[::-1]
        img = axes.scatter3D(find_nearest(ix_new, zj[iZ_1])[0], i_X, i_Y, c=U_1 if is_energy != 1 else U_1 ** 2, cmap=cmap_3d, # ix_new.tolist().index(zj[iZ_1])
                             alpha=math.e ** (-1 * alpha))
        img = axes.scatter3D(find_nearest(ix_new, zj[iZ_2])[0], i_X, i_Y, c=U_2 if is_energy != 1 else U_2 ** 2, cmap=cmap_3d, # ix_new.tolist().index(zj[iZ_2])
                             alpha=math.e ** (-1 * alpha))

        if is_show_structure_face == 1:
            img = axes.scatter3D(find_nearest(ix_new, zj[iZ_structure_front])[0], i_X, i_Y, # ix_new.tolist().index(zj[iZ_structure_front])
                                 c=U_structure_front if is_energy != 1 else U_structure_front ** 2, cmap=cmap_3d,
                                 alpha=math.e ** (-1 * alpha))
            img = axes.scatter3D(find_nearest(ix_new, zj[iZ_structure_end])[0], i_X, i_Y, # ix_new.tolist().index(zj[iZ_structure_end])
                                 c=U_structure_end if is_energy != 1 else U_structure_end ** 2, cmap=cmap_3d,
                                 alpha=math.e ** (-1 * alpha))
    else:
        i_Z, i_XY = np.meshgrid([i for i in range(Iz_new)], [j for j in range(Ixy)])
        i_XY = i_XY[::-1]
        img = axes.scatter3D(i_Z, iX, i_XY, c=U_YZ_new if is_energy != 1 else U_YZ_new ** 2, cmap=cmap_3d,
                             alpha=math.e ** (-1 * alpha), vmin=vmin if is_energy != 1 else vmin ** 2,
                             vmax=vmax if is_energy != 1 else vmax ** 2)
        i_XY = i_XY[::-1]
        img = axes.scatter3D(i_Z, i_XY, iY, c=U_XZ_new if is_energy != 1 else U_XZ_new ** 2, cmap=cmap_3d,
                             alpha=math.e ** (-1 * alpha), vmin=vmin if is_energy != 1 else vmin ** 2,
                             vmax=vmax if is_energy != 1 else vmax ** 2)

        i_X, i_Y = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        i_Y = i_Y[::-1]
        img = axes.scatter3D(find_nearest(ix_new, zj[iZ_1])[0], i_X, i_Y, c=U_1 if is_energy != 1 else U_1 ** 2, cmap=cmap_3d, # ix_new.tolist().index(zj[iZ_1])
                             alpha=math.e ** (-1 * alpha), vmin=vmin if is_energy != 1 else vmin ** 2,
                             vmax=vmax if is_energy != 1 else vmax ** 2)
        img = axes.scatter3D(find_nearest(ix_new, zj[iZ_2])[0], i_X, i_Y, c=U_2 if is_energy != 1 else U_2 ** 2, cmap=cmap_3d, # ix_new.tolist().index(zj[iZ_2])
                             alpha=math.e ** (-1 * alpha), vmin=vmin if is_energy != 1 else vmin ** 2,
                             vmax=vmax if is_energy != 1 else vmax ** 2)

        if is_show_structure_face == 1:
            img = axes.scatter3D(find_nearest(ix_new, zj[iZ_structure_front])[0], i_X, i_Y, # ix_new.tolist().index(zj[iZ_structure_front])
                                 c=U_structure_front if is_energy != 1 else U_structure_front ** 2, cmap=cmap_3d,
                                 alpha=math.e ** (-1 * alpha), vmin=vmin if is_energy != 1 else vmin ** 2,
                                 vmax=vmax if is_energy != 1 else vmax ** 2)
            img = axes.scatter3D(find_nearest(ix_new, zj[iZ_structure_end])[0], i_X, i_Y, # ix_new.tolist().index(zj[iZ_structure_end])
                                 c=U_structure_end if is_energy != 1 else U_structure_end ** 2, cmap=cmap_3d,
                                 alpha=math.e ** (-1 * alpha), vmin=vmin if is_energy != 1 else vmin ** 2,
                                 vmax=vmax if is_energy != 1 else vmax ** 2)

    if is_colorbar_on == 1:
        cax = add_right_cax(axes, pad=0.05, width=0.05)
        cb = fig.colorbar(img, cax=cax)
        # cb = fig.colorbar(img, cax=cax, extend='both')
        cb.ax.tick_params(labelsize=fontsize)  # 设置 colorbar 刻度字体；字体大小 labelsize=fontsize。 # Text 对象没有 fontdict 标签
        if is_self_colorbar != 1:  # np.round(np.linspace(vmin if is_energy != 1 else vmin**2, vmax if is_energy != 1 else vmax**2, ticks_num + 1), 2) 保留 2 位小数，改为 保留 2 位 有效数字
            cb.set_ticks([float(format(x, '.2g')) for x in
                          np.linspace(vmin if is_energy != 1 else vmin ** 2, vmax if is_energy != 1 else vmax ** 2,
                                      ticks_num + 1)])  # range(vmin if is_energy != 1 else vmin**2, vmax if is_energy != 1 else vmax**2, round((vmax-vmin) / ticks_num ,2)) 其中 range 步长不支持 非整数，只能用 np.arange 或 np.linspace
            cb.set_ticklabels([float(format(x, '.2g')) for x in
                               np.linspace(vmin if is_energy != 1 else vmin ** 2, vmax if is_energy != 1 else vmax ** 2,
                                           ticks_num + 1)])
        cb.set_label('', fontsize=fontsize, fontdict=font)  # 设置 colorbar 的 标签名、标签字体；字体大小 fontsize=fontsize

    plt.show()

    if is_title_on == 0 and is_axes_on == 0 and is_colorbar_on == 0:
        axes.margins(0, 0, 0)
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
                is_energy=1, vmax=1, vmin=0):

    # %%

    fig = plt.figure(figsize=(size_fig * 10, size_fig * 10), dpi=dpi)
    axes = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if is_title_on:
        axes.set_title(img_title if is_energy != 1 else img_title + "_Squared", fontsize=fontsize, fontdict=font)

    Ix, Iy = U_z_stored[:, :, 0].shape[1], U_z_stored[:, :, 0].shape[0]
    Iz = len(zj)
    Iz_new = (Iz-1)*sample+1 # zj 区间范围 保持不变，分段数 乘以 sample 后，新划分出的 刻度的个数
    
    ix, iy = zj, [j for j in range(Iy)]
    ix_new, iy_new = np.linspace(0, zj[-1], Iz_new), iy
    
    if is_axes_on == 0:
        axes.axis('off')
    else:
        axes.set_xticks(range(0, Iz_new, Iz_new // ticks_num))  # 按理 等价于 np.linspace(0, Ix, ticks_num + 1)，但并不
        axes.set_yticks(range(0, Ix, Ix // ticks_num))  # 按理 等价于 np.linspace(0, Iy, ticks_num + 1)，但并不
        axes.set_zticks(range(0, Iy, Iy // ticks_num))
        if is_mm == 1:  # round(i * size_PerPixel,2) 保留 2 位小数，改为 保留 2 位 有效数字
            axes.set_xticklabels([float('%.2g' % i) for i in ix_new[list(range(0, Iz_new, Iz_new // ticks_num))]], 
                                 fontsize=fontsize, fontdict=font) # Pair 1
            # axes.set_xticklabels([float('%.2g' % (k * diz * size_PerPixel)) for k in range(0, Iz, Iz // ticks_num)],
            #                      fontsize=fontsize, fontdict=font)
            axes.set_yticklabels(
                [float('%.2g' % (i * size_PerPixel)) for i in range(- Ix // 2, Ix - Ix // 2, Ix // ticks_num)],
                fontsize=fontsize, fontdict=font)
            axes.set_zticklabels(
                [float('%.2g' % (j * size_PerPixel)) for j in range(- Iy // 2, Iy - Iy // 2, Iy // ticks_num)],
                fontsize=fontsize, fontdict=font)
        else:
            axes.set_xticklabels(range(0, Iz_new, Iz_new // ticks_num), fontsize=fontsize, fontdict=font)
            axes.set_yticklabels(range(0, Ix, Ix // ticks_num), fontsize=fontsize, fontdict=font)
            axes.set_zticklabels(range(0, Iy, Iy // ticks_num), fontsize=fontsize, fontdict=font)
        axes.set_xlabel('Z', fontsize=fontsize, fontdict=font)  # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        axes.set_ylabel('X', fontsize=fontsize, fontdict=font)  # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        axes.set_zlabel('Y', fontsize=fontsize, fontdict=font)  # 设置 z 轴的 标签名、标签字体；字体大小 fontsize=fontsize

    axes.view_init(elev=elev, azim=azim);  # 后一个为负 = 绕 z 轴逆时针

    sheets_stored_num = len(z_stored) - 1
    x_stretch_factor = sheets_stored_num ** 0.5 * 2
    # axes.get_proj = lambda: np.dot(Axes3D.get_proj(axes), np.diag([1 * x_stretch_factor, 1, 1, 1]))
    axes.get_proj = lambda: np.dot(Axes3D.get_proj(axes), np.diag([1, 1 / x_stretch_factor, 1 / x_stretch_factor, 1]))
    # axes.get_proj = lambda: np.dot(Axes3D.get_proj(axes), np.diag([1, 1/x_stretch_factor, 1/x_stretch_factor, 1/x_stretch_factor]))

    if is_self_colorbar == 1:
        i_X, i_Y = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        i_Y = i_Y[::-1]
        for sheet_stored_th in range(sheets_stored_num + 1):
            img = axes.scatter3D(find_nearest(ix_new, z_stored[sheet_stored_th])[0], i_X, i_Y, # ix_new.tolist().index(z_stored[sheet_stored_th])
                                 c=np.abs(U_z_stored[:, :, sheet_stored_th]) if is_energy != 1 else np.abs(
                                     U_z_stored[:, :, sheet_stored_th]) ** 2, cmap=cmap_3d,
                                 alpha=math.e ** -3 * math.e ** (-1 * alpha * sheet_stored_th / sheets_stored_num))
    else:
        i_X, i_Y = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        i_Y = i_Y[::-1]
        for sheet_stored_th in range(sheets_stored_num + 1):
            img = axes.scatter3D(find_nearest(ix_new, z_stored[sheet_stored_th])[0], i_X, i_Y, # ix_new.tolist().index(z_stored[sheet_stored_th])
                                 c=np.abs(U_z_stored[:, :, sheet_stored_th]) if is_energy != 1 else np.abs(
                                     U_z_stored[:, :, sheet_stored_th]) ** 2, cmap=cmap_3d,
                                 alpha=math.e ** -3 * math.e ** (-1 * alpha * sheet_stored_th / sheets_stored_num),
                                 vmin=vmin if is_energy != 1 else vmin ** 2, vmax=vmax if is_energy != 1 else vmax ** 2)

    if is_colorbar_on == 1:
        cax = add_right_cax(axes, pad=0.05, width=0.05)
        cb = fig.colorbar(img, cax=cax)
        # cb = fig.colorbar(img, cax=cax, extend='both')
        cb.ax.tick_params(labelsize=fontsize)  # 设置 colorbar 刻度字体；字体大小 labelsize=fontsize。 # Text 对象没有 fontdict 标签
        if is_self_colorbar != 1:  # np.round(np.linspace(vmin if is_energy != 1 else vmin**2, vmax if is_energy != 1 else vmax**2, ticks_num + 1), 2) 保留 2 位小数，改为 保留 2 位 有效数字
            cb.set_ticks([float(format(x, '.2g')) for x in
                          np.linspace(vmin if is_energy != 1 else vmin ** 2, vmax if is_energy != 1 else vmax ** 2,
                                      ticks_num + 1)])  # range(vmin if is_energy != 1 else vmin**2, vmax if is_energy != 1 else vmax**2, round((vmax-vmin) / ticks_num ,2)) 其中 range 步长不支持 非整数，只能用 np.arange 或 np.linspace
            cb.set_ticklabels([float(format(x, '.2g')) for x in
                               np.linspace(vmin if is_energy != 1 else vmin ** 2, vmax if is_energy != 1 else vmax ** 2,
                                           ticks_num + 1)])
        cb.set_label('', fontsize=fontsize, fontdict=font)  # 设置 colorbar 的 标签名、标签字体；字体大小 fontsize=fontsize

    plt.show()

    if is_title_on == 0 and is_axes_on == 0 and is_colorbar_on == 0:
        axes.margins(0, 0, 0)
        if is_save == 1:
            fig.savefig(img_address, transparent=True, pad_inches=0)  # 不包含图例等，且无白边
    else:
        if is_save == 1:
            fig.savefig(img_address, transparent=True, bbox_inches='tight')  # 包含图例等，但有白边
            # fig.savefig(img_address, transparent = True, bbox_inches='tight', pad_inches=0) # 包含图例，且无白边