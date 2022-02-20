# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 00:42:22 2022

@author: Xcz
"""

import os
import numpy as np
# import matplotlib.ticker as mticker
import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits import axes_grid1
from mpl_toolkits.mplot3d import Axes3D
import math

def plot_1d(Iz = 343, size_PerPixel = 0.007, diz = 0.774, 
            #%%
            array1D = 0, array1D_address = os.path.dirname(os.path.abspath(__file__)), array1D_title = '', 
            #%%
            is_save = 0, dpi = 100, size_fig_x = 3, size_fig_y = 3, 
            #%%
            color_1d='b', ticks_num = 6, is_title_on = 1, is_axes_on = 1, is_mm = 1, is_propagation = 0, 
            #%%
            fontsize = 9,
            font = {'family': 'Times New Roman', # 'serif'
                    'style': 'normal', # 'normal', 'italic', 'oblique'
                    'weight': 'normal',
                    'color': 'black', # 'black','gray','darkred'
                    },
            #%%
            vmax = 1, vmin = 0):
    
    # #%%
    # Iz = 343
    # size_PerPixel = 0.007
    # diz = 0.774
    # #%%
    # array1D = 0
    # array1D_address = os.path.dirname(os.path.abspath(__file__))
    # array1D_title = ''
    # #%%
    # is_save = 0
    # dpi = 100
    # size_fig_x, size_fig_y = 3, 3
    # #%%
    # color_1d='b'
    # ticks_num = 6 # 不包含 原点的 刻度数，也就是 区间数（植数问题）
    # is_title_on = 1
    # is_axes_on = 1
    # is_mm = 1
    # is_propagation = 0
    # #%%
    # fontsize = 9
    # font = {'family': 'Times New Roman', # 'serif'
    #         'style': 'normal', # 'normal', 'italic', 'oblique'
    #         'weight': 'normal',
    #         'color': 'black', # 'black','gray','darkred'
    #         'size': fontsize,
    #         }
    # #%%
    # vmax = 1
    # vmin = 0
    
    #%%
    
    fig, axes = plt.subplots(1, 1, figsize=(size_fig_x, size_fig_y), dpi=dpi)
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    if is_title_on:
        axes.set_title(array1D_title, fontsize=fontsize, fontdict=font)
    
    if is_axes_on == 0:
        axes.axis('off')
    else:
        axes.set_xticks(range(0, Iz, Iz // ticks_num)) # 按理 等价于 np.linspace(0, Ix, ticks_num + 1)，但并不
        axes.set_yticks([float(format(y, '.2g')) for y in np.linspace(vmin, vmax, ticks_num + 1)]) # 按理 等价于 np.linspace(0, Iy, ticks_num + 1)，但并不
        if is_mm == 1: # round(i * size_PerPixel,2) 保留 2 位小数，改为 保留 2 位 有效数字
            if is_propagation == 1:
                axes.set_xticklabels([float('%.2g' % (i * diz * size_PerPixel)) for i in range(0, Iz, Iz // ticks_num)], fontsize=fontsize, fontdict=font)
            else:
                axes.set_xticklabels([float('%.2g' % (i * size_PerPixel)) for i in range(- Iz // 2, Iz - Iz // 2, Iz // ticks_num)], fontsize=fontsize, fontdict=font)
            axes.set_yticklabels([float(format(y, '.2g')) for y in np.linspace(vmin, vmax, ticks_num + 1)], fontsize=fontsize, fontdict=font)
        else:
            axes.set_xticklabels(range(0, Iz, Iz // ticks_num), fontsize=fontsize, fontdict=font)
            axes.set_yticklabels([float(format(y, '.2g')) for y in np.linspace(vmin, vmax, ticks_num + 1)], fontsize=fontsize, fontdict=font)
        axes.set_xlabel('', fontsize=fontsize, fontdict=font) # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        axes.set_ylabel('', fontsize=fontsize, fontdict=font) # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize
    
    img = axes.plot(range(0, Iz), array1D, color=color_1d)
        
    plt.show()
    
    if is_title_on == 0 and is_axes_on == 0 :
        axes.margins(0, 0)
        if is_save == 1:
            fig.savefig(array1D_address, transparent = True, pad_inches=0) # 不包含图例等，且无白边
    else:
        if is_save == 1:
            fig.savefig(array1D_address, transparent = True, bbox_inches='tight') # 包含图例等，但有白边
            # fig.savefig(array1D_address, transparent = True, bbox_inches='tight', pad_inches=0) # 包含图例，且无白边

def plot_2d(Ix = 343, Iy = 343, size_PerPixel = 0.007, diz = 0.774, 
            #%%
            array2D = 0, array2D_address = os.path.dirname(os.path.abspath(__file__)), array2D_title = '', 
            #%%
            is_save = 0, dpi = 100, size_fig = 3,  
            #%%
            cmap_2d='viridis', ticks_num = 6, is_contourf = 0, is_title_on = 1, is_axes_on = 1, is_mm = 1, is_propagation = 0, 
            #%%
            fontsize = 9,
            font = {'family': 'Times New Roman', # 'serif'
                    'style': 'normal', # 'normal', 'italic', 'oblique'
                    'weight': 'normal',
                    'color': 'black', # 'black','gray','darkred'
                    },
            #%%
            is_self_colorbar = 1, is_colorbar_on = 1, vmax = 1, vmin = 0):
    
    # #%%
    # Ix = 343
    # Iy = 343
    # size_PerPixel = 0.007
    # diz = 0.774
    # #%%
    # array2D = 0
    # array2D_address = os.path.dirname(os.path.abspath(__file__))
    # array2D_title = ''
    # #%%
    # is_save = 0
    # dpi = 100
    # size_fig = 3
    # #%%
    # cmap_2d='viridis'
    # # cmap_2d.set_under('black')
    # # cmap_2d.set_over('red')
    # ticks_num = 6 # 不包含 原点的 刻度数，也就是 区间数（植数问题）
    # is_contourf = 0
    # is_title_on = 1
    # is_axes_on = 1
    # is_mm = 1
    # is_propagation = 0
    # #%%
    # fontsize = 9
    # font = {'family': 'Times New Roman', # 'serif'
    #         'style': 'normal', # 'normal', 'italic', 'oblique'
    #         'weight': 'normal',
    #         'color': 'black', # 'black','gray','darkred'
    #         'size': fontsize,
    #         }
    # #%%
    # is_self_colorbar = 1 # vmax 与 vmin 是否以 自己的 U 的 最大值 最小值 为 相应的值；是，则覆盖设定；否的话，需要自己设定。
    # is_colorbar_on = 1
    # vmax = 1
    # vmin = 0
    
    #%%
    
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
    
    fig, axes = plt.subplots(1, 1, figsize=(size_fig, size_fig), dpi=dpi)
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    if is_title_on:
        axes.set_title(array2D_title, fontsize=fontsize, fontdict=font)
    
    if is_axes_on == 0:
        axes.axis('off')
    else:
        # plt.xticks(range(0, Ix, Ix // ticks_num), fontsize=fontsize) # Text 对象没有 fontdict 标签
        # plt.yticks(range(0, Iy, Iy // ticks_num), fontsize=fontsize) # Text 对象没有 fontdict 标签
        axes.set_xticks(range(0, Ix, Ix // ticks_num)) # 按理 等价于 np.linspace(0, Ix, ticks_num + 1)，但并不
        axes.set_yticks(range(0, Iy, Iy // ticks_num)) # 按理 等价于 np.linspace(0, Iy, ticks_num + 1)，但并不
        if is_mm == 1: # round(i * size_PerPixel,2) 保留 2 位小数，改为 保留 2 位 有效数字
            if is_propagation == 1:
                axes.set_xticklabels([float('%.2g' % (i * diz * size_PerPixel)) for i in range(0, Ix, Ix // ticks_num)], fontsize=fontsize, fontdict=font)
            else:
                axes.set_xticklabels([float('%.2g' % (i * size_PerPixel)) for i in range(- Ix // 2, Ix - Ix // 2, Ix // ticks_num)], fontsize=fontsize, fontdict=font)
            axes.set_yticklabels([- float('%.2g' % (j * size_PerPixel)) for j in range(- Iy // 2, Iy - Iy // 2, Iy // ticks_num)], fontsize=fontsize, fontdict=font)
        else:
            axes.set_xticklabels(range(0, Ix, Ix // ticks_num), fontsize=fontsize, fontdict=font)
            axes.set_yticklabels(range(0, Iy, Iy // ticks_num), fontsize=fontsize, fontdict=font)
        axes.set_xlabel('', fontsize=fontsize, fontdict=font) # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        axes.set_ylabel('', fontsize=fontsize, fontdict=font) # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        # plt.xlabel('', fontsize=fontsize, fontdict=font) # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        # plt.ylabel('', fontsize=fontsize, fontdict=font) # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize
    
    if is_contourf == 1:
        if is_self_colorbar == 1:
            img = axes.contourf(array2D, cmap=cmap_2d)
        else:
            img = axes.contourf(array2D, cmap=cmap_2d, vmin=vmin, vmax=vmax)
    else:
        if is_self_colorbar == 1:
            img = axes.imshow(array2D, cmap=cmap_2d)
        else:
            img = axes.imshow(array2D, cmap=cmap_2d, vmin=vmin, vmax=vmax)
        
    if is_colorbar_on == 1:
        cax = add_right_cax(axes, pad=0.05, width=0.05)
        cb = fig.colorbar(img, cax=cax, extend='both')
        cb.ax.tick_params(labelsize=fontsize) # 设置 colorbar 刻度字体；字体大小 labelsize=fontsize。 # Text 对象没有 fontdict 标签
        if is_self_colorbar != 1: # np.round(np.linspace(vmin, vmax, ticks_num + 1), 2) 保留 2 位小数，改为 保留 2 位 有效数字
            cb.set_ticks([float(format(x, '.2g')) for x in np.linspace(vmin, vmax, ticks_num + 1)]) # range(vmin, vmax, round((vmax-vmin) / ticks_num ,2)) 其中 range 步长不支持 非整数，只能用 np.arange 或 np.linspace
            cb.set_ticklabels([float(format(x, '.2g')) for x in np.linspace(vmin, vmax, ticks_num + 1)]) 
        cb.set_label('', fontsize=fontsize, fontdict=font) # 设置 colorbar 的 标签名、标签字体；字体大小 fontsize=fontsize
    
    plt.show()
    
    if is_title_on == 0 and is_axes_on == 0 and is_colorbar_on == 0 :
        axes.margins(0, 0)
        if is_save == 1:
            fig.savefig(array2D_address, transparent = True, pad_inches=0) # 不包含图例等，且无白边
    else:
        if is_save == 1:
            fig.savefig(array2D_address, transparent = True, bbox_inches='tight') # 包含图例等，但有白边
            # fig.savefig(array2D_address, transparent = True, bbox_inches='tight', pad_inches=0) # 包含图例，且无白边

def plot_3d_XYZ(Iz = 700, Iy = 343, Ix = 343, size_PerPixel = 0.007, diz = 0.774, 
                #%%
                U_YZ = 0, U_XZ = 0, U_1 = 0, U_2 = 0, 
                U_structure_front = 0, U_structure_end = 0, is_show_structure_face = 1, 
                #%%
                img_address = os.path.dirname(os.path.abspath(__file__)), img_title = '', 
                #%%
                iX = 0, iY = 0, iZ_1 = 0, iZ_2 = 0, 
                iZ_structure_front = 0, iZ_structure_end = 0, 
                #%%
                is_save = 0, dpi = 100, size_fig = 3, 
                #%%
                cmap_3d='viridis', elev = 10, azim = -65, alpha = 2, 
                ticks_num = 6, is_title_on = 1, is_axes_on = 1, is_mm = 1,  
                #%%
                fontsize = 9,
                font = {'family': 'Times New Roman', # 'serif'
                        'style': 'normal', # 'normal', 'italic', 'oblique'
                        'weight': 'normal',
                        'color': 'black', # 'black','gray','darkred'
                        },
                #%%
                is_self_colorbar = 1, is_colorbar_on = 1, vmax = 1, vmin = 0):
    
    # #%%
    # Iz = 700
    # Iy = 343
    # Ix = 343
    # size_PerPixel = 0.007
    # diz = 0.774
    # #%%
    # U_YZ = 0
    # U_XZ = 0
    # U_1 = 0
    # U_2 = 0
    # #%%
    # U_structure_front = 0
    # U_structure_end = 0
    # is_show_structure_face = 1
    # #%%
    # img_address = os.path.dirname(os.path.abspath(__file__))
    # img_title = ''
    # #%%
    # iX = 0
    # iY = 0
    # iZ_1 = 0
    # iZ_2 = 0
    # #%%
    # iZ_structure_front = 0
    # iZ_structure_end = 0
    # #%%
    # is_save = 0
    # dpi = 100
    # size_fig = 3
    # #%%
    # cmap_3d='viridis'
    # # cmap_2d.set_under('black')
    # # cmap_2d.set_over('red')
    # elev, azim = 10, -65
    # alpha = 2
    # #%%
    # ticks_num = 6 # 不包含 原点的 刻度数，也就是 区间数（植数问题）
    # is_title_on = 1
    # is_axes_on = 1
    # is_mm = 1
    # #%%
    # fontsize = 9
    # font = {'family': 'Times New Roman', # 'serif'
    #         'style': 'normal', # 'normal', 'italic', 'oblique'
    #         'weight': 'normal',
    #         'color': 'black', # 'black','gray','darkred'
    #         'size': fontsize,
    #         }
    # #%%
    # is_self_colorbar = 1 # vmax 与 vmin 是否以 自己的 U 的 最大值 最小值 为 相应的值；是，则覆盖设定；否的话，需要自己设定。
    # is_colorbar_on = 1
    # vmax = 1
    # vmin = 0
    
    #%%
    
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
    
    fig = plt.figure(figsize=(size_fig * 10, size_fig * 10), dpi=dpi)
    axes = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    if is_title_on:
        axes.set_title(img_title, fontsize=fontsize, fontdict=font)
    
    if is_axes_on == 0:
        axes.axis('off')
    else:
        axes.set_xticks(range(0, Iz, Iz // ticks_num)) # 按理 等价于 np.linspace(0, Ix, ticks_num + 1)，但并不
        axes.set_yticks(range(0, Ix, Ix // ticks_num)) # 按理 等价于 np.linspace(0, Iy, ticks_num + 1)，但并不
        axes.set_zticks(range(0, Iy, Iy // ticks_num))
        if is_mm == 1: # round(i * size_PerPixel,2) 保留 2 位小数，改为 保留 2 位 有效数字
            axes.set_xticklabels([float('%.2g' % (k * diz * size_PerPixel)) for k in range(0, Iz, Iz // ticks_num)], fontsize=fontsize, fontdict=font)
            axes.set_yticklabels([float('%.2g' % (i * size_PerPixel)) for i in range(- Ix // 2, Ix - Ix // 2, Ix // ticks_num)], fontsize=fontsize, fontdict=font)
            axes.set_zticklabels([float('%.2g' % (j * size_PerPixel)) for j in range(- Iy // 2, Iy - Iy // 2, Iy // ticks_num)], fontsize=fontsize, fontdict=font)
        else:
            axes.set_xticklabels(range(0, Iz, Iz // ticks_num), fontsize=fontsize, fontdict=font)
            axes.set_yticklabels(range(0, Ix, Ix // ticks_num), fontsize=fontsize, fontdict=font)
            axes.set_zticklabels(range(0, Iy, Iy // ticks_num), fontsize=fontsize, fontdict=font)
        axes.set_xlabel('Z', fontsize=fontsize, fontdict=font) # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        axes.set_ylabel('X', fontsize=fontsize, fontdict=font) # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        axes.set_zlabel('Y', fontsize=fontsize, fontdict=font) # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize
    
    axes.view_init(elev=elev, azim=azim); # 后一个为负 = 绕 z 轴逆时针
    
    Ixy = Iy
    if is_self_colorbar == 1:
        i_Z, i_XY = np.meshgrid([i for i in range(Iz)], [j for j in range(Ixy)])
        i_XY = i_XY[::-1]
        img = axes.scatter3D(i_Z, iX, i_XY, c=U_YZ, cmap=cmap_3d, alpha = math.e**(-1 * alpha))
        i_XY = i_XY[::-1]
        img = axes.scatter3D(i_Z, i_XY, iY, c=U_XZ, cmap=cmap_3d, alpha = math.e**(-1 * alpha))
        
        i_X, i_Y = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        i_Y = i_Y[::-1]
        img = axes.scatter3D(iZ_1, i_X, i_Y, c=U_1, cmap=cmap_3d, alpha = math.e**(-1 * alpha))
        img = axes.scatter3D(iZ_2, i_X, i_Y, c=U_2, cmap=cmap_3d, alpha = math.e**(-1 * alpha))
        
        if is_show_structure_face == 1:
            img = axes.scatter3D(iZ_structure_front, i_X, i_Y, c=U_structure_front, cmap=cmap_3d, alpha = math.e**(-1 * alpha))
            img = axes.scatter3D(iZ_structure_end, i_X, i_Y, c=U_structure_end, cmap=cmap_3d, alpha = math.e**(-1 * alpha))
    else:
        i_Z, i_XY = np.meshgrid([i for i in range(Iz)], [j for j in range(Ixy)])
        i_XY = i_XY[::-1]
        img = axes.scatter3D(i_Z, iX, i_XY, c=U_YZ, cmap=cmap_3d, alpha = math.e**(-1 * alpha), vmin=vmin, vmax=vmax)
        i_XY = i_XY[::-1]
        img = axes.scatter3D(i_Z, i_XY, iY, c=U_XZ, cmap=cmap_3d, alpha = math.e**(-1 * alpha), vmin=vmin, vmax=vmax)
        
        i_X, i_Y = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        i_Y = i_Y[::-1]
        img = axes.scatter3D(iZ_1, i_X, i_Y, c=U_1, cmap=cmap_3d, alpha = math.e**(-1 * alpha), vmin=vmin, vmax=vmax)
        img = axes.scatter3D(iZ_2, i_X, i_Y, c=U_2, cmap=cmap_3d, alpha = math.e**(-1 * alpha), vmin=vmin, vmax=vmax)
        
        if is_show_structure_face == 1:
            img = axes.scatter3D(iZ_structure_front, i_X, i_Y, c=U_structure_front, cmap=cmap_3d, alpha = math.e**(-1 * alpha), vmin=vmin, vmax=vmax)
            img = axes.scatter3D(iZ_structure_end, i_X, i_Y, c=U_structure_end, cmap=cmap_3d, alpha = math.e**(-1 * alpha), vmin=vmin, vmax=vmax)
        
    if is_colorbar_on == 1:
        cax = add_right_cax(axes, pad=0.05, width=0.05)
        cb = fig.colorbar(img, cax=cax, extend='both')
        cb.ax.tick_params(labelsize=fontsize) # 设置 colorbar 刻度字体；字体大小 labelsize=fontsize。 # Text 对象没有 fontdict 标签
        if is_self_colorbar != 1: # np.round(np.linspace(vmin, vmax, ticks_num + 1), 2) 保留 2 位小数，改为 保留 2 位 有效数字
            cb.set_ticks([float(format(x, '.2g')) for x in np.linspace(vmin, vmax, ticks_num + 1)]) # range(vmin, vmax, round((vmax-vmin) / ticks_num ,2)) 其中 range 步长不支持 非整数，只能用 np.arange 或 np.linspace
            cb.set_ticklabels([float(format(x, '.2g')) for x in np.linspace(vmin, vmax, ticks_num + 1)]) 
        cb.set_label('', fontsize=fontsize, fontdict=font) # 设置 colorbar 的 标签名、标签字体；字体大小 fontsize=fontsize
    
    plt.show()
    
    if is_title_on == 0 and is_axes_on == 0 and is_colorbar_on == 0 :
        axes.margins(0, 0, 0)
        if is_save == 1:
            fig.savefig(img_address, transparent = True, pad_inches=0) # 不包含图例等，且无白边
    else:
        if is_save == 1:
            fig.savefig(img_address, transparent = True, bbox_inches='tight') # 包含图例等，但有白边
            # fig.savefig(img_address, transparent = True, bbox_inches='tight', pad_inches=0) # 包含图例，且无白边
            
def plot_3d_XYz(Iy = 343, Ix = 343, size_PerPixel = 0.007, diz = 0.774, 
                sheets_stored_num = 10, U_z_stored = 0, sheet_th_stored = [], 
                #%%
                img_address = os.path.dirname(os.path.abspath(__file__)), img_title = '', 
                #%%
                is_save = 0, dpi = 100, size_fig = 3, 
                #%%
                cmap_3d='viridis', elev = 10, azim = -65, alpha = 2, 
                ticks_num = 6, is_title_on = 1, is_axes_on = 1, is_mm = 1,  
                #%%
                fontsize = 9,
                font = {'family': 'Times New Roman', # 'serif'
                        'style': 'normal', # 'normal', 'italic', 'oblique'
                        'weight': 'normal',
                        'color': 'black', # 'black','gray','darkred'
                        },
                #%%
                is_self_colorbar = 1, is_colorbar_on = 1, vmax = 1, vmin = 0):
    
    # #%%
    # Iy, Ix = 343, 343
    # size_PerPixel = 0.007
    # diz = 0.774
    # #%%
    # sheets_stored_num = 10
    # U_z_stored = 0
    # sheet_th_stored = []
    # #%%
    # img_address = os.path.dirname(os.path.abspath(__file__))
    # img_title = ''
    # #%%
    # is_save = 0
    # dpi, size_fig = 100, 3
    # #%%
    # cmap_3d='viridis'
    # # cmap_2d.set_under('black')
    # # cmap_2d.set_over('red')
    # elev, azim = 10, -65
    # alpha = 2
    # #%%
    # ticks_num = 6 # 不包含 原点的 刻度数，也就是 区间数（植数问题）
    # is_title_on, is_axes_on = 1, 1
    # is_mm = 1
    # #%%
    # fontsize = 9
    # font = {'family': 'Times New Roman', # 'serif'
    #         'style': 'normal', # 'normal', 'italic', 'oblique'
    #         'weight': 'normal',
    #         'color': 'black', # 'black','gray','darkred'
    #         'size': fontsize,
    #         }
    # #%%
    # is_self_colorbar, is_colorbar_on = 1, 1 # vmax 与 vmin 是否以 自己的 U 的 最大值 最小值 为 相应的值；是，则覆盖设定；否的话，需要自己设定。
    # vmax, vmin = 1, 0
    
    #%%
    
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
    
    fig = plt.figure(figsize=(size_fig * 10, size_fig * 10), dpi=dpi)
    axes = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    if is_title_on:
        axes.set_title(img_title, fontsize=fontsize, fontdict=font)
    
    if is_axes_on == 0:
        axes.axis('off')
    else:
        Iz = sheet_th_stored[sheets_stored_num] + 1
        axes.set_xticks(range(0, Iz, Iz // ticks_num)) # 按理 等价于 np.linspace(0, Ix, ticks_num + 1)，但并不
        axes.set_yticks(range(0, Ix, Ix // ticks_num)) # 按理 等价于 np.linspace(0, Iy, ticks_num + 1)，但并不
        axes.set_zticks(range(0, Iy, Iy // ticks_num))
        if is_mm == 1: # round(i * size_PerPixel,2) 保留 2 位小数，改为 保留 2 位 有效数字
            axes.set_xticklabels([float('%.2g' % (k * diz * size_PerPixel)) for k in range(0, Iz, Iz // ticks_num)], fontsize=fontsize, fontdict=font)
            axes.set_yticklabels([float('%.2g' % (i * size_PerPixel)) for i in range(- Ix // 2, Ix - Ix // 2, Ix // ticks_num)], fontsize=fontsize, fontdict=font)
            axes.set_zticklabels([float('%.2g' % (j * size_PerPixel)) for j in range(- Iy // 2, Iy - Iy // 2, Iy // ticks_num)], fontsize=fontsize, fontdict=font)
        else:
            axes.set_xticklabels(range(0, Iz, Iz // ticks_num), fontsize=fontsize, fontdict=font)
            axes.set_yticklabels(range(0, Ix, Ix // ticks_num), fontsize=fontsize, fontdict=font)
            axes.set_zticklabels(range(0, Iy, Iy // ticks_num), fontsize=fontsize, fontdict=font)
        axes.set_xlabel('Z', fontsize=fontsize, fontdict=font) # 设置 x 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        axes.set_ylabel('X', fontsize=fontsize, fontdict=font) # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize
        axes.set_zlabel('Y', fontsize=fontsize, fontdict=font) # 设置 y 轴的 标签名、标签字体；字体大小 fontsize=fontsize
    
    axes.view_init(elev=elev, azim=azim); # 后一个为负 = 绕 z 轴逆时针
    
    x_stretch_factor = sheets_stored_num**0.5 * 2
    # axes.get_proj = lambda: np.dot(Axes3D.get_proj(axes), np.diag([1 * x_stretch_factor, 1, 1, 1]))
    axes.get_proj = lambda: np.dot(Axes3D.get_proj(axes), np.diag([1, 1/x_stretch_factor, 1/x_stretch_factor, 1]))
    # axes.get_proj = lambda: np.dot(Axes3D.get_proj(axes), np.diag([1, 1/x_stretch_factor, 1/x_stretch_factor, 1/x_stretch_factor]))
    
    if is_self_colorbar == 1:
        i_X, i_Y = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        i_Y = i_Y[::-1]
        for sheet_stored_th in range(sheets_stored_num + 1):
            img = axes.scatter3D(sheet_th_stored[sheet_stored_th], i_X, i_Y, c=np.abs(U_z_stored[:, :, sheet_stored_th]), cmap=cmap_3d, alpha = math.e**-3 * math.e**(-1* alpha * sheet_stored_th / sheets_stored_num))
    else:
        i_X, i_Y = np.meshgrid([i for i in range(Ix)], [j for j in range(Iy)])
        i_Y = i_Y[::-1]
        for sheet_stored_th in range(sheets_stored_num + 1):
            img = axes.scatter3D(sheet_th_stored[sheet_stored_th], i_X, i_Y, c=np.abs(U_z_stored[:, :, sheet_stored_th]), cmap=cmap_3d, alpha = math.e**-3 * math.e**(-1* alpha * sheet_stored_th / sheets_stored_num), vmin=vmin, vmax=vmax)
        
    if is_colorbar_on == 1:
        cax = add_right_cax(axes, pad=0.05, width=0.05)
        cb = fig.colorbar(img, cax=cax, extend='both')
        cb.ax.tick_params(labelsize=fontsize) # 设置 colorbar 刻度字体；字体大小 labelsize=fontsize。 # Text 对象没有 fontdict 标签
        if is_self_colorbar != 1: # np.round(np.linspace(vmin, vmax, ticks_num + 1), 2) 保留 2 位小数，改为 保留 2 位 有效数字
            cb.set_ticks([float(format(x, '.2g')) for x in np.linspace(vmin, vmax, ticks_num + 1)]) # range(vmin, vmax, round((vmax-vmin) / ticks_num ,2)) 其中 range 步长不支持 非整数，只能用 np.arange 或 np.linspace
            cb.set_ticklabels([float(format(x, '.2g')) for x in np.linspace(vmin, vmax, ticks_num + 1)]) 
        cb.set_label('', fontsize=fontsize, fontdict=font) # 设置 colorbar 的 标签名、标签字体；字体大小 fontsize=fontsize
    
    plt.show()
    
    if is_title_on == 0 and is_axes_on == 0 and is_colorbar_on == 0 :
        axes.margins(0, 0, 0)
        if is_save == 1:
            fig.savefig(img_address, transparent = True, pad_inches=0) # 不包含图例等，且无白边
    else:
        if is_save == 1:
            fig.savefig(img_address, transparent = True, bbox_inches='tight') # 包含图例等，但有白边
            # fig.savefig(img_address, transparent = True, bbox_inches='tight', pad_inches=0) # 包含图例，且无白边