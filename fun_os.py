# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""
import winreg
import os
import re
import cv2
import numpy as np
import math
from scipy.io import loadmat, savemat
from fun_plot import plot_1d, plot_2d, plot_3d_XYz, plot_3d_XYZ


# %%
# 获取 桌面路径（C 盘 原生）

def GetDesktopPath():  # 修改过 桌面位置 后，就不准了
    return os.path.join(os.path.expanduser("~"), 'Desktop')

# %%
# 获取 桌面路径（注册表）

def get_desktop():  # 无论如何都准，因为读的是注册表
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders')
    return winreg.QueryValueEx(key, "Desktop")[0]

# %%
# 获取 当前 py 文件 所在路径 Current Directory

def get_cd():
    return os.path.dirname(os.path.abspath(__file__))  # 其实不需要，默认就是在 相对路径下 读，只需要 文件名 即可

# %%
# 查找

# 查找 text 中的 数字部分
def find_nums(text):
    return re.findall('(\d+)', text)

# 查找 text 中的 非数字部分
def find_NOT_nums(text):
    return re.findall('(\D+)', text)

# 查找 含 s 的 字符串 part，part 为在 text 中 被 separator（分隔符） 分隔 的 文字
def find_part_has_s_in_text(text, s, separator):
    for i, part in enumerate(text.split(separator)):
        if s in part:  # 找到 第一个 part 之后，不加 含 z 的 part，就 跳出 for 循环
            return part

def find_ray_sequence(U1_name):
    if ' - ' in U1_name:
        part_2 = U1_name.split(' - ')[1] # 取 由 AST - U1_ ... 分割的 第二部分：U1_ ...
    else:
        part_2 = U1_name # 应该不存在 没有 method 而有 sequence 的可能（只有 文件夹 才有这 可能）
    part_1 = part_2.split('_')[0] # 再取 part_2 中的 第一部分 U1
    ray = find_nums(part_1) # U1 中找到 1
    return ray

def set_ray(U1_name, set_ray, **kwargs):

    if "U" not in kwargs:
        ray_sequence = find_ray_sequence(U1_name)  # 从 U1_name 中找到 ray_sequence
        ray = ray_sequence[0] + set_ray if len(ray_sequence) != 0 else set_ray
    else:
        ray = kwargs['ray'] + set_ray if "ray" in kwargs else set_ray # 传 'U' 的值 进来的同时，要传个 'ray' 键 及其 对应的值

    return ray

# %%
# 生成 part_1 （被 分隔符 分隔的 第一个） 字符串
def gan_part_1(U1_name, U_name, is_add_sequence,
               *args, ):  # args 是 method、"_phase" 或 '_amp'

    part_1_NOT_num = find_NOT_nums(U_name.split('_')[0])[0]

    if is_add_sequence == 1:
        if part_1_NOT_num == 'g':
            part_1_sequence = "3."
        elif part_1_NOT_num == 'H':
            part_1_sequence = "4."
        elif part_1_NOT_num == 'G':
            part_1_sequence = "5."
        elif part_1_NOT_num == 'U':
            part_1_sequence = "6."

        if len(args) == 2:  # 如果 还传入了 后缀 "_phase" 或 '_amp'
            suffix = args[1]
            if suffix == '_amp' or suffix == '_energy':
                part_1_sequence += '1.'
            elif suffix == '_phase':
                part_1_sequence += '2.'
            # suffix == '_' + suffix # 识别完后，给 suffix 前面 加上 下划线
        else:  # 如果 啥也没传，则 后缀 啥也不加
            suffix = ''

        part_1_sequence += ' '  # 数字序号后 都得加个空格
    else:
        part_1_sequence = ''

    # args[0] + ' - ' 要放在 上面几步 之后，且 放在 上面 is_add_sequence 之外，因为 哪怕不加序号，也可能加 method，比如在 U_print 的 时候
    if len(args) >= 1:  # 如果 传入了 method（第一个 总是 method，因为 有后缀 "phase" or "amp" 则必然 先有 method，不可能 只有 phase or amp 而没有 method）
        part_1_sequence += args[0] + ' - '

    # 如果 U1_name 被 _ 分割出的 第一部分 不是空的 且 含有数字，则将其 数字部分 取出，作为 part_1 的 数字部分（传染性）
    U1_name_part_1_nums = find_nums(U1_name.split('_')[0])
    if len(U1_name_part_1_nums) != 0:  # 如果 U1_name 第一部分 含有数字
        part_1_num = U1_name_part_1_nums[0]
    else:
        U_name_part_1_nums = find_nums(U_name.split('_')[0])
        if len(U_name_part_1_nums) != 0:  # 如果 U_name 第一部分 含有数字
            part_1_num = U_name_part_1_nums[0]  # 否则 用 U_name 第一部分 原本的 数字部分，作为 part_1 的 数字部分
        else:
            part_1_num = ""

    part_1 = part_1_sequence + part_1_NOT_num + part_1_num

    return part_1, part_1_NOT_num


def gan_part_1z(U1_name, U_name, is_add_sequence,
                is_auto, *args, ):  # args 是 z、method

    if is_auto == 0:
        part_1_NOT_num = find_NOT_nums(U_name.split('_')[0])[0]
        if len(args) >= 2:  # 如果 传入了 method（第一个 总是 method，因为 有后缀 "phase" or "amp" 则必然 先有 method，不可能 只有 phase or amp 而没有 method）
            U_new_name = args[1] + ' - ' + U_name
        else:
            U_new_name = U_name
    else:
        # %%
        # 生成 part_1
        # 如果传了 2 个及以上 参数进来，那么将 多传进来的 len(args) - 1 个参数全传入 gan_part_1
        # 剩下一个是 z 或 ()，它必然 传了进来（至少传了 1 个进来的 就是它）
        if len(args) >= 2:
            part_1, part_1_NOT_num = gan_part_1(U1_name, U_name, is_add_sequence,
                                                *args[1:], )  # 先对 tuple 排除第一个元素地 切片，切片后还是个 tuple。 # 然后解包，再传入 函数
        else:
            part_1, part_1_NOT_num = gan_part_1(U1_name, U_name, is_add_sequence, )  # 否则 没传 method 进来，也就不传 method 进去
        # %%
        # 查找 含 z 的 字符串 part_z 
        part_z = find_part_has_s_in_text(U_name, 'z', '_')

        U_new_name = U_name.replace(U_name.split('_')[0], part_1)  # 至少把 U_name 第一部分 替换成 part_1，作为 U_new_name
        if (U_name.find('z') != -1 or U_name.find('Z') != -1) and len(args) != 0 and args[0] != ():
            # 如果 找到 z 或 Z，且 传了 额外的 参数 进来，这个参数 解包后的 第一个参数 不是 空 tuple ()
            z = args[0]
            U_new_name = U_new_name.replace(part_z, str(float(
                '%.2g' % z)) + "mm")  # 把 原来含 z 的 part_z 替换为 str(float('%.2g' % z)) + "mm"
    return U_new_name, part_1_NOT_num


# %%

def U_energy_print(U1_name, is_print, is_auto,
                   U, U_name, method,
                   *args, ):  # args 是 z 或 ()

    U_full_name, part_1_NOT_num = gan_part_1z(U1_name, U_name, 0,  # 不加 序列号
                                              is_auto, args, method, )  # 要有 method （诸如 'AST'）
    # 这里 还不能是 *arg，这样会 抹除 信息：传了 2 个及以下参数的时候，传的是哪 2 个参数？ z、method、suffix 中的 任意 2 个，都有可能。

    is_print and print(U_full_name + ".total_energy = {}".format(np.sum(np.abs(U) ** 2)))


# %%

def U_dir(U1_name, U_name, is_auto,
          is_bulk, *args, ):  # args 是 z 或 ()

    folder_name, part_1_NOT_num = gan_part_1z(U1_name, U_name, 1,  # 要加 序列号
                                              is_auto, args, )  # 没有 method （诸如 'AST'）

    # %%
    desktop = get_desktop()
    folder_address = desktop + "\\" + folder_name

    if is_bulk == 0:
        if not os.path.isdir(folder_address):
            os.makedirs(folder_address)

    return folder_address


# %%

def U_amp_plot_address_and_title(U1_name, U_name, is_auto,
                                 method, folder_address, img_name_extension,
                                 *args, ):
    # %%
    # 绘制 U_amp
    suffix = '_amp'
    # %%
    # 生成 要储存的 图片名 和 地址
    U_amp_name, part_1_NOT_num = gan_part_1z(U1_name, U_name, 1,  # 要加 序列号
                                             is_auto, args, method, suffix, )  # 有 method 和 suffix
    U_amp_name += suffix  # 增加 后缀 "_amp" 或 "_phase"
    # %%
    # 生成 地址
    U_amp_full_name = U_amp_name + img_name_extension
    U_amp_plot_address = folder_address + "\\" + U_amp_full_name
    # %%
    # 生成 图片中的 title
    U_amp_title, part_1_NOT_num = gan_part_1z(U1_name, U_name, 0,  # 不加 序列号
                                              is_auto, args, method, suffix, )  # 有 method 和 suffix
    U_amp_title += suffix  # 增加 后缀 "_amp" 或 "_phase"

    return U_amp_plot_address, U_amp_title


def U_phase_plot_address_and_title(U1_name, U_name, is_auto,
                                   method, folder_address, img_name_extension,
                                   *args, ):
    # %%
    # 绘制 U_phase
    suffix = '_phase'
    # %%
    # 生成 要储存的 图片名 和 地址
    U_phase_name, part_1_NOT_num = gan_part_1z(U1_name, U_name, 1,  # 要加 序列号
                                               is_auto, args, method, suffix, )  # 有 method 和 suffix
    U_phase_name += suffix  # 增加 后缀 "_amp" 或 "_phase"
    # %%
    # 生成 地址
    U_phase_full_name = U_phase_name + img_name_extension
    U_phase_plot_address = folder_address + "\\" + U_phase_full_name
    # %%
    # 生成 图片中的 title
    U_phase_title, part_1_NOT_num = gan_part_1z(U1_name, U_name, 0,  # 不加 序列号
                                                is_auto, args, method, suffix, )  # 有 method 和 suffix
    U_phase_title += suffix  # 增加 后缀 "_amp" 或 "_phase"

    return U_phase_plot_address, U_phase_title


# %%

def U_amp_plot(U1_name, folder_address, is_auto,
               U, U_name, method,
               img_name_extension,
               # %%
               zj, sample, size_PerPixel,
               is_save, dpi, size_fig,
               # %%
               cmap_2d, ticks_num, is_contourf,
               is_title_on, is_axes_on, is_mm, is_propagation,
               fontsize, font,
               # %%
               is_self_colorbar, is_colorbar_on,
               is_energy, vmax, vmin,
               # %%
               *args, ):  # args 是 z 或 ()

    U_amp_plot_address, U_amp_title = U_amp_plot_address_and_title(U1_name, U_name, is_auto,
                                                                   method, folder_address, img_name_extension,
                                                                   *args, )
    # %%
    plot_2d(zj, sample, size_PerPixel,
            np.abs(U), U_amp_plot_address, U_amp_title,
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, is_propagation,
            fontsize, font,
            is_self_colorbar, is_colorbar_on,
            is_energy, vmax, vmin, )

    return U_amp_plot_address


# %%

def U_phase_plot(U1_name, folder_address, is_auto,
                 U, U_name, method,
                 img_name_extension,
                 # %%
                 zj, sample, size_PerPixel,
                 is_save, dpi, size_fig,
                 # %%
                 cmap_2d, ticks_num, is_contourf,
                 is_title_on, is_axes_on, is_mm, is_propagation,
                 fontsize, font,
                 # %%
                 is_self_colorbar, is_colorbar_on,
                 vmax, vmin,
                 # %%
                 *args, ):  # args 是 z 或 ()

    U_phase_plot_address, U_phase_title = U_phase_plot_address_and_title(U1_name, U_name, is_auto,
                                                                         method, folder_address, img_name_extension,
                                                                         *args, )
    # %%
    plot_2d(zj, sample, size_PerPixel,
            np.angle(U), U_phase_plot_address, U_phase_title,
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, is_propagation,
            fontsize, font,
            is_self_colorbar, is_colorbar_on,
            0, vmax, vmin, )  # 相位 不能有 is_energy = 1

    return U_phase_plot_address


# %%

def U_plot(U1_name, folder_address, is_auto,
           U, U_name, method,
           img_name_extension,
           # %%
           sample, size_PerPixel,
           is_save, dpi, size_fig,
           # %%
           cmap_2d, ticks_num, is_contourf,
           is_title_on, is_axes_on, is_mm,
           fontsize, font,
           # %%
           is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
           # %%                          何况 一般默认 is_self_colorbar = 1...
           *args, ):  # args 是 z 或 ()

    U_amp_plot_address = U_amp_plot(U1_name, folder_address, is_auto,
                                    U, U_name, method,
                                    img_name_extension,
                                    # %%
                                    [], sample, size_PerPixel,
                                    is_save, dpi, size_fig,
                                    # %%
                                    cmap_2d, ticks_num, is_contourf,
                                    is_title_on, is_axes_on, is_mm, 0,
                                    fontsize, font,
                                    # %%
                                    1, is_colorbar_on,
                                    is_energy, 1, 0,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                                    # %% 何况 一般默认 is_self_colorbar = 1...
                                    *args, )

    U_phase_plot_address = U_phase_plot(U1_name, folder_address, is_auto,
                                        U, U_name, method,
                                        img_name_extension,
                                        # %%
                                        [], sample, size_PerPixel,
                                        is_save, dpi, size_fig,
                                        # %%
                                        cmap_2d, ticks_num, is_contourf,
                                        is_title_on, is_axes_on, is_mm, 0,
                                        fontsize, font,
                                        # %%
                                        1, is_colorbar_on,
                                        1, 0,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                                        # %% 何况 一般默认 is_self_colorbar = 1...
                                        *args, )

    return U_amp_plot_address, U_phase_plot_address


# %%

def U_plot_save(U1_name, is_print, is_auto,
                U, U_name, method,
                img_name_extension,
                # %%
                size_PerPixel,
                is_save, is_save_txt, dpi, size_fig,
                # %%
                cmap_2d, ticks_num, is_contourf,
                is_title_on, is_axes_on, is_mm,
                fontsize, font,
                # %%
                is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                # %%                          何况 一般默认 is_self_colorbar = 1...
                *args, ):  # *args = z 或 ()

    U_energy_print(U1_name, is_print, is_auto,
                   U, U_name, method,
                   *args, )

    if is_save == 1:
        folder_address = U_dir(U1_name, U_name, is_auto,
                               0, *args, )
    else:
        folder_address = ''

    # %%
    # 绘图：U

    U_amp_plot_address, U_phase_plot_address = U_plot(U1_name, folder_address, is_auto,
                                                      U, U_name, method,
                                                      img_name_extension,
                                                      # %%
                                                      1, size_PerPixel,
                                                      is_save, dpi, size_fig,
                                                      cmap_2d, ticks_num, is_contourf,
                                                      is_title_on, is_axes_on, is_mm,
                                                      fontsize, font,
                                                      is_colorbar_on, is_energy,
                                                      # %%
                                                      *args, )

    # %%
    # 储存 U 到 txt 文件

    if is_save == 1:
        U_address = U_save(U1_name, folder_address, is_auto,
                           U, U_name, method,
                           is_save_txt, *args, )
    else:
        U_address = ''

    return folder_address
    # return folder_address, U_address, U_amp_plot_address, U_phase_plot_address


def GHU_plot_save(U1_name, is_energy_evolution_on,  # 默认 全自动 is_auto = 1
                  G, G_name, method,
                  G_energy,
                  H, H_name,
                  U, U_name,
                  U_energy,
                  img_name_extension,
                  # %%
                  zj, sample, size_PerPixel,
                  is_save, is_save_txt, dpi, size_fig,
                  # %%
                  color_1d, cmap_2d,
                  ticks_num, is_contourf,
                  is_title_on, is_axes_on, is_mm,
                  fontsize, font,
                  # %%
                  is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                  # %%                          何况 一般默认 is_self_colorbar = 1...
                  z, ):  # 默认必须给 z

    folder_address = U_plot_save(U1_name, 0, 1,
                                 G, G_name, method,
                                 img_name_extension,
                                 # %%
                                 size_PerPixel,
                                 is_save, is_save_txt, dpi, size_fig,
                                 # %%
                                 cmap_2d, ticks_num, is_contourf,
                                 is_title_on, is_axes_on, is_mm,
                                 fontsize, font,
                                 # %%
                                 is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                                 # %%                          何况 一般默认 is_self_colorbar = 1...
                                 z, )

    if is_energy_evolution_on == 1:
        U_energy_plot_address = U_energy_plot(U1_name, folder_address, 1,
                                              G_energy, G_name + "_energy", method,
                                              img_name_extension,
                                              # %%
                                              zj, sample, size_PerPixel,
                                              is_save, dpi, size_fig * 10, size_fig,
                                              color_1d, ticks_num,
                                              is_title_on, is_axes_on, is_mm,
                                              fontsize, font,
                                              # %%
                                              z, )

    folder_address = U_plot_save(U1_name, 0, 1,
                                 H, H_name, method,
                                 img_name_extension,
                                 # %%
                                 size_PerPixel,
                                 is_save, is_save_txt, dpi, size_fig,
                                 # %%
                                 cmap_2d, ticks_num, is_contourf,
                                 is_title_on, is_axes_on, is_mm,
                                 fontsize, font,
                                 # %%
                                 is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                                 # %%                          何况 一般默认 is_self_colorbar = 1...
                                 z, )

    folder_address = U_plot_save(U1_name, 1, 1,
                                 U, U_name, method,
                                 img_name_extension,
                                 # %%
                                 size_PerPixel,
                                 is_save, is_save_txt, dpi, size_fig,
                                 # %%
                                 cmap_2d, ticks_num, is_contourf,
                                 is_title_on, is_axes_on, is_mm,
                                 fontsize, font,
                                 # %%
                                 is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                                 # %%                          何况 一般默认 is_self_colorbar = 1...
                                 z, )

    if is_energy_evolution_on == 1:
        U_energy_plot_address = U_energy_plot(U1_name, folder_address, 1,
                                              U_energy, U_name + "_energy", method,
                                              img_name_extension,
                                              # %%
                                              zj, sample, size_PerPixel,
                                              is_save, dpi, size_fig * 10, size_fig,
                                              color_1d, ticks_num,
                                              is_title_on, is_axes_on, is_mm,
                                              fontsize, font,
                                              # %%
                                              z, )

#%%

# def GHU_plot_save(U1_name, is_energy_evolution_on,  # 默认 全自动 is_auto = 1
#                   G, G_name, method,
#                   G_energy,
#                   H, H_name,
#                   U, U_name,
#                   U_energy,
#                   img_name_extension,
#                   # %%
#                   zj, sample, size_PerPixel,
#                   is_save, is_save_txt, dpi, size_fig,
#                   # %%
#                   color_1d, cmap_2d,
#                   ticks_num, is_contourf,
#                   is_title_on, is_axes_on, is_mm,
#                   fontsize, font,
#                   # %%
#                   is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
#                   # %%                          何况 一般默认 is_self_colorbar = 1...
#                   z, ):

# %%

def U_slices_plot(U1_name, folder_address, is_auto,
                  U_XZ, U_XZ_name, method,
                  U_YZ, U_YZ_name,
                  img_name_extension,
                  # %%
                  zj, sample, size_PerPixel,
                  is_save, dpi, size_fig,
                  # %%
                  cmap_2d, ticks_num, is_contourf,
                  is_title_on, is_axes_on, is_mm,
                  fontsize, font,
                  # %%
                  is_colorbar_on, is_energy,
                  # %%
                  X, Y, ):  # args 是 X 和 Y

    U_YZ_XZ_amp_max = np.max([np.max(np.abs(U_YZ)), np.max(np.abs(U_XZ))])
    U_YZ_XZ_amp_min = np.min([np.min(np.abs(U_YZ)), np.min(np.abs(U_XZ))])

    U_amp_plot(U1_name, folder_address, is_auto,
               U_YZ, U_YZ_name, method,
               img_name_extension,
               # %%
               zj, sample, size_PerPixel,
               is_save, dpi, size_fig,
               # %%
               cmap_2d, ticks_num, is_contourf,
               is_title_on, is_axes_on, is_mm, 1,
               fontsize, font,
               # %%
               0, is_colorbar_on,
               is_energy, U_YZ_XZ_amp_max, U_YZ_XZ_amp_min,
               # %%
               X, )

    U_amp_plot(U1_name, folder_address, is_auto,
               U_XZ, U_XZ_name, method,
               img_name_extension,
               # %%
               zj, sample, size_PerPixel,
               is_save, dpi, size_fig,
               # %%
               cmap_2d, ticks_num, is_contourf,
               is_title_on, is_axes_on, is_mm, 1,
               fontsize, font,
               # %%
               0, is_colorbar_on,
               is_energy, U_YZ_XZ_amp_max, U_YZ_XZ_amp_min,
               # %%
               Y, )

    U_YZ_XZ_phase_max = np.max([np.max(np.angle(U_YZ)), np.max(np.angle(U_XZ))])
    U_YZ_XZ_phase_min = np.min([np.min(np.angle(U_YZ)), np.min(np.angle(U_XZ))])

    U_phase_plot(U1_name, folder_address, is_auto,
                 U_YZ, U_YZ_name, method,
                 img_name_extension,
                 # %%
                 zj, sample, size_PerPixel,
                 is_save, dpi, size_fig,
                 # %%
                 cmap_2d, ticks_num, is_contourf,
                 is_title_on, is_axes_on, is_mm, 1,
                 fontsize, font,
                 # %%
                 0, is_colorbar_on,
                 U_YZ_XZ_phase_max, U_YZ_XZ_phase_min,
                 # %%
                 X, )

    U_phase_plot(U1_name, folder_address, is_auto,
                 U_XZ, U_XZ_name, method,
                 img_name_extension,
                 # %%
                 zj, sample, size_PerPixel,
                 is_save, dpi, size_fig,
                 # %%
                 cmap_2d, ticks_num, is_contourf,
                 is_title_on, is_axes_on, is_mm, 1,
                 fontsize, font,
                 # %%
                 0, is_colorbar_on,
                 U_YZ_XZ_phase_max, U_YZ_XZ_phase_min,
                 # %%
                 Y, )

    return U_YZ_XZ_amp_max, U_YZ_XZ_amp_min, U_YZ_XZ_phase_max, U_YZ_XZ_phase_min


# %%

def U_selects_plot(U1_name, folder_address, is_auto,
                   U_1, U_1_name, method,
                   U_2, U_2_name,
                   U_f, U_f_name,
                   U_e, U_e_name,
                   img_name_extension,
                   # %%
                   sample, size_PerPixel,
                   is_save, dpi, size_fig,
                   # %%
                   cmap_2d, ticks_num, is_contourf,
                   is_title_on, is_axes_on, is_mm,
                   fontsize, font,
                   # %%
                   is_colorbar_on, is_energy, is_show_structure_face,
                   # %%
                   z_1, z_2, z_f, z_e, ):  # args 是 z_1, z_2, z_f, z_e,

    if is_show_structure_face == 1:
        U_amps_max = np.max(
            [np.max(np.abs(U_1)), np.max(np.abs(U_2)),
             np.max(np.abs(U_f)), np.max(np.abs(U_e))])
        U_amps_min = np.min(
            [np.min(np.abs(U_1)), np.min(np.abs(U_2)),
             np.min(np.abs(U_f)), np.min(np.abs(U_e))])
    else:
        U_amps_max = np.max(
            [np.max(np.abs(U_1)), np.max(np.abs(U_2))])
        U_amps_min = np.min(
            [np.min(np.abs(U_1)), np.min(np.abs(U_2))])

    U_amp_plot(U1_name, folder_address, is_auto,
               U_1, U_1_name, method,
               img_name_extension,
               # %%
               [], sample, size_PerPixel,
               is_save, dpi, size_fig,
               # %%
               cmap_2d, ticks_num, is_contourf,
               is_title_on, is_axes_on, is_mm, 0,
               fontsize, font,
               # %%
               0, is_colorbar_on,
               is_energy, U_amps_max, U_amps_min,
               # %%
               z_1, )

    U_amp_plot(U1_name, folder_address, is_auto,
               U_2, U_2_name, method,
               img_name_extension,
               # %%
               [], sample, size_PerPixel,
               is_save, dpi, size_fig,
               # %%
               cmap_2d, ticks_num, is_contourf,
               is_title_on, is_axes_on, is_mm, 0,
               fontsize, font,
               # %%
               0, is_colorbar_on,
               is_energy, U_amps_max, U_amps_min,
               # %%
               z_2, )
    if is_show_structure_face == 1:
        U_amp_plot(U1_name, folder_address, is_auto,
                   U_f, U_f_name, method,
                   img_name_extension,
                   # %%
                   [], sample, size_PerPixel,
                   is_save, dpi, size_fig,
                   # %%
                   cmap_2d, ticks_num, is_contourf,
                   is_title_on, is_axes_on, is_mm, 0,
                   fontsize, font,
                   # %%
                   0, is_colorbar_on,
                   is_energy, U_amps_max, U_amps_min,
                   # %%
                   z_f, )

        U_amp_plot(U1_name, folder_address, is_auto,
                   U_e, U_e_name, method,
                   img_name_extension,
                   # %%
                   [], sample, size_PerPixel,
                   is_save, dpi, size_fig,
                   # %%
                   cmap_2d, ticks_num, is_contourf,
                   is_title_on, is_axes_on, is_mm, 0,
                   fontsize, font,
                   # %%
                   0, is_colorbar_on,
                   is_energy, U_amps_max, U_amps_min,
                   # %%
                   z_e, )

    if is_show_structure_face == 1:
        U_phases_max = np.max(
            [np.max(np.angle(U_1)), np.max(np.angle(U_2)),
             np.max(np.angle(U_f)), np.max(np.angle(U_e))])
        U_phases_min = np.min(
            [np.min(np.angle(U_1)), np.min(np.angle(U_2)),
             np.min(np.angle(U_f)), np.min(np.angle(U_e))])
    else:
        U_phases_max = np.max(
            [np.max(np.angle(U_1)), np.max(np.angle(U_2))])
        U_phases_min = np.min(
            [np.min(np.angle(U_1)), np.min(np.angle(U_2))])

    U_phase_plot(U1_name, folder_address, is_auto,
                 U_1, U_1_name, method,
                 img_name_extension,
                 # %%
                 [], sample, size_PerPixel,
                 is_save, dpi, size_fig,
                 # %%
                 cmap_2d, ticks_num, is_contourf,
                 is_title_on, is_axes_on, is_mm, 0,
                 fontsize, font,
                 # %%
                 0, is_colorbar_on,
                 U_phases_max, U_phases_min,
                 # %%
                 z_1, )

    U_phase_plot(U1_name, folder_address, is_auto,
                 U_2, U_2_name, method,
                 img_name_extension,
                 # %%
                 [], sample, size_PerPixel,
                 is_save, dpi, size_fig,
                 # %%
                 cmap_2d, ticks_num, is_contourf,
                 is_title_on, is_axes_on, is_mm, 0,
                 fontsize, font,
                 # %%
                 0, is_colorbar_on,
                 U_phases_max, U_phases_min,
                 # %%
                 z_2, )

    if is_show_structure_face == 1:
        U_phase_plot(U1_name, folder_address, is_auto,
                     U_f, U_f_name, method,
                     img_name_extension,
                     # %%
                     [], sample, size_PerPixel,
                     is_save, dpi, size_fig,
                     # %%
                     cmap_2d, ticks_num, is_contourf,
                     is_title_on, is_axes_on, is_mm, 0,
                     fontsize, font,
                     # %%
                     0, is_colorbar_on,
                     U_phases_max, U_phases_min,
                     # %%
                     z_f, )

        U_phase_plot(U1_name, folder_address, is_auto,
                     U_e, U_e_name, method,
                     img_name_extension,
                     # %%
                     [], sample, size_PerPixel,
                     is_save, dpi, size_fig,
                     # %%
                     cmap_2d, ticks_num, is_contourf,
                     is_title_on, is_axes_on, is_mm, 0,
                     fontsize, font,
                     # %%
                     0, is_colorbar_on,
                     U_phases_max, U_phases_min,
                     # %%
                     z_e, )

    return U_amps_max, U_amps_min, U_phases_max, U_phases_min


# %%

def U_amps_z_plot(U1_name, folder_address, is_auto,
                  U, U_name, method,
                  img_name_extension,
                  # %%
                  sample, size_PerPixel,
                  is_save, dpi, size_fig,
                  # %%
                  cmap_2d, ticks_num, is_contourf,
                  is_title_on, is_axes_on, is_mm,
                  fontsize, font,
                  # %%
                  is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
                  # %%
                  z_stored, ):  # 必须要传 z 序列 参数 进来

    U_amp_max = np.max(np.abs(U))
    U_amp_min = np.min(np.abs(U))

    for sheet_stored_th in range(U.shape[2]):  # 就不返回 address 了，大的 def 也不返回
        U_amp_plot(U1_name, folder_address, is_auto,  # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
                   U[:, :, sheet_stored_th], U_name, method,
                   img_name_extension,
                   # %%
                   [], sample, size_PerPixel,
                   is_save, dpi, size_fig,
                   # %%
                   cmap_2d, ticks_num, is_contourf,
                   is_title_on, is_axes_on, is_mm, 0,
                   fontsize, font,
                   # %%
                   0, is_colorbar_on,  # is_self_colorbar = 0，统一 colorbar
                   is_energy, U_amp_max, U_amp_min,  # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
                   # %%
                   z_stored[sheet_stored_th], )


# %%

def U_phases_z_plot(U1_name, folder_address, is_auto,
                    U, U_name, method,
                    img_name_extension,
                    # %%
                    sample, size_PerPixel,
                    is_save, dpi, size_fig,
                    # %%
                    cmap_2d, ticks_num, is_contourf,
                    is_title_on, is_axes_on, is_mm,
                    fontsize, font,
                    # %%
                    is_colorbar_on,  # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
                    # %%
                    z_stored, ):  # 必须要传 z 序列 参数 进来

    U_phase_max = np.max(np.angle(U))
    U_phase_min = np.min(np.angle(U))

    for sheet_stored_th in range(U.shape[2]):  # 就不返回 address 了，大的 def 也不返回
        U_phase_plot(U1_name, folder_address, is_auto,
                     U[:, :, sheet_stored_th], U_name, method,
                     img_name_extension,
                     # %%
                     [], sample, size_PerPixel,
                     is_save, dpi, size_fig,
                     # %%
                     cmap_2d, ticks_num, is_contourf,
                     is_title_on, is_axes_on, is_mm, 0,
                     fontsize, font,
                     # %%
                     0, is_colorbar_on,  # is_self_colorbar = 0，统一 colorbar
                     U_phase_max, U_phase_min,
                     # %%
                     z_stored[sheet_stored_th], )


# %%

def U_amp_plot_3d_XYz(U1_name, folder_address, is_auto,
                      U, U_name, method,
                      img_name_extension,
                      # %%
                      sample, size_PerPixel,
                      is_save, dpi, size_fig,
                      elev, azim, alpha,
                      # %%
                      cmap_3d, ticks_num,
                      is_title_on, is_axes_on, is_mm,
                      fontsize, font,
                      # %%
                      is_colorbar_on, is_energy,
                      # %%
                      zj, z_stored, ):  # args 是 z 或 ()，但 z 可从 z_stored 中 提取，所以这里 省略了 *args，外面不用传 z 进来

    U_amp_plot_address, U_amp_title = U_amp_plot_address_and_title(U1_name, U_name, is_auto,
                                                                   method, folder_address, img_name_extension,
                                                                   z_stored[-1], )

    U_amp_max = np.max(np.abs(U))  # is_self_colorbar = 1 并设置这个，没有意义，不如直接设置 1 0？ 试试就知道，并不是。
    U_amp_min = np.min(np.abs(U))  # is_self_colorbar = 1 并设置这个，没有意义，不如直接设置 1 0？ 试试就知道，并不是。

    plot_3d_XYz(zj, sample, size_PerPixel,
                np.abs(U), z_stored,
                U_amp_plot_address, U_amp_title,
                is_save, dpi, size_fig,
                cmap_3d, elev, azim, alpha,
                ticks_num, is_title_on, is_axes_on, is_mm,
                fontsize, font,
                1, is_colorbar_on,  # is_self_colorbar = 1 看上去 直接就 colorbar 统一了，但不是
                is_energy, U_amp_max, U_amp_min, )

    return U_amp_plot_address


# %%

def U_phase_plot_3d_XYz(U1_name, folder_address, is_auto,
                        U, U_name, method,
                        img_name_extension,
                        # %%
                        sample, size_PerPixel,
                        is_save, dpi, size_fig,
                        elev, azim, alpha,
                        # %%
                        cmap_3d, ticks_num,
                        is_title_on, is_axes_on, is_mm,
                        fontsize, font,
                        # %%
                        is_colorbar_on,
                        # %%
                        zj, z_stored, ):  # args 是 z 或 ()，但 z 可从 z_stored 中 提取，所以这里 省略了 *args，外面不用传 z 进来

    U_phase_plot_address, U_phase_title = U_phase_plot_address_and_title(U1_name, U_name, is_auto,
                                                                         method, folder_address, img_name_extension,
                                                                         z_stored[-1], )

    U_phase_max = np.max(np.angle(U))  # is_self_colorbar = 1 并设置这个，没有意义，不如直接设置 1 0？ 试试就知道，并不是。
    U_phase_min = np.min(np.angle(U))  # is_self_colorbar = 1 并设置这个，没有意义，不如直接设置 1 0？ 试试就知道，并不是。

    plot_3d_XYz(zj, sample, size_PerPixel,
                np.angle(U), z_stored,
                U_phase_plot_address, U_phase_title,
                is_save, dpi, size_fig,
                cmap_3d, elev, azim, alpha,
                ticks_num, is_title_on, is_axes_on, is_mm,
                fontsize, font,
                1, is_colorbar_on,  # is_self_colorbar = 1
                0, U_phase_max, U_phase_min, )  # 相位 不能有 is_energy = 1

    return U_phase_plot_address


# %%

def U_amp_plot_3d_XYZ(U1_name, folder_address, is_auto,
                      U_name, method,
                      U_YZ, U_XZ,
                      U_1, U_2,
                      U_f, U_e,
                      th_X, th_Y,
                      th_1, th_2,
                      th_f, th_e,
                      img_name_extension,
                      # %%
                      sample, size_PerPixel,
                      is_save, dpi, size_fig,
                      elev, azim, alpha,
                      # %%
                      cmap_3d, ticks_num,
                      is_title_on, is_axes_on, is_mm,
                      fontsize, font,
                      # %%
                      is_colorbar_on, is_energy, is_show_structure_face,
                      vmax, vmin,
                      # %%
                      zj, *args, ):  # args 是 z 或 ()

    U_amp_plot_address, U_amp_title = U_amp_plot_address_and_title(U1_name, U_name, is_auto,
                                                                   method, folder_address, img_name_extension,
                                                                   *args, )

    plot_3d_XYZ(zj, sample, size_PerPixel,
                np.abs(U_YZ), np.abs(U_XZ),
                np.abs(U_1), np.abs(U_2),
                np.abs(U_f), np.abs(U_e), is_show_structure_face,
                U_amp_plot_address, U_amp_title,
                th_X, th_Y,
                th_1, th_2,
                th_f, th_e,
                is_save, dpi, size_fig,
                cmap_3d, elev, azim, alpha,
                ticks_num, is_title_on, is_axes_on, is_mm,
                fontsize, font,
                1, is_colorbar_on,  # is_self_colorbar = 1
                is_energy, vmax, vmin, )

    return U_amp_plot_address


# %%

def U_phase_plot_3d_XYZ(U1_name, folder_address, is_auto,
                        U_name, method,
                        U_YZ, U_XZ,
                        U_1, U_2,
                        U_f, U_e,
                        th_X, th_Y,
                        th_1, th_2,
                        th_f, th_e,
                        img_name_extension,
                        # %%
                        sample, size_PerPixel,
                        is_save, dpi, size_fig,
                        elev, azim, alpha,
                        # %%
                        cmap_3d, ticks_num,
                        is_title_on, is_axes_on, is_mm,
                        fontsize, font,
                        # %%
                        is_colorbar_on, is_show_structure_face,
                        vmax, vmin,
                        # %%
                        zj, *args, ):  # args 是 z 或 ()，但 z 可从 z_stored 中 提取，所以这里 省略了 *args，外面不用传 z 进来

    U_phase_plot_address, U_phase_title = U_phase_plot_address_and_title(U1_name, U_name, is_auto,
                                                                         method, folder_address, img_name_extension,
                                                                         *args, )

    plot_3d_XYZ(zj, sample, size_PerPixel,
                np.angle(U_YZ), np.angle(U_XZ),
                np.angle(U_1), np.angle(U_2),
                np.angle(U_f), np.angle(U_e), is_show_structure_face,
                U_phase_plot_address, U_phase_title,
                th_X, th_Y,
                th_1, th_2,
                th_f, th_e,
                is_save, dpi, size_fig,
                cmap_3d, elev, azim, alpha,
                ticks_num, is_title_on, is_axes_on, is_mm,
                fontsize, font,
                1, is_colorbar_on,  # is_self_colorbar = 1
                0, vmax, vmin, )  # 相位 不能有 is_energy = 1

    return U_phase_plot_address


def U_SSI_plot(U1_name, folder_address,
               G_stored, G_name, method,
               U_stored, U_name,
               G_YZ, G_XZ,
               U_YZ, U_XZ,
               G_1, G_2,
               G_f, G_e,
               U_1, U_2,
               U_f, U_e,
               th_X, th_Y,
               th_1, th_2,
               th_f, th_e,
               img_name_extension,
               # %%
               sample, size_PerPixel,
               is_save, dpi, size_fig,
               elev, azim, alpha,
               # %%
               cmap_2d, cmap_3d,
               ticks_num, is_contourf,
               is_title_on, is_axes_on, is_mm,
               fontsize, font,
               # %%
               is_colorbar_on, is_energy, is_show_structure_face,
               # %%
               X, Y,
               z_1, z_2,
               z_f, z_e,
               zj, z_stored, z, ):

    is_plot_3d_XYz = 0
    is_plot_selective = 0
    is_plot_3d_XYZ = ""

    #%%

    if is_save == 1:
        folder_address = U_dir(U1_name, G_name + "_sheets_stored", 1,
                               0, z, )

    # -------------------------

    U_amps_z_plot(U1_name, folder_address, 1,
                  G_stored, G_name, method,
                  img_name_extension,
                  # %%
                  sample, size_PerPixel,
                  is_save, dpi, size_fig,
                  # %%
                  cmap_2d, ticks_num, is_contourf,
                  is_title_on, is_axes_on, is_mm,
                  fontsize, font,
                  # %%
                  is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                  # %%                          何况 一般默认 is_self_colorbar = 1...
                  z_stored, )

    U_phases_z_plot(U1_name, folder_address, 1,
                    G_stored, G_name, method,
                    img_name_extension,
                    # %%
                    sample, size_PerPixel,
                    is_save, dpi, size_fig,
                    # %%
                    cmap_2d, ticks_num, is_contourf,
                    is_title_on, is_axes_on, is_mm,
                    fontsize, font,
                    # %%
                    is_colorbar_on,  # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
                    # %%
                    z_stored, )

    # -------------------------

    if is_save == 1:
        folder_address = U_dir(U1_name, U_name + "_sheets_stored", 1,
                               0, z, )

    U_amps_z_plot(U1_name, folder_address, 1,
                  U_stored, U_name, method,
                  img_name_extension,
                  # %%
                  sample, size_PerPixel,
                  is_save, dpi, size_fig,
                  # %%
                  cmap_2d, ticks_num, is_contourf,
                  is_title_on, is_axes_on, is_mm,
                  fontsize, font,
                  # %%
                  is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                  # %%                          何况 一般默认 is_self_colorbar = 1...
                  z_stored, )

    U_phases_z_plot(U1_name, folder_address, 1,
                    U_stored, U_name, method,
                    img_name_extension,
                    # %%
                    sample, size_PerPixel,
                    is_save, dpi, size_fig,
                    # %%
                    cmap_2d, ticks_num, is_contourf,
                    is_title_on, is_axes_on, is_mm,
                    fontsize, font,
                    # %%
                    is_colorbar_on,  # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
                    # %%
                    z_stored, )

    # %%
    # 这 sheets_stored_num 层 也可以 画成 3D，就是太丑了，所以只 整个 U1_amp 示意一下即可

    if is_plot_3d_XYz == 1:

        U_amp_plot_address = U_amp_plot_3d_XYz(U1_name, folder_address, 1,
                                               U_stored, U_name + "_sheets_stored", method,
                                               img_name_extension,
                                               # %%
                                               sample, size_PerPixel,
                                               is_save, dpi, size_fig,
                                               elev, azim, alpha,
                                               # %%
                                               cmap_3d, ticks_num,
                                               is_title_on, is_axes_on, is_mm,
                                               fontsize, font,
                                               # %%
                                               is_colorbar_on, is_energy,
                                               # %%
                                               zj, z_stored, )

    # %%

    if is_plot_selective == 1:

        if is_save == 1:
            folder_address = U_dir(U1_name, G_name + "_sheets_selective_stored", 1,
                                   0, z, )

        # ------------------------- 储存 G1_section_1_shift_amp、G1_section_1_shift_amp、G1_structure_frontface_shift_amp、G1_structure_endface_shift_amp
        # ------------------------- 储存 G1_section_1_shift_phase、G1_section_1_shift_phase、G1_structure_frontface_shift_phase、G1_structure_endface_shift_phase

        G_amps_max, G_amps_min, G_phases_max, G_phases_min = \
            U_selects_plot(U1_name, folder_address, 1,
                           G_1, G_name + "_sec1", method,
                           G_2, G_name + "_sec2",
                           G_f, G_name + "_front",
                           G_e, G_name + "_end",
                           img_name_extension,
                           # %%
                           sample, size_PerPixel,
                           is_save, dpi, size_fig,
                           # %%
                           cmap_2d, ticks_num, is_contourf,
                           is_title_on, is_axes_on, is_mm,
                           fontsize, font,
                           # %%
                           is_colorbar_on, is_energy, is_show_structure_face,
                           # %%
                           z_1, z_2, z_f, z_e, )

        # %%

        if is_save == 1:
            folder_address = U_dir(U1_name, U_name + "_sheets_selective_stored", 1,
                                   0, z, )

        # ------------------------- 储存 U1_section_1_amp、U1_section_1_amp、U1_structure_frontface_amp、U1_structure_endface_amp
        # ------------------------- 储存 U1_section_1_phase、U1_section_1_phase、U1_structure_frontface_phase、U1_structure_endface_phase

        U_amps_max, U_amps_min, U_phases_max, U_phases_min = \
            U_selects_plot(U1_name, folder_address, 1,
                           U_1, U_name + "_sec1", method,
                           U_2, U_name + "_sec2",
                           U_f, U_name + "_front",
                           U_e, U_name + "_end",
                           img_name_extension,
                           # %%
                           sample, size_PerPixel,
                           is_save, dpi, size_fig,
                           # %%
                           cmap_2d, ticks_num, is_contourf,
                           is_title_on, is_axes_on, is_mm,
                           fontsize, font,
                           # %%
                           is_colorbar_on, is_energy, is_show_structure_face,
                           # %%
                           z_1, z_2, z_f, z_e, )

    # %%

    if is_save == 1:
        folder_address = U_dir(U1_name, G_name + "_YZ_XZ_stored", 1,
                               0, z, )

    # ========================= G1_shift_YZ_stored_amp、G1_shift_XZ_stored_amp
    # ------------------------- G1_shift_YZ_stored_phase、G1_shift_XZ_stored_phase

    G_YZ_XZ_amp_max, G_YZ_XZ_amp_min, G_YZ_XZ_phase_max, G_YZ_XZ_phase_min = \
        U_slices_plot(U1_name, folder_address, 1,
                      G_YZ, G_name + "_YZ", method,
                      G_XZ, G_name + "_XZ",
                      img_name_extension,
                      # %%
                      zj, sample, size_PerPixel,
                      is_save, dpi, size_fig,
                      # %%
                      cmap_2d, ticks_num, is_contourf,
                      is_title_on, is_axes_on, is_mm,
                      fontsize, font,
                      # %%
                      is_colorbar_on, is_energy,
                      # %%
                      X, Y, )

    # %%
    # 绘制 G1_amp 的 侧面 3D 分布图，以及 初始 和 末尾的 G1_amp（现在 可以 任选位置 了）

    if ("G" in is_plot_3d_XYZ and "a" in is_plot_3d_XYZ) or is_plot_3d_XYZ == "G" or is_plot_3d_XYZ == "a":

        U_amp_plot_address = U_amp_plot_3d_XYZ(U1_name, folder_address, 1,
                                               G_name + "_XYZ", method,
                                               G_YZ, G_XZ,
                                               G_1, G_2,
                                               G_f, G_e,
                                               th_X, th_Y,
                                               th_1, th_2,
                                               th_f, th_e,
                                               img_name_extension,
                                               # %%
                                               sample, size_PerPixel,
                                               is_save, dpi, size_fig,
                                               elev, azim, alpha,
                                               # %%
                                               cmap_3d, ticks_num,
                                               is_title_on, is_axes_on, is_mm,
                                               fontsize, font,
                                               # %%
                                               is_colorbar_on, is_energy, is_show_structure_face,
                                               np.max([G_YZ_XZ_amp_max, G_amps_max]),
                                               np.min([G_YZ_XZ_amp_min, G_amps_min]),
                                               # %%
                                               zj, z, )

    # %%
    # 绘制 G1_phase 的 侧面 3D 分布图，以及 初始 和 末尾的 G1_phase

    if ("G" in is_plot_3d_XYZ and "p" in is_plot_3d_XYZ) or is_plot_3d_XYZ == "G" or is_plot_3d_XYZ == "p":

        U_phase_plot_address = U_phase_plot_3d_XYZ(U1_name, folder_address, 1,
                                                   G_name + "_XYZ", method,
                                                   G_YZ, G_XZ,
                                                   G_1, G_2,
                                                   G_f, G_e,
                                                   th_X, th_Y,
                                                   th_1, th_2,
                                                   th_f, th_e,
                                                   img_name_extension,
                                                   # %%
                                                   sample, size_PerPixel,
                                                   is_save, dpi, size_fig,
                                                   elev, azim, alpha,
                                                   # %%
                                                   cmap_3d, ticks_num,
                                                   is_title_on, is_axes_on, is_mm,
                                                   fontsize, font,
                                                   # %%
                                                   is_colorbar_on, is_show_structure_face,
                                                   np.max([G_YZ_XZ_phase_max, G_phases_max]),
                                                   np.min([G_YZ_XZ_phase_min, G_phases_min]),
                                                   # %%
                                                   zj, z, )

    # %%

    if is_save == 1:
        folder_address = U_dir(U1_name, U_name + "_YZ_XZ_stored", 1,
                               0, z, )

    # ========================= U1_YZ_stored_amp、U1_XZ_stored_amp
    # ------------------------- U1_YZ_stored_phase、U1_XZ_stored_phase

    U_YZ_XZ_amp_max, U_YZ_XZ_amp_min, U_YZ_XZ_phase_max, U_YZ_XZ_phase_min = \
        U_slices_plot(U1_name, folder_address, 1,
                      U_YZ, U_name + "_YZ", method,
                      U_XZ, U_name + "_XZ",
                      img_name_extension,
                      # %%
                      zj, sample, size_PerPixel,
                      is_save, dpi, size_fig,
                      # %%
                      cmap_2d, ticks_num, is_contourf,
                      is_title_on, is_axes_on, is_mm,
                      fontsize, font,
                      # %%
                      is_colorbar_on, is_energy,
                      # %%
                      X, Y, )

    # %%
    # 绘制 U1_amp 的 侧面 3D 分布图，以及 初始 和 末尾的 U1_amp

    if ("U" in is_plot_3d_XYZ and "a" in is_plot_3d_XYZ) or is_plot_3d_XYZ == "U" or is_plot_3d_XYZ == "a":

        U_amp_plot_address = U_amp_plot_3d_XYZ(U1_name, folder_address, 1,
                                               U_name + "_XYZ", method,
                                               U_YZ, U_XZ,
                                               U_1, U_2,
                                               U_f, U_e,
                                               th_X, th_Y,
                                               th_1, th_2,
                                               th_f, th_e,
                                               img_name_extension,
                                               # %%
                                               sample, size_PerPixel,
                                               is_save, dpi, size_fig,
                                               elev, azim, alpha,
                                               # %%
                                               cmap_3d, ticks_num,
                                               is_title_on, is_axes_on, is_mm,
                                               fontsize, font,
                                               # %%
                                               is_colorbar_on, is_energy, is_show_structure_face,
                                               np.max([U_YZ_XZ_amp_max, U_amps_max]),
                                               np.min([U_YZ_XZ_amp_min, U_amps_min]),
                                               # %%
                                               zj, z, )

    # %%
    # 绘制 U1_phase 的 侧面 3D 分布图，以及 初始 和 末尾的 U1_phase

    if ("U" in is_plot_3d_XYZ and "p" in is_plot_3d_XYZ) or is_plot_3d_XYZ == "U" or is_plot_3d_XYZ == "p":

        U_phase_plot_address = U_phase_plot_3d_XYZ(U1_name, folder_address, 1,
                                                   U_name + "_XYZ", method,
                                                   U_YZ, U_XZ,
                                                   U_1, U_2,
                                                   U_f, U_e,
                                                   th_X, th_Y,
                                                   th_1, th_2,
                                                   th_f, th_e,
                                                   img_name_extension,
                                                   # %%
                                                   sample, size_PerPixel,
                                                   is_save, dpi, size_fig,
                                                   elev, azim, alpha,
                                                   # %%
                                                   cmap_3d, ticks_num,
                                                   is_title_on, is_axes_on, is_mm,
                                                   fontsize, font,
                                                   # %%
                                                   is_colorbar_on, is_show_structure_face,
                                                   np.max([U_YZ_XZ_phase_max, U_phases_max]),
                                                   np.min([U_YZ_XZ_phase_min, U_phases_min]),
                                                   # %%
                                                   zj, z, )


# %%

def U_save(U1_name, folder_address, is_auto,
           U, U_name, method,
           is_save_txt, *args, ):
    U_full_name, part_1_NOT_num = gan_part_1z(U1_name, U_name, 1,  # 要加 序列号
                                              is_auto, args, method, )  # 要有 method （诸如 'AST'）

    file_name = U_full_name + (is_save_txt and ".txt" or ".mat")
    U_address = folder_address + "\\" + file_name
    np.savetxt(U_address, U) if is_save_txt else savemat(U_address, {part_1_NOT_num: U})

    return U_address


# %%

def U_energy_plot(U1_name, folder_address, is_auto,
                  U, U_name, method,
                  img_name_extension,
                  # %%
                  zj, sample, size_PerPixel,
                  is_save, dpi, size_fig_x, size_fig_y,
                  color_1d, ticks_num, is_title_on, is_axes_on, is_mm,
                  fontsize, font,  # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                  # %%
                  *args, ):
    # %%
    # 绘制 U_amp
    suffix = '_energy'
    # %%
    # 生成 要储存的 图片名 和 地址
    U_energy_name, part_1_NOT_num = gan_part_1z(U1_name, U_name, 1,  # 要加 序列号
                                                is_auto, args, method, suffix, )  # 有 method 和 suffix
    # U_energy_name += suffix  # 增加 后缀 "_energy" （才怪，suffix 只 help 辅助 加 5.1 这种序号，原 U_name 里已有 _energy 了）
    # %%
    # 生成 地址
    U_energy_full_name = U_energy_name + img_name_extension
    U_energy_plot_address = folder_address + "\\" + U_energy_full_name
    # %%
    # 生成 图片中的 title
    U_energy_title, part_1_NOT_num = gan_part_1z(U1_name, U_name, 0,  # 不加 序列号
                                                 is_auto, args, method, suffix, )  # 有 method 和 suffix
    # U_energy_title += suffix  # 增加 后缀 "_evolution" （才怪，suffix 只 help 辅助 加 5.1 这种序号，原 U_name 里已有 _energy 了）
    # %%
    U_energy_max = np.max(U)  # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar）
    U_energy_min = np.min(U)  # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar）

    plot_1d(zj, sample, size_PerPixel,
            U, U_energy_plot_address, U_energy_title,
            is_save, dpi, size_fig_x, size_fig_y,
            color_1d, ticks_num, is_title_on, is_axes_on, is_mm, 1,
            fontsize, font,
            0, U_energy_max, U_energy_min)

    return U_energy_plot_address


# %%

def Info_img(img_full_name):
    img_name = os.path.splitext(img_full_name)[0]
    img_name_extension = os.path.splitext(img_full_name)[1]

    cdir = get_cd()
    desktop = get_desktop()

    img_address = cdir + "\\" + img_full_name  # 默认 在 相对路径下 读，只需要 文件名 即可：读于内
    img_squared_address = desktop + "\\" + "1." + img_name + "_squared" + img_name_extension  # 除 原始文件 以外，生成的文件 均放在桌面：写出于外
    img_squared_bordered_address = desktop + "\\" + "2." + img_name + "_squared" + "_bordered" + img_name_extension

    return img_name, img_name_extension, img_address, img_squared_address, img_squared_bordered_address


# %%
# 导入 方形，以及 加边框 的 图片

def img_squared_bordered_Read(img_full_name,
                              U_NonZero_size, dpi,
                              is_phase_only, ):
    img_name, img_name_extension, img_address, img_squared_address, img_squared_bordered_address = Info_img(
        img_full_name)

    img_squared = cv2.imdecode(np.fromfile(img_squared_address, dtype=np.uint8), 0)  # 按 相对路径 + 灰度图 读取图片
    img_squared_bordered = cv2.imdecode(np.fromfile(img_squared_bordered_address, dtype=np.uint8),
                                        0)  # 按 相对路径 + 灰度图 读取图片

    size_PerPixel = U_NonZero_size / img_squared.shape[0]  # Unit: mm / 个 每个 像素点 的 尺寸，相当于 △x = △y = △z
    size_fig = img_squared_bordered.shape[0] / dpi
    Ix, Iy = img_squared_bordered.shape[0], img_squared_bordered.shape[1]

    if is_phase_only == 1:
        U = np.power(math.e, (img_squared_bordered.astype(np.complex128()) / 255 * 2 * math.pi - math.pi) * 1j)  # 变成相位图
    else:
        U = img_squared_bordered.astype(np.complex128)

    return img_name, img_name_extension, img_squared, size_PerPixel, size_fig, Ix, Iy, U


# %%
# 导入 方形 图片，以及 U

def U_Read(U_name, img_full_name,
           U_NonZero_size, dpi,
           is_save_txt, ):
    desktop = get_desktop()

    U_full_name = U_name + (is_save_txt and ".txt" or ".mat")
    U_address = desktop + "\\" + U_full_name
    img_name, img_name_extension, img_address, img_squared_address, img_squared_bordered_address = Info_img(
        img_full_name)

    img_squared = cv2.imdecode(np.fromfile(img_squared_address, dtype=np.uint8), 0)  # 按 相对路径 + 灰度图 读取图片
    U = np.loadtxt(U_address, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U_full_name)['U']  # 加载 复振幅场

    size_PerPixel = U_NonZero_size / img_squared.shape[0]  # Unit: mm / 个 每个 像素点 的 尺寸，相当于 △x = △y = △z
    size_fig = U.shape[0] / dpi
    Ix, Iy = U.shape[0], U.shape[1]

    return img_name, img_name_extension, img_squared, size_PerPixel, size_fig, Ix, Iy, U
