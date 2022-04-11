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
from fun_gif_video import imgs2gif_imgio, imgs2gif_PIL, imgs2gif_art

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
def find_nums(text): # 这个没用了，也就是 ray
    return re.findall('(\d+)', text)


# 查找 text 中的 非数字部分
def find_NOT_nums(text): # 这个没用了，也就是 ugHGU
    return re.findall('(\D+)', text)


# 查找 含 s 的 字符串 part，part 为在 text 中 被 separator（分隔符） 分隔 的 文字
def find_part_has_s_in_text(text, s, separator):
    for i, part in enumerate(text.split(separator)):
        if s in part:  # 找到 第一个 part 之后，不加 含 z 的 part，就 跳出 for 循环
            return part

# 查找 ray_sequence
def split_parts(U_name):
    if ' - ' in U_name:
        Part_1 = U_name.split(' - ')[0] # 取出 seq + method + way 的 method_and_way
        method_and_way = Part_1.split(" ")[1] if " " in Part_1 else Part_1 # 去掉 seq，只保留 method + way
        Part_2 = U_name.split(' - ')[1]  # 取 由 AST - U0_ ... 分割的 第二部分：U0_ ...
    else:
        method_and_way = ""
        Part_2 = U_name  # 应该不存在 没有 method 而有 sequence 的可能（只有 文件夹 才有这 可能）
    if '_' in Part_2:
        part_1 = Part_2.split('_')[0] # 取 part_2 中的 第一部分 U1
    elif ' ' in Part_2:
        part_1 = Part_2.split(" ")[1]
    else:
        part_1 = Part_2
    ray_seq = part_1[1:] if len(part_1[1:]) > 0 else "" # 取出 U0_name 第一部分 第一个字符之后的东西
    ugHGU = part_1[0] if len(part_1) > 0 else ""
    U_name_no_seq = method_and_way + (' - ' if method_and_way != "" else "") + Part_2

    # print(U_name)
    # print(U_name_no_seq, method_and_way, Part_2, ugHGU, ray_seq)

    from fun_global_var import Set
    Set("method_and_way", method_and_way)

    return U_name_no_seq, method_and_way, Part_2, ugHGU, ray_seq

# 查找 ray （ 要么从 U_name 里传 ray 和 U 进来，要么 单独传个 U 和 ray ）
def set_ray(U0_name, ray_new, **kwargs): # U0_name 只在这有用，用于获取 其 ray，获取一次后就不需 U0_name 了
    if "ray" not in kwargs: # 传 'U' 的值 进来的同时，要传个 'ray' 键 及其 对应的值
        U_name_no_seq, method_and_way, Part_2, ugHGU, ray_seq = split_parts(U0_name)  # 从 U0_name 中找到 ray_sequence
        ray = ray_seq[0] + ray_new if len(ray_seq) != 0 else ray_new
    else:
        ray = kwargs['ray'] + ray_new if "ray" in kwargs else ray_new  # 传 'U' 的值 进来的同时，要传个 'ray' 键 及其 对应的值

    return ray

# # 查找 ray（已废弃）
# def get_ray(U0_name, U_name, ): # 这个没用了，已经被 split_parts 和 set_ray 替代了
#     method_and_way, ugHGU, ray_seq = split_parts(U0_name)  # 从 U0_name 中找到 ray_sequence
#     # 如果 U0_name 被 _ 分割出的 第一部分 不是空的 且 含有数字，则将其 数字部分 取出，暂作为 part_1 的 数字部分（传染性）
#     ray = ray_seq[0] if len(ray_seq) != 0 else ""
#
#     # U_name_rays = find_nums(U_name.split('_')[0])
#     U_name_rays = U_name.split('_')[0][1:] # 取出 U_name 第一部分 第一个字符之后的东西
#     # 如果 U_name 第一部分 含有数字，则 在 part_1 后面 追加 U_name 第一部分 原本的 数字部分
#     ray += U_name_rays[0] if len(U_name_rays) != 0 else ""
#     return ray

#%%
# 替换

def replace_p_ray(title, ugHGU, ray): # ugHGU 起到了 标识符的作用，防止误 replace 了 其他字符串
    return title.replace(ugHGU + ray, ugHGU + ray.replace("0", "p"))

def subscript_ray(title, ugHGU, ray): # ugHGU 起到了 标识符的作用，防止误 replace 了 其他字符串
    # return title.replace(ugHGU + ray, ugHGU + "$_{" + ray.replace("0", "p") + "}$")
    return title.replace(ugHGU + ray, ugHGU + "$_{" + ray + "}$")

def subscript_way(title, method_and_way): # method 起到了 标识符的作用，防止误 replace 了 其他字符串
    if '_' in method_and_way:
        method = method_and_way.split("_")[0]
        way = method_and_way.split("_")[1]
        return title.replace(method + "_" + way, method + "$_{" + way + "}$")
    else:
        return title

def add___between_ugHGU_and_ray(Uz_name, ugHGU, ray):
    return Uz_name.replace(ugHGU + ray, ugHGU + "_" + ray)


    # %%
# 生成 part_1 （被 分隔符 分隔的 第一个） 字符串
def gan_seq(U_name, is_add_sequence, # 就 2 功能，加序号，减 method_and_way
            **kwargs, ):  # kwargs 是 “suffix”

    # ugHGU = find_NOT_nums(U_name.split('_')[0])[0]
    # ugHGU = U_name.split('_')[0][0] if len(U_name.split('_')[0]) != 0 else ""
    U_name_no_seq, method_and_way, Part_2, ugHGU, ray = split_parts(U_name)
    # print(ugHGU)

    seq = ''
    # 模为 1 即有 seq
    # >= 0 即有 method_and_way
    if abs(is_add_sequence) == 1:
        if ugHGU == 'g':
            seq = "3."
        elif ugHGU == 'H':
            seq = "4."
        elif ugHGU == 'G':
            seq = "3." if method_and_way == "PUMP" else "5."
        elif ugHGU == 'U':
            seq = "2." if method_and_way == "PUMP" else "6."
        elif ugHGU == "χ" or ugHGU == "n":
            seq = "0."

        if "suffix" in kwargs:  # 如果 还传入了 后缀 "_phase" 或 '_amp'
            suffix = kwargs["suffix"]
            if suffix == '_amp' or suffix == '_amp_error' or suffix == '_energy' :
                seq += '1.'
            elif suffix == '_phase' or suffix == '_phase_error':
                seq += '2.'

        if seq != "":
            seq += ' '  # 数字序号后 都得加个空格

    return seq


def gan_Uz_name(U_name, is_add_sequence, **kwargs, ):  # args 是 z 或 () 和 suffix

    U_name_no_seq, method_and_way, Part_2, ugHGU, ray = split_parts(U_name)
    seq = gan_seq(U_name, is_add_sequence, **kwargs, ) # is_add_sequence 模为 1 即有 seq
    U_new_name = seq + U_name_no_seq
    # %%
    # 查找 含 z 的 字符串 part_z
    part_z = find_part_has_s_in_text(Part_2, 'z', '_')
    if (U_name.find('z') != -1 or U_name.find('Z') != -1) and 'z' in kwargs:
        # 如果 找到 z 或 Z，且 传了 额外的 参数 进来，这个参数 解包后的 第一个参数 不是 空 tuple ()
        z = kwargs['z']
        U_new_name = U_new_name.replace(part_z, str(float('%.2g' % z)) + "mm")
        # 把 原来含 z 的 part_z 替换为 str(float('%.2g' % z)) + "mm"

    if is_add_sequence < 0:  # is_add_sequence >= 0 即有 method_and_way = method + way
        U_new_name = U_new_name.replace(method_and_way + " - ", "")

    return U_new_name, U_name_no_seq, method_and_way, Part_2, ugHGU, ray


def gan_Uz_plot_address(folder_address, img_name_extension,
                        U_name, suffix, **kwargs):
    Uz_name, U_name_no_seq, method_and_way, Part_2, ugHGU, ray = gan_Uz_name(U_name, 1, suffix=suffix,
                                                                            **kwargs, )  # 要加 序列号 # 有 method 和 suffix
    Uz_name += suffix if suffix != "_energy" else ""
    Uz_name = add___between_ugHGU_and_ray(Uz_name, ugHGU, ray)
    Uz_full_name = Uz_name + img_name_extension
    Uz_plot_address = folder_address + "\\" + Uz_full_name

    return Uz_full_name, Uz_plot_address


def gan_Uz_title(U_name, suffix, **kwargs):
    Uz_title, U_name_no_seq, method_and_way, Part_2, ugHGU, ray = gan_Uz_name(U_name, 0, suffix=suffix,
                                                                             **kwargs, )  # 不加 序列号 # 有 method 和 suffix
    Uz_title += suffix if suffix != "_energy" else ""
    Uz_title = subscript_ray(Uz_title, ugHGU, ray)
    Uz_title = subscript_way(Uz_title, method_and_way)
    return Uz_title

def gan_Uz_save_address(U_name, folder_address, is_save_txt,
                        **kwargs):
    U_full_name, U_name_no_seq, method_and_way, Part_2, ugHGU, ray = gan_Uz_name(U_name, 1,
                                                                                 **kwargs, )  # 要加 序列号 # 要有 method （诸如 'AST'）
    U_full_name = add___between_ugHGU_and_ray(U_full_name, ugHGU, ray)
    file_name = U_full_name + (is_save_txt and ".txt" or ".mat")
    U_address = folder_address + "\\" + file_name
    return U_address, ugHGU

def gan_Uz_dir_address(U_name, **kwargs, ):
    folder_name, U_name_no_seq, method_and_way, Part_2, ugHGU, ray = gan_Uz_name(U_name, -1,
                                                                                 **kwargs, )  # 要加 序列号 # 没有 method （诸如 'AST'）
    folder_name = add___between_ugHGU_and_ray(folder_name, ugHGU, ray)
    desktop = get_desktop()
    folder_address = desktop + "\\" + folder_name
    return folder_address

# %%

def U_energy_print(U, U_name, is_print,
                   **kwargs, ):  # kwargs 是 z

    U_full_name, U_name_no_seq, method_and_way, Part_2, ugHGU, ray = gan_Uz_name(U_name, 0, **kwargs, ) # 不加 序列号 # 要有 method （诸如 'AST'）

    is_print and print(U_full_name + ".total_energy = {}".format(np.sum(np.abs(U) ** 2)))

def U_rsd_print(U, U_name, is_print,
                **kwargs, ):  # kwargs 是 z

    U_full_name, U_name_no_seq, method_and_way, Part_2, ugHGU, ray = gan_Uz_name(U_name, 0, **kwargs, ) # 不加 序列号 # 要有 method （诸如 'AST'）

    is_print and is_print-1 and print(U_full_name + ".rsd = {}".format(np.std(np.abs(U)) / np.mean(np.abs(U)))) # is_print 是 1 和 0 都不行，得是 2 等才行...

# %%

def U_dir(U_name, is_save,
          **kwargs, ):  # kwargs 是 z

    folder_address = gan_Uz_dir_address(U_name, **kwargs, )

    if is_save == 1:
        if not os.path.isdir(folder_address):
            os.makedirs(folder_address)

    return folder_address

# %%

def U_amp_plot_address_and_title(U_name, folder_address, img_name_extension,
                                 **kwargs, ): # kwargs 是 z
    # %%
    # 绘制 U_amp
    suffix = '_amp'
    # %%
    # 生成 要储存的 图片名 和 地址
    U_amp_full_name, U_amp_plot_address = gan_Uz_plot_address(folder_address, img_name_extension,
                                                            U_name, suffix, **kwargs)
    # %%
    # 生成 图片中的 title
    U_amp_title = gan_Uz_title(U_name, suffix, **kwargs)  # 增加 后缀 "_amp" 或 "_phase"


    return U_amp_plot_address, U_amp_title


#%%

def U_amp_error_plot_address_and_title(U_name, folder_address, img_name_extension,
                                 **kwargs, ): # kwargs 是 z
    # %%
    # 绘制 U_amp
    suffix = '_amp_error'
    # %%
    # 生成 要储存的 图片名 和 地址
    U_amp_error_full_name, U_amp_error_plot_address = gan_Uz_plot_address(folder_address, img_name_extension,
                                                            U_name, suffix, **kwargs)
    # %%
    # 生成 图片中的 title
    U_amp_error_title = gan_Uz_title(U_name, suffix, **kwargs)  # 增加 后缀 "_amp" 或 "_phase"


    return U_amp_error_plot_address, U_amp_error_title

#%%

def U_phase_plot_address_and_title(U_name, folder_address, img_name_extension,
                                   **kwargs, ):
    # %%
    # 绘制 U_phase
    suffix = '_phase'
    # %%
    # 生成 要储存的 图片名 和 地址
    U_phase_full_name, U_phase_plot_address = gan_Uz_plot_address(folder_address, img_name_extension,
                                                            U_name, suffix, **kwargs)
    # %%
    # 生成 图片中的 title
    U_phase_title = gan_Uz_title(U_name, suffix, **kwargs)  # 增加 后缀 "_amp" 或 "_phase"

    return U_phase_plot_address, U_phase_title

#%%

def U_phase_error_plot_address_and_title(U_name, folder_address, img_name_extension,
                                   **kwargs, ):
    # %%
    # 绘制 U_phase
    suffix = '_phase_error'
    # %%
    # 生成 要储存的 图片名 和 地址
    U_phase_error_full_name, U_phase_error_plot_address = gan_Uz_plot_address(folder_address, img_name_extension,
                                                            U_name, suffix, **kwargs)
    # %%
    # 生成 图片中的 title
    U_phase_error_title = gan_Uz_title(U_name, suffix, **kwargs)  # 增加 后缀 "_amp" 或 "_phase"

    return U_phase_error_plot_address, U_phase_error_title

# %%

def U_amp_plot(folder_address,
               U, U_name,
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
               **kwargs, ):  # args 是 z 或 ()

    U_amp_plot_address, U_amp_title = U_amp_plot_address_and_title(U_name, folder_address, img_name_extension,
                                                                   **kwargs, )
    # %%

    plot_2d(zj, sample, size_PerPixel,
            np.abs(U), U_amp_plot_address, U_amp_title,
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, is_propagation,
            fontsize, font,
            is_self_colorbar, is_colorbar_on,
            is_energy, vmax, vmin, )

    return U_amp_plot_address, U_amp_title

#%%

def U_amp_error_plot(folder_address,
               U, U_name,
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
               **kwargs, ):  # args 是 z 或 ()

    U_amp_error_plot_address, U_amp_error_title = U_amp_error_plot_address_and_title(U_name, folder_address, img_name_extension,
                                                                   **kwargs, )
    # %%

    plot_2d(zj, sample, size_PerPixel,
            np.abs(U), U_amp_error_plot_address, U_amp_error_title,
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, is_propagation,
            fontsize, font,
            is_self_colorbar, is_colorbar_on,
            is_energy, vmax, vmin, )

    return U_amp_error_plot_address, U_amp_error_title

# %%

def U_phase_plot(folder_address,
                 U, U_name,
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
                 **kwargs, ):  # args 是 z 或 ()

    U_phase_plot_address, U_phase_title = U_phase_plot_address_and_title(U_name, folder_address, img_name_extension,
                                                                         **kwargs, )
    # %%

    plot_2d(zj, sample, size_PerPixel,
            np.angle(U), U_phase_plot_address, U_phase_title,
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, is_propagation,
            fontsize, font,
            is_self_colorbar, is_colorbar_on,
            0, vmax, vmin, )  # 相位 不能有 is_energy = 1

    return U_phase_plot_address, U_phase_title

#%%

def U_phase_error_plot(folder_address,
                 U, U_name,
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
                 **kwargs, ):  # args 是 z 或 ()

    U_phase_error_plot_address, U_phase_error_title = U_phase_error_plot_address_and_title(U_name, folder_address, img_name_extension,
                                                                         **kwargs, )
    # %%

    plot_2d(zj, sample, size_PerPixel,
            np.angle(U), U_phase_error_plot_address, U_phase_error_title,
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, is_propagation,
            fontsize, font,
            is_self_colorbar, is_colorbar_on,
            0, vmax, vmin, )  # 相位 不能有 is_energy = 1

    return U_phase_error_plot_address, U_phase_error_title

# %%

def U_plot(folder_address,
           U, U_name,
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
           **kwargs, ):  # args 是 z 或 ()

    U_amp_plot_address = U_amp_plot(folder_address,
                                    U, U_name,
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
                                    **kwargs, )

    U_phase_plot_address = U_phase_plot(folder_address,
                                        U, U_name,
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
                                        **kwargs, )

    return U_amp_plot_address, U_phase_plot_address

#%%

def U_error_plot(folder_address,
           U, U_0,
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
           **kwargs, ):  # args 是 z 或 ()

    from fun_global_var import fkey

    U_amp_error = np.abs(U) - np.abs(U_0)
    U_phase_error = np.abs(U) - np.angle(U_0)

    U_amp_error_plot_address = U_amp_error_plot(folder_address,
                                    U_amp_error, fkey("U"),
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
                                    **kwargs, )

    U_phase_error_plot_address = U_phase_error_plot(folder_address,
                                        U_phase_error, fkey("U"),
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
                                        **kwargs, )

    return U_amp_error_plot_address, U_phase_error_plot_address

# %%

def U_plot_save(U, U_name, is_print,
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
                **kwargs, ):  # **kwargs = z

    U_energy_print(U, U_name, is_print,
                   **kwargs, )
    U_rsd_print(U, U_name, is_print,
                **kwargs, )

    folder_address = U_dir(U_name, is_save, **kwargs, )

    # %%
    # 绘图：U

    U_amp_plot_address, U_phase_plot_address = U_plot(folder_address,
                                                      U, U_name,
                                                      img_name_extension,
                                                      # %%
                                                      1, size_PerPixel,
                                                      is_save, dpi, size_fig,
                                                      cmap_2d, ticks_num, is_contourf,
                                                      is_title_on, is_axes_on, is_mm,
                                                      fontsize, font,
                                                      is_colorbar_on, is_energy,
                                                      # %%
                                                      **kwargs, )

    # %%
    # 储存 U 到 txt 文件

    U_save(U, U_name, folder_address,
           is_save, is_save_txt, **kwargs, )

    return folder_address
    # return folder_address, U_address, U_amp_plot_address, U_phase_plot_address

#%%

def U_error_plot_save(U, U_0, is_print,
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
                      **kwargs, ):  # **kwargs = z

    from fun_global_var import fkey

    U_error = U - U_0
    U_error_name = fkey("U") + "_error"

    folder_address = U_dir(U_error_name, is_save, **kwargs, )

    #%%
    U_amp_error = np.abs(U) - np.abs(U_0)
    U_amp_error_name = fkey("U") + "_amp_error"
    U_energy_print(U_amp_error, U_amp_error_name, is_print,
                   **kwargs, )
    U_rsd_print(U_amp_error, U_amp_error_name, is_print,
                **kwargs, )

    U_phase_error = np.abs(U) - np.angle(U_0)
    U_phase_error_name = fkey("U") + "_phase_error"
    U_energy_print(U_phase_error, U_phase_error_name, is_print,
                   **kwargs, )
    U_rsd_print(U_phase_error, U_phase_error_name, is_print,
                **kwargs, )

    # %%
    # 绘图：U

    U_amp_error_plot_address, U_phase_error_plot_address = U_error_plot(folder_address,
                                                      U, U_0,
                                                      img_name_extension,
                                                      # %%
                                                      1, size_PerPixel,
                                                      is_save, dpi, size_fig,
                                                      cmap_2d, ticks_num, is_contourf,
                                                      is_title_on, is_axes_on, is_mm,
                                                      fontsize, font,
                                                      is_colorbar_on, is_energy,
                                                      # %%
                                                      **kwargs, )

    # %%
    # 储存 U 到 txt 文件

    U_save(U_amp_error, U_amp_error_name, folder_address,
           is_save, is_save_txt, **kwargs, )
    U_save(U_phase_error, U_phase_error_name, folder_address,
           is_save, is_save_txt, **kwargs, )

    return folder_address


def GHU_plot_save(G, G_name, is_energy_evolution_on,  # 默认 全自动 is_auto_seq_and_z = 1
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

    folder_address = U_plot_save(G, G_name, 0,
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
                                 z=z, )

    if is_energy_evolution_on == 1:
        U_energy_plot_address = U_energy_plot(folder_address,
                                              G_energy, G_name + "_energy",
                                              img_name_extension,
                                              # %%
                                              zj, sample, size_PerPixel,
                                              is_save, dpi, size_fig * 10, size_fig,
                                              color_1d, ticks_num,
                                              is_title_on, is_axes_on, is_mm,
                                              fontsize, font,
                                              # %%
                                              z=z, )

    folder_address = U_plot_save(H, H_name, 0,
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
                                 z=z, )

    folder_address = U_plot_save(U, U_name, 1,
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
                                 z=z, )

    if is_energy_evolution_on == 1:
        U_energy_plot_address = U_energy_plot(folder_address,
                                              U_energy, U_name + "_energy",
                                              img_name_extension,
                                              # %%
                                              zj, sample, size_PerPixel,
                                              is_save, dpi, size_fig * 10, size_fig,
                                              color_1d, ticks_num,
                                              is_title_on, is_axes_on, is_mm,
                                              fontsize, font,
                                              # %%
                                              z=z, )


# %%

def U_slices_plot(folder_address,
                  U_XZ, U_XZ_name,
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

    U_amp_plot(folder_address,
               U_YZ, U_YZ_name,
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
               z=X, )

    U_amp_plot(folder_address,
               U_XZ, U_XZ_name,
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
               z=Y, )

    U_YZ_XZ_phase_max = np.max([np.max(np.angle(U_YZ)), np.max(np.angle(U_XZ))])
    U_YZ_XZ_phase_min = np.min([np.min(np.angle(U_YZ)), np.min(np.angle(U_XZ))])

    U_phase_plot(folder_address,
                 U_YZ, U_YZ_name,
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
                 z=X, )

    U_phase_plot(folder_address,
                 U_XZ, U_XZ_name,
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
                 z=Y, )

    return U_YZ_XZ_amp_max, U_YZ_XZ_amp_min, U_YZ_XZ_phase_max, U_YZ_XZ_phase_min


# %%

def U_selects_plot(folder_address,
                   U_1, U_1_name,
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

    U_amp_plot(folder_address,
               U_1, U_1_name,
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
               z=z_1, )

    U_amp_plot(folder_address,
               U_2, U_2_name,
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
               z=z_2, )
    if is_show_structure_face == 1:
        U_amp_plot(folder_address,
                   U_f, U_f_name,
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
                   z=z_f, )

        U_amp_plot(folder_address,
                   U_e, U_e_name,
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
                   z=z_e, )

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

    U_phase_plot(folder_address,
                 U_1, U_1_name,
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
                 z=z_1, )

    U_phase_plot(folder_address,
                 U_2, U_2_name,
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
                 z=z_2, )

    if is_show_structure_face == 1:
        U_phase_plot(folder_address,
                     U_f, U_f_name,
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
                     z=z_f, )

        U_phase_plot(folder_address,
                     U_e, U_e_name,
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
                     z=z_e, )

    return U_amps_max, U_amps_min, U_phases_max, U_phases_min


# %%

def U_amps_z_plot(folder_address,
                  U, U_name,
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
                  z_stored, is_animated,
                  duration, fps, loop, ):  # 必须要传 z 序列、is_animated 进来

    U_amp_max = np.max(np.abs(U))
    U_amp_min = np.min(np.abs(U))

    # global imgs_address_list, titles_list
    imgs_address_list = []
    titles_list = []
    for sheet_stored_th in range(U.shape[2]):
        U_amp_plot_address, U_amp_title = U_amp_plot(folder_address,
                                                     # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
                                                     U[:, :, sheet_stored_th], U_name,
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
                                                     is_energy, U_amp_max, U_amp_min,
                                                     # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
                                                     # %%
                                                     z=z_stored[sheet_stored_th], )
        imgs_address_list.append(U_amp_plot_address)
        titles_list.append(U_amp_title)  # 每张图片都用单独list的形式加入到图片序列中

    if is_save == 1:  # 只有 储存后，才能根据 储存的图片 生成 gif

        """ plot2d 无法多线程，因为会挤占 同一个 fig 这个 全局的画布资源？ 注释了 plt.show() 也没用，应该不是它的锅。
        不过其实可以在 U_amp_plot 里面搞多线程，因为 获取 address 和 title 不是全局的 """
        # def fun1(for_th, fors_num, *arg, ):
        #     U_amp_plot_address, U_amp_title = U_amp_plot(U0_name, folder_address, is_auto_seq_and_z,
        #                                                  # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
        #                                                  U[:, :, for_th], U_name, method,
        #                                                  img_name_extension,
        #                                                  # %%
        #                                                  [], sample, size_PerPixel,
        #                                                  is_save, dpi, size_fig,
        #                                                  # %%
        #                                                  cmap_2d, ticks_num, is_contourf,
        #                                                  is_title_on, is_axes_on, is_mm, 0,
        #                                                  fontsize, font,
        #                                                  # %%
        #                                                  0, is_colorbar_on,  # is_self_colorbar = 0，统一 colorbar
        #                                                  is_energy, U_amp_max, U_amp_min,
        #                                                  # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
        #                                                  # %%
        #                                                  z_stored[for_th], )
        #     return U_amp_plot_address, U_amp_title
        #
        # def fun2(for_th, fors_num, U_amp_plot_address, U_amp_title, *args, **kwargs, ):
        #     global imgs_address_list, titles_list
        #     imgs_address_list.append(U_amp_plot_address)
        #     titles_list.append(U_amp_title)  # 每张图片都用单独list的形式加入到图片序列中
        #
        # my_thread(10, U.shape[2],
        #           fun1, fun2, noop,
        #           is_ordered=1, is_print=0, )

        if fps > 0: duration = 1 / fps  # 如果传入了 fps，则可 over write duration
        gif_address = imgs_address_list[-1].replace(img_name_extension, ".gif")
        if is_animated == 0:
            imgs2gif_imgio(imgs_address_list, gif_address,
                           duration, fps, loop, )
        elif is_animated == -1:
            imgs2gif_PIL(imgs_address_list, gif_address,
                         duration, fps, loop, )
        else:
            imgs2gif_art(imgs_address_list, gif_address,
                         duration, fps, loop, )

        return gif_address


# %%

def U_phases_z_plot(folder_address,
                    U, U_name,
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
                    z_stored, is_animated,
                    duration, fps, loop, ):  # 必须要传 z 序列、is_animated 进来

    U_phase_max = np.max(np.angle(U))
    U_phase_min = np.min(np.angle(U))

    # global imgs_address_list, titles_list
    imgs_address_list = []
    titles_list = []
    for sheet_stored_th in range(U.shape[2]):
        U_phase_plot_address, U_phase_title = U_phase_plot(folder_address,
                                                           U[:, :, sheet_stored_th], U_name,
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
                                                           z=z_stored[sheet_stored_th], )
        imgs_address_list.append(U_phase_plot_address)
        titles_list.append(U_phase_title)  # 每张图片都用单独list的形式加入到图片序列中

    if is_save == 1:  # 只有 储存后，才能根据 储存的图片 生成 gif

        """ plot2d 无法多线程，因为会挤占 同一个 fig 这个 全局的画布资源？ 注释了 plt.show() 也没用，应该不是它的锅。
        不过其实可以在 U_amp_plot 里面搞多线程，因为 获取 address 和 title 不是全局的 """
        # def fun1(for_th, fors_num, *arg, ):
        #     U_phase_plot_address, U_phase_title = U_phase_plot(U0_name, folder_address, is_auto_seq_and_z,
        #                                                        U[:, :, for_th], U_name, method,
        #                                                        img_name_extension,
        #                                                        # %%
        #                                                        [], sample, size_PerPixel,
        #                                                        is_save, dpi, size_fig,
        #                                                        # %%
        #                                                        cmap_2d, ticks_num, is_contourf,
        #                                                        is_title_on, is_axes_on, is_mm, 0,
        #                                                        fontsize, font,
        #                                                        # %%
        #                                                        0, is_colorbar_on,  # is_self_colorbar = 0，统一 colorbar
        #                                                        U_phase_max, U_phase_min,
        #                                                        # %%
        #                                                        z_stored[for_th], )
        #     return U_phase_plot_address, U_phase_title
        #
        # def fun2(for_th, fors_num, U_phase_plot_address, U_phase_title, *args, **kwargs, ):
        #     global imgs_address_list, titles_list
        #     imgs_address_list.append(U_phase_plot_address)
        #     titles_list.append(U_phase_title)  # 每张图片都用单独list的形式加入到图片序列中
        #
        # my_thread(10, U.shape[2],
        #           fun1, fun2, noop,
        #           is_ordered=1, is_print=0, )

        if fps > 0: duration = 1 / fps  # 如果传入了 fps，则可 over write duration
        gif_address = imgs_address_list[-1].replace(img_name_extension, ".gif")
        if is_animated == 0:
            imgs2gif_imgio(imgs_address_list, gif_address,
                           duration, fps, loop, )
        elif is_animated == -1:
            imgs2gif_PIL(imgs_address_list, gif_address,
                         duration, fps, loop, )
        else:
            imgs2gif_art(imgs_address_list, gif_address,
                         duration, fps, loop, )

        return gif_address


# %%

def U_amp_plot_3d_XYz(folder_address,
                      U, U_name,
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

    U_amp_plot_address, U_amp_title = U_amp_plot_address_and_title(U_name, folder_address, img_name_extension,
                                                                   z=z_stored[-1], )

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

def U_phase_plot_3d_XYz(folder_address,
                        U, U_name,
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

    U_phase_plot_address, U_phase_title = U_phase_plot_address_and_title(U_name, folder_address, img_name_extension,
                                                                         z=z_stored[-1], )

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

def U_amp_plot_3d_XYZ(folder_address,
                      U_name,
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
                      zj, **kwargs, ):  # args 是 z 或 ()

    U_amp_plot_address, U_amp_title = U_amp_plot_address_and_title(U_name, folder_address, img_name_extension,
                                                                   **kwargs, )

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

def U_phase_plot_3d_XYZ(folder_address,
                        U_name,
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
                        zj, **kwargs, ):  # args 是 z 或 ()

    U_phase_plot_address, U_phase_title = U_phase_plot_address_and_title(U_name, folder_address, img_name_extension,
                                                                         **kwargs, )

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


# %%

def U_EVV_plot(G_stored, G_name,
               U_stored, U_name,
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
               is_colorbar_on, is_energy,
               # %%
               plot_group, is_animated,
               loop, duration, fps,
               # %%
               is_plot_3d_XYz,
               # %%
               zj, z_stored, z, ):
    folder_address = U_dir(G_name + "_sheets", is_save, z=z, )

    # -------------------------

    if ("G" in plot_group and "a" in plot_group):
        gif_address = U_amps_z_plot(folder_address,
                                    G_stored, G_name,
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
                                    z_stored, is_animated,
                                    duration, fps, loop, )

    if ("G" in plot_group and "p" in plot_group):
        gif_address = U_phases_z_plot(folder_address,
                                      G_stored, G_name,
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
                                      z_stored, is_animated,
                                      duration, fps, loop, )

    # -------------------------

    folder_address = U_dir(U_name + "_sheets", is_save, z=z, )

    if ("U" in plot_group and "a" in plot_group):
        gif_address = U_amps_z_plot(folder_address,
                                    U_stored, U_name,
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
                                    z_stored, is_animated,
                                    duration, fps, loop, )

    if ("U" in plot_group and "p" in plot_group):
        gif_address = U_phases_z_plot(folder_address,
                                      U_stored, U_name,
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
                                      z_stored, is_animated,
                                      duration, fps, loop, )

    # %%
    # 这 sheets_stored_num 层 也可以 画成 3D，就是太丑了，所以只 整个 U0_amp 示意一下即可

    if ("U" in plot_group and "a" in plot_group) and is_plot_3d_XYz == 1:
        U_amp_plot_address = U_amp_plot_3d_XYz(folder_address,
                                               U_stored, U_name + "_sheets",
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

def U_SSI_plot(G_stored, G_name,
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
               plot_group, is_animated,
               loop, duration, fps,
               # %%
               is_plot_3d_XYz, is_plot_selective,
               is_plot_YZ_XZ, is_plot_3d_XYZ,
               # %%
               X, Y,
               z_1, z_2,
               z_f, z_e,
               zj, z_stored, z, ):
    # %%

    U_EVV_plot(G_stored, G_name,
               U_stored, U_name,
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
               is_colorbar_on, is_energy,
               # %%
               plot_group, is_animated,
               loop, duration, fps,
               # %%
               is_plot_3d_XYz,
               # %%
               zj, z_stored, z, )

    # %%

    if is_plot_selective == 1:

        if "G" in plot_group:
            folder_address = U_dir(G_name + "_sheets_selective", is_save, z=z, )

            # ------------------------- 储存 G1_section_1_shift_amp、G1_section_1_shift_amp、G1_structure_frontface_shift_amp、G1_structure_endface_shift_amp
            # ------------------------- 储存 G1_section_1_shift_phase、G1_section_1_shift_phase、G1_structure_frontface_shift_phase、G1_structure_endface_shift_phase

            G_amps_max, G_amps_min, G_phases_max, G_phases_min = \
                U_selects_plot(folder_address,
                               G_1, G_name + "_sec1",
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

        if "U" in plot_group:
            folder_address = U_dir(U_name + "_sheets_selective", is_save, z=z, )

            # ------------------------- 储存 U0_section_1_amp、U0_section_1_amp、U0_structure_frontface_amp、U0_structure_endface_amp
            # ------------------------- 储存 U0_section_1_phase、U0_section_1_phase、U0_structure_frontface_phase、U0_structure_endface_phase

            U_amps_max, U_amps_min, U_phases_max, U_phases_min = \
                U_selects_plot(folder_address,
                               U_1, U_name + "_sec1",
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

    if is_plot_YZ_XZ == 1:

        folder_address = U_dir(G_name + "_YZ_XZ", is_save, z=z, )

        # ========================= G1_shift_YZ_stored_amp、G1_shift_XZ_stored_amp
        # ------------------------- G1_shift_YZ_stored_phase、G1_shift_XZ_stored_phase

        if "G" in plot_group:
            G_YZ_XZ_amp_max, G_YZ_XZ_amp_min, G_YZ_XZ_phase_max, G_YZ_XZ_phase_min = \
                U_slices_plot(folder_address,
                              G_YZ, G_name + "_YZ",
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

        if is_plot_3d_XYZ == 1:
            # %%
            # 绘制 G1_amp 的 侧面 3D 分布图，以及 初始 和 末尾的 G1_amp（现在 可以 任选位置 了）

            if ("G" in plot_group and "a" in plot_group):
                U_amp_plot_address = U_amp_plot_3d_XYZ(folder_address,
                                                       G_name + "_XYZ",
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
                                                       zj, z=z, )

            # %%
            # 绘制 G1_phase 的 侧面 3D 分布图，以及 初始 和 末尾的 G1_phase

            if ("G" in plot_group and "p" in plot_group):
                U_phase_plot_address = U_phase_plot_3d_XYZ(folder_address,
                                                           G_name + "_XYZ",
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
                                                           zj, z=z, )

        # %%

        folder_address = U_dir(U_name + "_YZ_XZ", is_save, z=z, )

        # ========================= U0_YZ_stored_amp、U0_XZ_stored_amp
        # ------------------------- U0_YZ_stored_phase、U0_XZ_stored_phase

        if "U" in plot_group:
            U_YZ_XZ_amp_max, U_YZ_XZ_amp_min, U_YZ_XZ_phase_max, U_YZ_XZ_phase_min = \
                U_slices_plot(folder_address,
                              U_YZ, U_name + "_YZ",
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

        if is_plot_3d_XYZ == 1:
            # %%
            # 绘制 U0_amp 的 侧面 3D 分布图，以及 初始 和 末尾的 U0_amp

            if ("U" in plot_group and "a" in plot_group):
                U_amp_plot_address = U_amp_plot_3d_XYZ(folder_address,
                                                       U_name + "_XYZ",
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
                                                       zj, z=z, )

            # %%
            # 绘制 U0_phase 的 侧面 3D 分布图，以及 初始 和 末尾的 U0_phase

            if ("U" in plot_group and "p" in plot_group):
                U_phase_plot_address = U_phase_plot_3d_XYZ(folder_address,
                                                           U_name + "_XYZ",
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
                                                           zj, z=z, )

# %%

def U_save(U, U_name, folder_address,
           is_save, is_save_txt, **kwargs, ):
    U_address, ugHGU = gan_Uz_save_address(U_name, folder_address, is_save_txt,
                                            **kwargs)
    if is_save == 1:
        np.savetxt(U_address, U) if is_save_txt else savemat(U_address, {ugHGU: U})

    return U_address


# %%

def U_energy_plot(folder_address,
                  U, U_name,
                  img_name_extension,
                  # %%
                  zj, sample, size_PerPixel,
                  is_save, dpi, size_fig_x, size_fig_y,
                  color_1d, ticks_num, is_title_on, is_axes_on, is_mm,
                  fontsize, font,  # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                  # %%
                  **kwargs, ):
    # %%
    # 绘制 U_amp
    suffix = '_energy'
    # %%
    # 生成 要储存的 图片名 和 地址
    U_energy_full_name, U_energy_plot_address = gan_Uz_plot_address(folder_address, img_name_extension,
                                                                U_name, suffix, **kwargs)
    # %%
    # 生成 图片中的 title
    U_energy_title = gan_Uz_title(U_name, suffix, **kwargs) # 增加 后缀 "_evolution" （才怪，suffix 只 help 辅助 加 5.1 这种序号，原 U_name 里已有 _energy 了）
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
    img_squared_address = desktop + "\\" + "1. " + img_name + "_squared" + img_name_extension  # 除 原始文件 以外，生成的文件 均放在桌面：写出于外
    img_squared_bordered_address = desktop + "\\" + "2. " + img_name + "_squared" + "_bordered" + img_name_extension

    return img_name, img_name_extension, img_address, img_squared_address, img_squared_bordered_address


# %%

def img_squared_Read(img_full_name, U_NonZero_size):
    img_name, img_name_extension, img_address, img_squared_address, img_squared_bordered_address = Info_img(
        img_full_name)
    img_squared = cv2.imdecode(np.fromfile(img_squared_address, dtype=np.uint8), 0)  # 按 相对路径 + 灰度图 读取图片
    size_PerPixel = U_NonZero_size / img_squared.shape[0]  # Unit: mm / 个 每个 像素点 的 尺寸，相当于 △x = △y = △z

    return img_name, img_name_extension, img_address, \
           img_squared_address, img_squared_bordered_address, \
           img_squared, size_PerPixel


# %%
# 导入 方形，以及 加边框 的 图片

def img_squared_bordered_Read(img_full_name,
                              U_NonZero_size, dpi,
                              is_phase_only, ):
    img_name, img_name_extension, img_address, \
    img_squared_address, img_squared_bordered_address, \
    img_squared, size_PerPixel = img_squared_Read(img_full_name, U_NonZero_size)

    img_squared_bordered = cv2.imdecode(np.fromfile(img_squared_bordered_address, dtype=np.uint8),
                                        0)  # 按 相对路径 + 灰度图 读取图片
    size_fig = img_squared_bordered.shape[0] / dpi
    Ix, Iy = img_squared_bordered.shape[0], img_squared_bordered.shape[1]

    if is_phase_only == 1:
        U = np.power(math.e, (img_squared_bordered.astype(np.complex128()) / 255 * 2 * math.pi - math.pi) * 1j)  # 变成相位图
    else:
        U = img_squared_bordered.astype(np.complex128)

    return img_name, img_name_extension, img_squared, size_PerPixel, size_fig, Ix, Iy, U


# %%

def U_read_only(U_name, is_save_txt):
    desktop = get_desktop()
    U_full_name = U_name + (is_save_txt and ".txt" or ".mat")
    U_address = desktop + "\\" + U_full_name
    U = np.loadtxt(U_address, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U_full_name)['U']  # 加载 复振幅场

    return U


# %%
# 导入 方形 图片，以及 U

def U_Read(U_name, img_full_name,
           U_NonZero_size, dpi,
           is_save_txt, ):
    U = U_read_only(U_name, is_save_txt)
    size_fig = U.shape[0] / dpi
    Ix, Iy = U.shape[0], U.shape[1]

    img_name, img_name_extension, img_address, \
    img_squared_address, img_squared_bordered_address, \
    img_squared, size_PerPixel = img_squared_Read(img_full_name, U_NonZero_size)

    return img_name, img_name_extension, img_squared, size_PerPixel, size_fig, Ix, Iy, U
