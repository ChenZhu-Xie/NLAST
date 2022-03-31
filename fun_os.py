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
from fun_plot import plot_2d

#%%
# 获取 桌面路径（C 盘 原生）

def GetDesktopPath(): # 修改过 桌面位置 后，就不准了
    return os.path.join(os.path.expanduser("~"), 'Desktop')

#%%
# 获取 桌面路径（注册表）

def get_desktop(): # 无论如何都准，因为读的是注册表
  key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders')
  return winreg.QueryValueEx(key, "Desktop")[0]

#%%
# 获取 当前 py 文件 所在路径 Current Directory

def get_cd():
    return os.path.dirname(os.path.abspath(__file__)) # 其实不需要，默认就是在 相对路径下 读，只需要 文件名 即可

#%%
# 查找

# 查找 text 中的 数字部分
def find_nums(text):
    return re.findall('(\d+)', text)

# 查找 text 中的 非数字部分
def find_NOT_nums(text):
    return re.findall('(\D+)', text)

#%%
# 查找 含 s 的 字符串 part，part 为在 text 中 被 separator（分隔符） 分隔 的 文字
def find_part_has_s_in_text(text, s, separator):
    for i, part in enumerate(text.split(separator)):
        if s in part: # 找到 第一个 part 之后，不加 含 z 的 part，就 跳出 for 循环
            return part

#%%
# 生成 part_1 （被 分隔符 分隔的 第一个） 字符串
def gan_part_1(U1_name, U_name, is_add_sequence, 
               *args, ): # args 是 method、"_phase" 或 '_amp'
    
    part_1_NOT_num = find_NOT_nums(U_name.split('_')[0])[0]
    
    if is_add_sequence == 1:
        if part_1_NOT_num  == 'g':
            part_1_sequence = "3."
        elif part_1_NOT_num  == 'H':
            part_1_sequence = "4."
        elif part_1_NOT_num  == 'G':
            part_1_sequence = "5."
        elif part_1_NOT_num  == 'U':
            part_1_sequence = "6."
        
        
        if len(args) == 2: # 如果 还传入了 后缀 "_phase" 或 '_amp'
            suffix = args[1]
            if suffix == '_amp':
                part_1_sequence += '1.'
            elif suffix == '_phase':
                part_1_sequence += '2.'
            # suffix == '_' + suffix # 识别完后，给 suffix 前面 加上 下划线
        else: # 如果 啥也没传，则 后缀 啥也不加
            suffix = ''
        
        part_1_sequence += ' ' # 数字序号后 都得加个空格
    else:
        part_1_sequence = ''
        
    # args[0] + ' - ' 要放在 上面几步 之后，且 放在 上面 is_add_sequence 之外，因为 哪怕不加序号，也可能加 method，比如在 U_print 的 时候
    if len(args) >= 1: # 如果 传入了 method（第一个 总是 method，因为 有后缀 "phase" or "amp" 则必然 先有 method，不可能 只有 phase or amp 而没有 method）
        part_1_sequence += args[0] + ' - '
    
    # 如果 U1_name 被 _ 分割出的 第一部分 不是空的 且 含有数字，则将其 数字部分 取出，作为 part_1 的 数字部分（传染性）
    U1_name_part_1_nums = find_nums(U1_name.split('_')[0])
    if len(U1_name_part_1_nums) != 0: # 如果 第一部分 含有数字
        part_1_num = U1_name_part_1_nums[0]
    else:
        part_1_num = find_nums(U_name.split('_')[0])[0] # 否则 用 U_name 第一部分 原本的 数字部分，作为 part_1 的 数字部分
    
    part_1 = part_1_sequence + part_1_NOT_num + part_1_num
    
    return part_1, part_1_NOT_num

def gan_part_1z(U1_name, U_name, is_add_sequence, 
                is_auto, *args, ): # args 是 method、z
    
    if is_auto == 0:
        U_new_name = U_name
        part_1_NOT_num = find_NOT_nums(U_name.split('_')[0])[0]
    else:
        #%%
        # 生成 part_1
        # 如果传了 2 个及以上 参数进来，那么将 多传进来的 len(args) - 1 个参数全传入 gan_part_1
        # 剩下一个是 z 或 ()，它必然 传了进来（至少传了 1 个进来的 就是它）
        if len(args) >= 2:
            part_1, part_1_NOT_num = gan_part_1(U1_name, U_name, is_add_sequence, 
                                                *args[1:], ) # 先对 tuple 排除第一个元素地 切片，切片后还是个 tuple。 # 然后解包，再传入 函数
        else:
            part_1, part_1_NOT_num = gan_part_1(U1_name, U_name, is_add_sequence, ) # 否则 没传 method 进来，也就不传 method 进去
        #%%
        # 查找 含 z 的 字符串 part_z 
        part_z = find_part_has_s_in_text(U_name, 'z', '_')
        
        U_new_name = U_name.replace(U_name.split('_')[0], part_1) # 至少把 U_name 第一部分 替换成 part_1，作为 U_new_name
        if U_name.find('z') != -1 and len(args) != 0 and args[0] != (): # 如果 找到 z，且 传了 额外的 参数 进来，这个参数 解包后的 第一个参数 不是 空 tuple ()
            z = args[0]
            U_new_name = U_new_name.replace(part_z, str(float('%.2g' % z)) + "mm") # 把 原来含 z 的 part_z 替换为 str(float('%.2g' % z)) + "mm"
    return U_new_name, part_1_NOT_num

#%%

def U_energy_print(U1_name, is_print, is_auto, 
                   U, U_name, method, 
                   *args, ): # args 是 z 或 ()
    
    U_full_name, part_1_NOT_num = gan_part_1z(U1_name, U_name, 0, # 不加 序列号
                                              is_auto, args, method, ) # 要有 method （诸如 'AST'）
    # 这里 还不能是 *arg，这样会 抹除 信息：传了 2 个及以下参数的时候，传的是哪 2 个参数？ z、method、suffix 中的 任意 2 个，都有可能。
        
    is_print and print(U_full_name + ".total_energy = {}".format(np.sum(np.abs(U) ** 2)))

#%%

def U_dir(U1_name, U_name, is_auto, 
          *args, ): # args 是 z 或 ()
    
    folder_name, part_1_NOT_num = gan_part_1z(U1_name, U_name, 1, # 要加 序列号
                                              is_auto, args, ) # 没有 method （诸如 'AST'）
    
    #%%
    desktop = get_desktop()
    folder_address = desktop + "\\" + folder_name
    
    if not os.path.isdir(folder_address):
        os.makedirs(folder_address)
    
    return folder_address

#%%

def U_plot(U1_name, folder_address, is_auto, 
           U, U_name, method, 
           img_name_extension, 
           #%%
           zj, sample, size_PerPixel,
           is_save, dpi, size_fig,
           cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, is_propagation,
           fontsize, font,
           is_self_colorbar, is_colorbar_on, is_energy, vmax, vmin, 
           #%%
           *args, ): # args 是 z 或 ()
    
    #%%
    # 绘制 U_amp
    suffix = '_amp'
    #%%
    # 生成 要储存的 图片名 和 地址
    U_amp_name, part_1_NOT_num = gan_part_1z(U1_name, U_name, 1, # 要加 序列号
                                             is_auto, args, method, suffix, ) # 有 method 和 suffix
    U_amp_name += suffix # 增加 后缀 "_amp" 或 "_phase"
    #%%
    # 生成 地址
    U_amp_full_name = U_amp_name + img_name_extension
    U_amp_plot_address = folder_address + "\\" + U_amp_full_name
    #%%
    # 生成 图片中的 title
    U_amp_title, part_1_NOT_num = gan_part_1z(U1_name, U_name, 0, # 不加 序列号
                                              is_auto, args, method, suffix, ) # 有 method 和 suffix
    U_amp_title += suffix # 增加 后缀 "_amp" 或 "_phase"
    #%%
    plot_2d(zj, sample, size_PerPixel,
            np.abs(U), U_amp_plot_address, U_amp_title,
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, is_propagation,
            fontsize, font,
            is_self_colorbar, is_colorbar_on, is_energy, vmax, vmin)
    
    #%%
    # 绘制 U_phase
    suffix = '_phase'
    #%%
    # 生成 要储存的 图片名 和 地址
    U_phase_name, part_1_NOT_num = gan_part_1z(U1_name, U_name, 1, # 要加 序列号
                                             is_auto, args, method, suffix, ) # 有 method 和 suffix
    U_phase_name += suffix # 增加 后缀 "_amp" 或 "_phase"
    #%%
    # 生成 地址
    U_phase_full_name = U_phase_name + img_name_extension
    U_phase_plot_address = folder_address + "\\" + U_phase_full_name
    #%%
    # 生成 图片中的 title
    U_phase_title, part_1_NOT_num = gan_part_1z(U1_name, U_name, 0, # 不加 序列号
                                              is_auto, args, method, suffix, ) # 有 method 和 suffix
    U_phase_title += suffix # 增加 后缀 "_amp" 或 "_phase"
    #%%
    plot_2d(zj, sample, size_PerPixel,
            np.angle(U), U_phase_plot_address, U_phase_title,
            is_save, dpi, size_fig,
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, is_propagation,
            fontsize, font,
            is_self_colorbar, is_colorbar_on, 0, vmax, vmin) # 相位 不能有 is_energy = 1
    
    return U_amp_plot_address, U_phase_plot_address

#%%

def U_save(U1_name, folder_address, is_auto, 
           U, U_name, method, 
           is_save_txt, *args, ):
    
    U_full_name, part_1_NOT_num = gan_part_1z(U1_name, U_name, 1, # 要加 序列号
                                              is_auto, args, method, ) # 要有 method （诸如 'AST'）
    
    file_name = U_full_name + (is_save_txt and ".txt" or ".mat")
    U_address = folder_address + "\\" + file_name
    np.savetxt(U_address, U) if is_save_txt else savemat(U_address, {part_1_NOT_num: U})
    
    return U_address

#%%

def Info_img(img_full_name):
    
    img_name = os.path.splitext(img_full_name)[0]
    img_name_extension = os.path.splitext(img_full_name)[1]
    
    cdir = get_cd()
    desktop = get_desktop()

    img_address = cdir + "\\" + img_full_name # 默认 在 相对路径下 读，只需要 文件名 即可：读于内
    img_squared_address = desktop + "\\" + "1." + img_name + "_squared" + img_name_extension # 除 原始文件 以外，生成的文件 均放在桌面：写出于外
    img_squared_bordered_address = desktop + "\\" + "2." + img_name + "_squared" + "_bordered" + img_name_extension
    
    return img_name, img_name_extension, img_address, img_squared_address, img_squared_bordered_address

#%%
# 导入 方形，以及 加边框 的 图片

def img_squared_bordered_Read(img_full_name, 
                              U_NonZero_size, dpi, 
                              is_phase_only, ):

    img_name, img_name_extension, img_address, img_squared_address, img_squared_bordered_address = Info_img(img_full_name)

    img_squared = cv2.imdecode(np.fromfile(img_squared_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
    img_squared_bordered = cv2.imdecode(np.fromfile(img_squared_bordered_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片

    size_PerPixel = U_NonZero_size / img_squared.shape[0] # Unit: mm / 个 每个 像素点 的 尺寸，相当于 △x = △y = △z
    size_fig = img_squared_bordered.shape[0] / dpi
    Ix, Iy = img_squared_bordered.shape[0], img_squared_bordered.shape[1]

    if is_phase_only == 1:
        U = np.power(math.e, (img_squared_bordered.astype(np.complex128()) / 255 * 2 * math.pi - math.pi) * 1j) # 变成相位图
    else:
        U = img_squared_bordered.astype(np.complex128)
        
    return img_name, img_name_extension, img_squared, size_PerPixel, size_fig, Ix, Iy, U

#%%
# 导入 方形 图片，以及 U

def U_Read(U_name, img_full_name, 
           U_NonZero_size, dpi, 
           is_save_txt, ):
    
    desktop = get_desktop()
    
    U_full_name = U_name + (is_save_txt and ".txt" or ".mat")
    U_address = desktop + "\\" + U_full_name
    img_name, img_name_extension, img_address, img_squared_address, img_squared_bordered_address = Info_img(img_full_name)

    img_squared = cv2.imdecode(np.fromfile(img_squared_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
    U = np.loadtxt(U_address, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U_full_name)['U'] # 加载 复振幅场

    size_PerPixel = U_NonZero_size / img_squared.shape[0] # Unit: mm / 个 每个 像素点 的 尺寸，相当于 △x = △y = △z
    size_fig = U.shape[0] / dpi
    Ix, Iy = U.shape[0], U.shape[1]
    
    return img_name, img_name_extension, img_squared, size_PerPixel, size_fig, Ix, Iy, U


