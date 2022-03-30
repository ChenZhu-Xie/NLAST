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
def dir_text_1(U1_name, U_name, is_add_sequence, 
               *args, ):
    
    part_1_NOT_num = find_NOT_nums(U_name.split('_')[0])[0]
    
    if is_add_sequence == 1:
        if part_1_NOT_num  == 'g':
            part_1_sequence = "3. "
        elif part_1_NOT_num  == 'H':
            part_1_sequence = "4. "
        elif part_1_NOT_num  == 'G':
            part_1_sequence = "5. "
        elif part_1_NOT_num  == 'U':
            part_1_sequence = "6. "
    else:
        part_1_sequence = ''
    
    # 如果 U1_name 被 _ 分割出的 第一部分 不是空的 且 含有数字，则将其 数字部分 取出，作为 part_1 的 数字部分（传染性）
    U1_name_part_1_nums = find_nums(U1_name.split('_')[0])
    if len(U1_name_part_1_nums) != 0: # 如果 第一部分 含有数字
        part_1_num = U1_name_part_1_nums[0]
    else:
        part_1_num = find_nums(U_name.split('_')[0])[0] # 否则 用 U_name 第一部分 原本的 数字部分，作为 part_1 的 数字部分
    
    if len(args) == 0:
        part_1 = part_1_sequence + part_1_NOT_num + part_1_num
    else:
        part_1 = part_1_sequence + args[0] + ' - ' + part_1_NOT_num + part_1_num
    
    return part_1, part_1_NOT_num

#%%

def U_energy_print(U1_name, is_print, is_auto, 
                   U, U_name, method, 
                   *args, ):
    
    if is_auto == 0:
        U_full_name = U_name
    else:
        #%%
        # 生成 part_1
        part_1, part_1_NOT_num = dir_text_1(U1_name, U_name, 0, 
                                            method, )
        #%%
        # 查找 含 z 的 字符串 part_z 
        part_z = find_part_has_s_in_text(U_name, 'z', '_')
        
        U_full_name = U_name.replace(U_name.split('_')[0], part_1) # 至少把 U_name 第一部分 替换成 part_1，作为 U_full_name
        if U_name.find('z') != -1 and len(args) != 0: # 如果 找到 z，且 传了 额外的 参数 进来
            z = args[0]
            U_full_name = U_full_name.replace(part_z, str(float('%.2g' % z)) + "mm") # 把 原来含 z 的 part_z 替换为 str(float('%.2g' % z)) + "mm"
        
    is_print and print(U_full_name + ".total_energy = {}".format(np.sum(np.abs(U) ** 2)))

#%%

def U_dir(U1_name, U_name, is_auto, 
          *args, ):
    
    if is_auto == 0:
        folder_name = U_name
    else:
        #%%
        # 生成 part_1
        part_1, part_1_NOT_num = dir_text_1(U1_name, U_name, 1, )
        #%%
        # 查找 含 z 的 字符串 part_z 
        part_z = find_part_has_s_in_text(U_name, 'z', '_')
        
        folder_name = U_name.replace(U_name.split('_')[0], part_1) # 至少把 U_name 第一部分 替换成 part_1，作为 folder name
        if U_name.find('z') != -1 and len(args) != 0: # 如果 找到 z，且 传了 额外的 参数 进来
            z = args[0]
            folder_name = folder_name.replace(part_z, str(float('%.2g' % z)) + "mm") # 把 原来含 z 的 part_z 替换为 str(float('%.2g' % z)) + "mm"
    
    #%%
    desktop = get_desktop()
    folder_address = desktop + "\\" + folder_name
    
    if not os.path.isdir(folder_address):
        os.makedirs(folder_address)
    
    return folder_address

#%%

def U_save(U1_name, folder_address, is_auto, 
           U, U_name, method, 
           is_save_txt, *args, ):
    
    if is_auto == 0:
        U_full_name = U_name
    else:
        #%%
        # 生成 part_1
        part_1, part_1_NOT_num = dir_text_1(U1_name, U_name, 1, 
                                            method, )
        #%%
        # 查找 含 z 的 字符串 part_z 
        part_z = find_part_has_s_in_text(U_name, 'z', '_')
        
        U_full_name = U_name.replace(U_name.split('_')[0], part_1) # 至少把 U_name 第一部分 替换成 part_1，作为 U_full_name
        if U_name.find('z') != -1 and len(args) != 0: # 如果 找到 z，且 传了 额外的 参数 进来
            z = args[0]
            U_full_name = U_full_name.replace(part_z, str(float('%.2g' % z)) + "mm") # 把 原来含 z 的 part_z 替换为 str(float('%.2g' % z)) + "mm"
    
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


