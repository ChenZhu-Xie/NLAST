# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

import os
import cv2
import numpy as np
import math
from scipy.io import loadmat

#%%

def Info_img(img_full_name):
    
    img_name = os.path.splitext(img_full_name)[0]
    img_name_extension = os.path.splitext(img_full_name)[1]
    
    location = os.path.dirname(os.path.abspath(__file__)) # 其实不需要，默认就是在 相对路径下 读，只需要 文件名 即可

    img_address = location + "\\" + img_full_name
    img_squared_address = location + "\\" + "1." + img_name + "_squared" + img_name_extension
    img_squared_bordered_address = location + "\\" + "2." + img_name + "_squared" + "_bordered" + img_name_extension
    
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
    
    U_full_name = U_name + (is_save_txt and ".txt" or ".mat")
    img_name, img_name_extension, img_address, img_squared_address, img_squared_bordered_address = Info_img(img_full_name)

    img_squared = cv2.imdecode(np.fromfile(img_squared_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
    U = np.loadtxt(U_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U_full_name)['U'] # 加载 复振幅场

    size_PerPixel = U_NonZero_size / img_squared.shape[0] # Unit: mm / 个 每个 像素点 的 尺寸，相当于 △x = △y = △z
    size_fig = U.shape[0] / dpi
    Ix, Iy = U.shape[0], U.shape[1]
    
    return img_name, img_name_extension, img_squared, size_PerPixel, size_fig, Ix, Iy, U