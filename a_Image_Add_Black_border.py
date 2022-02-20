# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

#%%

import os
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 10E10 #Image 的 默认参数 无法处理那么大的图片

#%%
def Image_Add_Black_border(file_full_name = "Grating.png", border_percentage = 0.5):
    # file_full_name = "Grating.png"
    # border_percentage = 0.5 # 边框 占图片的 百分比，也即 图片 放大系数
    def image_border(src, dst, loc='a', width=3, color=(0, 0, 0, 255)):
        '''
        src: (str) 需要加边框的图片路径
        dst: (str) 加边框的图片保存路径
        loc: (str) 边框添加的位置, 默认是'a'(
            四周: 'a' or 'all'
            上: 't' or 'top'
            右: 'r' or 'rigth'
            下: 'b' or 'bottom'
            左: 'l' or 'left'
        )
        width: (int) 边框宽度 (默认是3)
        color: (int or 3-tuple) 边框颜色 (默认是0, 表示黑色; 也可以设置为三元组表示RGB颜色)
        '''
        # 读取图片
        img_ori = Image.open(src)
        w = img_ori.size[0]
        h = img_ori.size[1]
    
        # 添加边框
        if loc in ['a', 'all']:
            w += 2*width
            h += 2*width
            img_new = Image.new('RGBA', (w, h), color)
            img_new.paste(img_ori, (width, width))
        elif loc in ['t', 'top']:
            h += width
            img_new = Image.new('RGBA', (w, h), color)
            img_new.paste(img_ori, (0, width, w, h))
        elif loc in ['r', 'right']:
            w += width
            img_new = Image.new('RGBA', (w, h), color)
            img_new.paste(img_ori, (0, 0, w-width, h))
        elif loc in ['b', 'bottom']:
            h += width
            img_new = Image.new('RGBA', (w, h), color)
            img_new.paste(img_ori, (0, 0, w, h-width))
        elif loc in ['l', 'left']:
            w += width
            img_new = Image.new('RGBA', (w, h), color)
            img_new.paste(img_ori, (width, 0, w, h))
        else:
            pass
    
        # 保存图片
        img_new.save(dst)
    
    #%%
    # 预处理 导入图片 为方形，并加边框
    
    file_name = os.path.splitext(file_full_name)[0]
    file_name_extension = os.path.splitext(file_full_name)[1]
    
    location = os.path.dirname(os.path.abspath(__file__))
    file_address = location + "\\" + file_full_name
    file_squared_address = location + "\\" + "1." + file_name + "_squared" + file_name_extension
    file_squared_bordered_address = location + "\\" + "2." + file_name + "_squared" + "_bordered" + file_name_extension
    
    img_original = cv2.imdecode(np.fromfile(file_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
    print("img_original.shape = {}".format(img_original.shape))
    
    if img_original.shape[0] != img_original.shape[1]: # 如果 高 ≠ 宽
        if img_original.shape[0] < img_original.shape[1]:# 如果 图片很宽，就上下加 黑色_不透明_边框
            image_border(file_address, file_squared_address, loc='t', width=( img_original.shape[1] - img_original.shape[0] ) // 2, color=(0, 0, 0, 255))
            if ( img_original.shape[1] - img_original.shape[0] ) % 2 == 1: # 如果 宽高差 是 奇数，则 下边框 多加一个 像素
                image_border(file_squared_address, file_squared_address, loc='b', width=( img_original.shape[1] - img_original.shape[0] ) // 2 + 1, color=(0, 0, 0, 255))
            else:
                image_border(file_squared_address, file_squared_address, loc='b', width=( img_original.shape[1] - img_original.shape[0] ) // 2, color=(0, 0, 0, 255))
        else: # 如果 图片很高，就左右加 黑色_不透明_边框
            image_border(file_address, file_squared_address, loc='l', width=( img_original.shape[0] - img_original.shape[1] ) // 2, color=(0, 0, 0, 255))
            if ( img_original.shape[0] - img_original.shape[1] ) % 2 == 1: # 如果 高宽差 是 奇数，则 右边框 多加一个 像素
                image_border(file_squared_address, file_squared_address, loc='r', width=( img_original.shape[0] - img_original.shape[1] ) // 2 + 1, color=(0, 0, 0, 255))
            else:
                image_border(file_squared_address, file_squared_address, loc='r', width=( img_original.shape[0] - img_original.shape[1] ) // 2, color=(0, 0, 0, 255))
    else:
        image_border(file_address, file_squared_address, loc='a', width=0, color=(0, 0, 0, 255))
    
    img_squared = cv2.imdecode(np.fromfile(file_squared_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
    print("img_squared.shape = {}".format(img_squared.shape))
    
    border_width = int(img_squared.shape[0] * border_percentage)
    image_border(file_squared_address, file_squared_bordered_address, loc='a', width=border_width, color=(0, 0, 0, 255))
    img_squared_bordered = cv2.imdecode(np.fromfile(file_squared_bordered_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
    print("U1.shape = U2.shape = img_squared_bordered.shape = {}".format(img_squared_bordered.shape))
    
# Image_Add_Black_border(border_percentage = 0.5)