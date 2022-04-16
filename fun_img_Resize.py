# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

# %%

import cv2
import numpy as np
from PIL import Image
from fun_os import get_desktop, Info_img

Image.MAX_IMAGE_PIXELS = 10E10  # Image 的 默认参数 无法处理那么大的图片


# %%

def image_Border(src, dst, loc='a', width=3, color=(0, 0, 0, 255)):
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
        w += 2 * width
        h += 2 * width
        img_new = Image.new('RGBA', (w, h), color)
        img_new.paste(img_ori, (width, width))
    elif loc in ['t', 'top']:
        h += width
        img_new = Image.new('RGBA', (w, h), color)
        img_new.paste(img_ori, (0, width, w, h))
    elif loc in ['r', 'right']:
        w += width
        img_new = Image.new('RGBA', (w, h), color)
        img_new.paste(img_ori, (0, 0, w - width, h))
    elif loc in ['b', 'bottom']:
        h += width
        img_new = Image.new('RGBA', (w, h), color)
        img_new.paste(img_ori, (0, 0, w, h - width))
    elif loc in ['l', 'left']:
        w += width
        img_new = Image.new('RGBA', (w, h), color)
        img_new.paste(img_ori, (width, 0, w, h))
    else:
        pass

    # 保存图片
    img_new.save(dst)


# %%

def image_Add_black_border(img_full_name="Grating.png",
                           border_percentage=0.5,
                           is_print=1, ):
    # img_full_name = "Grating.png"
    # border_percentage = 0.5 # 边框 占图片的 百分比，也即 图片 放大系数

    is_print and print("    >·>·>·>·>·>·>·>·>·> 加黑边 start >·>·>·>·>·>·>·>·>·>")

    # %%
    # 预处理 导入图片 为方形，并加边框

    img_name, img_name_extension, img_address, img_squared_address, img_squared_bordered_address = Info_img(
        img_full_name)

    img = cv2.imdecode(np.fromfile(img_address, dtype=np.uint8), 0)  # 按 相对路径 + 灰度图 读取图片
    is_print and print("img.shape = {}".format(img.shape))

    if img.shape[0] != img.shape[1]:  # 如果 高 ≠ 宽
        if img.shape[0] < img.shape[1]:  # 如果 图片很宽，就上下加 黑色_不透明_边框
            image_Border(img_address, img_squared_address, loc='t', width=(img.shape[1] - img.shape[0]) // 2,
                         color=(0, 0, 0, 255))
            if (img.shape[1] - img.shape[0]) % 2 == 1:  # 如果 宽高差 是 奇数，则 下边框 多加一个 像素
                image_Border(img_squared_address, img_squared_address, loc='b',
                             width=(img.shape[1] - img.shape[0]) // 2 + 1, color=(0, 0, 0, 255))
            else:
                image_Border(img_squared_address, img_squared_address, loc='b',
                             width=(img.shape[1] - img.shape[0]) // 2, color=(0, 0, 0, 255))
        else:  # 如果 图片很高，就左右加 黑色_不透明_边框
            image_Border(img_address, img_squared_address, loc='l', width=(img.shape[0] - img.shape[1]) // 2,
                         color=(0, 0, 0, 255))
            if (img.shape[0] - img.shape[1]) % 2 == 1:  # 如果 高宽差 是 奇数，则 右边框 多加一个 像素
                image_Border(img_squared_address, img_squared_address, loc='r',
                             width=(img.shape[0] - img.shape[1]) // 2 + 1, color=(0, 0, 0, 255))
            else:
                image_Border(img_squared_address, img_squared_address, loc='r',
                             width=(img.shape[0] - img.shape[1]) // 2, color=(0, 0, 0, 255))
    else:
        image_Border(img_address, img_squared_address, loc='a', width=0, color=(0, 0, 0, 255))

    img_squared = cv2.imdecode(np.fromfile(img_squared_address, dtype=np.uint8), 0)  # 按 绝对路径 + 灰度图 读取图片
    is_print and print("img_squared.shape = {}".format(img_squared.shape))

    border_width = int(img_squared.shape[0] * border_percentage)
    image_Border(img_squared_address, img_squared_bordered_address, loc='a', width=border_width, color=(0, 0, 0, 255))

    img_squared_bordered = cv2.imdecode(np.fromfile(img_squared_bordered_address, dtype=np.uint8),
                                        0)  # 按 相对路径 + 灰度图 读取图片
    is_print and print("U.shape = img_squared_bordered.shape = {}".format(img_squared_bordered.shape))

    is_print and print("    >·>·>·>·>·>·>·>·>·> 加黑边 end >·>·>·>·>·>·>·>·>·>")


# %%

def if_image_Add_black_border(U_name, img_full_name,
                              is_name_main, is_print, **kwargs, ):
    if (type(U_name) != str) or U_name == "" and "U" not in kwargs:
        if is_name_main:
            border_percentage = kwargs["border_percentage"] if "border_percentage" in kwargs else 0.1

            image_Add_black_border(img_full_name,  # 预处理 导入图片 为方形，并加边框
                                   border_percentage,
                                   is_print, )


# %%
# 需要先将 目标 U_0_NonZero = img_squared 给 放大 或 缩小 到 与 全息图（结构） 横向尺寸 Ix_structure, Iy_structure 相同，才能开始 之后的工作

def img_squared_Resize(img_name, img_name_extension, img_squared,
                       Ix_structure, Iy_structure, Ix,
                       is_print=1, ):
    desktop = get_desktop()

    img_squared_resize = cv2.resize(img_squared, (Ix_structure, Iy_structure), interpolation=cv2.INTER_AREA)
    img_squared_resize_full_name = "1. " + img_name + "_squared" + "_resize" + img_name_extension
    img_squared_resize_address = desktop + "\\" + img_squared_resize_full_name
    # cv2.imwrite(img_squared_resize_address, img_squared_resize) # 保存 img_squared_resize，但不能有 中文路径
    cv2.imencode(img_squared_resize_address, img_squared_resize)[1].tofile(img_squared_resize_address)
    is_print and print("img_squared_resize.shape = {}".format(img_squared_resize.shape))

    img_squared_resize_bordered_address = desktop + "\\" + "2. " + img_name + "_squared" + "_resize" + "_bordered" + img_name_extension
    border_width = (Ix - Ix_structure) // 2
    image_Border(img_squared_resize_address, img_squared_resize_bordered_address, loc='a', width=border_width,
                 color=(0, 0, 0, 255))
    img_squared_resize_bordered = cv2.imdecode(np.fromfile(img_squared_resize_bordered_address, dtype=np.uint8),
                                               0)  # 按 相对路径 + 灰度图 读取图片
    is_print and print(
        "structure_squared.shape = img_squared_resize_bordered.shape = {}".format(img_squared_resize_bordered.shape))

    return border_width, img_squared_resize_full_name, img_squared_resize
