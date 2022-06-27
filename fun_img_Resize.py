# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

# %%

import cv2
import numpy as np
from PIL import Image
from fun_os import Info_img
from fun_global_var import tree_print, init_GLV_DICT

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
                           is_print=1, **kwargs):
    # img_full_name = "Grating.png"
    # border_percentage = 0.5 # 边框 占图片的 百分比，也即 图片 放大系数

    is_print and print(tree_print(kwargs.get("is_end", 0), kwargs.get("add_level", 0)) + "加黑边")
    kwargs.pop("is_end", 0)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。
    # 其实 外面的 kwargs 没传进来，所以这里 直接就是 is_end = 0，add_level = 0
    # %%
    # 预处理 导入图片 为方形，并加边框

    img_name, img_name_extension, img_address, folder_address, img_squared_address, img_squared_bordered_address \
        = Info_img(img_full_name)

    img = cv2.imdecode(np.fromfile(img_address, dtype=np.uint8), 0)  # 按 相对路径 + 灰度图 读取图片
    is_print and print(tree_print() + "img.shape = {}".format(img.shape))

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
    is_print and print(tree_print() + "img_squared.shape = {}".format(img_squared.shape))

    border_width = int(img_squared.shape[0] * border_percentage / 2)
    image_Border(img_squared_address, img_squared_bordered_address, loc='a', width=border_width, color=(0, 0, 0, 255))

    img_squared_bordered = cv2.imdecode(np.fromfile(img_squared_bordered_address, dtype=np.uint8),
                                        0)  # 按 相对路径 + 灰度图 读取图片
    is_print and print(tree_print(kwargs.get("is_end_last", 1)) + "U.shape = img_squared_bordered.shape = {}".format(img_squared_bordered.shape))


# %%

def if_image_Add_black_border(U_name, img_full_name,
                              is_name_main, is_print, **kwargs, ):  # 没有 该函数作为起始的 py 文件，需要加 init_GLV_DICT

    if is_name_main:  # 等价于：如果是 第一次 进入该程序
        # %% 开始 加边框
        # print(1, kwargs)
        # init_GLV_DICT(**kwargs) # 这里初始化的 init_GLV 传了参数进去 —— 没懂为什么得是 **....，不然传进去变成位置参数 args 中的一元素了
        kwargs.pop("is_end", 0)
        if ((type(U_name) != str) or U_name == "") and ("U" not in kwargs and "U1" not in kwargs):
            border_percentage = kwargs["border_percentage"] if "border_percentage" in kwargs else 0.1
            kwargs.pop("border_percentage", None)

            is_end_last=-1 if kwargs.get('ray', "2") == "3" else 1
            image_Add_black_border(img_full_name,  # 预处理 导入图片 为方形，并加边框
                                   border_percentage,
                                   is_print, is_end_last=is_end_last)  # 没把 kwargs 传进来，因此 外面的 is_end = 1 不会进来，也就不会 使加黑边 为 末尾

        if kwargs.get('ray', "2") == "3":
            U2_name, img2_full_name = kwargs.get("U2_name", U_name), kwargs.get("img2_full_name", img_full_name)
            if ((type(U2_name) != str) or U2_name == "") and ("U2" not in kwargs):
                border_percentage = kwargs["border_percentage"] if "border_percentage" in kwargs else 0.1
                kwargs.pop("border_percentage", None)

                image_Add_black_border(img2_full_name,  # 预处理 导入图片 为方形，并加边框
                                       border_percentage,
                                       is_print, add_level=1)  # 没把 kwargs 传进来，因此 外面的 is_end = 1 不会进来，也就不会 使加黑边 为 末尾


# %%
# 需要先将 目标 U_0 = img_squared 给 放大 或 缩小 到 与 全息图（结构） 横向尺寸 Ix_structure, Iy_structure 相同，才能开始 之后的工作

def img_squared_Resize(img_full_name, img_squared,
                       Ix_structure, Iy_structure, Ix, Iy,
                       is_print=1, **kwargs):
    img_name, img_name_extension, img_address, folder_address, img_squared_address, img_squared_bordered_address \
        = Info_img(img_full_name)

    is_print and print(tree_print(kwargs.get("is_end", 0), kwargs.get("add_level", 0)) + "图片裁剪")
    kwargs.pop("is_end", 0)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    img_squared_resize = cv2.resize(img_squared, (Iy_structure, Ix_structure), interpolation=cv2.INTER_AREA)
    # 使用cv2.imread()读取图片之后,数据的形状和维度布局是(H,W,C),但是使用函数cv2.resize()进行缩放时候,传入的目标形状是(W,H)
    img_squared_resize_full_name = "1.1. " + img_name + "_squared" + "_resize" + img_name_extension
    img_squared_resize_address = folder_address + "\\" + img_squared_resize_full_name
    # cv2.imwrite(img_squared_resize_address, img_squared_resize) # 保存 img_squared_resize，但不能有 中文路径
    cv2.imencode(img_squared_resize_address, img_squared_resize)[1].tofile(img_squared_resize_address)
    is_print and print(tree_print() + "img_squared_resize.shape = {}".format(img_squared_resize.shape))

    img_squared_resize_bordered_full_name = "2.2. " + img_name + "_squared" + "_resize" + "_bordered" + img_name_extension
    img_squared_resize_bordered_address = folder_address + "\\" + img_squared_resize_bordered_full_name
    border_width_x = (Ix - Ix_structure) // 2
    border_width_y = (Iy - Iy_structure) // 2
    # 上下加边框 border_width_x
    image_Border(img_squared_resize_address, img_squared_resize_bordered_address, loc='t', width=border_width_x,
                 color=(0, 0, 0, 255))
    image_Border(img_squared_resize_bordered_address, img_squared_resize_bordered_address, loc='b', width=border_width_x,
                 color=(0, 0, 0, 255))
    # 左右加边框 border_width_y
    image_Border(img_squared_resize_bordered_address, img_squared_resize_bordered_address, loc='r', width=border_width_y,
                 color=(0, 0, 0, 255))
    image_Border(img_squared_resize_bordered_address, img_squared_resize_bordered_address, loc='l', width=border_width_y,
                 color=(0, 0, 0, 255))
    # 上: 't' or 'top'
    # 右: 'r' or 'rigth'
    # 下: 'b' or 'bottom'
    # 左: 'l' or 'left'
    from fun_global_var import Set
    Set("border_width_x", border_width_x)
    Set("border_width_y", border_width_y)
    img_squared_resize_bordered = cv2.imdecode(np.fromfile(img_squared_resize_bordered_address, dtype=np.uint8),
                                               0)  # 按 相对路径 + 灰度图 读取图片
    is_print and print(tree_print(1) + "structure_squared.shape = img_squared_resize_bordered.shape = {}"
                       .format(img_squared_resize_bordered.shape))

    return border_width_x, border_width_y, img_squared_resize_full_name, img_squared_resize

# %%

def U_resize(U, U_pixels_x, U_pixels_y, Ix, Iy):
    if U_pixels_x > 0:
        U = U_resize_x(U, U_pixels_x, Iy)
    if U_pixels_y > 0:
        U = U_resize_y(U, U_pixels_y, Ix)
    return U

def U_resize_x(U, U_pixels_x, Iy):
    if U_pixels_x > Iy:  # 如果 U_pixels_x 比 图片宽，则需要 np.pad 填充零
        one_side_pixels_x = (U_pixels_x - Iy) // 2
        other_side_pixels_x = (U_pixels_x - Iy) - one_side_pixels_x
        U = np.pad(U, ((0, 0), (one_side_pixels_x, other_side_pixels_x)),
                   'constant', constant_values=(0, 0))  # ((行前, 行后) 填充行, (列前, 列后) 填充列)
    elif U_pixels_x < Iy:  # 如果 U_pixels_x 比 图片窄，则需要 裁剪图片
        one_side_pixels_x = (Iy - U_pixels_x) // 2
        other_side_pixels_x = (Iy - U_pixels_x) - one_side_pixels_x
        U = U[:, one_side_pixels_x:(-other_side_pixels_x)]  # 去掉前 one_side_pixels_x 个，和后 other_side_pixels_x 个 列
    return U

def U_resize_y(U, U_pixels_y, Ix):
    if U_pixels_y > Ix:  # 如果 U_pixels_x 比 图片宽，则需要 np.pad 填充零
        one_side_pixels_y = (U_pixels_y - Ix) // 2
        other_side_pixels_x = (U_pixels_y - Ix) - one_side_pixels_y
        U = np.pad(U, ((one_side_pixels_y, other_side_pixels_x), (0, 0)),
                   'constant', constant_values=(0, 0))  # ((行前, 行后) 填充行, (列前, 列后) 填充列)
    elif U_pixels_y < Ix:  # 如果 U_pixels_y 比 图片矮，则需要 裁剪图片
        one_side_pixels_y = (Ix - U_pixels_y) // 2
        other_side_pixels_x = (Ix - U_pixels_y) - one_side_pixels_y
        U = U[one_side_pixels_y:(-other_side_pixels_x), :]  # 去掉前 one_side_pixels_y 个，和后 other_side_pixels_x 个 行
    return U