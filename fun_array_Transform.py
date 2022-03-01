# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

import numpy as np

#%%
# ( 先转置，后上下翻转 )**2 = ( 先上下翻转，后转置 )**2
# = 先上下翻转，后左右翻转 = 先左右翻转，后上下翻转
# = 绕垂直纸面朝外的 z 轴，旋转 180 度

def Rotate_180(U):
    
    U = U.transpose((1,0))
    U = U[::-1]
    U = U.transpose((1,0))
    U = U[::-1]
    
    return U

#%%

def Roll_xy(U, 
            roll_x, roll_y, 
            is_linear_convolution, ):
    
    # # 往下（行） 线性平移 roll_x 像素：先对 将要平移出框的 roll_x 个行 取零，再 循环平移；先后 顺序可反，但 取零的行的 上下 也得 同时反
    U = np.roll(U, roll_x, axis=0)
    
    if is_linear_convolution == 1:
        if roll_x < 0: # n2_x + Gx / (2 * math.pi) * I2_x 小于 (I2_x - 1) - I2_x//2 时，后半部分 取零，只剩 前半部分
            U[roll_x:, :] = 0
        elif roll_x > 0: # n2_x + Gx / (2 * math.pi) * I2_x 大于 (I2_x - 1) - I2_x//2 时，前半部分 取零，只剩 后半部分
            U[:roll_x, :] = 0
            
    # # 往右（列） 线性平移 roll_y 像素：先对 将要平移出框的 roll_y 个列 取零，再 循环平移；先后 顺序可反，但 取零的列的 左右 也得 同时反
    U = np.roll(U, roll_y, axis=1)
    
    if is_linear_convolution == 1:
        if roll_y < 0:
            U[:, roll_y:] = 0
        elif roll_y > 0:
            U[:, :roll_y] = 0
            
    return U