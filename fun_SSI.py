# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 20:23:31 2022

@author: Xcz
"""

import numpy as np

#%%
# 定义 调制区域切片厚度 的 纵向实际像素、调制区域切片厚度 的 实际纵向尺寸

def Cal_diz(deff_structure_sheet_expect, deff_structure_length_expect, size_PerPixel, 
            Tz, mz,
            is_print = 1):
    
    #%%
    if mz != 0: # 如过你想 让结构 提供 z 向倒格矢
        if deff_structure_sheet_expect >= 0.1 * Tz or deff_structure_sheet_expect <= 0 or (type(deff_structure_sheet_expect) != float and type(deff_structure_sheet_expect) != int): # 则 deff_structure_sheet_expect 不能超过 0.1 * Tz（以保持 良好的 占空比）
            deff_structure_sheet_expect = 0.1 * Tz # Unit: μm
    else:
        if deff_structure_sheet_expect >= 0.01 * deff_structure_length_expect * 1000 or deff_structure_sheet_expect <= 0 or (type(deff_structure_sheet_expect) != float and type(deff_structure_sheet_expect) != int): # 则 deff_structure_sheet_expect 不能超过 0.01 * deff_structure_length_expect（以保持 良好的 精度）
            deff_structure_sheet_expect = 0.01 * deff_structure_length_expect * 1000 # Unit: μm
            
    diz = deff_structure_sheet_expect / 1000 / size_PerPixel # Unit: mm
    # diz = int( deff_structure_sheet_expect / 1000 / size_PerPixel )
    deff_structure_sheet = diz * size_PerPixel * 1000 # Unit: μm 调制区域切片厚度 的 实际纵向尺寸
    is_print and print("deff_structure_sheet = {} μm".format(deff_structure_sheet))
    
    return diz, deff_structure_sheet
    
#%%
# 定义 结构前端面 距离 晶体前端面 的 纵向实际像素、结构前端面 距离 晶体前端面 的 实际纵向尺寸

def Cal_Iz_frontface(diz, 
                     z0_structure_frontface_expect, L0_Crystal, size_PerPixel, 
                     is_print = 1):  
    
    #%%
    if z0_structure_frontface_expect <=0 or z0_structure_frontface_expect >= L0_Crystal or (type(z0_structure_frontface_expect) != float and type(z0_structure_frontface_expect) != int):
        Iz_frontface = 0
    else:
        Iz_frontface = z0_structure_frontface_expect / size_PerPixel

    sheets_num_frontface = int(Iz_frontface // diz)
    Iz_frontface = sheets_num_frontface * diz
    z0_structure_frontface = Iz_frontface * size_PerPixel
    is_print and print("z0_structure_frontface = {} mm".format(z0_structure_frontface))
    
    return sheets_num_frontface, Iz_frontface, z0_structure_frontface

#%%
# 定义 调制区域 的 纵向实际像素、调制区域 的 实际纵向尺寸

def Cal_Iz_structure(diz, 
                     deff_structure_length_expect, size_PerPixel, 
                     is_print = 1): 
    
    #%%
    Iz_structure = deff_structure_length_expect / size_PerPixel # Iz_structure 对应的是 期望的（连续的），而不是 实际的（discrete 离散的）？不，就得是离散的。
    # Iz_structure = int( deff_structure_length_expect / size_PerPixel )
    # sheets_num = Iz_structure // diz
    # Iz_structure = sheets_num * diz
    # deff_structure_length = Iz_structure * size_PerPixel # Unit: mm 调制区域 的 实际纵向尺寸
    # print("deff_structure_length = {} mm".format(deff_structure_length))

    sheets_num_structure = int(Iz_structure // diz)
    Iz_structure = sheets_num_structure * diz # Iz_structure 对应的是 实际的（discrete 离散的），而不是 期望的（连续的）。
    deff_structure_length = Iz_structure * size_PerPixel # Unit: mm 传播距离 = 调制区域 的 实际纵向尺寸
    # deff_structure_length = sheets_num * diz * size_PerPixel # Unit: mm 调制区域 的 实际纵向尺寸
    is_print and print("deff_structure_length = {} mm".format(deff_structure_length))
    
    return sheets_num_structure, Iz_structure, deff_structure_length

#%%
# 定义 结构后端面 距离 晶体前端面 的 纵向实际像素、结构后端面 距离 晶体前端面 的 实际纵向尺寸

def Cal_Iz_endface(sheets_num_frontface, sheets_num_structure, 
                   Iz_frontface, Iz_structure, diz, 
                   size_PerPixel, 
                   is_print = 1): 
    
    #%%
    Iz_endface = Iz_frontface + Iz_structure

    sheets_num_endface = sheets_num_frontface + sheets_num_structure
    Iz_endface = sheets_num_endface * diz
    z0_structure_endface = Iz_endface * size_PerPixel
    is_print and print("z0_structure_endface = {} mm".format(z0_structure_endface))
    
    return sheets_num_endface, Iz_endface, z0_structure_endface

#%%
# 定义 晶体 的 纵向实际像素、晶体 的 实际纵向尺寸

def Cal_Iz(diz, 
           L0_Crystal, size_PerPixel, 
           is_print = 1): 
    
    #%%
    Iz = L0_Crystal / size_PerPixel

    sheets_num = int(Iz // diz) + int(not np.mod(Iz,diz) == 0)
    # Iz = sheets_num * diz
    # z0 = L0_Crystal = Iz * size_PerPixel
    is_print and print("z0 = L0_Crystal = {} mm".format(L0_Crystal))
    
    return sheets_num, Iz

#%%
# 定义 需要展示的截面 1 距离晶体前端面 的 纵向实际像素、需要展示的截面 1 距离晶体前端面 的 实际纵向尺寸

def Cal_iz_1(diz, 
             z0_section_1f_expect, size_PerPixel, 
             is_print = 1): 
    
    #%%
    Iz_1f = z0_section_1f_expect / size_PerPixel

    sheets_num_section_1 = int(Iz_1f // diz)
    sheet_th_section_1 = sheets_num_section_1 # 将要计算的是 距离 Iz_1f 最近的（但在其前面的） 那个面，它是 前面一层 的 后端面，所以这里 用的是 e = endface（但是这一层的前端面，所以取消了这里的 e 的标记，容易引起混淆）
    if sheets_num_section_1 == 0:
        sheets_num_section_1 = 1 # 怕 0 - 1 减出负来了，这样 即使 Iz_1f = 0，前端面也 至少给出的是 dz 处的 场分布
    sheet_th_section_1f = sheets_num_section_1 - 1 # 但需要 前面一层 的 前端面 的 序数 来计算
    iz_1 = sheet_th_section_1 * diz
    z0_1 = iz_1 * size_PerPixel
    is_print and print("z0_section_1 = {} mm".format(z0_1))
    
    return sheet_th_section_1, sheet_th_section_1f, iz_1, z0_1

#%%
# 定义 需要展示的截面 2 距离晶体后端面 的 纵向实际像素、需要展示的截面 2 距离晶体后端面 的 实际纵向尺寸

def Cal_iz_2(sheets_num, 
             Iz, diz, 
             z0_section_2f_expect, size_PerPixel, 
             is_print = 1): 
    
    #%%
    Iz_2f = z0_section_2f_expect / size_PerPixel

    sheets_num_section_2 = sheets_num - int((Iz_2f - np.mod(Iz,diz)) // diz) * int(Iz_2f > np.mod(Iz,diz)) - int(Iz_2f > np.mod(Iz,diz)) # 距离 后端面 的 距离，转换为 距离 前端面 的 距离 （但要稍微 更靠 后端面 一点）
    sheet_th_section_2 = sheets_num_section_2
    sheet_th_section_2f = sheets_num_section_2 - 1
    iz_2 = (sheet_th_section_2 - int(Iz_2f <= np.mod(Iz,diz))) * diz + int(Iz_2f <= np.mod(Iz,diz)) * np.mod(Iz,diz)
    z0_2 = iz_2 * size_PerPixel
    is_print and print("z0_section_2 = {} mm".format(z0_2))
    
    return sheet_th_section_2, sheet_th_section_2f, iz_2, z0_2

#%%
# 定义 调制区域 的 横向实际像素、调制区域 的 实际横向尺寸

def Cal_IxIy(I1_x, I1_y, 
             deff_structure_size_expect, size_PerPixel, 
             is_print = 1):
    
    Ix, Iy = int( deff_structure_size_expect / size_PerPixel ), int( deff_structure_size_expect / size_PerPixel )
    # Ix, Iy 需要与 I1_x, I1_y 同奇偶性，这样 加边框 才好加（对称地加 而不用考虑 左右两边加的量 可能不一样）
    Ix, Iy = Ix + np.mod(I1_x - Ix,2), Iy + np.mod(I1_y - Iy,2)
    deff_structure_size = Ix * size_PerPixel # Unit: mm 不包含 边框，调制区域 的 实际横向尺寸
    is_print and print("deff_structure_size = {} mm".format(deff_structure_size))
    
    return Ix, Iy, deff_structure_size















