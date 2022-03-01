# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

#%%

import math
import numpy as np

#%%

def Step_U(U, mode, 
           Duty_Cycle_x, Duty_Cycle_y, 
           is_positive_xy):
    
    if mode == 'x':
        return ( U > (2 * is_positive_xy - 1) * np.cos(Duty_Cycle_x * math.pi) ).astype(np.int8()) # uint8 会导致 之后 structure 和 modulation 也变成 无符号 整形，以致于 在 0 - 1 时 变成 255 而不是 -1...
    elif mode == 'y':
        return ( U > (2 * is_positive_xy - 1) * np.cos(Duty_Cycle_y * math.pi) ).astype(np.int8()) # uint8 会导致 之后 structure 和 modulation 也变成 无符号 整形，以致于 在 0 - 1 时 变成 255 而不是 -1...

#%%

def CGH(U, mode, 
        Duty_Cycle_x, Duty_Cycle_y, 
        is_positive_xy, 
        #%%
        Gx, Gy, 
        is_Gauss, l, 
        is_continuous, ):
    
    i1_x0, i1_y0 = np.meshgrid([i for i in range(U.shape[0])], [j for j in range(U.shape[1])])
    i1_x0_shift, i1_y0_shift = i1_x0 - U.shape[0] // 2, i1_y0 - U.shape[1] // 2
    if is_Gauss == 1 and l == 0:
        if mode == 'x*y':
            cgh = np.cos(Gx * i1_x0_shift)
            cgh_x = Step_U(cgh, 'x', 
                           Duty_Cycle_x, Duty_Cycle_y, 
                           is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
            cgh = np.cos(Gy * i1_y0_shift)
            cgh_y = Step_U(cgh, 'y', 
                           Duty_Cycle_x, Duty_Cycle_y, 
                           is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
            cgh = cgh_x * cgh_y
        elif mode == 'x':
            cgh = np.cos(Gx * i1_x0_shift)
            cgh = Step_U(cgh, 'x', 
                         Duty_Cycle_x, Duty_Cycle_y, 
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'y':
            cgh = np.cos(Gy * i1_y0_shift)
            cgh = Step_U(cgh, 'y', 
                         Duty_Cycle_x, Duty_Cycle_y, 
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'x+y':
            cgh = np.cos(Gx * i1_x0_shift + Gy * i1_y0_shift)
            cgh = Step_U(cgh, 'x', 
                         Duty_Cycle_x, Duty_Cycle_y, 
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh # 在所有方向的占空比都认为是 Duty_Cycle_x
        return cgh
    else:
        if mode == 'x*y':
            cgh = np.cos(Gx * i1_x0_shift - ( np.angle(U) + math.pi )) - np.cos( np.arcsin( np.abs(U) / np.max( np.abs(U) ) ) )
            cgh_x = Step_U(cgh, 'x', 
                           Duty_Cycle_x, Duty_Cycle_y, 
                           is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
            cgh = np.cos(Gy * i1_y0_shift - ( np.angle(U) + math.pi )) - np.cos( np.arcsin( np.abs(U) / np.max( np.abs(U) ) ) )
            cgh_y = Step_U(cgh, 'y', 
                           Duty_Cycle_x, Duty_Cycle_y, 
                           is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
            cgh = cgh_x * cgh_y
        elif mode == 'x':
            cgh = np.cos(Gx * i1_x0_shift - ( np.angle(U) + math.pi )) - np.cos( np.arcsin( np.abs(U) / np.max( np.abs(U) ) ) )
            cgh = Step_U(cgh, 'x', 
                         Duty_Cycle_x, Duty_Cycle_y, 
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'y':
            cgh = np.cos(Gy * i1_y0_shift - ( np.angle(U) + math.pi )) - np.cos( np.arcsin( np.abs(U) / np.max( np.abs(U) ) ) )
            cgh = Step_U(cgh, 'y', 
                         Duty_Cycle_x, Duty_Cycle_y, 
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
        elif mode == 'x+y':
            cgh = np.cos(Gx * i1_x0_shift + Gy * i1_y0_shift - ( np.angle(U) + math.pi )) - np.cos( np.arcsin( np.abs(U) / np.max( np.abs(U) ) ) )
            cgh = Step_U(cgh,'x', 
                         Duty_Cycle_x, Duty_Cycle_y, 
                         is_positive_xy) if is_continuous == 0 else 0.5 + 0.5 * cgh
        return cgh

#%%

def structure_Generate(U, mode, 
                       Duty_Cycle_x, Duty_Cycle_y, 
                       is_positive_xy, 
                       #%%
                       Gx, Gy, 
                       is_Gauss, l, 
                       is_continuous, 
                       #%%
                       is_target_far_field, is_transverse_xy, is_reverse_xy, ):
    
    if is_target_far_field == 0: # 如果 想要的 U1_0 是近场（晶体后端面）分布
        
        g = np.fft.fft2(U)
        g_shift = np.fft.fftshift(g)
        
        if is_transverse_xy == 1:
            structure = CGH(g_shift, mode, 
                            Duty_Cycle_x, Duty_Cycle_y, 
                            is_positive_xy, 
                            #%%
                            Gx, Gy, 
                            is_Gauss, l, 
                            is_continuous, ).T # 转置（沿 右下 对角线 翻转）
        else:
            structure = CGH(g_shift, mode, 
                            Duty_Cycle_x, Duty_Cycle_y, 
                            is_positive_xy, 
                            #%%
                            Gx, Gy, 
                            is_Gauss, l, 
                            is_continuous, )[::-1] # 上下翻转
            
    else: # 如果 想要的 U1_0 是远场分布
        if is_transverse_xy == 1:
            structure = CGH(U, mode, 
                            Duty_Cycle_x, Duty_Cycle_y, 
                            is_positive_xy, 
                            #%%
                            Gx, Gy, 
                            is_Gauss, l, 
                            is_continuous, ).T # 转置（沿 右下 对角线 翻转）
        else:
            structure = CGH(U, mode, 
                            Duty_Cycle_x, Duty_Cycle_y, 
                            is_positive_xy, 
                            #%%
                            Gx, Gy, 
                            is_Gauss, l, 
                            is_continuous, )[::-1] # 上下翻转
        
    if is_reverse_xy == 1:
        structure = 1 - structure

    return structure