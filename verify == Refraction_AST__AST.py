# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:57 2021

@author: Xcz
"""

#%%

import numpy as np
np.seterr(divide='ignore',invalid='ignore')
from fun_img_Resize import image_Add_black_border
from b_1_AST import AST

def Refraction_AST__AST(U1_name = "", 
                        img_full_name = "Grating.png", 
                        border_percentage = 0.3, 
                        is_phase_only = 0, 
                        #%%
                        z_pump = 0, 
                        is_LG = 0, is_Gauss = 0, is_OAM = 0, 
                        l = 0, p = 0, 
                        theta_x = 0, theta_y = 0, 
                        is_H_l = 0, is_H_theta = 0, 
                        #%%
                        U1_0_NonZero_size = 1, w0 = 0.3,
                        z0_1 = 1, z0_n = 5, 
                        #%%
                        lam1 = 0.8, is_air_pump = 0, is_air = 0, T = 25, 
                        #%%
                        is_save = 0, is_save_txt = 0, dpi = 100, 
                        #%%
                        cmap_2d = 'viridis', 
                        #%%
                        ticks_num = 6, is_contourf = 0, 
                        is_title_on = 1, is_axes_on = 1, 
                        is_mm = 1, is_propagation = 0, 
                        #%%
                        fontsize = 9, 
                        font = {'family': 'serif',
                                'style': 'normal', # 'normal', 'italic', 'oblique'
                                'weight': 'normal',
                                'color': 'black', # 'black','gray','darkred'
                                }, 
                        #%%
                        is_self_colorbar = 0, is_colorbar_on = 1, 
                        is_energy = 1, vmax = 1, vmin = 0, 
                        #%%
                        is_print = 1, ):
    
    #%%
    # 线性 惠更斯 菲涅尔 原理
    
    image_Add_black_border(img_full_name, 
                           border_percentage, 
                           is_print, )
    
    # %%
    # 先以 n 衍射 z0_n 后 以 1 衍射 z0_1
    
    # AST('', 
    #     img_full_name, 
    #     is_phase_only, 
    #     #%%
    #     z_pump, 
    #     is_LG, is_Gauss, is_OAM, 
    #     l, p, 
    #     theta_x, theta_y, 
    #     is_H_l, is_H_theta, 
    #     #%%
    #     U1_0_NonZero_size, w0,
    #     z0_n, 
    #     #%%
    #     lam1, is_air_pump, is_air, T, 
    #     #%%
    #     is_save, is_save_txt, dpi, 
    #     #%%
    #     cmap_2d, 
    #     #%%
    #     ticks_num, is_contourf, 
    #     is_title_on, is_axes_on, 
    #     is_mm, is_propagation, 
    #     #%%
    #     fontsize, font, 
    #     #%%
    #     is_self_colorbar, is_colorbar_on, 
    #     is_energy, vmax, vmin, 
    #     #%%
    #     is_print)
    
    # U1_name = "6. AST - U1_" + str(float('%.2g' % z0_n)) + "mm"
    # # U1_full_name = U1_name + ".txt"
    # # U1_short_name = U1_name.replace('6. AST - ', '')
    
    # AST(U1_name, 
    #     img_full_name, 
    #     is_phase_only, 
    #     #%%
    #     z_pump, 
    #     is_LG, is_Gauss, is_OAM, 
    #     l, p, 
    #     theta_x, theta_y, 
    #     is_H_l, is_H_theta, 
    #     #%%
    #     U1_0_NonZero_size, w0,
    #     z0_1, 
    #     #%%
    #     lam1, 1, 1, T, 
    #     #%%
    #     is_save, is_save_txt, dpi, 
    #     #%%
    #     cmap_2d, 
    #     #%%
    #     ticks_num, is_contourf, 
    #     is_title_on, is_axes_on, 
    #     is_mm, is_propagation, 
    #     #%%
    #     fontsize, font, 
    #     #%%
    #     is_self_colorbar, is_colorbar_on, 
    #     is_energy, vmax, vmin, 
    #     #%%
    #     is_print)
    
    #%%
    # 先以 1 衍射 z0_1 后 以 n 衍射 z0_n
    
    AST('', 
        img_full_name, 
        is_phase_only, 
        #%%
        z_pump, 
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        is_H_l, is_H_theta, 
        #%%
        U1_0_NonZero_size, w0,
        z0_1, 
        #%%
        lam1, 1, 1, T, 
        #%%
        is_save, is_save_txt, dpi, 
        #%%
        cmap_2d, 
        #%%
        ticks_num, is_contourf, 
        is_title_on, is_axes_on, 
        is_mm, is_propagation, 
        #%%
        fontsize, font, 
        #%%
        is_self_colorbar, is_colorbar_on, 
        is_energy, vmax, vmin, 
        #%%
        is_print)
    
    U1_name = "6. AST - U1_" + str(float('%.2g' % z0_1)) + "mm"
    # U1_full_name = U1_name + ".txt"
    # U1_short_name = U1_name.replace('6. AST - ', '')
    
    AST(U1_name, 
        img_full_name, 
        is_phase_only, 
        #%%
        z_pump, 
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        is_H_l, is_H_theta, 
        #%%
        U1_0_NonZero_size, w0,
        z0_n, 
        #%%
        lam1, is_air_pump, is_air, T, 
        #%%
        is_save, is_save_txt, dpi, 
        #%%
        cmap_2d, 
        #%%
        ticks_num, is_contourf, 
        is_title_on, is_axes_on, 
        is_mm, is_propagation, 
        #%%
        fontsize, font, 
        #%%
        is_self_colorbar, is_colorbar_on, 
        is_energy, vmax, vmin, 
        #%%
        is_print)
    
    #%%
    
Refraction_AST__AST(U1_name = "", 
                    img_full_name = "Grating.png", 
                    border_percentage = 0.3, 
                    is_phase_only = 0, 
                    #%%
                    z_pump = 0, 
                    is_LG = 0, is_Gauss = 1, is_OAM = 1, 
                    l = 3, p = 0, 
                    theta_x = 1, theta_y = 0, 
                    is_H_l = 0, is_H_theta = 0, 
                    #%%
                    U1_0_NonZero_size = 1, w0 = 0.3,
                    z0_1 = 5, z0_n = 5, 
                    #%%
                    lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, 
                    #%%
                    is_save = 0, is_save_txt = 0, dpi = 100, 
                    #%%
                    cmap_2d = 'viridis', 
                    #%%
                    ticks_num = 6, is_contourf = 0, 
                    is_title_on = 1, is_axes_on = 1, 
                    is_mm = 1, is_propagation = 0, 
                    #%%
                    fontsize = 9, 
                    font = {'family': 'serif',
                            'style': 'normal', # 'normal', 'italic', 'oblique'
                            'weight': 'normal',
                            'color': 'black', # 'black','gray','darkred'
                            }, 
                    #%%
                    is_self_colorbar = 0, is_colorbar_on = 1, 
                    is_energy = 1, vmax = 1, vmin = 0, 
                    #%%
                    is_print = 1, )