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
from b_3_NLA import NLA

def PRL_NLA__AST(U1_name = "", 
                 img_full_name = "Grating.png", 
                 border_percentage = 0.3, 
                 is_phase_only = 0, 
                 #%%
                 z_pump = 0, 
                 is_LG = 0, is_Gauss = 0, is_OAM = 0, 
                 l = 0, p = 0, 
                 theta_x = 0, theta_y = 0, 
                 #%%
                 is_random_phase = 0, 
                 is_H_l = 0, is_H_theta = 0, is_H_random_phase = 0, 
                 #%%
                 U1_0_NonZero_size = 1, w0 = 0.3,
                 z0_AST = 1, z0_NLA = 5, 
                 # %%
                 lam1=0.8, is_air_pump=0, is_air=0, T=25,
                 deff=30, is_fft = 1, fft_mode = 0, 
                 is_linear_convolution = 0,
                 #%%
                 Tx=10, Ty=10, Tz="2*lc",
                 mx=0, my=0, mz=0,
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
                 is_print = 1, is_contours = 1, n_TzQ = 1, Gz_max_Enhance = 1, match_mode = 1, ):
    
    #%%
    # start
    
    image_Add_black_border(img_full_name, 
                           border_percentage, 
                           is_print, )
    
    #%%
    # 先晶体内 倍频 z0_NLA，后空气中 衍射 z0_AST
    
    NLA('', 
        img_full_name, 
        is_phase_only, 
        #%%
        z_pump, 
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        is_random_phase, 
        is_H_l, is_H_theta, is_H_random_phase, 
        #%%
        U1_0_NonZero_size, w0,
        z0_NLA, 
        #%%
        lam1, is_air_pump, is_air, T, 
        deff, is_fft, fft_mode,
        is_linear_convolution, 
        Tx, Ty, Tz, 
        mx, my, mz, 
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
        is_print, is_contours, n_TzQ, Gz_max_Enhance, match_mode, )
    
    U2_txt_name = "6. NLA - U2_" + str(float('%.2g' % z0_NLA)) + "mm"
    # U2_txt_full_name = U2_txt_name + ".txt"
    # U2_txt_short_name = U2_txt_name.replace('6. NLA - ', '')
    
    AST(U2_txt_name, 
        img_full_name, 
        is_phase_only, 
        #%%
        z_pump, 
        is_LG, is_Gauss, is_OAM, 
        l, p, 
        theta_x, theta_y, 
        is_random_phase, 
        is_H_l, is_H_theta, is_H_random_phase, 
        #%%
        U1_0_NonZero_size, w0,
        z0_AST, 
        #%%
        lam1, is_air_pump, 1, T, 
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
        is_print, )
    
    U2_AST_txt_name = "6. AST - U2_" + str(float('%.2g' % z0_AST)) + "mm"
    U2_AST_txt_full_name = U2_AST_txt_name + (is_save_txt and ".txt" or ".mat")
    # U2_AST_txt_short_name = U2_AST_txt_name.replace('6. AST - ', '')
    U2_AST_txt_short_name = U2_AST_txt_name.replace('6. AST - ', 'AST - ')
    
    #%%

# 基波 1 度 斜向上（图右），倍频 -1 度 斜向下（图左）。   
# PRL_NLA__AST(img_full_name = "grating.png", border_percentage = 0.3, is_LG = 0, is_Gauss = 1, is_OAM = 1, is_H_l = 0, l = 1, p = 0, is_phase_only = 0, is_H_theta = 0, theta_x = 1, theta_y = 0, U1_0_NonZero_size = 0.5, w0 = 0.3, z0_NLA = 0.1, z0_AST = 5, X = 0, Y = 0, lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, deff = 30, Tx = 9.98, Ty = 10, Tz = 17.997, mx = -1, my = 0, mz = 1, is_linear_convolution = 0, save = 0, dpi = 100)
# PRL_NLA__AST(img_full_name = "grating.png", border_percentage = 0.3, is_LG = 0, is_Gauss = 1, is_OAM = 1, is_H_l = 0, l = 1, p = 0, is_phase_only = 0, is_H_theta = 0, theta_x = 1, theta_y = 0, U1_0_NonZero_size = 0.5, w0 = 0.3, z0_NLA = 1, z0_AST = 20, X = 0, Y = 0, lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, deff = 30, Tx = 9.98, Ty = 10, Tz = 17.997, mx = -1, my = 0, mz = 1, is_linear_convolution = 0, save = 0, dpi = 100)
# 基波 1 度 斜向上（图右），倍频 0 度 不偏（正中）。
PRL_NLA__AST(U1_name = "", 
             img_full_name = "Grating.png", 
             border_percentage = 0.3, 
             is_phase_only = 0, 
             #%%
             z_pump = 0, 
             is_LG = 0, is_Gauss = 1, is_OAM = 1, 
             l = 3, p = 0, 
             theta_x = 1, theta_y = 0, 
             is_random_phase = 0, 
             is_H_l = 0, is_H_theta = 0, is_H_random_phase = 0, 
             #%%
             U1_0_NonZero_size = 0.5, w0 = 0.3,
             z0_AST = 5, z0_NLA = 0.1, 
             # %%
             lam1=0.8, is_air_pump=0, is_air=0, T=25,
             deff=30, is_fft = 1, fft_mode = 0, 
             is_linear_convolution = 0,
             Tx=10, Ty=10, Tz="2*lc",
             mx=0, my=0, mz=0,
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
             is_print = 1, is_contours = 1, n_TzQ = 1, Gz_max_Enhance = 1, match_mode = 1, )
# PRL_NLA__AST(img_full_name = "grating.png", border_percentage = 0.3, is_LG = 0, is_Gauss = 1, is_OAM = 1, is_H_l = 0, l = 1, p = 0, is_phase_only = 0, is_H_theta = 0, theta_x = 1, theta_y = 0, U1_0_NonZero_size = 0.5, w0 = 0.3, z0_NLA = 2, z0_AST = 15, X = 0, Y = 0, lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, deff = 30, Tx = 20.155, Ty = 10, Tz = 17.885, mx = -1, my = 0, mz = 1, is_linear_convolution = 0, save = 0, dpi = 100)
# 基波 0 度 不偏（正中），倍频 1 度 斜向上（图右）。
# PRL_NLA__AST(img_full_name = "grating.png", border_percentage = 0.3, is_LG = 0, is_Gauss = 1, is_OAM = 1, is_H_l = 0, l = 1, p = 0, is_phase_only = 0, is_H_theta = 0, theta_x = 0, theta_y = 0, U1_0_NonZero_size = 0.5, w0 = 0.3, z0_NLA = 0.1, z0_AST = 5, X = 0, Y = 0, lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, deff = 30, Tx = 19.769, Ty = 10, Tz = 18.139, mx = 1, my = 0, mz = 1, is_linear_convolution = 0, save = 0, dpi = 100)

# PRL_NLA__AST(img_full_name = "grating.png", border_percentage = 0.3, is_LG = 0, is_Gauss = 1, is_OAM = 1, is_H_l = 0, l = 1, p = 0, is_phase_only = 0, is_H_theta = 0, theta_x = 8, theta_y = 0, U1_0_NonZero_size = 0.5, w0 = 0.3, z0_NLA = 0.1, z0_AST = 5, X = 0, Y = 0, lam1 = 2, is_air_pump = 0, is_air = 0, T = 25, deff = 30, Tx = 4.737, Ty = 10, Tz = 18.139, mx = 1, my = 0, mz = 0, is_linear_convolution = 0, save = 0, dpi = 100)
