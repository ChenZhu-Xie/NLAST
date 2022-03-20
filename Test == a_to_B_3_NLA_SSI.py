# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:38:57 2021

@author: Xcz
"""

#%%

import numpy as np
np.seterr(divide='ignore',invalid='ignore')
from fun_img_Resize import image_Add_black_border
from fun_os import img_squared_bordered_Read, U_Read
from fun_pump import pump_LG
from fun_linear import Cal_n, Cal_kz
from A_3_structure_Generate_NLC import structure_NLC
from B_3_NLA_SSI import NLA_SSI
from B_3_SFM_SSI import SFM_SSI

def a_to_B_3_NLA_SSI(U1_name_Structure = "", 
                     border_percentage = 0.3, 
                     is_phase_only_Structure = 0, 
                     #%%
                     z_pump_Structure = 0, 
                     is_LG_Structure = 0, is_Gauss_Structure = 1, is_OAM_Structure = 1, 
                     l_Structure = 0, p_Structure = 0, 
                     theta_x_Structure = 0, theta_y_Structure = 0, 
                     is_random_phase_Structure = 0, 
                     is_H_l_Structure = 0, is_H_theta_Structure = 0, is_H_random_phase_Structure = 0, 
                     #%%
                     U1_name = "", 
                     img_full_name = "l=1.png", 
                     is_phase_only = 0, 
                     #%%
                     z_pump = 0, 
                     is_LG = 0, is_Gauss = 1, is_OAM = 1, 
                     l = 1, p = 0, 
                     theta_x = 1, theta_y = 0, 
                     is_random_phase = 0, 
                     is_H_l = 0, is_H_theta = 0, is_H_random_phase = 0, 
                     #%%---------------------------------------------------------------------
                     #%%
                     U1_0_NonZero_size = 0.5, w0 = 0.1, w0_Structure = 5, structure_size_Enlarge = 0.1, 
                     L0_Crystal = 2, z0_structure_frontface_expect = 0.5, deff_structure_length_expect = 1, 
                     deff_structure_sheet_expect = 1.8, sheets_stored_num = 10, 
                     z0_section_1f_expect = 1, z0_section_2f_expect = 1, X = 0, Y = 0, 
                     Duty_Cycle_x = 0.5, Duty_Cycle_y = 0.5, Duty_Cycle_z = 0.5, structure_xy_mode = 'x', Depth = 2, 
                     #%%
                     is_continuous = 0, is_target_far_field = 1, is_transverse_xy = 0, is_reverse_xy = 0, is_positive_xy = 1, 
                     #%%
                     is_bulk = 1, is_no_backgroud = 1, 
                     is_stored = 0, is_show_structure_face = 1, is_energy_evolution_on = 1, 
                     #%%
                     lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, 
                     deff = 30, 
                     Tx = 19.769, Ty = 20, Tz = 18.139, 
                     mx = -1, my = 0, mz = 1, 
                     #%%
                     is_save = 0, is_save_txt = 0, dpi = 100, 
                     #%%
                     color_1d = 'b', cmap_2d = 'viridis', cmap_3d = 'rainbow', 
                     elev = 10, azim = -65, alpha = 2, 
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
                     is_print = 1, is_contours = 1, n_TzQ = 1, Gz_max_Enhance = 1, match_mode = 1, 
                     #%%
                     is_NLA = 1, ):
    
    #%%
    # a_Image_Add_Black_border
    
    image_Add_black_border(img_full_name, 
                           border_percentage, 
                           is_print, )
    
    #%%
    # 为了生成 U1_0 和 g1_shift
    
    if (type(U1_name) != str) or U1_name == "":
        
        #%%
        # 导入 方形，以及 加边框 的 图片
        
        img_name, img_name_extension, img_squared, size_PerPixel, size_fig, I1_x, I1_y, U1_0 = img_squared_bordered_Read(img_full_name, 
                                                                                                                         U1_0_NonZero_size, dpi, 
                                                                                                                         is_phase_only)
        
        #%%
        # 预处理 输入场
        
        n1, k1 = Cal_n(size_PerPixel, 
                       is_air_pump, 
                       lam1, T, p = "e")
        
        U1_0 = pump_LG(img_full_name, 
                       I1_x, I1_y, size_PerPixel, 
                       U1_0, w0, k1, z_pump, 
                       is_LG, is_Gauss, is_OAM, 
                       l, p, 
                       theta_x, theta_y, 
                       is_random_phase, 
                       is_H_l, is_H_theta, is_H_random_phase, 
                       is_save, is_save_txt, dpi, 
                       cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                       fontsize, font, 
                       1, is_colorbar_on, is_energy, vmax, vmin, 
                       is_print, ) 
        
    else:

        #%%
        # 导入 方形 的 图片，以及 U
        
        img_name, img_name_extension, img_squared, size_PerPixel, size_fig, I1_x, I1_y, U1_0 = U_Read(U1_name, img_full_name, 
                                                                                                      U1_0_NonZero_size, dpi, 
                                                                                                      is_save_txt, )
        
    #%%
    # 线性 角谱理论 - 基波 begin

    g1 = np.fft.fft2(U1_0)
    g1_shift = np.fft.fftshift(g1)
    
    #%%
    
    if is_contours != 0:
        
        is_print and print("===== 描边 start =====")
        
        if deff_structure_length_expect <= L0_Crystal + deff_structure_sheet_expect / 1000:
            deff_structure_length_expect = L0_Crystal + deff_structure_sheet_expect / 1000
            is_print and print("deff_structure_length_expect = {} mm".format(deff_structure_length_expect))
            
        is_print and print("===== 描边 end =====")
    
    #%%
    
    structure_NLC(U1_name_Structure, 
                  img_full_name, 
                  is_phase_only_Structure, 
                  #%%
                  z_pump_Structure, 
                  is_LG_Structure, is_Gauss_Structure, is_OAM_Structure, 
                  l_Structure, p_Structure, 
                  theta_x_Structure, theta_y_Structure, 
                  is_random_phase_Structure, 
                  is_H_l_Structure, is_H_theta_Structure, is_H_random_phase_Structure, 
                  #%%
                  U1_0_NonZero_size, w0_Structure, structure_size_Enlarge, 
                  deff_structure_length_expect, deff_structure_sheet_expect, 
                  Duty_Cycle_x, Duty_Cycle_y, Duty_Cycle_z, structure_xy_mode, Depth, 
                  #%%
                  is_continuous, is_target_far_field, is_transverse_xy, is_reverse_xy, is_positive_xy, is_no_backgroud, 
                  #%%
                  lam1, is_air_pump, is_air, T, 
                  deff, 
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
                  is_print, is_contours, n_TzQ, Gz_max_Enhance, match_mode, 
                  #%%
                  g1_shift, )
    
    #%%
    # B_3_NLA_SSI
    
    arg = [ U1_name, 
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
            L0_Crystal, z0_structure_frontface_expect, deff_structure_length_expect, 
            deff_structure_sheet_expect, sheets_stored_num, 
            z0_section_1f_expect, z0_section_2f_expect, X, Y, 
            #%%
            is_bulk, is_no_backgroud, 
            is_stored, is_show_structure_face, is_energy_evolution_on, 
            #%%
            lam1, is_air_pump, is_air, T, 
            deff, 
            Tx, Ty, Tz, 
            mx, my, mz, 
            #%%
            is_save, is_save_txt, dpi, 
            #%%
            color_1d, cmap_2d, cmap_3d, 
            elev, azim, alpha, 
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
            is_print, is_contours, n_TzQ, Gz_max_Enhance, match_mode, ]
    
    if is_NLA == 1:
        NLA_SSI(*arg)
    else:
        SFM_SSI(*arg)

a_to_B_3_NLA_SSI(U1_name_Structure = "", 
                 border_percentage = 0.1, 
                 is_phase_only_Structure = 0, 
                 #%%
                 z_pump_Structure = 0, 
                 is_LG_Structure = 0, is_Gauss_Structure = 1, is_OAM_Structure = 0, 
                 l_Structure = 0, p_Structure = 0, 
                 theta_x_Structure = 0, theta_y_Structure = 0, 
                 is_random_phase_Structure = 0, 
                 is_H_l_Structure = 0, is_H_theta_Structure = 0, is_H_random_phase_Structure = 0, 
                 #%%
                 U1_name = "", 
                 img_full_name = "grating.png", 
                 is_phase_only = 0, 
                 #%%
                 z_pump = 0, 
                 is_LG = 0, is_Gauss = 0, is_OAM = 0, 
                 l = 0, p = 0, 
                 theta_x = 0, theta_y = 0, 
                 is_random_phase = 0, 
                 is_H_l = 0, is_H_theta = 0, is_H_random_phase = 0, 
                 #%%---------------------------------------------------------------------
                 #%%
                 U1_0_NonZero_size = 0.9, w0 = 0.5, w0_Structure = 0, structure_size_Enlarge = 0.1, 
                 L0_Crystal = 4, z0_structure_frontface_expect = 0, deff_structure_length_expect = 0.5, 
                 deff_structure_sheet_expect = 1, sheets_stored_num = 10, 
                 z0_section_1f_expect = 0, z0_section_2f_expect = 0, X = 0, Y = 0, 
                 Duty_Cycle_x = 0.5, Duty_Cycle_y = 0.5, Duty_Cycle_z = 0.5, structure_xy_mode = 'x', Depth = 2, 
                 #%%
                 is_continuous = 0, is_target_far_field = 1, is_transverse_xy = 0, is_reverse_xy = 0, is_positive_xy = 1, 
                 #%%
                 is_bulk = 0, is_no_backgroud = 0, 
                 is_stored = 0, is_show_structure_face = 0, is_energy_evolution_on = 1, 
                 #%%
                 lam1 = 1, is_air_pump = 0, is_air = 0, T = 25, 
                 deff = 30, 
                 Tx = 13.703, Ty = 13.703, Tz = 6.943, 
                 mx = 0, my = 0, mz = 1, 
                 #%%
                 is_save = 1, is_save_txt = 0, dpi = 100, 
                 #%%
                 color_1d = 'b', cmap_2d = 'viridis', cmap_3d = 'rainbow', 
                 elev = 10, azim = -65, alpha = 2, 
                 #%%
                 ticks_num = 6, is_contourf = 0, 
                 is_title_on = 1, is_axes_on = 1, 
                 is_mm = 1, is_propagation = 0, 
                 #%%
                 fontsize = 7, 
                 font = {'family': 'serif', 
                         'style': 'normal', # 'normal', 'italic', 'oblique'
                         'weight': 'normal',
                         'color': 'black', # 'black','gray','darkred'
                         }, 
                 #%%
                 is_self_colorbar = 0, is_colorbar_on = 1, 
                 is_energy = 1, vmax = 1, vmin = 0, 
                 #%%
                 is_print = 1, is_contours = 2, n_TzQ = 1, Gz_max_Enhance = 1, match_mode = 0, 
                 #%%
                 is_NLA = 0, )