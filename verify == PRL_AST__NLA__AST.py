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

def PRL_AST__NLA__AST(U1_name = "", 
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
                      z0_AST = 1, z0_NLA = 5, 
                      #%%
                      lam1 = 0.8, is_air_pump = 0, is_air = 0, T = 25, 
                      deff = 30, is_linear_convolution = 0, 
                      Tx = 10, Ty = 10, Tz = "2*lc", 
                      mx = 0, my = 0, mz = 0, 
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
    # 先空气中 衍射 z0_AST，后晶体内 倍频 z0_NLA
    
    # AST(img_full_name, is_Gauss, is_OAM, is_H_l, l, is_phase_only, is_H_theta, theta_x, theta_y, U1_0_NonZero_size, w0, z0_AST, X, Y, lam1, is_air_pump, 1, T, save, dpi)
    # is_air = 1 空气入射晶体，会发生折射... k1 关于 界面的 切向分量 得相同（所以 折射前后 G 谱是一样的），而晶体内的 k1 会变大 2 倍左右，因此 晶体内的 k1 会偏折向 更靠近 光轴
    # 以致于，在空气中看来，入射角是 1 度，但 在晶体中，入射角会 < 1 度，那么 若在晶体内 提供 同样大的 y 向 倒格矢，则原来 会将倍频光线折到 0 度，现在却会 偏折到 - 0.几 度
    # 原来 集中的分量 在 图中间的， 现在就 在图左边了
    # 可以用 两次 不同 n 的 线性衍射 来 确认这件事
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
        z0_AST, 
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
        is_print, )
    
    U1_name = "6. AST - U1_" + str(float('%.2g' % z0_AST)) + "mm"
    # U1_full_name = U1_name + ".txt"
    # U1_short_name = U1_name.replace('6. AST - ', '')
    
    NLA(U1_name, 
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
        z0_NLA, 
        #%%
        lam1, is_air_pump, is_air, T, 
        deff, is_linear_convolution, 
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
    
    #%%
    # 再空气中 衍射 z0_AST
    
    AST(U2_txt_name, 
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
# PRL_AST__NLA__AST(img_full_name = "grating.png", border_percentage = 0.3, is_LG = 0, is_Gauss = 1, is_OAM = 1, is_H_l = 0, l = 1, p = 0, is_phase_only = 0, is_H_theta = 0, theta_x = 1, theta_y = 0, U1_0_NonZero_size = 0.5, w0 = 0.3, z0_NLA = 0.1, z0_AST = 5, X = 0, Y = 0, lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, deff = 30, Tx = 9.98, Ty = 10, Tz = 17.997, mx = -1, my = 0, mz = 1, is_linear_convolution = 0, save = 0, dpi = 100)
# PRL_AST__NLA__AST(img_full_name = "grating.png", border_percentage = 0.3, is_LG = 0, is_Gauss = 1, is_OAM = 1, is_H_l = 0, l = 1, p = 0, is_phase_only = 0, is_H_theta = 0, theta_x = 1, theta_y = 0, U1_0_NonZero_size = 0.5, w0 = 0.3, z0_NLA = 1, z0_AST = 20, X = 0, Y = 0, lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, deff = 30, Tx = 9.98, Ty = 10, Tz = 17.997, mx = -1, my = 0, mz = 1, is_linear_convolution = 0, save = 0, dpi = 100)

# 基波 1 度 斜向上（图右），倍频 0 度 不偏（正中）。
# theta_x = np.arcsin( LN_n(1.5, 25, "e") * np.sin(1 / 180 * math.pi) ) / math.pi * 180
# PRL_AST__NLA__AST(img_full_name = "grating.png", border_percentage = 0.3, is_LG = 0, is_Gauss = 1, is_OAM = 1, is_H_l = 0, l = 1, p = 0, is_phase_only = 0, is_H_theta = 0, theta_x = theta_x, theta_y = 0, U1_0_NonZero_size = 0.5, w0 = 0.3, z0_NLA = 1, z0_AST = 5, X = 0, Y = 0, lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, deff = 30, Tx = 20.155, Ty = 10, Tz = 17.885, mx = -1, my = 0, mz = 1, is_linear_convolution = 0, save = 0, dpi = 100)
PRL_AST__NLA__AST(U1_name = "", 
                  img_full_name = "Grating.png", 
                  border_percentage = 0.3, 
                  is_phase_only = 0, 
                  #%%
                  z_pump = 0, 
                  is_LG = 0, is_Gauss = 1, is_OAM = 1, 
                  l = 1, p = 0, 
                  theta_x = 1, theta_y = 0, 
                  is_H_l = 0, is_H_theta = 0, 
                  #%%
                  U1_0_NonZero_size = 0.5, w0 = 0.3,
                  z0_AST = 5, z0_NLA = 0.1, 
                  #%%
                  lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, 
                  deff = 30, is_linear_convolution = 0, 
                  Tx = 20.155, Ty = 10, Tz = 17.885, 
                  mx = -1, my = 0, mz = 1, 
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
# PRL_AST__NLA__AST(img_full_name = "grating.png", border_percentage = 0.3, is_LG = 0, is_Gauss = 1, is_OAM = 1, is_H_l = 0, l = 1, p = 0, is_phase_only = 0, is_H_theta = 0, theta_x = 1, theta_y = 0, U1_0_NonZero_size = 0.5, w0 = 0.3, z0_NLA = 2, z0_AST = 15, X = 0, Y = 0, lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, deff = 30, Tx = 20.155, Ty = 10, Tz = 17.885, mx = -1, my = 0, mz = 1, is_linear_convolution = 0, save = 0, dpi = 100)
# 基模 高斯光 Tx = 20.155 最佳 z 向匹配 Tz = 17.4 ~ 17.6 之间。
# 对于 纵向匹配，确实 应该在 预测的 布拉格衍射点 附近。
# PRL_AST__NLA__AST(img_full_name = "grating.png", border_percentage = 0.3, is_LG = 0, is_Gauss = 1, is_OAM = 1, is_H_l = 0, l = 0, p = 0, is_phase_only = 0, is_H_theta = 0, theta_x = 1, theta_y = 0, U1_0_NonZero_size = 0.5, w0 = 0.3, z0_NLA = 0.5, z0_AST = 1, X = 0, Y = 0, lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, deff = 30, Tx = 20.155, Ty = 10, Tz = 17.5, mx = -1, my = 0, mz = 1, is_linear_convolution = 0, save = 0, dpi = 100)
# 基模 高斯光 Tz = 17.885 最佳 x 向匹配 Tx > 50，随着 Tx 从 20.155 下降，横向倒格矢 |Gx| 增大，能量是下降的；但随着 Tx 从 20.155 ~ 50，横向倒格矢 |Gx| 减小，中心频率 右移，能量一直上涨到 0.00299628531145771；随着 Tx 从 50 涨到 500，能量只下降 一点点，到 0.002774217859003981。
# 对于 横向匹配：
# 能量最大时，并不是 将倍频 中心频率 移到零频时。
# 把 倍频 在 频域中心频率 移到零频时，能量并不最大。
# PRL_AST__NLA__AST(img_full_name = "grating.png", border_percentage = 0.3, is_LG = 0, is_Gauss = 1, is_OAM = 1, is_H_l = 0, l = 0, p = 0, is_phase_only = 0, is_H_theta = 0, theta_x = 1, theta_y = 0, U1_0_NonZero_size = 0.5, w0 = 0.3, z0_NLA = 0.5, z0_AST = 1, X = 0, Y = 0, lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, deff = 30, Tx = 500, Ty = 10, Tz = 17.885, mx = -1, my = 0, mz = 1, is_linear_convolution = 0, save = 0, dpi = 100)

# 基波 0 度 不偏（正中），倍频 1 度 斜向上（图右）。
# PRL_AST__NLA__AST(img_full_name = "grating.png", border_percentage = 0.3, is_LG = 0, is_Gauss = 1, is_OAM = 1, is_H_l = 0, l = 1, p = 0, is_phase_only = 0, is_H_theta = 0, theta_x = 0, theta_y = 0, U1_0_NonZero_size = 0.5, w0 = 0.3, z0_NLA = 1, z0_AST = 5, X = 0, Y = 0, lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, deff = 30, Tx = 19.769, Ty = 10, Tz = 18.139, mx = 1, my = 0, mz = 1, is_linear_convolution = 0, save = 0, dpi = 100)

# 基波 0 度 不偏（正中），倍频 11 度 斜向上（图右）。
# PRL_AST__NLA__AST(img_full_name = "lena.png", border_percentage = 0.3, is_LG = 0, is_Gauss = 1, is_OAM = 1, is_H_l = 0, l = 3, p = 0, is_phase_only = 0, is_H_theta = 0, theta_x = 0, theta_y = 0, U1_0_NonZero_size = 0.5, w0 = 0.3, z0_NLA = 1, z0_AST = 5, X = 0, Y = 0, lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, deff = 30, Tx = 1.808, Ty = 10, Tz = 18.139, mx = 1, my = 0, mz = 0, is_linear_convolution = 0, save = 0, dpi = 100)
# 横向失配，纵向匹配：？
# PRL_AST__NLA__AST(img_full_name = "lena.png", border_percentage = 0.3, is_LG = 0, is_Gauss = 1, is_OAM = 1, is_H_l = 0, l = 3, p = 0, is_phase_only = 0, is_H_theta = 0, theta_x = 0, theta_y = 0, U1_0_NonZero_size = 0.5, w0 = 0.3, z0_NLA = 1, z0_AST = 5, X = 0, Y = 0, lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, deff = 30, Tx = 2, Ty = 10, Tz = 18.139, mx = 1, my = 0, mz = 0, is_linear_convolution = 0, save = 0, dpi = 100)
# 横向匹配，纵向失配：拉曼奈斯
# PRL_AST__NLA__AST(img_full_name = "lena.png", border_percentage = 0.3, is_LG = 0, is_Gauss = 1, is_OAM = 1, is_H_l = 0, l = 3, p = 0, is_phase_only = 0, is_H_theta = 0, theta_x = 0, theta_y = 0, U1_0_NonZero_size = 0.5, w0 = 0.3, z0_NLA = 1, z0_AST = 5, X = 0, Y = 0, lam1 = 1.5, is_air_pump = 0, is_air = 0, T = 25, deff = 30, Tx = 1.808, Ty = 10, Tz = 18.139, mx = 1, my = 0, mz = 1, is_linear_convolution = 0, save = 0, dpi = 100)
