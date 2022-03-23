# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 20:23:31 2022

@author: Xcz
"""

import math
import numpy as np
from fun_array_Transform import Roll_xy
from fun_linear import Cal_kz, fft2, ifft2, Uz_AST, Find_energy_Dropto_fraction
from fun_pump import incline_profile

#%%

def Sinc(x):
    return np.nan_to_num( np.sin(x) / x ) + np.isnan( np.sin(x) / x ).astype(np.int8)

#%%

def Cosc(x):
    return np.nan_to_num( (np.cos(x) - 1) / x )
# return np.nan_to_num( (np.cos(x) - 1) / x ) * ( 1 - np.isnan( (np.cos(x) - 1) / x ).astype(np.int8) ) 不够聪明

#%%
# 定义 对于 kz 的 类似 Sinc 的 函数：( e^ikz - 1 ) / kz

def Eikz(x):
    return Cosc(x) + 1j * Sinc(x)

#%%
# 定义 m 级次 的 倒格波 系数 Cm

def C_m(m):
    if m == 0:
        return 1
    else:
        return Sinc(math.pi*m / 2) - Sinc(math.pi*m)

#%%

def Cal_lc_SHG(k1, k2, Tz, size_PerPixel, 
               is_print = 1):
    
    dk = 2*k1 - k2 # Unit: 1 / mm
    lc = math.pi / abs(dk) * size_PerPixel # Unit: mm
    is_print and print("相干长度 = {} μm".format(lc * 1000))
    
    # print(type(Tz) != np.float64)
    # print(type(Tz) != float) # float = np.float ≠ np.float64
    if (type(Tz) != float and type(Tz) != np.float64 and type(Tz) != int) or Tz <= 0: # 如果 传进来的 Tz 既不是 float 也不是 int，或者 Tz <= 0，则给它 安排上 2*lc
        Tz = 2*lc * 1000  # Unit: um
        
    return dk, lc, Tz

#%%

def Cal_GxGyGz(mx, my, mz,
               Tx, Ty, Tz, size_PerPixel, 
               is_print = 1):
    
    Gx = 2 * math.pi * mx * size_PerPixel / (Tx / 1000) # Tz / 1000 即以 mm 为单位
    Gy = 2 * math.pi * my * size_PerPixel / (Ty / 1000) # Tz / 1000 即以 mm 为单位
    Gz = 2 * math.pi * mz * size_PerPixel / (Tz / 1000) # Tz / 1000 即以 mm 为单位
        
    return Gx, Gy, Gz

#%%

def Cal_dk_z_Q_shift_SHG(k1, 
                         k1_z_shift, k2_z_shift, 
                         mesh_k1_x_k1_y_shift, mesh_k2_x_k2_y_shift, 
                         n2_x, n2_y, 
                         Gx, Gy, Gz, ):
    
    # n2_x_n2_y 的 mesh 才用 Gy / (2 * math.pi) * I2_y)，这里是 k2_x_k2_y 的 mesh，所以用 Gy 才对应
    dk_x_shift = mesh_k2_x_k2_y_shift[n2_x, n2_y, 0] - mesh_k1_x_k1_y_shift[:, :, 0] - Gy
    # 其实 mesh_k2_x_k2_y_shift[:, :, 0]、mesh_n2_x_n2_y_shift[:, :, 0]、mesh_n2_x_n2_y[:, :, 0]、 n2_x 均只和 y，即 [:, :] 中的 第 2 个数字 有关，
    # 只由 列 y、ky 决定，与行 即 x、kx 无关
    # 而 Gy 得与 列 y、ky 发生关系,
    # 所以是 - Gy 而不是 Gx
    # 并且这里的 dk_x_shift 应写为 dk_y_shift
    dk_y_shift = mesh_k2_x_k2_y_shift[n2_x, n2_y, 1] - mesh_k1_x_k1_y_shift[:, :, 1] - Gx
    k1_z_shift_dk_x_dk_y = (k1**2 - dk_x_shift**2 - dk_y_shift**2 + 0j )**0.5
    
    dk_z_shift = k1_z_shift + k1_z_shift_dk_x_dk_y - k2_z_shift[n2_x, n2_y]
    dk_z_Q_shift = dk_z_shift + Gz
    
    return dk_z_Q_shift
    
#%%

def Cal_roll_xy(Gx, Gy,
                Ix, Iy, 
                *args ):
    if len(args) >= 2:
        nx, ny = args[0], args[1]
        roll_x = np.floor( Ix//2 - (Ix - 1) + nx - Gy / (2 * math.pi) * Iy ).astype(np.int64)
        roll_y = np.floor( Iy//2 - (Iy - 1) + ny - Gx / (2 * math.pi) * Ix ).astype(np.int64)
        # 之后要平移列，而 Gx 才与列有关...
    else:
        roll_x = np.floor( Gy / (2 * math.pi) * Iy).astype(np.int64)
        roll_y = np.floor( Gx / (2 * math.pi) * Ix).astype(np.int64)
    
    return roll_x, roll_y

#%%

def G2_z_modulation_NLAST(k1, k2, Gz, 
                          modulation, U1_0, iz, const, ):
    
    k2_z_shift, mesh_k2_x_k2_y_shift = Cal_kz(U1_0.shape[0], U1_0.shape[1], k2)
    
    kiiz_shift = k1 + k2_z_shift + Gz

    U1_z_Squared_modulated = fft2(
        ifft2(fft2(modulation) / (kiiz_shift ** 2 - k2 ** 2)) * Uz_AST(U1_0, k1, iz) ** 2)

    U1_0_Squared_modulated = fft2(
        ifft2(fft2(modulation) / (kiiz_shift ** 2 - k2 ** 2)) * U1_0 ** 2)

    G2_z_shift = const * (U1_z_Squared_modulated * math.e ** (Gz * iz * 1j) \
                           - U1_0_Squared_modulated * math.e ** (k2_z_shift * iz * 1j))
    
    # G2_z_shift = const * U1_z_Squared_modulated * math.e ** (Gz * iz * 1j)
    # G2_z_shift = const * U1_0_Squared_modulated * math.e ** (k2_z_shift * iz * 1j)
    
    return G2_z_shift

def G2_z_NLAST(k1, k2, Gx, Gy, Gz, 
               U1_0, iz, const, 
               is_linear_convolution, ):
    
    Ix, Iy = U1_0.shape[0], U1_0.shape[1]
    k2_z_shift, mesh_k2_x_k2_y_shift = Cal_kz(Ix, Iy, k2)

    G_U1_z0_Squared_shift = fft2(Uz_AST(U1_0, k1, iz) ** 2)
    g_U1_0_Squared_shift = fft2(U1_0 ** 2)

    roll_x, roll_y = Cal_roll_xy(Gx, Gy,
                                 Ix, Iy, )

    G_U1_z0_Squared_shift_Q = Roll_xy(G_U1_z0_Squared_shift,
                                      roll_x, roll_y,
                                      is_linear_convolution, )
    g_U1_0_Squared_shift_Q = Roll_xy(g_U1_0_Squared_shift,
                                     roll_x, roll_y,
                                     is_linear_convolution, )

    molecule = G_U1_z0_Squared_shift_Q * math.e ** (Gz * iz * 1j) \
                - g_U1_0_Squared_shift_Q * math.e ** (k2_z_shift * iz * 1j) 
    
    # molecule = G_U1_z0_Squared_shift_Q * math.e ** (Gz * iz * 1j)
    # molecule = g_U1_0_Squared_shift_Q
    # molecule = g_U1_0_Squared_shift_Q * math.e ** (k2_z_shift * iz * 1j)
    
    #%%
    
    # Gz_shift, mesh_dont_care = Cal_kz(I1_x, I1_y, Gz)
    # molecule = G_U1_z0_Squared_shift_Q * math.e ** (Gz_shift * iz * 1j) \
    #            - g_U1_0_Squared_shift_Q * math.e ** (k2_z_shift * iz * 1j)
    
    #%%
    
    # U = G_U1_z0_Squared_shift_Q * math.e ** (Gz * iz * 1j)
    # U = incline_profile(I1_x, I1_y, 
    #                     U, k2, 
    #                     - np.arcsin(Gx/k2) / math.pi * 180, - np.arcsin(Gy/k2) / math.pi * 180, )
    # molecule = U - g_U1_0_Squared_shift_Q * math.e ** (k2_z_shift * iz * 1j) 

    #%%

    # roll_x, roll_y = Cal_roll_xy(Gx, Gy,
    #                               I2_x, I2_y, )
    
    # g1_shift_roll = Roll_xy(g1_shift,
    #                     roll_x//2, roll_y//2,
    #                     is_linear_convolution, )
    # G_U1_z0_Squared_shift_Q = fft2(Uz_AST(ifft2(g1_shift_roll), k1, i1_z0) ** 2)
    
    # g_U1_0_Squared_shift = fft2(U1_0 ** 2)
    # g_U1_0_Squared_shift_Q = Roll_xy(g_U1_0_Squared_shift,
    #                                   roll_x, roll_y,
    #                                   is_linear_convolution, )

    # molecule = G_U1_z0_Squared_shift_Q * math.e ** (Gz * iz * 1j) \
    #             - g_U1_0_Squared_shift_Q * math.e ** (k2_z_shift * iz * 1j) 
    
    #%%
    
    # if Gx == 0 and Gy == 0:
    #     molecule = G_U1_z0_Squared_shift_Q * math.e ** (Gz * iz * 1j) \
    #                 - g_U1_0_Squared_shift_Q * math.e ** (k2_z_shift * iz * 1j) 
    # else:
    #     molecule = g_U1_0_Squared_shift_Q * math.e ** (k2_z_shift * iz * 1j) 
    
    # %% denominator: dk_shift_Squared

    # n2_x_n2_y 的 mesh 才用 Gy / (2 * math.pi) * I2_y)，这里是 k2_x_k2_y 的 mesh，所以用 Gy 才对应
    k1izQ_shift = (k1 ** 2 - (mesh_k2_x_k2_y_shift[:, :, 0] - Gy) ** 2 - (
            mesh_k2_x_k2_y_shift[:, :, 1] - Gx) ** 2 + 0j) ** 0.5

    kizQ_shift = k1 + k1izQ_shift + Gz
    # kizQ_shift = k1 + k2_z_shift + Gz
    denominator = kizQ_shift ** 2 - k2_z_shift ** 2

    kizQ_shift = k1 + (k1 ** 2 - Gx ** 2 - Gy ** 2) ** 0.5 + Gz
    denominator = kizQ_shift ** 2 - k2 ** 2

    # %% G2_z0_shift

    G2_z_shift = 2 * const * molecule / denominator
    
    return G2_z_shift

#%%

def G2_z_NLAST_false(k1, k2, Gx, Gy, Gz, 
                     U1_0, iz, const, 
                     is_linear_convolution, ):
    
    Ix, Iy = U1_0.shape[0], U1_0.shape[1]
    k1_z_shift, mesh_k1_x_k1_y_shift = Cal_kz(Ix, Iy, k1)
    k2_z_shift, mesh_k2_x_k2_y_shift = Cal_kz(Ix, Iy, k2)

    G_U1_z0_Squared_shift = fft2(Uz_AST(U1_0, k1, iz) ** 2)
    g_U1_0_Squared_shift = fft2(U1_0 ** 2)

    dG_Squared_shift = G_U1_z0_Squared_shift \
                       - g_U1_0_Squared_shift * math.e ** (k2_z_shift * iz * 1j)

    # %% denominator: dk_shift_Squared

    kiizQ_shift = k1 + k1_z_shift + Gz

    dk_shift_Squared = kiizQ_shift ** 2 - k2_z_shift ** 2

    # %% fractional

    fractional = dG_Squared_shift / dk_shift_Squared

    roll_x, roll_y = Cal_roll_xy(Gx, Gy,
                                 Ix, Iy, )

    fractional_Q = Roll_xy(fractional,
                           roll_x, roll_y,
                           is_linear_convolution, )

    # %% G2_z0_shift

    G2_z_shift = 2 * const * fractional_Q * math.e ** (Gz * iz * 1j)
    
    return G2_z_shift

#%%
# 提供 查找 边缘的，参数的 提示 or 帮助信息 msg

def Info_find_contours_SHG(g1_shift, k1_z_shift, k2_z_shift, Tz, mz, 
                           z0, size_PerPixel, deff_structure_length_expect, deff_structure_sheet_expect, 
                           is_print = 1, is_contours = 1, n_TzQ = 1, Gz_max_Enhance = 1, match_mode = 1):
    
    #%%
    # 描边
    
    if is_contours != 0:
        
        is_print and print("===== 描边 start =====")
        
        dk = 2 * np.max(np.abs(k1_z_shift)) - np.max(np.abs(k2_z_shift))
        # print(k2_z_shift[0,0])
        is_print and print("dk = {} / μm, {}".format(dk/size_PerPixel/1000, dk))
        lc = math.pi / abs(dk) * size_PerPixel * 1000 # Unit: um
        # print("相干长度 = {} μm".format(lc))
        # print("Tz_max = {} μm <= 畴宽 = {} μm ".format(lc*2, Tz))
        # print("畴宽_max = 相干长度 = {} μm <= 畴宽 = {} μm ".format(lc, Tz/2))
        if (type(Tz) != float and type(Tz) != int) or Tz <= 0: # 如果 传进来的 Tz 既不是 float 也不是 int，或者 Tz <= 0，则给它 安排上 2*lc
            Tz = 2*lc  # Unit: um
        
        Gz = 2 * math.pi * mz * size_PerPixel / (Tz / 1000) # Tz / 1000 即以 mm 为单位
        
        dkQ = dk + Gz
        lcQ = math.pi / abs(dkQ) * size_PerPixel # Unit: mm
        # print("相干长度_Q = {} mm".format(lcQ))
        TzQ = 2*lcQ
    
        #%%
        
        # print("k2_z_min = {} / μm, k1_z_min = {} / μm".format(np.min(np.abs(k2_z_shift))/size_PerPixel/1000, np.min(np.abs(k1_z_shift))/size_PerPixel/1000))
        # print(np.abs(k2_z_shift))
        if match_mode == 1:
            ix, iy, scale, energy_fraction = Find_energy_Dropto_fraction(g1_shift, 2/3, 0.1)
            Gz_max = np.abs(k2_z_shift[ix, 0]) - 2 * np.abs(k1_z_shift[ix, 0])
            is_print and print("scale = {}, energy_fraction = {}".format(scale, energy_fraction))
        else:
            Gz_max = np.min(np.abs(k2_z_shift)) - 2 * np.min(np.abs(k1_z_shift))
            
        Gz_max = Gz_max * Gz_max_Enhance
        is_print and print("Gz_max = {} / μm, {}".format(Gz_max/size_PerPixel/1000, Gz_max))
        Tz_min = 2 * math.pi * mz * size_PerPixel / (abs(Gz_max) / 1000) # 以使 lcQ >= lcQ_exp = (wc**2 + z0**2)**0.5 - z0
        # print("Tz_min = {} μm".format(Tz_min))
        
        dkQ_max = dk + Gz_max
        lcQ_min = math.pi / abs(dkQ_max) * size_PerPixel
        # print("lcQ_min = {} mm".format(lcQ_min))
        TzQ_min = 2*lcQ_min
        
        z0_min = TzQ_min
        
        #%%
        
        if is_contours != 2:
            is_print and print("===== 描边 1：若无额外要求 =====") # 波长定，Tz 定 (lcQ 定)，z0 不定
            
            is_print and print("z0_exp = {} mm".format(z0_min * n_TzQ))
            is_print and print("Tz_exp = {} μm".format(Tz_min))
        
        #%%
        
        if is_contours != 1:
            is_print and print("===== 描边 2：若希望 mod( 现 z0, TzQ_exp ) = 0 =====") # 波长定，z0 定，Tz 不定 (lcQ 不定)
            
            is_print and print("lcQ_min = {} mm".format(lcQ_min))
            TzQ_exp = z0 / (z0 // TzQ_min) # 满足 Tz_min <= · <= Tz_max = 原 Tz， 且 能使 z0 整除 TzQ 中，最小（最接近 TzQ_min）的 TzQ
            lcQ_exp = TzQ_exp/2
            is_print and print("lcQ_exp = {} mm".format(lcQ_exp))
            is_print and print("lcQ     = {} mm".format(lcQ))
            is_print and print("lc = {} μm".format(lc))
            # print("TzQ_min = {} mm".format(TzQ_min))
            # print("TzQ_exp = {} mm".format(TzQ_exp))
            # print("TzQ     = {} mm".format(TzQ))
            
            is_print and print("z0_min = {} mm # ==> 1.先调 z0 >= z0_min".format(z0_min)) # 先使 TzQ_exp 不遇分母 为零的错误，以 正确预测 lcQ_exp，以及后续的 Tz_exp
            z0_exp = TzQ_exp # 满足 >= TzQ_min， 且 能整除 TzQ_exp 中，最小的 z0
            # z0_exp = TzQ # 满足 >= TzQ_min， 且 能整除 TzQ_exp 中，最小的 z0
            is_print and print("z0_exp = {} mm # ==> 2.再调 z0 = z0_exp".format(z0_exp * n_TzQ))
            # print("z0_exp = {} * n mm # ==> 3.最后调 z0 = z0_exp".format(z0_exp))
            is_print and print("z0     = {} mm".format(z0))
        
            dkQ_exp = math.pi / lcQ_exp * size_PerPixel
            Gz_exp = dkQ_exp - dk
            Tz_exp = 2 * math.pi * mz * size_PerPixel / (abs(Gz_exp) / 1000) # 以使 lcQ >= lcQ_exp = (wc**2 + z0**2)**0.5 - z0
            is_print and print("Tz_min = {} μm".format(Tz_min))
            is_print and print("Tz_exp = {} μm # ==> 2.同时 Tz = Tz_exp".format(Tz_exp))
            is_print and print("Tz     = {} μm".format(Tz))
            is_print and print("Tz_max = {} μm".format(lc*2))
        
            domain_min = Tz_min / 2
            is_print and print("畴宽_min = {} μm".format(domain_min))
            domain_exp = Tz_exp / 2
            is_print and print("畴宽_exp = {} μm".format(domain_exp))
            is_print and print("畴宽     = {} μm".format(Tz/2))
            is_print and print("畴宽_max = {} μm".format(lc))
            
        #%%
        
    if is_contours == 1:
        z0_recommend = z0_min * n_TzQ
        Tz_recommend = Tz_min
    elif is_contours == 2:
        z0_recommend = z0_exp * n_TzQ
        Tz_recommend = Tz_exp
    else:
        z0_recommend = z0
        Tz_recommend = Tz
    
    if is_contours != 0:
        
        if deff_structure_length_expect <= z0_recommend + deff_structure_sheet_expect / 1000:
            deff_structure_length_expect = z0_recommend + deff_structure_sheet_expect / 1000
            is_print and print("deff_structure_length_expect = {} mm".format(deff_structure_length_expect))
            
        is_print and print("===== 描边 end =====")
    
    return z0_recommend, Tz_recommend, deff_structure_length_expect
        
#%%
# 提供 查找 边缘的，参数的 提示 or 帮助信息 msg
# 注：旧版本，已经过时，当时并 未想清楚。

def Info_find_contours(dk, Tz, mz, 
                       U_NonZero_size, w0, z0, size_PerPixel,
                       is_print = 1):
    
    #%%
    # 描边
    if is_print == 1: # 由于这个 函数不 return，只提供信息；因此 如果不 print，相当于什么都没做
        
        lc = math.pi / abs(dk) * size_PerPixel * 1000 # Unit: um
        # print("相干长度 = {} μm".format(lc))
        # print("Tz_max = {} μm <= 畴宽 = {} μm ".format(lc*2, Tz))
        # print("畴宽_max = 相干长度 = {} μm <= 畴宽 = {} μm ".format(lc, Tz/2))
        if (type(Tz) != float and type(Tz) != int) or Tz <= 0: # 如果 传进来的 Tz 既不是 float 也不是 int，或者 Tz <= 0，则给它 安排上 2*lc
            Tz = 2*lc  # Unit: um
        
        Gz = 2 * math.pi * mz * size_PerPixel / (Tz / 1000) # Tz / 1000 即以 mm 为单位

        dkQ = dk + Gz
        lcQ = math.pi / abs(dkQ) * size_PerPixel # Unit: mm
        # print("相干长度_Q = {} mm".format(lcQ))
        TzQ = 2*lcQ
    
        if (type(w0) == float or type(w0) == int) and w0 > 0: # 如果引入了 高斯限制
            wc = w0
        else:
            wc = U_NonZero_size / 2
    
        #%%
    
        print("===== 描边必需 1 =====") # 波长定，z0 定，Tz 不定 (lcQ 不定)
    
        lcQ_min = (wc**2 + z0**2)**0.5 - z0
        print("相干长度_Q_min = {} mm".format(lcQ_min))
        TzQ_min = 2*lcQ_min
        TzQ_exp = z0 / (z0 // TzQ_min) # 满足 Tz_min <= · <= Tz_max = 原 Tz， 且 能使 z0 整除 TzQ 中，最小（最接近 TzQ_min）的 TzQ
        lcQ_exp = TzQ_exp/2
        print("相干长度_Q_exp = {} mm".format(lcQ_exp))
        print("相干长度_Q     = {} mm".format(lcQ))
    
        dkQ_max_abs = math.pi / lcQ_min * size_PerPixel
        Gz_max = dkQ_max_abs - dk
        Tz_min = 2 * math.pi * mz * size_PerPixel / (abs(Gz_max) / 1000) # 以使 lcQ >= lcQ_exp = (wc**2 + z0**2)**0.5 - z0
        print("Tz_min = {} μm".format(Tz_min))
    
        dkQ_exp_abs = math.pi / lcQ_exp * size_PerPixel
        Gz_exp = dkQ_exp_abs - dk
        Tz_exp = 2 * math.pi * mz * size_PerPixel / (abs(Gz_exp) / 1000) # 以使 lcQ >= lcQ_exp = (wc**2 + z0**2)**0.5 - z0
        print("Tz_exp = {} μm # ==> 3.最后调 Tz = Tz_exp ".format(Tz_exp))
        print("Tz     = {} μm # ==> 1.先调 Tz < Tz_max".format(Tz))
        print("Tz_max = {} μm".format(lc*2))
    
        domain_min = Tz_min / 2
        print("畴宽_min = {} μm".format(domain_min))
        domain_exp = Tz_exp / 2
        print("畴宽_exp = {} μm".format(domain_exp))
        print("畴宽     = {} μm".format(Tz/2))
        print("畴宽_max = {} μm".format(lc))
        print("相干 长度 = {} μm".format(lc))
        
        #%%
            
        print("===== 描边必需 2 =====") # 波长定，Tz 定 (lcQ 定)，z0 不定
    
        z0_min = (wc**2 - lcQ**2)/(2*lcQ) # 以使 (wc**2 + z0**2)**0.5 - z0 = lcQ_exp <= lcQ
        # 这个玩意其实还得保证 >= TzQ_min，以先使 TzQ_exp 不遇分母 为零的错误，以 正确预测 lcQ_exp，以及后续的 Tz_exp
        print("z0_min = {} mm".format(z0_min))
        z0_exp = z0_min - np.mod(z0_min, TzQ) + TzQ # 满足  >= TzQ_min， 且 能整除 TzQ_exp 中，最小的 z0
        print("z0_exp = {} mm".format(z0_exp))
        print("z0     = {} mm # ==> 2.接着调 z0 = z0_exp".format(z0))
        
        print("===== 描边 end =====")