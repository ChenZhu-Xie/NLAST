# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 20:23:31 2022

@author: Xcz
"""

import math
import numpy as np

#%%

def help_find_contours(dk, Tz, mz, 
                       U1_0_NonZero_size, w0, z0, size_PerPixel,
                       is_print = 1):
    
    # 提供 查找 边缘的，参数的 暗示信息 msg
    
    #%%
    # 描边
    if is_print == 1:
        
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
    
        if (type(w0) == float or type(w0) == int) and w0 > 0: # 如果引入了 高斯限制
            wc = w0
        else:
            wc = U1_0_NonZero_size / 2
    
        #%%
    
        print("===== 描边必需 1 =====") # 波长定，z0 定，Tz 不定 (lcQ 不定)
    
        lcQ_min = (wc**2 + z0**2)**0.5 - z0
        print("相干长度_Q_min = {} mm".format(lcQ_min))
        lcQ_exp = z0 / (z0 // lcQ_min) # 满足 lc_min <= · <= lc_max = 原 lc， 且 能使 z0 整除 lcQ 中，最小的 lcQ
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
        print("Tz     = {} μm # ==> 1.先调 Tz <= Tz_max".format(Tz))
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
    
        z0_min = (wc**2 - lcQ**2)/(2*lcQ) # 以使 (wc**2 + z0**2)**0.5 - z0 = lcQ_exp <= 
        print("z0_min = {} mm".format(z0_min))
        z0_exp = z0_min - np.mod(z0_min, lcQ) + lcQ # 满足 >= z0_min， 且 能整除 lcQ 中，最小的 z0
        print("z0_exp = {} mm".format(z0_exp))
        print("z0     = {} mm # ==> 2.接着调 z0 = z0_exp".format(z0))
        
        print("===== 描边 end =====")
        