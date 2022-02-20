# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

#%%

import os
import cv2
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import math
# import copy
from PIL import Image
Image.MAX_IMAGE_PIXELS = 10E10 #Image 的 默认参数 无法处理那么大的图片
from n_dispersion import LN_n, KTP_n
import threading
# import scipy
from scipy.io import loadmat, savemat
import time
from fun_plot import plot_1d, plot_2d, plot_3d_XYZ, plot_3d_XYz
from fun_pump import pump_LG

#%%

def NLA(U1_txt_name = "", 
        file_full_name = "Grating.png", 
        phase_only = 0, 
        #%%
        is_LG = 0, is_Gauss = 0, is_OAM = 0, 
        l = 0, p = 0, 
        theta_x = 0, theta_y = 0, 
        is_H_l = 0, is_H_theta = 0, 
        #%%
        U1_0_NonZero_size = 1, w0 = 0.3,
        z0 = 1, 
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
        vmax = 1, vmin = 0):
    
    # #%%
    # U1_txt_name = ""
    # file_full_name = "lena.png"
    # #%%
    # phase_only = 0
    # is_LG, is_Gauss, is_OAM = 0, 1, 1
    # l, p = 1, 0
    # theta_x, theta_y = 1, 0
    # is_H_l, is_H_theta = 0, 0
    # # 正空间：右，下 = +, +
    # # 倒空间：左, 上 = +, +
    # # 朝着 x, y 轴 分别偏离 θ_1_x, θ_1_y 度
    # #%%
    # U1_0_NonZero_size = 1 # Unit: mm 不包含边框，图片 的 实际尺寸
    # w0 = 0.5 # Unit: mm 束腰（z = 0 处）
    # z0 = 10 # Unit: mm 传播距离
    # # size_modulate = 1e-3 # Unit: mm χ2 调制区域 的 横向尺寸，即 公式中的 d
    # #%%
    # lam1 = 1.064 # Unit: um 基波波长
    # is_air, T = 0, 25 # is_air = 0, 1, 2 分别表示 LN, 空气, KTP；T 表示 温度
    # #%%
    # deff = 30 # pm / V
    # Tx, Ty, Tz = 10, 10, "2*lc" # Unit: um
    # mx, my, mz = 0, 0, 0
    # # 倒空间：右, 下 = +, +
    # is_linear_convolution = 0 # 0 代表 循环卷积，1 代表 线性卷积
    # #%%
    # is_save = 0
    # is_save_txt = 0
    # dpi = 100
    # #%%
    # cmap_2d='viridis'
    # # cmap_2d.set_under('black')
    # # cmap_2d.set_over('red')
    # #%%
    # ticks_num = 6 # 不包含 原点的 刻度数，也就是 区间数（植数问题）
    # is_contourf = 0
    # is_title_on, is_axes_on = 1, 1
    # is_mm, is_propagation = 1, 0
    # #%%
    # fontsize = 9
    # font = {'family': 'serif',
    #         'style': 'normal', # 'normal', 'italic', 'oblique'
    #         'weight': 'normal',
    #         'color': 'black', # 'black','gray','darkred'
    #         }
    # #%%
    # is_self_colorbar, is_colorbar_on = 0, 1 # vmax 与 vmin 是否以 自己的 U 的 最大值 最小值 为 相应的值；是，则覆盖设定；否的话，需要自己设定。
    # vmax, vmin = 1, 0

    if (type(U1_txt_name) != str) or U1_txt_name == "":
        #%%
        # 导入 方形，以及 加边框 的 图片
    
        file_name = os.path.splitext(file_full_name)[0]
        file_name_extension = os.path.splitext(file_full_name)[1]
    
        location = os.path.dirname(os.path.abspath(__file__))
        file_squared_address = location + "\\" + "1." + file_name + "_squared" + file_name_extension
        file_squared_bordered_address = location + "\\" + "2." + file_name + "_squared" + "_bordered" + file_name_extension
    
        img_squared = cv2.imdecode(np.fromfile(file_squared_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
        img_squared_bordered = cv2.imdecode(np.fromfile(file_squared_bordered_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
    
        size_fig = img_squared_bordered.shape[0] / dpi
    
        #%%
        # 线性 角谱理论 - 基波 begin
    
        size_PerPixel = U1_0_NonZero_size / img_squared.shape[0] # Unit: mm / 个 每个 像素点 的 尺寸，相当于 △x = △y = △z
        I1_x, I1_y = img_squared_bordered.shape[0], img_squared_bordered.shape[1]
        # U1_size = I1_x * size_PerPixel # Unit: mm 包含 边框 后，图片 的 实际尺寸
        # print("U1_size = U2_size = {} mm".format(U1_size))
        # print("U1_size = {} mm".format(U1_size))
        # print("%f mm" %(U1_size))
    
        #%%
        # U1_0 = U(x, y, 0) = img_squared_bordered
    
        # I_img_squared_bordered = np.empty([I1_x, I1_y], dtype=np.uint64)
        # I_img_squared_bordered = copy.deepcopy(img_squared_bordered) # 但这 深拷贝 也不行，因为把 最底层的 数据类型 uint8 也拷贝了，我 tm 无语，不如直接 astype 算了
        if phase_only == 1:
            U1_0 = np.power(math.e, (img_squared_bordered.astype(np.complex128()) / 255 * 2 * math.pi - math.pi) * 1j) # 变成相位图
        else:
            U1_0 = img_squared_bordered.astype(np.complex128)
        
        #%%
        # 预处理 输入场
        
        if is_air_pump == 1:
            n1 = 1
        elif is_air_pump == 0:
            n1 = LN_n(lam1, T, "e")
        else:
            n1 = KTP_n(lam1, T, "e")

        k1 = 2 * math.pi * size_PerPixel / (lam1 / 1000 / n1) # lam / 1000 即以 mm 为单位
        
        U1_0 = pump_LG(file_full_name, 
                       I1_x, I1_y, size_PerPixel, 
                       U1_0, w0, k1, 0, 
                       is_LG, is_Gauss, is_OAM, 
                       l, p, 
                       theta_x, theta_y, 
                       is_H_l, is_H_theta, 
                       is_save, is_save_txt, dpi, 
                       cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                       fontsize, font, 
                       1, is_colorbar_on, vmax, vmin) 
        
    else:
    
        #%%
        # 导入 方形，以及 加边框 的 图片
        
        U1_txt_full_name = U1_txt_name + (is_save_txt and ".txt" or ".mat")
        U1_txt_short_name = U1_txt_name.replace('6. AST - ', '')
        
        file_name = os.path.splitext(file_full_name)[0]
        file_name_extension = os.path.splitext(file_full_name)[1]

        location = os.path.dirname(os.path.abspath(__file__))
        file_squared_address = location + "\\" + "1." + file_name + "_squared" + file_name_extension

        img_squared = cv2.imdecode(np.fromfile(file_squared_address, dtype=np.uint8), 0) # 按 相对路径 + 灰度图 读取图片
        U1_0 = np.loadtxt(U1_txt_full_name, dtype=np.complex128()) if is_save_txt == 1 else loadmat(U1_txt_full_name)['U'] # 加载 复振幅场

        size_fig = U1_0.shape[0] / dpi

        #%%
        # 线性 角谱理论 - 基波 begin

        size_PerPixel = U1_0_NonZero_size / img_squared.shape[0] # Unit: mm / 个 每个 像素点 的 尺寸，相当于 △x = △y = △z
        I1_x, I1_y = U1_0.shape[0], U1_0.shape[1]
        # U1_size = I1_x * size_PerPixel # Unit: mm 包含 边框 后，图片 的 实际尺寸
        # print("U1_size = U2_size = {} mm".format(U1_size))
        # print("U1_size = {} mm".format(U1_size))
        # print("%f mm" %(U1_size))

    #%%

    if is_air == 1:
        n1 = 1
    elif is_air == 0:
        n1 = LN_n(lam1, T, "e")
    else:
        n1 = KTP_n(lam1, T, "e")

    k1 = 2 * math.pi * size_PerPixel / (lam1 / 1000 / n1) # lam / 1000 即以 mm 为单位

    #%%
    # 线性 角谱理论 - 基波 begin
    g1 = np.fft.fft2(U1_0)
    g1_shift = np.fft.fftshift(g1)
    
    z1_0 = z0
    i1_z0 = z1_0 / size_PerPixel

    n1_x, n1_y = np.meshgrid([i for i in range(I1_x)], [j for j in range(I1_y)])
    Mesh_n1_x_n1_y = np.dstack((n1_x, n1_y))
    Mesh_n1_x_n1_y_shift = Mesh_n1_x_n1_y - (I1_x // 2, I1_y // 2)
    Mesh_k1_x_k1_y_shift = np.dstack((2 * math.pi * Mesh_n1_x_n1_y_shift[:, :, 0] / I1_x, 2 * math.pi * Mesh_n1_x_n1_y_shift[:, :, 1] / I1_y))

    k1_z_shift = (k1**2 - np.square(Mesh_k1_x_k1_y_shift[:, :, 0]) - np.square(Mesh_k1_x_k1_y_shift[:, :, 1]) + 0j )**0.5
    H1_z0_shift = np.power(math.e, k1_z_shift * i1_z0 * 1j)
    
    G1_z0_shift = g1_shift * H1_z0_shift
    G1_z0 = np.fft.ifftshift(G1_z0_shift)
    U1_z0 = np.fft.ifft2(G1_z0)
    
    #%%

    def Sinc(x):
        return np.nan_to_num( np.sin(x) / x ) + np.isnan( np.sin(x) / x ).astype(np.int8)

    def Cosc(x):
        return np.nan_to_num( (np.cos(x) - 1) / x )
    # return np.nan_to_num( (np.cos(x) - 1) / x ) * ( 1 - np.isnan( (np.cos(x) - 1) / x ).astype(np.int8) ) 不够聪明

    # 定义 对于 kz 的 类似 Sinc 的 函数：( e^ikz - 1 ) / kz
    def Eikz(x):
        return Cosc(x) + 1j * Sinc(x)

    # 定义 m 级次 的 倒格波 系数 Cm
    def C_m(m):
        if m == 0:
            return 1
        else:
            return Sinc(math.pi*m / 2) - Sinc(math.pi*m)
    
    #%%
    # 非线性 角谱理论 - 无近似 begin

    I2_x, I2_y = U1_0.shape[0], U1_0.shape[1]

    #%%
    # const

    lam2 = lam1 / 2

    if is_air == 1:
        n2 = 1
    elif is_air == 0:
        n2 = LN_n(lam2, T, "e")
    else:
        n2 = KTP_n(lam2, T, "e")

    k2 = 2 * math.pi * size_PerPixel / (lam2 / 1000 / n2) # lam / 1000 即以 mm 为单位
    deff = C_m(mx) * C_m(my) * C_m(mz) * deff * 1e-12 # pm / V 转换成 m / V
    const = (k2 / size_PerPixel / n2)**2 * deff

    #%%
    # G2_z0_shift

    n2_x, n2_y = np.meshgrid([i for i in range(I2_x)], [j for j in range(I2_y)])
    Mesh_n2_x_n2_y = np.dstack((n2_x, n2_y))
    Mesh_n2_x_n2_y_shift = Mesh_n2_x_n2_y - (I2_x // 2, I2_y // 2)

    #%%
    # 引入 倒格矢，对 k2 的 方向 进行调整，其实就是对 k2 的 k2x, k2y, k2z 网格的 中心频率 从 (0, 0, k2z) 移到 (Gx, Gy, k2z + Gz)

    dk = 2*k1 - k2 # Unit: 1 / mm
    lc = math.pi / abs(dk) * size_PerPixel # Unit: mm
    print("相干长度 = {} μm".format(lc * 1000))
    if (type(Tz) != float and type(Tz) != int) or Tz <= 0: # 如果 传进来的 Tz 既不是 float 也不是 int，或者 Tz <= 0，则给它 安排上 2*lc
        Tz = 2*lc * 1000  # Unit: um

    Gx = 2 * math.pi * mx * size_PerPixel / (Tx / 1000) # Tz / 1000 即以 mm 为单位
    Gy = 2 * math.pi * my * size_PerPixel / (Ty / 1000) # Tz / 1000 即以 mm 为单位
    Gz = 2 * math.pi * mz * size_PerPixel / (Tz / 1000) # Tz / 1000 即以 mm 为单位

    Mesh_k2_x_k2_y_shift = np.dstack((2 * math.pi * Mesh_n2_x_n2_y_shift[:, :, 0] / I2_x - Gx, 2 * math.pi * Mesh_n2_x_n2_y_shift[:, :, 1] / I2_y - Gy))
    k2_z_shift = (k2**2 - np.square(Mesh_k2_x_k2_y_shift[:, :, 0]) - np.square(Mesh_k2_x_k2_y_shift[:, :, 1]) + 0j )**0.5

    z2_0 = z0
    i2_z0 = z2_0 / size_PerPixel

    integrate_z0_shift = np.zeros( (I2_x,I2_y) ,dtype=np.complex128() )

    g1_shift_rotate_180 = g1_shift
    # ( 先转置，后上下翻转 )**2 = ( 先上下翻转，后转置 )**2 = 先上下翻转，后左右翻转 = 先左右翻转，后上下翻转 = 绕垂直纸面朝外的 z 轴，旋转 180 度
    g1_shift_rotate_180 = g1_shift_rotate_180.transpose((1,0))
    g1_shift_rotate_180 = g1_shift_rotate_180[::-1]
    g1_shift_rotate_180 = g1_shift_rotate_180.transpose((1,0))
    g1_shift_rotate_180 = g1_shift_rotate_180[::-1]

    #%%
    # 多线程
    
    global thread_th, for_th # 好怪，作为 被封装 和 被引用的 函数，还得在 这一层 声明 全局变量，光是在 内层 子线程 里声明 的话，没用。
    thread_th = 0 # 生产出的 第几个 / 一共几个 线程，全局
    for_th = 0 # 正在计算到的 第几个 for 循环的序数，全局（非顺序的情况下，这个的含义只是计数，即一共计算了几个 序数 了）
    con = threading.Condition() # 锁不必定义为全局变量

    class Producer(threading.Thread):
        """线程 生产者"""
        def __init__(self, threads_num, fors_num):
            self.threads_num = threads_num
            self.fors_num = fors_num
            self.for_th = 0 # 生产出的 第几个 for_th，这个不必定义为全局变量
            super().__init__()
        
        def run(self):
            global thread_th
            
            con.acquire()
            while True:
                
                if self.for_th >= self.fors_num and for_th == self.fors_num: # 退出生产线程 的 条件：p 线程 完成 其母线程功能，且 最后一个 t 线程 完成其子线程功能
                    break
                else:
                    if thread_th >= self.threads_num or self.for_th == self.fors_num : # 暂停生产线程 的 条件： 运行线程数 达到 设定，或 p 线程 完成 其母线程功能
                        con.notify()
                        con.wait()
                    else:
                        # print(self.for_th)
                        t = Customer('thread:%s' % thread_th, self.for_th)
                        t.setDaemon(True)
                        t.start()
                        
                        thread_th += 1
                        # print('已生产了 共 {} 个 线程'.format(thread_th))
                        self.for_th += 1
                        # print('已算到了 第 {} 个 for_th'.format(self.for_th))
                        # time.sleep(1)
            con.release()

    class Customer(threading.Thread):
        """线程 消费者"""
        def __init__(self, name, for_th):
            self.thread_name = name
            self.for_th = for_th
            super().__init__()
            
        def run(self):
            global thread_th, for_th
            """----- 你 需累积的 全局变量，替换 最末一个 g2_z_plus_dz_shift -----"""
                
            """----- your code begin 1 -----"""
            for n2_y in range(I2_y):
                dk_x_shift = Mesh_k2_x_k2_y_shift[self.for_th, n2_y, 0] - Mesh_k1_x_k1_y_shift[:, :, 0] - Gy
                # 其实 Mesh_k2_x_k2_y_shift[:, :, 0]、Mesh_n2_x_n2_y_shift[:, :, 0]、Mesh_n2_x_n2_y[:, :, 0]、 n2_x = self.for_th 均只和 y，即 [:, :] 中的 第 2 个数字 有关，
                # 只由 列 y、ky 决定，与行 即 x、kx 无关
                # 而 Gy 得与 列 y、ky 发生关系,
                # 所以是 - Gy 而不是 Gx
                # 并且这里的 dk_x_shift 应写为 dk_y_shift
                dk_y_shift = Mesh_k2_x_k2_y_shift[self.for_th, n2_y, 1] - Mesh_k1_x_k1_y_shift[:, :, 1] - Gx
                k1_z_shift_dk_x_dk_y = (k1**2 - np.square(dk_x_shift) - np.square(dk_y_shift) + 0j )**0.5
                
                dk_z_shift = k1_z_shift + k1_z_shift_dk_x_dk_y - k2_z_shift[self.for_th, n2_y]
                dk_z_Q_shift = dk_z_shift + Gz

                g1_shift_dk_x_dk_y = g1_shift_rotate_180
                # g1_shift_dk_x_dk_y = g1_shift_rotate_180_shift
                roll_x = np.floor( self.for_th + I2_x//2 - (I2_x - 1) - Gy / (2 * math.pi) * I2_y ).astype(np.int64)
                roll_y = np.floor( n2_y + I2_y//2 - (I2_y - 1) - Gx / (2 * math.pi) * I2_x ).astype(np.int64)
                # 之后要平移列，而 Gx 才与列有关...
                # # 往下（行） 线性平移 roll_x 像素：先对 将要平移出框的 roll_x 个行 取零，再 循环平移；先后 顺序可反，但 取零的行的 上下 也得 同时反
                g1_shift_dk_x_dk_y = np.roll(g1_shift_dk_x_dk_y, roll_x, axis=0)
                if is_linear_convolution == 1:
                    if roll_x < 0: # n2_x + Gx / (2 * math.pi) * I2_x 小于 (I2_x - 1) - I2_x//2 时，后半部分 取零，只剩 前半部分
                        g1_shift_dk_x_dk_y[roll_x:, :] = 0
                    elif roll_x > 0: # n2_x + Gx / (2 * math.pi) * I2_x 大于 (I2_x - 1) - I2_x//2 时，前半部分 取零，只剩 后半部分
                        g1_shift_dk_x_dk_y[:roll_x, :] = 0
                # # 往右（列） 线性平移 roll_y 像素：先对 将要平移出框的 roll_y 个列 取零，再 循环平移；先后 顺序可反，但 取零的列的 左右 也得 同时反
                g1_shift_dk_x_dk_y = np.roll(g1_shift_dk_x_dk_y, roll_y, axis=1)
                if is_linear_convolution == 1:
                    if roll_y < 0:
                        g1_shift_dk_x_dk_y[:, roll_y:] = 0
                    elif roll_y > 0:
                        g1_shift_dk_x_dk_y[:, :roll_y] = 0
                
                integrate_z0_shift[self.for_th, n2_y] = np.sum(g1_shift * g1_shift_dk_x_dk_y * Eikz(dk_z_Q_shift * i2_z0) * i2_z0 * size_PerPixel * (2 / (dk_z_Q_shift / k2_z_shift[self.for_th, n2_y] + 2)))
            """----- your code end 1 -----"""
            
            con.acquire() # 上锁
            
            if is_ordered == 1:
                
                while True:
                    if for_th == self.for_th:
                        # print(self.for_th)
                        """----- your code begin 2 -----"""
                        
                        """----- your code end 2 -----"""
                        for_th += 1
                        break
                    else:
                        con.notify()
                        con.wait() # 但只有当 for_th 不等于 self.for_th， 才等待
            else:
                
                # print(self.for_th)
                """----- your code begin 2 -----"""
                
                """----- your code end 2 -----"""
                for_th += 1
            
            thread_th -= 1 # 在解锁之前 减少 1 个线程数量，以便 p 线程 收到消息后，生产 1 个 线程出来
            con.notify() # 无论如何 都得通知一下 其他线程，让其别 wait() 了
            con.release() # 解锁

    """----- your code begin 0 -----"""
    is_ordered = 0 # for_th 是否 按顺序执行
    threads_num = 10 # 需要开启 多少个线程 持续计算
    fors_num = I2_x # 需要计算 for 循环中 多少个 序数
    """----- your code end 0 -----"""

    tick_start = time.time()

    p = Producer(threads_num, fors_num)
    p.setDaemon(True)
    p.start()
    p.join() # 添加join使 p 线程执行完

    print("----- consume time: {} s -----".format(time.time() - tick_start))

    # integrate_z0_shift = integrate_z0_shift * (2 * math.pi / I2_x / size_PerPixel) * (2 * math.pi / I2_y / size_PerPixel)
    g2_z0_shift = const * integrate_z0_shift / k2_z_shift * size_PerPixel

    G2_z0_shift = g2_z0_shift * np.power(math.e, k2_z_shift * i2_z0 * 1j)

    G2_z0_shift_amp = np.abs(G2_z0_shift)
    # print(np.max(G2_z0_shift_amp))
    G2_z0_shift_phase = np.angle(G2_z0_shift)
    if is_save == 1:
        if not os.path.isdir("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_shift"):
            os.makedirs("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_shift")

    #%%
    #绘图：G2_z0_shift_amp

    G2_z0_shift_amp_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_shift" + "\\" + "5.1. NLA - " + "G2_" + str(float('%.2g' % z0)) + "mm" + "_shift_amp" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, 0, 
            G2_z0_shift_amp, G2_z0_shift_amp_address, "G2_" + str(float('%.2g' % z0)) + "mm" + "_shift_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, vmax, vmin)

    #%%
    #绘图：G2_z0_shift_phase

    G2_z0_shift_phase_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_shift" + "\\" + "5.2. NLA - " + "G2_" + str(float('%.2g' % z0)) + "mm" + "_shift_phase" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, 0, 
            G2_z0_shift_phase, G2_z0_shift_phase_address, "G2_" + str(float('%.2g' % z0)) + "mm" + "_shift_phase", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, vmax, vmin)
    
    #%%
    # 储存 G2_z0_shift 到 txt 文件

    if is_save == 1:
        G2_z0_shift_full_name = "5. NLA - G2_" + str(float('%.2g' % z0)) + "mm" + "_shift" + (is_save_txt and ".txt" or ".mat")
        G2_z0_shift_txt_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_shift" + "\\" + G2_z0_shift_full_name
        np.savetxt(G2_z0_shift_txt_address, G2_z0_shift) if is_save_txt else savemat(G2_z0_shift_txt_address, {"G":G2_z0_shift})
        
    #%%
    # G2_z0 = G2_z0(k1_x, k1_y) → IFFT2 → U2(x0, y0, z0) = U2_z0

    G2_z0 = np.fft.ifftshift(G2_z0_shift)
    U2_z0 = np.fft.ifft2(G2_z0)
    # 2 维 坐标空间 中的 复标量场，是 i2_x0, i2_y0 的函数
    # U2_z0 = U2_z0 * scale_down_factor # 归一化
    U2_z0_amp = np.abs(U2_z0)
    # print(np.max(U2_z0_amp))
    U2_z0_phase = np.angle(U2_z0)

    print("NLA - U2_{}mm.total_energy = {}".format(z0, np.sum(U2_z0_amp**2)))

    if is_save == 1:
        if not os.path.isdir("6. U2_" + str(float('%.2g' % z0)) + "mm"):
            os.makedirs("6. U2_" + str(float('%.2g' % z0)) + "mm")

    #%%
    #绘图：U2_z0_amp

    U2_z0_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_amp" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, 0, 
            U2_z0_amp, U2_z0_amp_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, vmax, vmin)

    #%%
    #绘图：U2_z0_phase

    U2_z0_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_phase" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, 0, 
            U2_z0_phase, U2_z0_phase_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_phase", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, vmax, vmin)
    
    #%%
    # 储存 U2_z0 到 txt 文件

    U2_z0_full_name = "6. NLA - U2_" + str(float('%.2g' % z0)) + "mm" + (is_save_txt and ".txt" or ".mat")
    if is_save == 1:
        U2_z0_txt_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "\\" + U2_z0_full_name
        np.savetxt(U2_z0_txt_address, U2_z0) if is_save_txt else savemat(U2_z0_txt_address, {"U":U2_z0})

        #%%
        #再次绘图：U2_z0_amp
    
        U2_z0_amp_address = location + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_amp" + file_name_extension
    
        plot_2d(I2_x, I2_y, size_PerPixel, 0, 
                U2_z0_amp, U2_z0_amp_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_amp", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, vmax, vmin)
    
        #再次绘图：U2_z0_phase
    
        U2_z0_phase_address = location + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_phase" + file_name_extension
    
        plot_2d(I2_x, I2_y, size_PerPixel, 0, 
                U2_z0_phase, U2_z0_phase_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_phase", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                1, is_colorbar_on, vmax, vmin)

    #%%
    # 储存 U2_z0 到 txt 文件

    # if is_save == 1:
    np.savetxt(U2_z0_full_name, U2_z0) if is_save_txt else savemat(U2_z0_full_name, {"U":U2_z0})
    