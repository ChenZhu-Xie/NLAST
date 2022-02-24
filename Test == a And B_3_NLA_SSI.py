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
# import matplotlib.pyplot as plt
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
from a_Image_Add_Black_border import Image_Add_Black_border

#%%
U1_txt_name = ""
file_full_name = "lena.png"
border_percentage = 0.3 # 边框 占图片的 百分比，也即 图片 放大系数
#%%
phase_only = 0
is_LG, is_Gauss, is_OAM = 0, 0, 0
l, p = 0, 0
theta_x, theta_y = -0.5, 0
is_H_l, is_H_theta = 0, 0
# 正空间：右，下 = +, +
# 倒空间：左, 上 = +, +
# 朝着 x, y 轴 分别偏离 θ_1_x, θ_1_y 度
#%%
U1_0_NonZero_size = 0.5 # Unit: mm 不包含边框，图片 的 实际尺寸
w0 = 0 # Unit: mm 束腰（z = 0 处）
L0_Crystal_expect = 15 # Unit: mm 晶体长度
z0_structure_frontface_expect = 0.5 # Unit: mm 结构 前端面，距离 晶体 前端面 的 距离
deff_structure_length_expect = 1 # Unit: mm 调制区域 z 向长度（类似 z）
deff_structure_sheet_expect = 1.8 # Unit: μm z 向 切片厚度
sheets_stored_num = 10 # 储存片数 （不包含 最末：因为 最末，作为结果 已经单独 呈现了）；每一步 储存的 实际上不是 g_z，而是 g_z+dz
z0_section_1f_expect = 0 # Unit: mm z 向 需要展示的截面 1 距离晶体前端面 的 距离
z0_section_2f_expect = 0 # Unit: mm z 向 需要展示的截面 2 距离晶体后端面 的 距离
X, Y = 0, 0 # Unit: mm 切片 中心点 平移 矢量（逆着 z 正向看去，矩阵的行 x 是向下的，矩阵的列 y 是向右的；这里的 Y 是 矩阵的行 x 的反向，这里的 X 是矩阵的列 y 的正向）
# X 增加，则 从 G2_z_shift 中 读取的 列 向右移，也就是 xz 面向 列 增加的方向（G2_z_shift 的 右侧）移动
# Y 增加，则 从 G2_z_shift 中 读取的 行 向上移，也就是 yz 面向 行 减小的方向（G2_z_shift 的 上侧）移动
# size_modulate = 1e-3 # Unit: mm χ2 调制区域 的 横向尺寸，即 公式中的 d
is_bulk = 1 # 是否 不读取 结构，1 为 不读取，即 均一晶体；0 为 读取结构
is_no_backgroud = 0 # 1 -1 调制，改为 0 -2 调制
is_stored = 0 # 如果要储存中间结果，则不能多线程，只能单线程
is_show_structure_face = 0 # 如果要显示 结构 前后端面 的 场分布，就打开这个
is_energy_evolution_on = 1 # 储存 能量 随 z 演化 的 曲线
#%%
lam1 = 0.8 # Unit: um 基波波长
is_air_pump, is_air, T = 0, 0, 25 # is_air = 0, 1, 2 分别表示 LN, 空气, KTP；T 表示 温度
#%%
deff = 30 # pm / V
Tx, Ty, Tz = 6.633, 20, 18.437 # Unit: um "2*lc"，测试： 0 度 - 20.155, 20, 17.885 、 -2 度 ： 6.633, 20, 18.437 、-3 度 ： 4.968, 20, 19.219
mx, my, mz = -1, 0, 1
# 倒空间：右, 下 = +, +
#%%
is_save = 0
is_save_txt = 0
dpi = 100
#%%
color_1d='b'
cmap_2d='viridis'
# cmap_2d.set_under('black')
# cmap_2d.set_over('red')
cmap_3d='rainbow' # 3D 图片 colormap # cm.coolwarm, cm.viridis, viridis, cmap.to_rgba(i), 'rainbow', 'winter', 'Greens', 'b', 'c', 'm'
# cmap_3d.set_under('black')
# cmap_3d.set_over('red')
elev, azim = 10, -65 # 3D camera 相机视角：前一个为正 即 俯视，后一个为负 = 绕 z 轴逆时针（右手 螺旋法则，z 轴 与 拇指 均向上）
alpha = 2
#%%
ticks_num = 6 # 不包含 原点的 刻度数，也就是 区间数（植数问题）
is_contourf = 0
is_title_on, is_axes_on = 1, 1
is_mm, is_propagation = 1, 0
#%%
fontsize = 9
font = {'family': 'serif',
        'style': 'normal', # 'normal', 'italic', 'oblique'
        'weight': 'normal',
        'color': 'black', # 'black','gray','darkred'
        'size': fontsize,
        }
#%%
is_self_colorbar, is_colorbar_on = 0, 1 # vmax 与 vmin 是否以 自己的 U 的 最大值 最小值 为 相应的值；是，则覆盖设定；否的话，需要自己设定。
vmax, vmin = 1, 0

if (type(U1_txt_name) != str) or U1_txt_name == "":
    #%%
    # 预处理 导入图片 为方形，并加边框

    Image_Add_Black_border(file_full_name, border_percentage)
    
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
    # 线性 角谱理论 - 基波^2 begin
    
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
# 非线性 角谱理论 - SSI begin

I2_x, I2_y = U1_0.shape[0], U1_0.shape[1]

#%%
# 引入 倒格矢，对 k2 的 方向 进行调整，其实就是对 k2 的 k2x, k2y, k2z 网格的 中心频率 从 (0, 0, k2z) 移到 (Gx, Gy, k2z + Gz)

lam2 = lam1 / 2

if is_air == 1:
    n2 = 1
elif is_air == 0:
    n2 = LN_n(lam2, T, "e")
else:
    n2 = KTP_n(lam2, T, "e")

k2 = 2 * math.pi * size_PerPixel / (lam2 / 1000 / n2) # lam / 1000 即以 mm 为单位

dk = 2*k1 - k2 # Unit: 1 / mm
lc = math.pi / abs(dk) * size_PerPixel # Unit: mm
print("相干长度 = {} μm".format(lc * 1000))
if (type(Tz) != float and type(Tz) != int) or Tz <= 0: # 如果 传进来的 Tz 既不是 float 也不是 int，或者 Tz <= 0，则给它 安排上 2*lc
    Tz = 2*lc * 1000  # Unit: um

Gx = 2 * math.pi * mx * size_PerPixel / (Tx / 1000) # Tz / 1000 即以 mm 为单位
Gy = 2 * math.pi * my * size_PerPixel / (Ty / 1000) # Tz / 1000 即以 mm 为单位
Gz = 2 * math.pi * mz * size_PerPixel / (Tz / 1000) # Tz / 1000 即以 mm 为单位

#%%
# 定义 调制区域切片厚度 的 纵向实际像素、调制区域切片厚度 的 实际纵向尺寸

if mz != 0: # 如过你想 让结构 提供 z 向倒格矢
    if deff_structure_sheet_expect >= 0.1 * Tz or deff_structure_sheet_expect <= 0 or (type(deff_structure_sheet_expect) != float and type(deff_structure_sheet_expect) != int): # 则 deff_structure_sheet_expect 不能超过 0.1 * Tz（以保持 良好的 占空比）
        deff_structure_sheet_expect = 0.1 * Tz # Unit: μm
else:
    if deff_structure_sheet_expect >= 0.01 * deff_structure_length_expect * 1000 or deff_structure_sheet_expect <= 0 or (type(deff_structure_sheet_expect) != float and type(deff_structure_sheet_expect) != int): # 则 deff_structure_sheet_expect 不能超过 0.01 * deff_structure_length_expect（以保持 良好的 精度）
        deff_structure_sheet_expect = 0.01 * deff_structure_length_expect * 1000 # Unit: μm
        
diz = deff_structure_sheet_expect / 1000 / size_PerPixel # Unit: mm
# diz = int( deff_structure_sheet_expect / 1000 / size_PerPixel )
deff_structure_sheet = diz * size_PerPixel # Unit: mm 调制区域切片厚度 的 实际纵向尺寸
# print("deff_structure_sheet = {} mm".format(deff_structure_sheet))

#%%
# 定义 结构前端面 距离 晶体前端面 的 纵向实际像素、结构前端面 距离 晶体前端面 的 实际纵向尺寸

Iz_frontface = z0_structure_frontface_expect / size_PerPixel

sheets_num_frontface = int(Iz_frontface // diz)
Iz_frontface = sheets_num_frontface * diz
z0_structure_frontface = Iz_frontface * size_PerPixel
print("z0_structure_frontface = {} mm".format(z0_structure_frontface))

#%%
# 定义 调制区域 的 纵向实际像素、调制区域 的 实际纵向尺寸

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
print("deff_structure_length = {} mm".format(deff_structure_length))

#%%
# 定义 结构后端面 距离 晶体后端面 的 纵向实际像素、结构后端面 距离 晶体后端面 的 实际纵向尺寸

# Iz_endface = (L0_Crystal_expect - z0_structure_frontface - deff_structure_length) / size_PerPixel

# sheets_num_endface = int(Iz_endface // diz)
# Iz_endface = sheets_num_endface * diz
# z0_structure_endface = Iz_endface * size_PerPixel
# print("z0_structure_endface = {} mm".format(z0_structure_endface))

#%%
# 定义 晶体 的 纵向实际像素、晶体 的 实际纵向尺寸

# # Iz = (z0_structure_frontface + deff_structure_length + z0_structure_endface) / size_PerPixel
# Iz = Iz_frontface + Iz_structure + Iz_endface

# # sheets_num = int(Iz // diz)
# sheets_num = sheets_num_frontface + sheets_num_structure + sheets_num_endface
# Iz = sheets_num * diz
# z0 = L0_Crystal = Iz * size_PerPixel
# print("z0 = L0_Crystal = {} mm".format(L0_Crystal))

#%%
# 定义 结构后端面 距离 晶体前端面 的 纵向实际像素、结构后端面 距离 晶体前端面 的 实际纵向尺寸

Iz_endface = Iz_frontface + Iz_structure

sheets_num_endface = sheets_num_frontface + sheets_num_structure
Iz_endface = sheets_num_endface * diz
z0_structure_endface = Iz_endface * size_PerPixel
print("z0_structure_endface = {} mm".format(z0_structure_endface))

#%%
# 定义 晶体 的 纵向实际像素、晶体 的 实际纵向尺寸

Iz = L0_Crystal_expect / size_PerPixel

sheets_num = int(Iz // diz)
Iz = sheets_num * diz
z0 = L0_Crystal = Iz * size_PerPixel
print("z0 = L0_Crystal = {} mm".format(L0_Crystal))

#%%
# 定义 需要展示的截面 1 距离晶体前端面 的 纵向实际像素、需要展示的截面 1 距离晶体前端面 的 实际纵向尺寸

Iz_1f = z0_section_1f_expect / size_PerPixel

sheets_num_section_1 = int(Iz_1f // diz)
sheet_th_section_1 = sheets_num_section_1 # 将要计算的是 距离 Iz_1f 最近的（但在其前面的） 那个面，它是 前面一层 的 后端面，所以这里 用的是 e = endface（但是这一层的前端面，所以取消了这里的 e 的标记，容易引起混淆）
if sheets_num_section_1 == 0:
    sheets_num_section_1 = 1 # 怕 0 - 1 减出负来了，这样 即使 Iz_1f = 0，前端面也 至少给出的是 dz 处的 场分布
sheet_th_section_1f = sheets_num_section_1 - 1 # 但需要 前面一层 的 前端面 的 序数 来计算
iz_1 = sheet_th_section_1 * diz
z0_1 = iz_1 * size_PerPixel
print("z0_section_1 = {} mm".format(z0_1))

#%%
# 定义 需要展示的截面 2 距离晶体后端面 的 纵向实际像素、需要展示的截面 2 距离晶体后端面 的 实际纵向尺寸

Iz_2f = z0_section_2f_expect / size_PerPixel

sheets_num_section_2 = sheets_num - int(Iz_2f // diz) # 距离 后端面 的 距离，转换为 距离 前端面 的 距离 （但要稍微 更靠 后端面 一点）
sheet_th_section_2 = sheets_num_section_2
sheet_th_section_2f = sheets_num_section_2 - 1
iz_2 = sheet_th_section_2 * diz
z0_2 = iz_2 * size_PerPixel
print("z0_section_2 = {} mm".format(z0_2))

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

U1_z0_Squared = U1_z0**2
U1_z0_Squared_amp = np.abs(U1_z0_Squared)

#%%
# const

deff = deff * 1e-12 # pm / V 转换成 m / V
const = (k2 / size_PerPixel / n2)**2 * deff

#%%
# G2_z0_shift

n2_x, n2_y = np.meshgrid([i for i in range(I2_x)], [j for j in range(I2_y)])
Mesh_n2_x_n2_y = np.dstack((n2_x, n2_y))
Mesh_n2_x_n2_y_shift = Mesh_n2_x_n2_y - (I2_x // 2, I2_y // 2)
Mesh_k2_x_k2_y_shift = np.dstack((2 * math.pi * Mesh_n2_x_n2_y_shift[:, :, 0] / I2_x, 2 * math.pi * Mesh_n2_x_n2_y_shift[:, :, 1] / I2_y))
k2_z_shift = (k2**2 - np.square(Mesh_k2_x_k2_y_shift[:, :, 0]) - np.square(Mesh_k2_x_k2_y_shift[:, :, 1]) + 0j )**0.5

global thread_th, for_th, G2_z_plus_dz_shift # 好怪，作为 被封装 和 被引用的 函数，还得在 这一层 声明 全局变量，光是在 内层 子线程 里声明 的话，没用。
thread_th = 0 # 生产出的 第几个 / 一共几个 线程，全局
for_th = 0 # 正在计算到的 第几个 for 循环的序数，全局（非顺序的情况下，这个的含义只是计数，即一共计算了几个 序数 了）
G2_z_plus_dz_shift = 0
U2_z_plus_dz = 0

if is_energy_evolution_on == 1:
    G2_z_shift_energy = np.empty( (sheets_num + 1), dtype=np.float64() )
    U2_z_energy = np.empty( (sheets_num + 1), dtype=np.float64() )
G2_z_shift_energy[0] = np.sum(np.abs(G2_z_plus_dz_shift)**2)
U2_z_energy[0] = np.sum(np.abs(U2_z_plus_dz)**2)

H2_z_shift_k2_z = (np.power(math.e, k2_z_shift * diz * 1j) - 1) / k2_z_shift**2 * size_PerPixel**2 # 注意 这里的 传递函数 的 指数是 正的 ！！！
H2_z_plus_dz_shift_k2_z = np.power(math.e, k2_z_shift * diz * 1j) # 注意 这里的 传递函数 的 指数是 正的 ！！！
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

if is_stored == 1:
    
    # sheet_stored_th = np.empty( (sheets_stored_num + 1), dtype=np.int64() ) # 这个其实 就是 0123...
    sheet_th_stored = np.empty( int(sheets_stored_num + 1), dtype=np.int64() )
    iz_stored = np.empty( int(sheets_stored_num + 1), dtype=np.float64() )
    z_stored = np.empty( int(sheets_stored_num + 1), dtype=np.float64() )
    G2_z_shift_stored = np.empty( (I2_x, I2_y, int(sheets_stored_num + 1)), dtype=np.complex128() )
    U2_z_stored = np.empty( (I2_x, I2_y, int(sheets_stored_num + 1)), dtype=np.complex128() )

    # 小写的 x,y 表示 电脑中 矩阵坐标系，大写 X,Y 表示 笛卡尔坐标系
    # G2_shift_xz_stored = np.empty( (I2_x, sheets_num + 1), dtype=np.complex128() )
    # G2_shift_yz_stored = np.empty( (I2_y, sheets_num + 1), dtype=np.complex128() )
    # U2_xz_stored = np.empty( (I2_x, sheets_num + 1), dtype=np.complex128() )
    # U2_yz_stored = np.empty( (I2_y, sheets_num + 1), dtype=np.complex128() )
    G2_shift_YZ_stored = np.empty( (I2_x, sheets_num + 1), dtype=np.complex128() )
    G2_shift_XZ_stored = np.empty( (I2_y, sheets_num + 1), dtype=np.complex128() )
    U2_YZ_stored = np.empty( (I2_x, sheets_num + 1), dtype=np.complex128() )
    U2_XZ_stored = np.empty( (I2_y, sheets_num + 1), dtype=np.complex128() )

class Customer(threading.Thread):
    """线程 消费者"""
    def __init__(self, name, for_th):
        self.thread_name = name
        self.for_th = for_th
        self.modulation_squared_z = 1 - is_no_backgroud
        super().__init__()
        
    def run(self):
        global thread_th, for_th, G2_z_plus_dz_shift
        """----- 你 需累积的 全局变量，替换 最末一个 g2_z_plus_dz_shift -----"""
        if is_stored == 1:
            global G2_structure_frontface_shift, U2_structure_frontface, G2_structure_endface_shift, U2_structure_endface, G2_section_1_shift, U2_section_1, G2_section_2_shift, U2_section_2
            
        """----- your code begin 1 -----"""
        iz = self.for_th * diz
        
        if is_bulk == 0:
            if self.for_th >= sheets_num_frontface and self.for_th <= sheets_num_endface - 1:
                modulation_squared_full_name = str(self.for_th - sheets_num_frontface) + ".mat"
                modulation_squared_address = location + "\\" + "0.χ2_modulation_squared" + "\\" + modulation_squared_full_name
                self.modulation_squared_z = loadmat(modulation_squared_address)['chi2_modulation_squared']
        else:
            self.modulation_squared_z = 1 - is_no_backgroud
        
        H1_z_shift = np.power(math.e, k1_z_shift * iz * 1j)
        G1_z_shift = g1_shift * H1_z_shift
        G1_z = np.fft.ifftshift(G1_z_shift)
        U1_z = np.fft.ifft2(G1_z)
        
        Q2_z = np.fft.fft2(self.modulation_squared_z * U1_z**2)
        Q2_z_shift = np.fft.fftshift(Q2_z)
        """----- your code end 1 -----"""
        
        con.acquire() # 上锁
        
        if is_ordered == 1:
            
            while True:
                if for_th == self.for_th:
                    # print(self.for_th)
                    """----- your code begin 2 -----"""
                    G2_z_plus_dz_shift = G2_z_plus_dz_shift * H2_z_plus_dz_shift_k2_z + const * Q2_z_shift * H2_z_shift_k2_z
                    G2_z_plus_dz_shift_temp = G2_z_plus_dz_shift
                    """----- your code end 2 -----"""
                    for_th += 1
                    break
                else:
                    con.notify()
                    con.wait() # 但只有当 for_th 不等于 self.for_th， 才等待
        else:
            
            # print(self.for_th)
            """----- your code begin 2 -----"""
            G2_z_plus_dz_shift = G2_z_plus_dz_shift * H2_z_plus_dz_shift_k2_z + const * Q2_z_shift * H2_z_shift_k2_z
            G2_z_plus_dz_shift_temp = G2_z_plus_dz_shift
            """----- your code end 2 -----"""
            for_th += 1
        
        thread_th -= 1 # 在解锁之前 减少 1 个线程数量，以便 p 线程 收到消息后，生产 1 个 线程出来
        con.notify() # 无论如何 都得通知一下 其他线程，让其别 wait() 了
        con.release() # 解锁
        
        #%%
        
        G2_z_plus_dz = np.fft.ifftshift(G2_z_plus_dz_shift_temp)
        U2_z_plus_dz = np.fft.ifft2(G2_z_plus_dz)
        
        if is_energy_evolution_on == 1:
            G2_z_shift_energy[self.for_th + 1] = np.sum(np.abs(G2_z_plus_dz_shift_temp)**2)
            U2_z_energy[self.for_th + 1] = np.sum(np.abs(U2_z_plus_dz)**2)
        
        if is_stored == 1:
            # 小写的 x,y 表示 电脑中 矩阵坐标系，大写 X,Y 表示 笛卡尔坐标系
            G2_shift_YZ_stored[:, self.for_th] = G2_z_plus_dz_shift_temp[:, I2_y // 2 + int(X / size_PerPixel) ] # X 增加，则 从 G2_z_shift 中 读取的 列 向右移，也就是 YZ 面向 列 增加的方向（G2_z_shift 的 右侧）移动
            G2_shift_XZ_stored[:, self.for_th] = G2_z_plus_dz_shift_temp[I2_x // 2 - int(Y / size_PerPixel), :] # Y 增加，则 从 G2_z_shift 中 读取的 行 向上移，也就是 XZ 面向 行 减小的方向（G2_z_shift 的 上侧）移动
            U2_YZ_stored[:, self.for_th] = U2_z_plus_dz[:, I2_y // 2 + int(X / size_PerPixel)]
            U2_XZ_stored[:, self.for_th] = U2_z_plus_dz[I2_x // 2 - int(Y / size_PerPixel), :]
            
            #%%
            
            if np.mod(self.for_th, sheets_num // sheets_stored_num) == 0: # 如果 self.for_th 是 sheets_num // sheets_stored_num 的 整数倍（包括零），则 储存之
                sheet_th_stored[int(self.for_th // (sheets_num // sheets_stored_num))] = self.for_th + 1
                iz_stored[int(self.for_th // (sheets_num // sheets_stored_num))] = iz + diz
                z_stored[int(self.for_th // (sheets_num // sheets_stored_num))] = (iz + diz) * size_PerPixel
                G2_z_shift_stored[:, :, int(self.for_th // (sheets_num // sheets_stored_num))] = G2_z_plus_dz_shift_temp #　储存的 第一层，实际上不是 G2_0，而是 G2_dz
                U2_z_stored[:, :, int(self.for_th // (sheets_num // sheets_stored_num))] = U2_z_plus_dz #　储存的 第一层，实际上不是 U2_0，而是 U2_dz
            
            if self.for_th == sheets_num_frontface: # 如果 self.for_th 是 sheets_num_frontface，则把结构 前端面 场分布 储存起来
                G2_structure_frontface_shift = G2_z_plus_dz_shift_temp
                U2_structure_frontface = U2_z_plus_dz
            if self.for_th == sheets_num_endface - 1: # 如果 self.for_th 是 sheets_num_endface - 1，则把结构 后端面 场分布 储存起来
                G2_structure_endface_shift = G2_z_plus_dz_shift_temp
                U2_structure_endface = U2_z_plus_dz
            if self.for_th == sheet_th_section_1f: # 如果 self.for_th 是 想要观察的 第一个面 前面那一层的 层序数，则 将储存之于 该层 前面那一层的 后端面（毕竟 算出来的是 z + dz） 分布中
                G2_section_1_shift = G2_z_plus_dz_shift_temp
                U2_section_1 = U2_z_plus_dz
            if self.for_th == sheet_th_section_2f: # 如果 self.for_th 是 想要观察的 第二个面 前面那一层的 层序数，则 将储存之于 该层 前面那一层的 后端面（毕竟 算出来的是 z + dz） 分布中
                G2_section_2_shift = G2_z_plus_dz_shift_temp
                U2_section_2 = U2_z_plus_dz

"""----- your code begin 0 -----"""
is_ordered = 1 # for_th 是否 按顺序执行
threads_num = 10 # 需要开启 多少个线程 持续计算
fors_num = sheets_num # 需要计算 for 循环中 多少个 序数
"""----- your code end 0 -----"""

tick_start = time.time()

p = Producer(threads_num, fors_num)
p.setDaemon(True)
p.start()
p.join() # 添加join使 p 线程执行完

print("----- consume time: {} s -----".format(time.time() - tick_start))
    
#%%

G2_z0_SSI_shift = G2_z_plus_dz_shift

G2_z0_SSI_shift_amp = np.abs(G2_z0_SSI_shift)
# print(np.max(G2_z0_SSI_shift_amp))
G2_z0_SSI_shift_phase = np.angle(G2_z0_SSI_shift)
if is_save == 1:
    if not os.path.isdir("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift"):
        os.makedirs("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift")

#%%
#绘图：G2_z0_SSI_shift_amp

G2_z0_SSI_shift_amp_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + "5.1. NLA - " + "G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_amp" + file_name_extension

plot_2d(I2_x, I2_y, size_PerPixel, diz, 
        G2_z0_SSI_shift_amp, G2_z0_SSI_shift_amp_address, "G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_amp", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        1, is_colorbar_on, vmax, vmin)

#%%
#绘图：G2_z0_SSI_shift_phase

G2_z0_SSI_shift_phase_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + "5.2. NLA - " + "G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_phase" + file_name_extension

plot_2d(I2_x, I2_y, size_PerPixel, diz, 
        G2_z0_SSI_shift_phase, G2_z0_SSI_shift_phase_address, "G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_phase", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        1, is_colorbar_on, vmax, vmin)

#%%
# 储存 G2_z0_SSI_shift 到 txt 文件

if is_save == 1:
    G2_z0_SSI_shift_full_name = "5. NLA - G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + (is_save_txt and ".txt" or ".mat")
    G2_z0_SSI_shift_txt_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + G2_z0_SSI_shift_full_name
    np.savetxt(G2_z0_SSI_shift_txt_address, G2_z0_SSI_shift) if is_save_txt else savemat(G2_z0_SSI_shift_txt_address, {'G':G2_z0_SSI_shift})

#%%    
# 绘制 G2_z_shift_energy 随 z 演化的 曲线

if is_energy_evolution_on == 1:
    
    vmax_G2_z_shift_energy = np.max(G2_z_shift_energy)
    vmin_G2_z_shift_energy = np.min(G2_z_shift_energy)
    
    G2_z_shift_energy_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "\\" + "5.1. NLA - " + "G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_energy_evolution" + file_name_extension
    
    plot_1d(sheets_num + 1, size_PerPixel, diz, 
            G2_z_shift_energy, G2_z_shift_energy_address, "G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_energy_evolution", 
            is_save, dpi, size_fig * 10, size_fig, 
            color_1d, ticks_num, is_title_on, is_axes_on, is_mm, 1, 
            fontsize, font, 
            vmax_G2_z_shift_energy, vmin_G2_z_shift_energy)
    
#%%
# G2_z0_SSI = G2_z0_SSI(k1_x, k1_y) → IFFT2 → U2(x0, y0, z0) = U2_z0_SSI

G2_z0_SSI = np.fft.ifftshift(G2_z0_SSI_shift)
U2_z0_SSI = np.fft.ifft2(G2_z0_SSI)
# 2 维 坐标空间 中的 复标量场，是 i2_x0, i2_y0 的函数
# U2_z0_SSI = U2_z0_SSI * scale_down_factor # 归一化

#%%

if is_stored == 1:

    sheet_th_stored[sheets_stored_num] = sheets_num
    iz_stored[sheets_stored_num] = Iz
    z_stored[sheets_stored_num] = Iz * size_PerPixel
    G2_z_shift_stored[:, :, sheets_stored_num] = G2_z0_SSI_shift #　储存的 第一层，实际上不是 G2_0，而是 G2_dz
    U2_z_stored[:, :, sheets_stored_num] = U2_z0_SSI #　储存的 第一层，实际上不是 U2_0，而是 U2_dz
    
    #%%
    
    if is_save == 1:
        if not os.path.isdir("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_stored"):
            os.makedirs("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_stored")
        if not os.path.isdir("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_stored"):
            os.makedirs("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_stored")
    
    #-------------------------
    
    vmax_G2_z_shift_stored_amp = np.max(np.abs(G2_z_shift_stored))
    vmin_G2_z_shift_stored_amp = np.min(np.abs(G2_z_shift_stored))
    
    for sheet_stored_th in range(sheets_stored_num + 1):
        
        G2_z_shift_sheet_stored_th_amp_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_stored" + "\\" + "5.1. NLA - " + "G2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI_shift" + "_amp" + file_name_extension
        
        plot_2d(I2_x, I2_y, size_PerPixel, diz, 
                np.abs(G2_z_shift_stored[:, :, sheet_stored_th]), G2_z_shift_sheet_stored_th_amp_address, "G2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI_shift" + "_amp", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                is_self_colorbar, is_colorbar_on, vmax_G2_z_shift_stored_amp, vmin_G2_z_shift_stored_amp)
        
    vmax_G2_z_shift_stored_phase = np.max(np.angle(G2_z_shift_stored))
    vmin_G2_z_shift_stored_phase = np.min(np.angle(G2_z_shift_stored))
        
    for sheet_stored_th in range(sheets_stored_num + 1):
        
        G2_z_shift_sheet_stored_th_phase_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_stored" + "\\" + "5.2. NLA - " + "G2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI_shift" + "_phase" + file_name_extension
    
        plot_2d(I2_x, I2_y, size_PerPixel, diz, 
                np.angle(G2_z_shift_stored[:, :, sheet_stored_th]), G2_z_shift_sheet_stored_th_phase_address, "G2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI_shift" + "_phase", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                is_self_colorbar, is_colorbar_on, vmax_G2_z_shift_stored_phase, vmin_G2_z_shift_stored_phase)
    
    #-------------------------    
    
    vmax_U2_z_stored_amp = np.max(np.abs(U2_z_stored))
    vmin_U2_z_stored_amp = np.min(np.abs(U2_z_stored))
    
    for sheet_stored_th in range(sheets_stored_num + 1):
        
        U2_z_sheet_stored_th_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_stored" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI" + "_amp" + file_name_extension
    
        plot_2d(I2_x, I2_y, size_PerPixel, diz, 
                np.abs(U2_z_stored[:, :, sheet_stored_th]), U2_z_sheet_stored_th_amp_address, "U2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI" + "_amp", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                is_self_colorbar, is_colorbar_on, vmax_U2_z_stored_amp, vmin_U2_z_stored_amp)
        
    vmax_U2_z_stored_phase = np.max(np.angle(U2_z_stored))
    vmin_U2_z_stored_phase = np.min(np.angle(U2_z_stored))
        
    for sheet_stored_th in range(sheets_stored_num + 1):
        
        U2_z_sheet_stored_th_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_stored" + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI" + "_phase" + file_name_extension
    
        plot_2d(I2_x, I2_y, size_PerPixel, diz, 
                np.angle(U2_z_stored[:, :, sheet_stored_th]), U2_z_sheet_stored_th_phase_address, "U2_" + str(float('%.2g' % z_stored[sheet_stored_th])) + "mm" + "_SSI" + "_phase", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
                fontsize, font,
                is_self_colorbar, is_colorbar_on, vmax_U2_z_stored_phase, vmin_U2_z_stored_phase)
    
    #%%
    # 这 sheets_stored_num 层 也可以 画成 3D，就是太丑了，所以只 整个 U2_amp 示意一下即可
    
    # U2_z_sheets_stored_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_stored" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_sheets_stored" + "_amp" + file_name_extension
    
    # plot_3d_XYz(I2_y, I2_x, size_PerPixel, diz, 
    #             sheets_stored_num, U2_z_stored, sheet_th_stored, 
    #             U2_z_sheets_stored_amp_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_sheets_stored" + "_amp", 
    #             is_save, dpi, size_fig, 
    #             cmap_3d, elev, azim, alpha, 
    #             ticks_num, is_title_on, is_axes_on, is_mm,  
    #             fontsize, font,
    #             is_self_colorbar, is_colorbar_on, vmax_U2_z_stored_amp, vmin_U2_z_stored_amp)
    
    #%%
    
    # 小写的 x,y 表示 电脑中 矩阵坐标系，大写 X,Y 表示 笛卡尔坐标系
    # G2_shift_xz_stored[:, sheets_num] = G2_z0_SSI_shift[:, I2_y // 2 + int(X / size_PerPixel)]
    # G2_shift_yz_stored[:, sheets_num] = G2_z0_SSI_shift[I2_x // 2 - int(Y / size_PerPixel), :]
    # U2_xz_stored[:, sheets_num] = U2_z0_SSI[:, I2_y // 2 + int(X / size_PerPixel)]
    # U2_yz_stored[:, sheets_num] = U2_z0_SSI[I2_x // 2 - int(Y / size_PerPixel), :]
    G2_shift_YZ_stored[:, sheets_num] = G2_z0_SSI_shift[:, I2_y // 2 + int(X / size_PerPixel)]
    G2_shift_XZ_stored[:, sheets_num] = G2_z0_SSI_shift[I2_x // 2 - int(Y / size_PerPixel), :]
    U2_YZ_stored[:, sheets_num] = U2_z0_SSI[:, I2_y // 2 + int(X / size_PerPixel)]
    U2_XZ_stored[:, sheets_num] = U2_z0_SSI[I2_x // 2 - int(Y / size_PerPixel), :]
    
    #%%
    # 再算一下 初始的 场分布，之后 绘 3D 用，因为 开启 多线程后，就不会 储存 中间层 了
    
    # modulation_squared_0 = 1
    
    # if is_bulk == 0:
    #     modulation_squared_full_name = str(0) + ".mat"
    #     modulation_squared_address = location + "\\" + "0.χ2_modulation_squared" + "\\" + modulation_squared_full_name
    #     modulation_squared_0 = loadmat(modulation_squared_address)['chi2_modulation_squared']
    
    # Q2_0 = np.fft.fft2(modulation_squared_0 * U1_0**2)
    # Q2_0_shift = np.fft.fftshift(Q2_0)
    
    # H2_0_shift_k2_z = 1 / k2_z_shift * size_PerPixel # 注意 这里的 传递函数 的 指数是 负的 ！！！（但 z = iz = 0，所以 指数项 变成 1 了）
    # g2_dz_shift = const * Q2_0_shift * H2_0_shift_k2_z
    
    # H2_dz_shift_k2_z = np.power(math.e, k2_z_shift * diz * 1j) # 注意 这里的 传递函数 的 指数是 正的 ！！！
    # G2_dz_shift = g2_dz_shift * H2_dz_shift_k2_z #　每一步 储存的 实际上不是 G2_z，而是 G2_z+dz
    # G2_dz = np.fft.ifftshift(G2_dz_shift)
    # U2_dz = np.fft.ifft2(G2_dz)
    
    #%%
    
    if is_save == 1:
        if not os.path.isdir("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored"):
            os.makedirs("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored")
        if not os.path.isdir("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored"):
            os.makedirs("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored")
    
    #========================= G2_shift_YZ_stored_amp、G2_shift_XZ_stored_amp
    
    vmax_G2_shift_YZ_XZ_stored_amp = np.max([np.max(np.abs(G2_shift_YZ_stored)), np.max(np.abs(G2_shift_XZ_stored))])
    vmin_G2_shift_YZ_XZ_stored_amp = np.min([np.min(np.abs(G2_shift_YZ_stored)), np.min(np.abs(G2_shift_XZ_stored))])
    
    G2_shift_YZ_stored_amp_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored" + "\\" + "5.1. NLA - " + "G2_" + str(float('%.2g' % X)) + "mm" + "_SSI_shift" + "_YZ" + "_amp" + file_name_extension
    
    plot_2d(sheets_num + 1, I2_x, size_PerPixel, diz, 
            np.abs(G2_shift_YZ_stored), G2_shift_YZ_stored_amp_address, "G2_" + str(float('%.2g' % X)) + "mm" + "_SSI_shift" + "_YZ" + "_amp", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_G2_shift_YZ_XZ_stored_amp, vmin_G2_shift_YZ_XZ_stored_amp)
    
    G2_shift_XZ_stored_amp_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored" + "\\" + "5.1. NLA - " + "G2_" + str(float('%.2g' % Y)) + "mm" + "_SSI_shift" + "_XZ" + "_amp" + file_name_extension
    
    plot_2d(sheets_num + 1, I2_y, size_PerPixel, diz, 
            np.abs(G2_shift_XZ_stored), G2_shift_XZ_stored_amp_address, "G2_" + str(float('%.2g' % Y)) + "mm" + "_SSI_shift" + "_XZ" + "_amp", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_G2_shift_YZ_XZ_stored_amp, vmin_G2_shift_YZ_XZ_stored_amp)
    
    #------------------------- G2_shift_YZ_stored_phase、G2_shift_XZ_stored_phase
    
    vmax_G2_shift_YZ_XZ_stored_phase = np.max([np.max(np.angle(G2_shift_YZ_stored)), np.max(np.angle(G2_shift_XZ_stored))])
    vmin_G2_shift_YZ_XZ_stored_phase = np.min([np.min(np.angle(G2_shift_YZ_stored)), np.min(np.angle(G2_shift_XZ_stored))])
    
    G2_shift_YZ_stored_phase_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored" + "\\" + "5.2. NLA - " + "G2_" + str(float('%.2g' % X)) + "mm" + "_SSI_shift" + "_YZ" + "_phase" + file_name_extension
    
    plot_2d(sheets_num + 1, I2_x, size_PerPixel, diz, 
            np.angle(G2_shift_YZ_stored), G2_shift_YZ_stored_phase_address, "G2_" + str(float('%.2g' % X)) + "mm" + "_SSI_shift" + "_YZ" + "_phase", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_G2_shift_YZ_XZ_stored_phase, vmin_G2_shift_YZ_XZ_stored_phase)
    
    G2_shift_XZ_stored_phase_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored" + "\\" + "5.2. NLA - " + "G2_" + str(float('%.2g' % Y)) + "mm" + "_SSI_shift" + "_XZ" + "_phase" + file_name_extension
    
    plot_2d(sheets_num + 1, I2_y, size_PerPixel, diz, 
            np.angle(G2_shift_XZ_stored), G2_shift_XZ_stored_phase_address, "G2_" + str(float('%.2g' % Y)) + "mm" + "_SSI_shift" + "_XZ" + "_phase", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_G2_shift_YZ_XZ_stored_phase, vmin_G2_shift_YZ_XZ_stored_phase)
    
    #========================= U2_YZ_stored_amp、U2_XZ_stored_amp
    
    vmax_U2_YZ_XZ_stored_amp = np.max([np.max(np.abs(U2_YZ_stored)), np.max(np.abs(U2_XZ_stored))])
    vmin_U2_YZ_XZ_stored_amp = np.min([np.min(np.abs(U2_YZ_stored)), np.min(np.abs(U2_XZ_stored))])
    
    U2_YZ_stored_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % X)) + "mm" + "_SSI" + "_YZ" + "_amp" + file_name_extension
    
    plot_2d(sheets_num + 1, I2_x, size_PerPixel, diz, 
            np.abs(U2_YZ_stored), U2_YZ_stored_amp_address, "U2_" + str(float('%.2g' % X)) + "mm" + "_SSI" + "_YZ" + "_amp", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_U2_YZ_XZ_stored_amp, vmin_U2_YZ_XZ_stored_amp)
    
    U2_XZ_stored_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % Y)) + "mm" + "_SSI" + "_XZ" + "_amp" + file_name_extension
    
    plot_2d(sheets_num + 1, I2_y, size_PerPixel, diz, 
            np.abs(U2_XZ_stored), U2_XZ_stored_amp_address, "U2_" + str(float('%.2g' % Y)) + "mm" + "_SSI" + "_XZ" + "_amp", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_U2_YZ_XZ_stored_amp, vmin_U2_YZ_XZ_stored_amp)
    
    #------------------------- U2_YZ_stored_phase、U2_XZ_stored_phase
    
    vmax_U2_YZ_XZ_stored_phase = np.max([np.max(np.angle(U2_YZ_stored)), np.max(np.angle(U2_XZ_stored))])
    vmin_U2_YZ_XZ_stored_phase = np.min([np.min(np.angle(U2_YZ_stored)), np.min(np.angle(U2_XZ_stored))])
    
    U2_YZ_stored_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored" + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % X)) + "mm" + "_SSI" + "_YZ" + "_phase" + file_name_extension
    
    plot_2d(sheets_num + 1, I2_x, size_PerPixel, diz, 
            np.angle(U2_YZ_stored), U2_YZ_stored_phase_address, "U2_" + str(float('%.2g' % X)) + "mm" + "_SSI" + "_YZ" + "_phase", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_U2_YZ_XZ_stored_phase, vmin_U2_YZ_XZ_stored_phase)
    
    U2_XZ_stored_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored" + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % Y)) + "mm" + "_SSI" + "_XZ" + "_phase" + file_name_extension
    
    plot_2d(sheets_num + 1, I2_y, size_PerPixel, diz, 
            np.angle(U2_XZ_stored), U2_XZ_stored_phase_address, "U2_" + str(float('%.2g' % Y)) + "mm" + "_SSI" + "_XZ" + "_phase", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 1, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_U2_YZ_XZ_stored_phase, vmin_U2_YZ_XZ_stored_phase)
    
    #%%
    
    if is_save == 1:
        if not os.path.isdir("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored"):
            os.makedirs("5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored")
        if not os.path.isdir("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored"):
            os.makedirs("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored")
    
    #------------------------- 储存 G2_section_1_shift_amp、G2_section_2_shift_amp、G2_structure_frontface_shift_amp、G2_structure_endface_shift_amp
    
    vmax_G2_section_1_2_front_end_shift_amp = np.max([np.max(np.abs(G2_section_1_shift)), np.max(np.abs(G2_section_2_shift)), np.max(np.abs(G2_structure_frontface_shift)), np.max(np.abs(G2_structure_endface_shift))])
    vmin_G2_section_1_2_front_end_shift_amp = np.min([np.min(np.abs(G2_section_1_shift)), np.min(np.abs(G2_section_2_shift)), np.min(np.abs(G2_structure_frontface_shift)), np.min(np.abs(G2_structure_endface_shift))])
    
    G2_section_1_shift_amp_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.1. NLA - " + "G2_" + str(float('%.2g' % z0_1)) + "mm" + "_SSI_shift" + "_amp" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.abs(G2_section_1_shift), G2_section_1_shift_amp_address, "G2_" + str(float('%.2g' % z0_1)) + "mm" + "_SSI_shift" + "_amp", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_G2_section_1_2_front_end_shift_amp, vmin_G2_section_1_2_front_end_shift_amp)
    
    G2_section_2_shift_amp_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.1. NLA - " + "G2_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_amp" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.abs(G2_section_2_shift), G2_section_2_shift_amp_address, "G2_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_amp", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_G2_section_1_2_front_end_shift_amp, vmin_G2_section_1_2_front_end_shift_amp)
    
    G2_structure_frontface_shift_amp_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.1. NLA - " + "G2_" + str(float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI_shift" + "_amp" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.abs(G2_structure_frontface_shift), G2_structure_frontface_shift_amp_address, "G2_" + str(float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI_shift" + "_amp", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_G2_section_1_2_front_end_shift_amp, vmin_G2_section_1_2_front_end_shift_amp)
    
    G2_structure_endface_shift_amp_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.1. NLA - " + "G2_" + str(float('%.2g' % z0_structure_endface)) + "mm" + "_SSI_shift" + "_amp" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.abs(G2_structure_endface_shift), G2_structure_endface_shift_amp_address, "G2_" + str(float('%.2g' % z0_structure_endface)) + "mm" + "_SSI_shift" + "_amp", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_G2_section_1_2_front_end_shift_amp, vmin_G2_section_1_2_front_end_shift_amp)
    
    #------------------------- 储存 G2_section_1_shift_phase、G2_section_2_shift_phase、G2_structure_frontface_shift_phase、G2_structure_endface_shift_phase
    
    vmax_G2_section_1_2_front_end_shift_phase = np.max([np.max(np.angle(G2_section_1_shift)), np.max(np.angle(G2_section_2_shift)), np.max(np.angle(G2_structure_frontface_shift)), np.max(np.angle(G2_structure_endface_shift))])
    vmin_G2_section_1_2_front_end_shift_phase = np.min([np.min(np.angle(G2_section_1_shift)), np.min(np.angle(G2_section_2_shift)), np.min(np.angle(G2_structure_frontface_shift)), np.min(np.angle(G2_structure_endface_shift))])
    
    G2_section_1_shift_phase_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.2. NLA - " + "G2_" + str(float('%.2g' % z0_1)) + "mm" + "_SSI_shift" + "_phase" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.angle(G2_section_1_shift), G2_section_1_shift_phase_address, "G2_" + str(float('%.2g' % z0_1)) + "mm" + "_SSI_shift" + "_phase", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_G2_section_1_2_front_end_shift_phase, vmin_G2_section_1_2_front_end_shift_phase)
    
    G2_section_2_shift_phase_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.2. NLA - " + "G2_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_phase" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.angle(G2_section_2_shift), G2_section_2_shift_phase_address, "G2_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_phase", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_G2_section_1_2_front_end_shift_phase, vmin_G2_section_1_2_front_end_shift_phase)
    
    G2_structure_frontface_shift_phase_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.2. NLA - " + "G2_" + str(float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI_shift" + "_phase" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.angle(G2_structure_frontface_shift), G2_structure_frontface_shift_phase_address, "G2_" + str(float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI_shift" + "_phase", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_G2_section_1_2_front_end_shift_phase, vmin_G2_section_1_2_front_end_shift_phase)
    
    G2_structure_endface_shift_phase_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_sheets_selective_stored" + "\\" + "5.2. NLA - " + "G2_" + str(float('%.2g' % z0_structure_endface)) + "mm" + "_SSI_shift" + "_phase" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.angle(G2_structure_endface_shift), G2_structure_endface_shift_phase_address, "G2_" + str(float('%.2g' % z0_structure_endface)) + "mm" + "_SSI_shift" + "_phase", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_G2_section_1_2_front_end_shift_phase, vmin_G2_section_1_2_front_end_shift_phase)
    
    #------------------------- 储存 U2_section_1_amp、U2_section_2_amp、U2_structure_frontface_amp、U2_structure_endface_amp
    
    vmax_U2_section_1_2_front_end_shift_amp = np.max([np.max(np.abs(U2_section_1)), np.max(np.abs(U2_section_2)), np.max(np.abs(U2_structure_frontface)), np.max(np.abs(U2_structure_endface))])
    vmin_U2_section_1_2_front_end_shift_amp = np.min([np.min(np.abs(U2_section_1)), np.min(np.abs(U2_section_2)), np.min(np.abs(U2_structure_frontface)), np.min(np.abs(U2_structure_endface))])
    
    U2_section_1_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z0_1)) + "mm" + "_SSI" + "_amp" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.abs(U2_section_1), U2_section_1_amp_address, "U2_" + str(float('%.2g' % z0_1)) + "mm" + "_SSI" + "_amp", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_U2_section_1_2_front_end_shift_amp, vmin_U2_section_1_2_front_end_shift_amp)
    
    U2_section_2_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_amp" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.abs(U2_section_2), U2_section_2_amp_address, "U2_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_amp", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_U2_section_1_2_front_end_shift_amp, vmin_U2_section_1_2_front_end_shift_amp)
    
    U2_structure_frontface_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI" + "_amp" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.abs(U2_structure_frontface), U2_structure_frontface_amp_address, "U2_" + str(float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI" + "_amp", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_U2_section_1_2_front_end_shift_amp, vmin_U2_section_1_2_front_end_shift_amp)
    
    U2_structure_endface_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z0_structure_endface)) + "mm" + "_SSI" + "_amp" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.abs(U2_structure_endface), U2_structure_endface_amp_address, "U2_" + str(float('%.2g' % z0_structure_endface)) + "mm" + "_SSI" + "_amp", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_U2_section_1_2_front_end_shift_amp, vmin_U2_section_1_2_front_end_shift_amp)
    
    #------------------------- 储存 U2_section_1_phase、U2_section_2_phase、U2_structure_frontface_phase、U2_structure_endface_phase
    
    vmax_U2_section_1_2_front_end_shift_phase = np.max([np.max(np.angle(U2_section_1)), np.max(np.angle(U2_section_2)), np.max(np.angle(U2_structure_frontface)), np.max(np.angle(U2_structure_endface))])
    vmin_U2_section_1_2_front_end_shift_phase = np.min([np.min(np.angle(U2_section_1)), np.min(np.angle(U2_section_2)), np.min(np.angle(U2_structure_frontface)), np.min(np.angle(U2_structure_endface))])
    
    U2_section_1_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % z0_1)) + "mm" + "_SSI" + "_phase" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.angle(U2_section_1), U2_section_1_phase_address, "U2_" + str(float('%.2g' % z0_1)) + "mm" + "_SSI" + "_phase", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_U2_section_1_2_front_end_shift_phase, vmin_U2_section_1_2_front_end_shift_phase)
    
    U2_section_2_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_phase" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.angle(U2_section_2), U2_section_2_phase_address, "U2_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_phase", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_U2_section_1_2_front_end_shift_phase, vmin_U2_section_1_2_front_end_shift_phase)
    
    U2_structure_frontface_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI" + "_phase" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.angle(U2_structure_frontface), U2_structure_frontface_phase_address, "U2_" + str(float('%.2g' % z0_structure_frontface)) + "mm" + "_SSI" + "_phase", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_U2_section_1_2_front_end_shift_phase, vmin_U2_section_1_2_front_end_shift_phase)
    
    U2_structure_endface_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_sheets_selective_stored" + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % z0_structure_endface)) + "mm" + "_SSI" + "_phase" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            np.angle(U2_structure_endface), U2_structure_endface_phase_address, "U2_" + str(float('%.2g' % z0_structure_endface)) + "mm" + "_SSI" + "_phase", 
            is_save, dpi, size_fig, 
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font, 
            is_self_colorbar, is_colorbar_on, vmax_U2_section_1_2_front_end_shift_phase, vmin_U2_section_1_2_front_end_shift_phase)
    
    #%%
    # 绘制 G2_amp 的 侧面 3D 分布图，以及 初始 和 末尾的 G2_amp（现在 可以 任选位置 了）
    
    vmax_G2_amp = np.max([vmax_G2_shift_YZ_XZ_stored_amp, vmax_G2_section_1_2_front_end_shift_amp])
    vmin_G2_amp = np.min([vmin_G2_shift_YZ_XZ_stored_amp, vmin_G2_section_1_2_front_end_shift_amp])
    
    G2_shift_XYZ_stored_amp_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored" + "\\" + "5.1. NLA - " + "G2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_XYZ" + "_amp" + file_name_extension
    
    plot_3d_XYZ(sheets_num + 1, I2_y, I2_x, size_PerPixel, diz, 
                np.abs(G2_shift_YZ_stored), np.abs(G2_shift_XZ_stored), np.abs(G2_section_1_shift), np.abs(G2_section_2_shift), 
                np.abs(G2_structure_frontface_shift), np.abs(G2_structure_endface_shift), is_show_structure_face, 
                G2_shift_XYZ_stored_amp_address, "G2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_XYZ" + "_amp", 
                I2_y // 2 + int(X / size_PerPixel), I2_x // 2 + int(Y / size_PerPixel), sheet_th_section_1, sheet_th_section_2, 
                sheets_num_frontface, sheets_num_endface - 1, 
                is_save, dpi, size_fig, 
                cmap_3d, elev, azim, alpha, 
                ticks_num, is_title_on, is_axes_on, is_mm,  
                fontsize, font, 
                is_self_colorbar, is_colorbar_on, vmax_G2_amp, vmin_G2_amp)
    
    #%%
    # 绘制 G2_phase 的 侧面 3D 分布图，以及 初始 和 末尾的 G2_phase
    
    # vmax_G2_phase = np.max([vmax_G2_shift_YZ_XZ_stored_phase, vmax_G2_section_1_2_front_end_shift_phase])
    # vmin_G2_phase = np.min([vmin_G2_shift_YZ_XZ_stored_phase, vmin_G2_section_1_2_front_end_shift_phase])
    
    # G2_shift_XYZ_stored_phase_address = location + "\\" + "5. G2_" + str(float('%.2g' % z0)) + "mm" + "_SSI_shift" + "_YZ_XZ_stored" + "\\" + "5.2. NLA - " + "G2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_XYZ" + "_phase" + file_name_extension
        
    # plot_3d_XYZ(sheets_num + 1, I2_y, I2_x, size_PerPixel, diz, 
    #             np.angle(G2_shift_YZ_stored), np.angle(G2_shift_XZ_stored), np.angle(G2_section_1_shift), np.angle(G2_section_2_shift), 
    #             np.angle(G2_structure_frontface_shift), np.angle(G2_structure_endface_shift), is_show_structure_face, 
    #             G2_shift_XYZ_stored_phase_address, "G2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI_shift" + "_XYZ" + "_phase", 
    #             I2_y // 2 + int(X / size_PerPixel), I2_x // 2 + int(Y / size_PerPixel), sheet_th_section_1, sheet_th_section_2, 
    #             sheets_num_frontface, sheets_num_endface - 1, 
    #             is_save, dpi, size_fig, 
    #             cmap_3d, elev, azim, alpha, 
    #             ticks_num, is_title_on, is_axes_on, is_mm,  
    #             fontsize, font, 
    #             is_self_colorbar, is_colorbar_on, vmax_G2_phase, vmin_G2_phase)
    
    #%%
    # 绘制 U2_amp 的 侧面 3D 分布图，以及 初始 和 末尾的 U2_amp
    
    vmax_U2_amp = np.max([vmax_U2_YZ_XZ_stored_amp, vmax_U2_section_1_2_front_end_shift_amp])
    vmin_U2_amp = np.min([vmin_U2_YZ_XZ_stored_amp, vmin_U2_section_1_2_front_end_shift_amp])
    
    U2_XYZ_stored_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_XYZ" + "_amp" + file_name_extension
        
    plot_3d_XYZ(sheets_num + 1, I2_y, I2_x, size_PerPixel, diz, 
                np.abs(U2_YZ_stored), np.abs(U2_XZ_stored), np.abs(U2_section_1), np.abs(U2_section_2), 
                np.abs(U2_structure_frontface), np.abs(U2_structure_endface), is_show_structure_face, 
                U2_XYZ_stored_amp_address, "U2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_XYZ" + "_amp", 
                I2_y // 2 + int(X / size_PerPixel), I2_x // 2 + int(Y / size_PerPixel), sheet_th_section_1, sheet_th_section_2, 
                sheets_num_frontface, sheets_num_endface - 1, 
                is_save, dpi, size_fig, 
                cmap_3d, elev, azim, alpha, 
                ticks_num, is_title_on, is_axes_on, is_mm,  
                fontsize, font, 
                is_self_colorbar, is_colorbar_on, vmax_U2_amp, vmin_U2_amp)
    
    #%%
    # 绘制 U2_phase 的 侧面 3D 分布图，以及 初始 和 末尾的 U2_phase
    
    # vmax_U2_phase = np.max([vmax_U2_YZ_XZ_stored_phase, vmax_U2_section_1_2_front_end_shift_phase])
    # vmin_U2_phase = np.min([vmin_U2_YZ_XZ_stored_phase, vmin_U2_section_1_2_front_end_shift_phase])
    
    # U2_XYZ_stored_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_YZ_XZ_stored" + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_XYZ" + "_phase" + file_name_extension
    
    # plot_3d_XYZ(sheets_num + 1, I2_y, I2_x, size_PerPixel, diz, 
    #             np.angle(U2_YZ_stored), np.angle(U2_XZ_stored), np.angle(U2_section_1), np.angle(U2_section_2), 
    #             np.angle(U2_structure_frontface), np.angle(U2_structure_endface), is_show_structure_face, 
    #             U2_XYZ_stored_phase_address, "U2_" + str(float('%.2g' % X)) + "mm" + "_" + str(float('%.2g' % Y)) + "mm" + "__" + str(float('%.2g' % z0_1)) + "mm" + "_" + str(float('%.2g' % z0_2)) + "mm" + "_SSI" + "_XYZ" + "_phase", 
    #             I2_y // 2 + int(X / size_PerPixel), I2_x // 2 + int(Y / size_PerPixel), sheet_th_section_1, sheet_th_section_2, 
    #             sheets_num_frontface, sheets_num_endface - 1, 
    #             is_save, dpi, size_fig, 
    #             cmap_3d, elev, azim, alpha, 
    #             ticks_num, is_title_on, is_axes_on, is_mm,  
    #             fontsize, font, 
    #             is_self_colorbar, is_colorbar_on, vmax_U2_phase, vmin_U2_phase)

#%%

U2_z0_SSI_amp = np.abs(U2_z0_SSI)
# print(np.max(U2_z0_SSI_amp))
U2_z0_SSI_phase = np.angle(U2_z0_SSI)

print("NLA - U2_{}mm_SSI.total_energy = {}".format(z0, np.sum(U2_z0_SSI_amp**2)))

if is_save == 1:
    if not os.path.isdir("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI"):
        os.makedirs("6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI")

#%%
#绘图：U2_z0_SSI_amp

U2_z0_SSI_amp_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_amp" + file_name_extension

plot_2d(I2_x, I2_y, size_PerPixel, diz, 
        U2_z0_SSI_amp, U2_z0_SSI_amp_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_amp", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        1, is_colorbar_on, vmax, vmin)

#%%
#绘图：U2_z0_SSI_phase

U2_z0_SSI_phase_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_phase" + file_name_extension

plot_2d(I2_x, I2_y, size_PerPixel, diz, 
        U2_z0_SSI_phase, U2_z0_SSI_phase_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_phase", 
        is_save, dpi, size_fig,  
        cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
        fontsize, font,
        1, is_colorbar_on, vmax, vmin)

#%%
# 储存 U2_z0_SSI 到 txt 文件

U2_z0_SSI_full_name = "6. NLA - U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + (is_save_txt and ".txt" or ".mat")
if is_save == 1:
    U2_z0_SSI_txt_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "\\" + U2_z0_SSI_full_name
    np.savetxt(U2_z0_SSI_txt_address, U2_z0_SSI) if is_save_txt else savemat(U2_z0_SSI_txt_address, {'U':U2_z0_SSI})
    
    #%%
    #再次绘图：U2_z0_SSI_amp

    U2_z0_SSI_amp_address = location + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_amp" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            U2_z0_SSI_amp, U2_z0_SSI_amp_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, vmax, vmin)

    #再次绘图：U2_z0_SSI_phase

    U2_z0_SSI_phase_address = location + "\\" + "6.2. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_phase" + file_name_extension

    plot_2d(I2_x, I2_y, size_PerPixel, diz, 
            U2_z0_SSI_phase, U2_z0_SSI_phase_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_phase", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
            fontsize, font,
            1, is_colorbar_on, vmax, vmin)

#%%
# 储存 U2_z0_SSI 到 txt 文件

if is_save == 1:
    np.savetxt(U2_z0_SSI_full_name, U2_z0_SSI) if is_save_txt else savemat(U2_z0_SSI_full_name, {'U':U2_z0_SSI})

#%%
# 绘制 U2_z_energy 随 z 演化的 曲线
    
if is_energy_evolution_on == 1:
    
    vmax_U2_z_energy = np.max(U2_z_energy)
    vmin_U2_z_energy = np.min(U2_z_energy)
    
    U2_z_energy_address = location + "\\" + "6. U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "\\" + "6.1. NLA - " + "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_energy_evolution" + file_name_extension
    
    plot_1d(sheets_num + 1, size_PerPixel, diz, 
            U2_z_energy, U2_z_energy_address, "U2_" + str(float('%.2g' % z0)) + "mm" + "_SSI" + "_energy_evolution", 
            is_save, dpi, size_fig * 10, size_fig, 
            color_1d, ticks_num, is_title_on, is_axes_on, is_mm, 1, 
            fontsize, font, 
            vmax_U2_z_energy, vmin_U2_z_energy)
