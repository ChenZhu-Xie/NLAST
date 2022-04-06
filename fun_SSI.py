# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 20:23:31 2022

@author: Xcz
"""

import numpy as np
from fun_algorithm import find_nearest
from fun_global_var import set, get

# %%
# 定义 调制区域切片厚度 的 纵向实际像素、调制区域切片厚度 的 实际纵向尺寸

def Cal_diz(deff_structure_sheet_expect, deff_structure_length_expect, size_PerPixel,
            Tz, mz,
            is_print=1, ):
    Tz_percentage = 0.1
    length_percentage = 0.01
    # %%
    if mz != 0:  # 如过你想 让结构 提供 z 向倒格矢
        if deff_structure_sheet_expect > Tz_percentage * Tz or deff_structure_sheet_expect <= 0 or (
                type(deff_structure_sheet_expect) != float and type(deff_structure_sheet_expect) != np.float64 and type(
            deff_structure_sheet_expect) != int):  # 则 deff_structure_sheet_expect 不能超过 0.1 * Tz（以保持 良好的 占空比）
            deff_structure_sheet_expect = Tz_percentage * Tz  # Unit: μm
    else:
        if deff_structure_sheet_expect > length_percentage * deff_structure_length_expect * 1000 or deff_structure_sheet_expect <= 0 or (
                type(deff_structure_sheet_expect) != float and type(deff_structure_sheet_expect) != np.float64 and type(
            deff_structure_sheet_expect) != int):  # 则 deff_structure_sheet_expect 不能超过 0.01 * deff_structure_length_expect（以保持 良好的 精度）
            deff_structure_sheet_expect = length_percentage * deff_structure_length_expect * 1000  # Unit: μm

    diz = deff_structure_sheet_expect / 1000 / size_PerPixel  # Unit: mm
    # diz = int( deff_structure_sheet_expect / 1000 / size_PerPixel )
    deff_structure_sheet = diz * size_PerPixel * 1000  # Unit: μm 调制区域切片厚度 的 实际纵向尺寸
    is_print and print("deff_structure_sheet = {} μm".format(deff_structure_sheet))

    return diz, deff_structure_sheet


# %%
# 定义 结构前端面 距离 晶体前端面 的 纵向实际像素、结构前端面 距离 晶体前端面 的 实际纵向尺寸

def Cal_Iz_frontface(diz,
                     z0_structure_frontface_expect, L0_Crystal, size_PerPixel,
                     is_print=1, ):
    # %%
    if z0_structure_frontface_expect <= 0 or z0_structure_frontface_expect >= L0_Crystal or (
            type(z0_structure_frontface_expect) != float and type(z0_structure_frontface_expect) != np.float64 and type(
        z0_structure_frontface_expect) != int):
        Iz_frontface = 0
    else:
        Iz_frontface = z0_structure_frontface_expect / size_PerPixel

    sheets_num_frontface = int(Iz_frontface // diz)
    sheet_th_frontface = sheets_num_frontface - 1 if sheets_num_frontface >= 1 else 0  # 但需要 前面一层 的 前端面 的 序数 来获取值
    Iz_frontface = sheets_num_frontface * diz
    z0_structure_frontface = Iz_frontface * size_PerPixel
    is_print and print("z0_structure_frontface = {} mm".format(z0_structure_frontface))

    return sheet_th_frontface, sheets_num_frontface, Iz_frontface, z0_structure_frontface


# %%
# 定义 调制区域 的 纵向实际像素、调制区域 的 实际纵向尺寸

def Cal_Iz_structure(diz,
                     deff_structure_length_expect, size_PerPixel,
                     is_print=1, ):
    # %%
    Iz_structure = deff_structure_length_expect / size_PerPixel  # Iz_structure 对应的是 期望的（连续的），而不是 实际的（discrete 离散的）？不，就得是离散的。
    # Iz_structure = int( deff_structure_length_expect / size_PerPixel )
    # sheets_num = Iz_structure // diz
    # Iz_structure = sheets_num * diz
    # deff_structure_length = Iz_structure * size_PerPixel # Unit: mm 调制区域 的 实际纵向尺寸
    # print("deff_structure_length = {} mm".format(deff_structure_length))

    sheets_num_structure = int(Iz_structure // diz)
    Iz_structure = sheets_num_structure * diz  # Iz_structure 对应的是 实际的（discrete 离散的），而不是 期望的（连续的）。
    deff_structure_length = Iz_structure * size_PerPixel  # Unit: mm 传播距离 = 调制区域 的 实际纵向尺寸
    # deff_structure_length = sheets_num * diz * size_PerPixel # Unit: mm 调制区域 的 实际纵向尺寸
    is_print and print("deff_structure_length = {} mm".format(deff_structure_length))

    return sheets_num_structure, Iz_structure, deff_structure_length


# %%
# 定义 结构后端面 距离 晶体前端面 的 纵向实际像素、结构后端面 距离 晶体前端面 的 实际纵向尺寸

def Cal_Iz_endface(sheets_num_frontface, sheets_num_structure,
                   diz, size_PerPixel,
                   is_print=1, ):
    # %%
    # Iz_endface = Iz_frontface + Iz_structure
    sheets_num_endface = sheets_num_frontface + sheets_num_structure
    sheet_th_endface = sheets_num_endface - 1 if sheets_num_endface >= 1 else 0  # 但需要 前面一层 的 前端面 的 序数 来获取值

    Iz_endface = sheets_num_endface * diz
    z0_structure_endface = Iz_endface * size_PerPixel
    is_print and print("z0_structure_endface = {} mm".format(z0_structure_endface))

    return sheet_th_endface, sheets_num_endface, Iz_endface, z0_structure_endface


# %%
# 定义 晶体 的 纵向实际像素、晶体 的 实际纵向尺寸

def Cal_Iz(diz,
           L0_Crystal, size_PerPixel,
           is_print=1, ):
    # %%
    Iz = L0_Crystal / size_PerPixel

    # sheets_num = int(Iz // diz) + int(not np.mod(Iz,diz) == 0) # mod(182,0.182) = 4.884981308350689e-15 != 0 也是牛逼
    sheets_num = int(Iz // diz) + int(not Iz / diz == Iz // diz)  # Iz - Iz//diz * diz 比 np.mod(Iz,diz) 好用
    # print(sheets_num)

    # Iz = sheets_num * diz
    # z0 = L0_Crystal = Iz * size_PerPixel
    is_print and print("z0 = L0_Crystal = {} mm".format(L0_Crystal))

    return sheets_num, Iz

# %%
# 定义 调制区域 的 横向实际像素、调制区域 的 实际横向尺寸

def Cal_IxIy(I1_x, I1_y,
             deff_structure_size_expect, size_PerPixel,
             is_print=1, ):
    Ix, Iy = int(deff_structure_size_expect / size_PerPixel), int(deff_structure_size_expect / size_PerPixel)
    # Ix, Iy 需要与 I1_x, I1_y 同奇偶性，这样 加边框 才好加（对称地加 而不用考虑 左右两边加的量 可能不一样）
    Ix, Iy = Ix + np.mod(I1_x - Ix, 2), Iy + np.mod(I1_y - Iy, 2)
    deff_structure_size = Ix * size_PerPixel  # Unit: mm 不包含 边框，调制区域 的 实际横向尺寸
    is_print and print("deff_structure_size = {} mm".format(deff_structure_size))

    return Ix, Iy, deff_structure_size

# %%

def slice_structure_SSI(deff_structure_sheet_expect, deff_structure_length_expect,
               Tz, mz, size_PerPixel,
               is_print):
    # %%
    # 定义 调制区域切片厚度 的 纵向实际像素、调制区域切片厚度 的 实际纵向尺寸

    diz, deff_structure_sheet = Cal_diz(deff_structure_sheet_expect, deff_structure_length_expect, size_PerPixel,
                                        Tz, mz,
                                        is_print)

    # %%
    # 定义 调制区域 的 纵向实际像素、调制区域 的 实际纵向尺寸

    sheets_num, Iz, deff_structure_length = Cal_Iz_structure(diz,
                                                             deff_structure_length_expect, size_PerPixel,
                                                             is_print)

    # %%

    Tz_unit = (Tz / 1000) / size_PerPixel

    return diz, deff_structure_sheet, sheets_num, \
           Iz, deff_structure_length, Tz_unit

# %%
# 等间距切片

def Slice_SSI(L0_Crystal, deff_structure_sheet_expect,
               z0_structure_frontface_expect, deff_structure_length_expect,
               z0_section_1_expect, z0_section_2_expect,
               Tz, mz, size_PerPixel,
               is_print):
    # %%
    # 定义 调制区域切片厚度 的 纵向实际像素、调制区域切片厚度 的 实际纵向尺寸

    diz, deff_structure_sheet = Cal_diz(deff_structure_sheet_expect, deff_structure_length_expect, size_PerPixel,
                                        Tz, mz,
                                        is_print)

    # %%
    # 定义 结构前端面 距离 晶体前端面 的 纵向实际像素、结构前端面 距离 晶体前端面 的 实际纵向尺寸

    sheet_th_frontface, sheets_num_frontface, Iz_frontface, z0_structure_frontface = Cal_Iz_frontface(diz,
                                                                                                      z0_structure_frontface_expect,
                                                                                                      L0_Crystal,
                                                                                                      size_PerPixel,
                                                                                                      is_print)

    # %%
    # 定义 结构后端面 距离 晶体前端面 的 纵向实际像素、结构后端面 距离 晶体前端面 的 实际纵向尺寸 1
    deff_structure_length, z0_structure_endface = cal_Iz_endface_1(z0_structure_frontface, deff_structure_length_expect,
                                                                   L0_Crystal,
                                                                   is_print)

    # %%
    # 定义 调制区域 的 纵向实际像素、调制区域 的 实际纵向尺寸

    sheets_num_structure, Iz_structure, deff_structure_length = Cal_Iz_structure(diz,
                                                                                 deff_structure_length,
                                                                                 size_PerPixel,
                                                                                 is_print)

    # %%
    # 定义 结构后端面 距离 晶体前端面 的 纵向实际像素、结构后端面 距离 晶体前端面 的 实际纵向尺寸 2

    sheet_th_endface, sheets_num_endface, Iz_endface, z0_structure_endface = \
        Cal_Iz_endface(sheets_num_frontface, sheets_num_structure,
                       diz, size_PerPixel,
                       is_print, )

    # %%
    # 定义 晶体 的 纵向实际像素、晶体 的 实际纵向尺寸

    sheets_num, Iz = Cal_Iz(diz,
                            L0_Crystal, size_PerPixel,
                            is_print)
    z0 = L0_Crystal

    # zj = np.zeros( (sheets_num + 1), dtype=np.float64() )
    # for i in range(sheets_num + 1):
    #     iz = i * diz
    #     zj[i] = iz * size_PerPixel
    zj = np.arange(sheets_num + 1, dtype=np.float64()) * diz * size_PerPixel
    zj[sheets_num] = Iz * size_PerPixel

    izj = zj / size_PerPixel  # 为循环 里使用
    dizj = izj[1:] - izj[:-1]  # 为循环 里使用

    # print(np.mod(Iz,diz))
    # print(Iz - Iz//diz * diz)
    leftover = Iz - Iz // diz * diz
    if leftover == 0:  # 触底反弹：如果 不剩（整除），则最后一步 保持 diz 不动，否则 沿用 leftover
        leftover = diz
    # print(leftover)

    # %%
    # 定义 需要展示的截面 1 距离晶体前端面 的 纵向实际像素、需要展示的截面 1 距离晶体前端面 的 实际纵向尺寸

    sheet_th_section_1, sheets_num_section_1, iz_1, z0_1 \
        = cal_iz_1(zj, z0_section_1_expect, size_PerPixel, is_print, )

    # %%
    # 定义 需要展示的截面 2 距离晶体后端面 的 纵向实际像素、需要展示的截面 2 距离晶体后端面 的 实际纵向尺寸

    sheet_th_section_2, sheets_num_section_2, iz_2, z0_2 \
        = cal_iz_2(zj, L0_Crystal, z0_section_2_expect, size_PerPixel, is_print, )

    #%%

    set("izj", izj)
    set("zj", zj)
    set("sheet_th_frontface", sheet_th_frontface)
    set("sheet_th_endface", sheet_th_endface)
    set("sheet_th_sec1", sheet_th_section_1)
    set("sheet_th_sec2", sheet_th_section_2)

    return diz, deff_structure_sheet, \
           sheet_th_frontface, sheets_num_frontface, Iz_frontface, z0_structure_frontface, \
           sheets_num_structure, Iz_structure, deff_structure_length, \
           sheets_num, Iz, z0, \
           dizj, izj, zj, \
           sheet_th_endface, sheets_num_endface, Iz_endface, z0_structure_endface, \
           sheet_th_section_1, sheets_num_section_1, iz_1, z0_1, \
           sheet_th_section_2, sheets_num_section_2, iz_2, z0_2


# %% 非等距切片 SSI ↓↓↓↓↓
# %% 非等距切片 SSI ↓↓↓↓↓
# %%
# 定义 结构前端面 距离 晶体前端面 的 纵向实际像素、结构前端面 距离 晶体前端面 的 实际纵向尺寸

def cal_Iz_frontface(z0_structure_frontface_expect, L0_Crystal, size_PerPixel,
                     is_print=1, ):
    if z0_structure_frontface_expect <= 0 or z0_structure_frontface_expect >= L0_Crystal or (
            type(z0_structure_frontface_expect) != float and type(z0_structure_frontface_expect) != np.float64 and type(
        z0_structure_frontface_expect) != int):
        z0_structure_frontface = 0
    else:
        z0_structure_frontface = z0_structure_frontface_expect
    Iz_frontface = z0_structure_frontface / size_PerPixel
    sheets_num_frontface = 1 if z0_structure_frontface > 0 else 0
    sheet_th_frontface = sheets_num_frontface - 1 if sheets_num_frontface >= 1 else 0

    is_print and print("z0_structure_frontface = {} mm".format(z0_structure_frontface))

    return sheet_th_frontface, sheets_num_frontface, Iz_frontface, z0_structure_frontface


# %%
# 定义 结构后端面 距离 晶体前端面 的 纵向实际像素、结构后端面 距离 晶体前端面 的 实际纵向尺寸 1

def cal_Iz_endface_1(z0_structure_frontface, deff_structure_length_expect, L0_Crystal,
                     is_print=1, ):
    z0_structure_endface = z0_structure_frontface + deff_structure_length_expect
    # print(z0_structure_endface)
    # print(deff_structure_length_expect) # 如果 deff_structure_length_expect 值不是 预期，则应该是 Info_find_contours_SHG 的 锅
    if z0_structure_endface > L0_Crystal:  # 设定 z0_structure_endface 的上限
        z0_structure_endface = L0_Crystal
        deff_structure_length = z0_structure_endface - z0_structure_frontface
    else:
        deff_structure_length = deff_structure_length_expect

    return deff_structure_length, z0_structure_endface

# %%

def cal_Iz_structure(deff_structure_length, size_PerPixel,
                     is_print=1, ):

    Iz_structure = deff_structure_length / size_PerPixel
    is_print and print("deff_structure_length = {} mm".format(deff_structure_length))

    return Iz_structure

# %%
# 定义 结构后端面 距离 晶体前端面 的 纵向实际像素、结构后端面 距离 晶体前端面 的 实际纵向尺寸 2

def cal_Iz_endface_2(sheets_num_frontface, sheets_num_structure,
                     z0_structure_endface, size_PerPixel,
                     is_print=1, ):

    # Iz_endface = Iz_frontface + Iz_structure
    sheets_num_endface = sheets_num_frontface + sheets_num_structure
    sheet_th_endface = sheets_num_endface - 1 if sheets_num_endface >= 1 else 0

    Iz_endface = z0_structure_endface / size_PerPixel
    is_print and print("z0_structure_endface = {} mm".format(z0_structure_endface))

    return sheet_th_endface, sheets_num_endface, Iz_endface


# %%
# 定义 晶体 的 纵向实际像素、晶体 的 实际纵向尺寸

def cal_Iz(sheets_num_endface, z0_structure_endface, L0_Crystal, size_PerPixel,
           is_print=1, ):
    z0 = L0_Crystal
    Iz = L0_Crystal / size_PerPixel
    sheets_num = sheets_num_endface + 1 if z0_structure_endface < L0_Crystal else sheets_num_endface

    is_print and print("z0 = L0_Crystal = {} mm".format(L0_Crystal))

    return sheets_num, Iz, z0


# %%
# 定义 调制区域切片厚度 的 纵向实际像素、调制区域切片厚度 的 实际纵向尺寸

def cal_diz(Duty_Cycle_z, Tz, Iz_structure, size_PerPixel,
            is_print=1, ):
    deff_structure_sheet = Tz / 1000
    # 不论是否 mz != 0，即 不论 结构 是否 提供 z 向倒格矢，都对 结构 分片
    # z 向无周期，分片干啥，就是为了看演化。不看演化不如直接用一步到位的 NLA 程序。
    # 看演化 也可以用 写出读入 结构版，但比较耗时，不过演化比 较均匀
    # ———— 不过通过 把这里结构覆盖整个长度，并且相邻畴不反转，但仍有周期，也可做到 相同效果。
    diz = deff_structure_sheet / size_PerPixel

    leftover = Iz_structure - Iz_structure // diz * diz
    leftover2 = leftover - leftover // (diz * Duty_Cycle_z) * (diz * Duty_Cycle_z)
    sheets_num_structure = int(Iz_structure // diz) * 2 + int(not leftover == 0) * \
                           (1 + int(not (leftover2 == 0 or leftover2 == leftover)))
    # 如果 0 < leftover2 < leftover，则 + 2，否则 + 1；如果 leftover = 0，则加 0
    if leftover == 0:  # 触底反弹：如果 不剩（整除），则最后一步 保持 diz 不动，否则 沿用 leftover
        # 如果 leftover = 0，则没有 leftover2 的事，
        leftover = diz
        leftover2 = diz * (1 - Duty_Cycle_z)  # 这一切都在保证 最后一步的步长 储存在 leftover2 中
    if leftover2 == 0:  # 如果 leftover > diz * Duty_Cycle_z，则 leftover = diz * Duty_Cycle_z + leftover2，
        # 否则 leftover = leftover2 = 小于等于 diz * Duty_Cycle_z 的值
        leftover2 = diz * Duty_Cycle_z  # 这一切都在保证 最后一步的步长 储存在 leftover2 中
        # 一共有 4 种可能：
        # leftover = 0 时为 diz * (1 - Duty_Cycle_z)
        # leftover2 > 0 即 leftover > diz * Duty_Cycle_z 时为 leftover - diz * Duty_Cycle_z
        # leftover2 == 0 时为 diz * Duty_Cycle_z
        # leftover2 < 0 即 leftover < diz * Duty_Cycle_z 时为 leftover

    is_print and print("deff_structure_sheet = {} μm".format(deff_structure_sheet * 1000))

    return sheets_num_structure, diz, deff_structure_sheet


# %%
# 生成 structure 各层 z 序列，以及 正负畴 序列信息 mj

def cal_zj_mj_structure(Duty_Cycle_z, deff_structure_sheet, sheets_num_structure, z0_structure_frontface,
                        z0_structure_endface,
                        is_stripe, mx, my, Tx, Ty, Tz, structure_xy_mode, size_PerPixel, ):
    # mj_structure = np.zeros((sheets_num_structure + 1), dtype=np.float64())
    # zj_structure = np.zeros((sheets_num_structure + 1), dtype=np.float64())

    # for j in range(sheets_num_structure + 1):
    #     if np.mod(j, 2) == 1:  # 如果 j 是奇数，则 在接下来的 deff_structure_sheet * (1 - Duty_Cycle_z) 内 输出 负畴
    #         zj_structure[j] = z0_structure_frontface + (j-1) // 2 * deff_structure_sheet + deff_structure_sheet * Duty_Cycle_z
    #         mj_structure[j] = -1
    #     else: # 如果 j 是偶数，则 在接下来的 deff_structure_sheet * Duty_Cycle_z 内 输出 正畴
    #         zj_structure[j] = z0_structure_frontface + j // 2 * deff_structure_sheet
    #         mj_structure[j] = 1

    zj_structure = z0_structure_frontface + np.arange(sheets_num_structure + 1,
                                                      dtype=np.float64()) * deff_structure_sheet / 2 \
                   - np.mod(np.arange(sheets_num_structure + 1, dtype=np.float64()), 2) * (
                           0.5 - Duty_Cycle_z) * deff_structure_sheet

    zj_structure[-1] = z0_structure_endface
    # print(zj_structure)

    Dzj_structure = zj_structure - z0_structure_frontface
    # dzj_structure = zj_structure[1:] - zj_structure[:-1] # 为了 对斜条纹时的 mj_structure 赋值
    # izj_structure = zj_structure / size_PerPixel # 为了 对斜条纹时的 mj_structure 赋值
    # dizj_structure = izj_structure[1:] - izj_structure[:-1] # 为了 对斜条纹时的 mj_structure 赋值
    # print("{} == {} ?".format(leftover2, dizj_structure[-1]))

    if is_stripe == 0:
        # mj_structure = - 2 * np.mod(np.arange(sheets_num_structure + 1, dtype=np.float64()), 2) + 1
        mj_structure = - 2 * np.mod(np.arange(sheets_num_structure + 1, dtype=np.int8()), 2) + 1
        mj_structure = mj_structure.astype(
            str)  # 字符 '-1','+1','0' 分别 表示 opposite，positive，以及 bulk，注意 astype 并不会改变 mj_structure，所以得 重新赋值给 mj_structure
    else:
        if structure_xy_mode == 'x' or structure_xy_mode == 'xy':
            xyj_structure = mx * Tx / Tz * Dzj_structure  # 本身 第一层 就不移
            # xj_structure = np.append(0, xj_structure) # 第一层 不移，其他层 才移
            # print(xj_structure)
        elif structure_xy_mode == 'y':
            xyj_structure = my * Ty / Tz * Dzj_structure

        mj_structure = (xyj_structure // size_PerPixel).astype(np.int64)
        # print(mj_structure)

    mj_structure = mj_structure.tolist()  # 转换为 list 才能储存 不同类型 的值
    # print(mj_structure)

    return zj_structure, mj_structure


# %%
# 生成 晶体内 各层 z 序列、izj、dizj，以及 正负畴 序列信息 mj

def cal_zj_izj_dizj_mj(zj_structure, mj_structure, z0_structure_frontface, z0_structure_endface, L0_Crystal,
                       size_PerPixel, ):
    # zj = np.zeros((sheets_num + 1), dtype=np.float64())
    zj = np.append(0, zj_structure) if z0_structure_frontface > 0 else zj_structure
    if z0_structure_endface < L0_Crystal: zj = np.append(zj, L0_Crystal)
    # 如果等于，就不 append 了，最后一个 z0_structure_endface 自己就是 L0_Crystal 了
    # print("{} == {} ?".format(len(zj), sheets_num + 1))
    # print(zj)

    izj = zj / size_PerPixel  # 为循环 里使用
    dizj = izj[1:] - izj[:-1]  # 为循环 里使用

    # mj = np.append('0', mj_structure) if z0_structure_frontface > 0 else mj_structure
    # if z0_structure_endface < L0_Crystal: mj = np.append(mj, '0')
    mj = mj_structure
    if z0_structure_frontface > 0: mj.insert(0, '0')  # 不像 np.array，对 list 增插 元素 之后，原 list 改变了
    if z0_structure_endface < L0_Crystal: mj.append('0')
    # print(mj,mj[0],type(mj),type(mj[0]))

    return zj, izj, dizj, mj


# %%
# 定义 需要展示的截面 1 距离晶体前端面 的 纵向实际像素、需要展示的截面 1 距离晶体前端面 的 实际纵向尺寸

def cal_iz_1(zj, z0_section_1_expect, size_PerPixel,
             is_print=1, ):
    sheets_num_section_1, z0_1 = find_nearest(zj, z0_section_1_expect)
    is_print and print("z0_section_1 = {} mm".format(z0_1))
    Iz_1 = z0_1 / size_PerPixel
    sheet_th_section_1 = sheets_num_section_1 - 1 if sheets_num_section_1 != 0 else 0

    return sheet_th_section_1, sheets_num_section_1, Iz_1, z0_1


# %%
# 定义 需要展示的截面 2 距离晶体后端面 的 纵向实际像素、需要展示的截面 2 距离晶体后端面 的 实际纵向尺寸

def cal_iz_2(zj, L0_Crystal, z0_section_2_expect, size_PerPixel,
             is_print=1, ):
    sheets_num_section_2, z0_2 = find_nearest(zj, L0_Crystal - z0_section_2_expect)
    is_print and print("z0_section_2 = {} mm".format(z0_2))
    Iz_2 = z0_2 / size_PerPixel
    sheet_th_section_2 = sheets_num_section_2 - 1 if sheets_num_section_2 >= 1 else 0

    return sheet_th_section_2, sheets_num_section_2, Iz_2, z0_2

#%%

# %%
# 非等间距切片

def slice_SSI(L0_Crystal, size_PerPixel,
               z0_structure_frontface_expect, deff_structure_length_expect,
               z0_section_1_expect, z0_section_2_expect,
               is_stripe, mx, my, Tx, Ty, Tz, Duty_Cycle_z, structure_xy_mode, 
               is_print):
    # %%
    # 定义 结构前端面 距离 晶体前端面 的 纵向实际像素、结构前端面 距离 晶体前端面 的 实际纵向尺寸

    sheet_th_frontface, sheets_num_frontface, Iz_frontface, z0_structure_frontface \
        = cal_Iz_frontface(z0_structure_frontface_expect, L0_Crystal, size_PerPixel,
                           is_print, )

    # %%
    # 定义 结构后端面 距离 晶体前端面 的 纵向实际像素、结构后端面 距离 晶体前端面 的 实际纵向尺寸 1

    deff_structure_length, z0_structure_endface \
        = cal_Iz_endface_1(z0_structure_frontface, deff_structure_length_expect, L0_Crystal,
                           is_print, )

    # %%
    # 定义 调制区域 的 纵向实际像素、调制区域 的 实际纵向尺寸

    Iz_structure = cal_Iz_structure(deff_structure_length, size_PerPixel,
                                    is_print=1, )

    # %%
    # 定义 调制区域切片厚度 的 纵向实际像素、调制区域切片厚度 的 实际纵向尺寸
    
    sheets_num_structure, diz, deff_structure_sheet \
        = cal_diz(Duty_Cycle_z, Tz, Iz_structure, size_PerPixel,
                  is_print, )

    # %%
    # 定义 结构后端面 距离 晶体前端面 的 纵向实际像素、结构后端面 距离 晶体前端面 的 实际纵向尺寸 2

    sheet_th_endface, sheets_num_endface, Iz_endface \
        = cal_Iz_endface_2(sheets_num_frontface, sheets_num_structure,
                           z0_structure_endface, size_PerPixel,
                           is_print, )

    # %%
    # 定义 晶体 的 纵向实际像素、晶体 的 实际纵向尺寸

    sheets_num, Iz, z0 \
        = cal_Iz(sheets_num_endface, z0_structure_endface, L0_Crystal, size_PerPixel,
                 is_print, )

    # %%
    # 生成 structure 各层 z 序列，以及 正负畴 序列信息 mj

    zj_structure, mj_structure \
        = cal_zj_mj_structure(Duty_Cycle_z, deff_structure_sheet, sheets_num_structure, z0_structure_frontface,
                              z0_structure_endface,
                              is_stripe, mx, my, Tx, Ty, Tz, structure_xy_mode, size_PerPixel, )

    # %%
    # 生成 晶体内 各层 z 序列、izj、dizj，以及 正负畴 序列信息 mj

    zj, izj, dizj, mj \
        = cal_zj_izj_dizj_mj(zj_structure, mj_structure, z0_structure_frontface, z0_structure_endface, L0_Crystal,
                             size_PerPixel)

    # %%
    # 定义 需要展示的截面 1 距离晶体前端面 的 纵向实际像素、需要展示的截面 1 距离晶体前端面 的 实际纵向尺寸

    sheet_th_section_1, sheets_num_section_1, Iz_1, z0_1 \
        = cal_iz_1(zj, z0_section_1_expect, size_PerPixel,
                   is_print, )

    # %%
    # 定义 需要展示的截面 2 距离晶体后端面 的 纵向实际像素、需要展示的截面 2 距离晶体后端面 的 实际纵向尺寸

    sheet_th_section_2, sheets_num_section_2, Iz_2, z0_2 \
        = cal_iz_2(zj, deff_structure_length_expect, z0_section_2_expect, size_PerPixel,
                   is_print, )

    #%%

    set("izj", izj)
    set("zj", zj)
    set("sheet_th_frontface", sheet_th_frontface)
    set("sheet_th_endface", sheet_th_endface)
    set("sheet_th_sec1", sheet_th_section_1)
    set("sheet_th_sec2", sheet_th_section_2)

    return diz, deff_structure_sheet, \
           sheet_th_frontface, sheets_num_frontface, Iz_frontface, z0_structure_frontface, \
           sheets_num_structure, Iz_structure, deff_structure_length, \
           sheets_num, Iz, z0, \
           mj, dizj, izj, zj, \
           sheet_th_endface, sheets_num_endface, Iz_endface, z0_structure_endface, \
           sheet_th_section_1, sheets_num_section_1, Iz_1, z0_1, \
           sheet_th_section_2, sheets_num_section_2, Iz_2, z0_2