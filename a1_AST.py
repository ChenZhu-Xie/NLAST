# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
from fun_img_Resize import if_image_Add_black_border
from fun_global_var import init_GLV_DICT, tree_print, init_GLV_rmw, end_AST, g_oea_vs_g_AST, \
    Get, Set, fget, fkey, fGHU_plot_save
from fun_pump import pump_pic_or_U
from fun_linear import init_AST_12oe, gan_g_p_xy

np.seterr(divide='ignore', invalid='ignore')


# %%

def define_lam_n_AST(lam1, **kwargs):
    n_name = "n"
    if "lam3" in kwargs:
        lam1 = kwargs["lam3"]
        n_name += "3"
    elif "h" in Get("ray"):  # 如果 ray 中含有 倍频 标识符
        lam1 = lam1 / 2
        n_name += "2"
    else:
        n_name += "1"
    Set("lam1", lam1)
    return lam1, n_name


# %%

def gan_p_xyp(**kwargs):
    p_x, p_y = np.array([1, 0, 0]), np.array([0, 1, 0])  # 生成 3 维 笛卡尔坐标系 下，线偏振基 的 单位矢量，与 p_p 同类
    if type(kwargs.get("phi_p", 0)) == str:
        phi = float(kwargs["phi_p"]) / 180 * np.pi
    else:
        phi = kwargs.get("phi_p", 90) / 180 * np.pi
    # phi = np.pi / 2 - phi  # x上y右z里 的 右手系 转换成 x右y上z里 笛卡尔 左手 坐标系
    p_ux, p_uy = np.cos(phi), np.sin(phi)  # 笛卡尔坐标系 下的 x,y 向 复振幅 占比
    # 基于 上述 规定的 线偏振方向 的 单位矢量，所以 理应先有 基矢，再有这个比例，即 p_ux, p_uy = np.dot(p_p, p_x), np.dot(p_p, p_y)
    p_p = np.array([p_ux, p_uy, 0])
    return p_p, p_x, p_y


def gan_p_g(phi_p_cover, **kwargs):  # 不与 kwargs 中的 phi_p 冲突
    kwargs["phi_p"] = phi_p_cover  # 覆盖 kwargs 中的 phi_p
    p_g = gan_p_xyp(**kwargs)[0]
    return p_g


def gan_g_p(g_shift, p_g=np.array([0, 1, 0]), **kwargs):  # polarizer # 默认 g 的 偏振 p_g = p_y
    p_p = gan_p_xyp(**kwargs)[0]
    g_p = g_shift * np.dot(p_p, p_g) # 默认 g 的 偏振 p_g = p_y = [0, 1, 0]
    return g_p, p_p


# %%

def gan_p_a(**kwargs):
    phi = kwargs.get("phi_a", 0) / 180 * np.pi
    # phi = np.pi / 2 - phi  # x上y右z里 的 右手系 转换成 x右y上z里 笛卡尔 左手 坐标系
    p_ux, p_uy = np.cos(phi), np.sin(phi)  # 笛卡尔坐标系 下的 x,y 向 复振幅 占比
    # 基于 上述 规定的 线偏振方向 的 单位矢量，所以 理应先有 基矢，再有这个比例，即 p_ux, p_uy = np.dot(p_p, p_x), np.dot(p_p, p_y)
    p_a = np.array([p_ux, p_uy, 0])
    return p_a


def gan_gp_a(g_oe, E_u, **kwargs):  # analyzer
    p_a = gan_p_a(**kwargs)
    g_a = g_oe * np.dot(E_u, p_a)  # E_u 投影到 p_a
    # 不能是 p_a * E_u，得是 E_u * p_a，因为 E_u 的 最末维度 是 2，而 p_a 的 第一个维度 也是 2
    return g_a


# %%

def gan_g_eoa(g_o, g_e, E_uo, E_ue, **kwargs):
    # %% 旧版
    g_oa = gan_gp_a(g_o, E_uo, **kwargs)
    g_ea = gan_gp_a(g_e, E_ue, **kwargs)
    g_a = g_ea + g_oa
    # %% 新版
    # p_a = gan_p_a(**kwargs)  # p_a 是 要输出的、xy 面 的 粗糙 检偏偏振，之后会被 外面 init_AST 使用
    # g_a = gan_g_p_xy(g_o, g_e, p_a, k_o, k_e, E_uo, E_ue)
    return g_a


# %%

def gp_p(g_shift, polar_g, **kwargs):
    if polar_g in "RrLl":
        from fun_linear import ifft2
        U_V, U_H, g_V, g_H = U_to_gU_LP(ifft2(g_shift), polar_g)
        # %% 旧版
        p_V = gan_p_g(90, **kwargs)  # 对应 V 方向 的 偏振矢量，给 g_V 用
        p_H = gan_p_g(0, **kwargs)  # 对应 H 方向 的 偏振矢量，给 g_H 用
        g_Vp, p_p = gan_g_p(g_V, p_V, **kwargs)
        g_Hp, p_p = gan_g_p(g_H, p_H, **kwargs)
        g_p = g_Vp + g_Hp
        # %% 新版
        # p_p = gan_p_xyp(**kwargs)[0]  # p_p 是 要输出的、xy 面 的 粗糙 起偏偏振，之后会被 外面 init_AST 使用
        # g_p = gan_g_p_xy(g_H, g_V, p_p)
    elif polar_g in "VvHh":
        # %% 旧版
        if polar_g in "Vv":
            p_g = gan_p_g(90, **kwargs)  # 若 g 的偏振 p_g 沿 y 方向：p_y
        elif polar_g in "Hh":  # polar_g in "Hh" or polar_g == "o"
            p_g = gan_p_g(0, **kwargs)  # 否则 默认 g 的偏振 p_g 沿 x 方向：p_x
        g_p, p_p = gan_g_p(g_shift, p_g, **kwargs)  # 朝 偏振方向 投影之后，g_p 大小改变，方向从 p_g 方向，变为 偏振片的 p_p 方向
        # %% 新版
        # if polar_g in "Vv":
        #     g_H, g_V = np.zeros((Get("Ix"), Get("Iy")), ), g_shift
        # elif polar_g in "Hh":
        #     g_H, g_V = g_shift, np.zeros((Get("Ix"), Get("Iy")), )
        # p_p = gan_p_xyp(**kwargs)[0]  # p_p 是 要输出的、xy 面 的 粗糙 起偏偏振，之后会被 外面 init_AST 使用
        # g_p = gan_g_p_xy(g_H, g_V, p_p)
    return g_p, p_p


def gan_gp_p(g_shift, polar_g, **kwargs):
    phi_p = kwargs.get("phi_p", 90)
    if type(phi_p) == str:  # 如果 是 str，则认为 不加偏振片，但 g 的 初始 线偏振 从 p_x 或 p_y 或 RL 均变为 p_p 方向，也就是 纯转成了 线偏振。
        p_g = gan_p_g(float(phi_p), **kwargs)
        g_p, p_p = gan_g_p(g_shift, p_g, **kwargs)  # 朝 偏振方向 投影之后，g_p 大小改变，方向从 p_g 方向，变为 偏振片的 p_p 方向
    else:
        g_p, p_p = gp_p(g_shift, polar_g, **kwargs)
    return g_p, p_p


def Gan_gp_p(is_HOPS, g_shift,
             U_0, U2_0, polar_2, **kwargs):  # 为了不与 kwargs 里 polar2 重复
    polar = kwargs.get("polar", "V")  # 默认 第一个 泵浦 是 竖直的
    if is_HOPS > 0 and is_HOPS < 1:
        g_p, p_p = gan_gp_p(g_shift, polar, **kwargs)
    else:
        if is_HOPS > 1 and is_HOPS < 2:
            U_V, U_H, g_V, g_H = U_12_to_gU_HOPS_CP(U_0, U2_0, polar, polar_2, **kwargs)
        elif is_HOPS > 2:
            U_V, U_H, g_V, g_H = U_12_to_gU_LP(U_0, U2_0, polar, polar_2)
        # %% 旧版
        g_Vp, p_p = gp_p(g_V, "V", **kwargs)
        g_Hp, p_p = gp_p(g_H, "H", **kwargs)
        g_p = g_Vp + g_Hp
        # %% 新版
        # p_p = gan_p_xyp(**kwargs)[0]  # p_p 是 要输出的、xy 面 的 粗糙 起偏偏振，之后会被 外面 init_AST 使用
        # g_p = gan_g_p_xy(g_H, g_V, p_p)
    return g_p, p_p


# %%

def Gan_gp_VH(is_HOPS, U_0, U2_0, polar2, **kwargs):  # 我觉得 圆偏 是个 假命题、不靠谱的基。。不然 g 的 z 分量怎么描述？
    # ...硬要描述也是可以，但还得转换为线偏基，才能用角谱描述
    polar = kwargs.get("polar", "V")  # 默认 第一个 泵浦 是 竖直的
    if is_HOPS == 0:
        U_V, U_H, g_V, g_H = U_to_gU_LP(U_0, polar)
    elif is_HOPS == 1:
        U_V, U_H, g_V, g_H = U_12_to_gU_HOPS_CP(U_0, U2_0, polar, polar2, **kwargs)
    elif is_HOPS == 2:
        U_V, U_H, g_V, g_H = U_12_to_gU_LP(U_0, U2_0, polar, polar2)
    p_V = gan_p_g(90, **kwargs)  # 对应 V 方向 的 偏振矢量，给 g_V 用
    p_H = gan_p_g(0, **kwargs)  # 对应 H 方向 的 偏振矢量，给 g_H 用
    return g_V, g_H, p_V, p_H


# %%  圆偏基 与 线偏基 （系数） 的 相互转换

# def projection_factor(target_polar_to_refer_polar):  # target_polar 朝 refer_polar 投影的 分量，'AB' 也就是 A 的 B 分量
#     VH_to_RL = 1 / 2 ** 0.5 * np.array([[1, -1j],
#                                         [1, 1j]])
#     if target_polar_to_refer_polar == 'RV':  # R 的 V 分量
#         return VH_to_RL[0, 0]
#     elif target_polar_to_refer_polar == 'RH':
#         return VH_to_RL[0, 1]
#     elif target_polar_to_refer_polar == 'LV':
#         return VH_to_RL[1, 0]
#     elif target_polar_to_refer_polar == 'LH':
#         return VH_to_RL[1, 1]
#     # %%
#     RL_to_VH = np.linalg.inv(np.array(VH_to_RL))
#     if target_polar_to_refer_polar == 'VR':
#         return RL_to_VH[0, 0]
#     elif target_polar_to_refer_polar == 'VL':
#         return RL_to_VH[0, 1]
#     elif target_polar_to_refer_polar == 'HR':
#         return RL_to_VH[1, 0]
#     elif target_polar_to_refer_polar == 'HL':
#         return RL_to_VH[1, 1]

def projection_factor(base_polar_Coeff_from_target_polar):  # target_polar 朝 refer_polar 投影的 分量，'AB' 也就是 B 的 A 分量 系数
    VH_to_RL = 1 / 2 ** 0.5 * np.array([[1, -1j],
                                        [1, 1j]])
    RL_to_VH = np.linalg.inv(np.array(VH_to_RL))
    # %%
    if base_polar_Coeff_from_target_polar == 'RV':  # V 的 R 分量
        return RL_to_VH[0, 0]
    elif base_polar_Coeff_from_target_polar == 'RH':
        return RL_to_VH[0, 1]
    elif base_polar_Coeff_from_target_polar == 'LV':
        return RL_to_VH[1, 0]
    elif base_polar_Coeff_from_target_polar == 'LH':
        return RL_to_VH[1, 1]
    # %%
    if base_polar_Coeff_from_target_polar == 'VR':  # R 的 V 分量
        return VH_to_RL[0, 0]
    elif base_polar_Coeff_from_target_polar == 'VL':
        return VH_to_RL[0, 1]
    elif base_polar_Coeff_from_target_polar == 'HR':
        return VH_to_RL[1, 0]
    elif base_polar_Coeff_from_target_polar == 'HL':
        return VH_to_RL[1, 1]
    # 按理 用 点乘 更 正宗。


# %%  基底 改变后，U 分量 在 新基底 下的 系数

def U_LP_to_LP(U, polar):  # 线偏 → 线偏，单入 双出
    if polar in "Vv":
        U_V = U
        U_H = np.zeros((U.shape[0], U.shape[1]))
    elif polar in "Hh":
        U_V = np.zeros((U.shape[0], U.shape[1]))
        U_H = U
    return U_V, U_H


def U_CP_to_CP(U, polar):  # 圆偏 → 圆偏，单入 双出
    if polar in "Rr":
        U_R = U
        U_L = np.zeros((U.shape[0], U.shape[1]))
    elif polar in "Ll":
        U_R = np.zeros((U.shape[0], U.shape[1]))
        U_L = U
    return U_R, U_L


def U_CP_to_LP(U, polar):  # 圆偏 → 线偏，单入 双出
    if polar in "Rr":
        U_V = projection_factor("VR") * U
        U_H = projection_factor("HR") * U
    if polar in "Ll":
        U_V = projection_factor("VL") * U
        U_H = projection_factor("HL") * U
    return U_V, U_H


def U_LP_to_CP(U, polar):  # 线偏 → 圆偏，单入 双出
    if polar in "Vv":
        U_R = projection_factor("RV") * U
        U_L = projection_factor("LV") * U
    if polar in "Hh":
        U_R = projection_factor("RH") * U
        U_L = projection_factor("LH") * U
    return U_R, U_L


def Reverse_U_LCP(U, polar):  # 线偏 ←→ 圆偏，单入 双出
    if polar in "RrLl":
        return U_CP_to_LP(U, polar)
    if polar in "VvHh":
        return U_LP_to_CP(U, polar)


# %% 2 个 U 合成后，总体在 线偏基 或 圆偏基 下的 2 个分量（基方向） 的 场叠加

def U_to_LP(U, polar):  # 线偏，圆偏 → 线偏，单入 双出
    if polar in "VvHh":
        U_V, U_H = U_LP_to_LP(U, polar)
    if polar in "RrLl":
        U_V, U_H = U_CP_to_LP(U, polar)
    return U_V, U_H


def U_to_CP(U, polar):  # 线偏，圆偏 → 圆偏，单入 双出
    if polar in "RrLl":
        U_R, U_L = U_CP_to_CP(U, polar)
    if polar in "VvHh":
        U_R, U_L = U_LP_to_CP(U, polar)
    return U_R, U_L


# %%

def U_12_to_LP(U1, U2, polar1, polar2):  # 线偏，圆偏 → 线偏，双入 双出
    U1_V, U1_H = U_to_LP(U1, polar1)
    U2_V, U2_H = U_to_LP(U2, polar2)
    U_V = U1_V + U2_V
    U_H = U1_H + U2_H
    return U_V, U_H


def U_12_to_CP(U1, U2, polar1, polar2):  # 线偏，圆偏 → 圆偏，双入 双出
    U1_R, U1_L = U_to_CP(U1, polar1)
    U2_R, U2_L = U_to_CP(U2, polar2)
    U_R = U1_R + U2_R
    U_L = U1_L + U2_L
    return U_R, U_L


# %%

def U_12_to_HOPS_LP(U_0, U2_0, p, p2, **kwargs):  # 系数
    Theta = kwargs.get("Theta", 0) / 180 * np.pi
    Phi = kwargs.get("Phi", 0) / 180 * np.pi
    V_factor = np.cos(Theta + np.pi / 4) * np.e ** (1j * Phi)
    H_factor = np.sin(Theta + np.pi / 4) * np.e ** (-1j * Phi)
    if p in "Vv":
        U_0 *= V_factor
    elif p in "Hh":
        U_0 *= H_factor
    if p2 in "Vv":
        U2_0 *= V_factor
    elif p2 in "Hh":
        U2_0 *= H_factor
    return U_0, U2_0


def U_12_to_HOPS_CP(U_0, U2_0, p, p2, **kwargs):  # 系数
    Theta = kwargs.get("Theta", 0) / 180 * np.pi
    Phi = kwargs.get("Phi", 0) / 180 * np.pi
    R_factor = np.cos(Theta + np.pi / 4) * np.e ** (1j * Phi)
    L_factor = np.sin(Theta + np.pi / 4) * np.e ** (-1j * Phi)
    if p in "Rr":
        U_0 *= R_factor
    elif p in "Ll":
        U_0 *= L_factor
    if p2 in "Rr":
        U2_0 *= R_factor
    elif p2 in "Ll":
        U2_0 *= L_factor
    return U_0, U2_0


# %%

def U_to_gU_LP(U_0, polar):
    from fun_linear import fft2
    U_V, U_H = U_to_LP(U_0, polar)
    g_V = fft2(U_V)
    g_H = fft2(U_H)
    return U_V, U_H, g_V, g_H


def U_to_gU_CP(U_0, polar):
    from fun_linear import fft2
    U_R, U_L = U_to_CP(U_0, polar)
    g_R = fft2(U_R)
    g_L = fft2(U_L)
    return U_R, U_L, g_R, g_L


def U_12_to_gU_LP(U_0, U2_0, polar, polar2):  # 投影
    from fun_linear import fft2
    U_V, U_H = U_12_to_LP(U_0, U2_0, polar, polar2)
    g_V = fft2(U_V)
    g_H = fft2(U_H)
    return U_V, U_H, g_V, g_H


def U_12_to_gU_CP(U_0, U2_0, polar, polar2):  # 投影
    from fun_linear import fft2
    U_R, U_L = U_12_to_CP(U_0, U2_0, polar, polar2)
    g_R = fft2(U_R)
    g_L = fft2(U_L)
    return U_R, U_L, g_R, g_L


def U_12_to_gU_HOPS_LP(U_0, U2_0, p, p2, **kwargs):  # 用 p 而不用 polar，来防止重名
    U_0, U2_0 = U_12_to_HOPS_LP(U_0, U2_0, p, p2, **kwargs)  # 系数
    return U_12_to_gU_LP(U_0, U2_0, p, p2)  # 投影


def U_12_to_gU_HOPS_CP(U_0, U2_0, p, p2, **kwargs):  # 用 p 而不用 polar，来防止重名
    U_0, U2_0 = U_12_to_HOPS_CP(U_0, U2_0, p, p2, **kwargs)  # 系数
    return U_12_to_gU_LP(U_0, U2_0, p, p2)  # 投影


# %%

def gan_nkgE_oe(g_p, p_p, is_print,
                args_init_AST, kwargs_init_AST,
                **kwargs):
    from fun_linear import init_AST_12oe
    kwargs["polar"] = "o"
    kwargs_init_AST["gp"] = g_p
    n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo = \
        init_AST_12oe(*args_init_AST, is_print,
                      p_p=p_p, is_end2=-1,  # 这里 p_ray=ray + "p" 中的 p 代表 polar
                      **kwargs_init_AST, **kwargs)
    theta_x_o = Get("theta_x")
    theta_y_o = Get("theta_y")
    # %%  晶体 abc 坐标系 -x y z 下的 kxy 网格上 各点的 k 单位矢量： kx 向 左 为正，ky 向 上 为正
    kwargs["polar"] = "e"
    kwargs_init_AST["gp"] = g_p
    n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue = \
        init_AST_12oe(*args_init_AST, is_print,
                      p_p=p_p, add_level=1,
                      **kwargs_init_AST, **kwargs)
    theta_x_e = Get("theta_x")
    theta_y_e = Get("theta_y")
    # %%
    from fun_global_var import Set  # 在 c1_SFG_NLA 中的 gan_args_SHG_oe 中的 accurate_args_SFG 中会用到，但也仅仅只是近似。
    Set("theta_x", theta_x_o)
    Set("theta_y", theta_y_o)
    Set("theta2_x", theta_x_e)
    Set("theta2_y", theta_y_e)
    return n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
           n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue


def gan_nkgE_VHoe(g_V, p_V, g_H, p_H, is_print,
                  args_init_AST, kwargs_init_AST,
                  **kwargs):
    from fun_linear import init_AST_12oe
    # %%  V 的 o 分量
    kwargs["polar"] = "o"
    kwargs_init_AST["gp"] = g_V
    n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo = \
        init_AST_12oe(*args_init_AST, is_print,
                      p_p=p_V, p_ray="V", is_end2=-1,
                      **kwargs_init_AST, **kwargs)
    theta_x_Vo = Get("theta_x")
    theta_y_Vo = Get("theta_y")
    # %%  V 的 e 分量
    kwargs["polar"] = "e"
    kwargs_init_AST["gp"] = g_V
    n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve = \
        init_AST_12oe(*args_init_AST, is_print,
                      p_p=p_V, p_ray="V", add_level=1, is_end2=-1,
                      **kwargs_init_AST, **kwargs)
    theta_x_Ve = Get("theta_x")
    theta_y_Ve = Get("theta_y")
    # %%  H 的 o 分量
    kwargs["polar"] = "o"
    kwargs_init_AST["gp"] = g_H
    n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho = \
        init_AST_12oe(*args_init_AST, is_print,
                      p_p=p_H, p_ray="H", add_level=1, is_end2=-1,
                      **kwargs_init_AST, **kwargs)
    theta_x_Ho = Get("theta_x")
    theta_y_Ho = Get("theta_y")
    # %%  H 的 e 分量
    kwargs["polar"] = "e"
    kwargs_init_AST["gp"] = g_H
    n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He = \
        init_AST_12oe(*args_init_AST, is_print,
                      p_p=p_H, p_ray="H", add_level=1,
                      **kwargs_init_AST, **kwargs)
    theta_x_He = Get("theta_x")
    theta_y_He = Get("theta_y")
    # %%
    from fun_global_var import Set  # 在 c1_SFG_NLA 中的 gan_args_SHG_VHoe 中的 accurate_args_SFG 中会用到，但也仅仅只是近似。
    Set("theta_x", (theta_x_Vo + theta_x_Ho) / 2)
    Set("theta_y", (theta_y_Vo + theta_y_Ho) / 2)
    Set("theta2_x", (theta_x_Ve + theta_x_He) / 2)
    Set("theta2_y", (theta_y_Ve + theta_y_He) / 2)
    # g_o = g_Vo + g_Ho  # 不知道 能不能 加在一起，他们的 D, k 方向一样，但 E, S 方向不一样
    # g_e = g_Ve + g_He  # 不知道 能不能 加在一起，他们的 D, k 方向一样，但 E, S 方向不一样
    return n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
           n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
           n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
           n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He


# %%

def plot_n_VHoe(n_name, is_save,
                is_add_polarizer,
                n1o, n1_Vo, n1_Ho,
                n1e, n1_Ve, n1_He,
                args_U_amp_plot_save,
                kwargs_U_amp_plot_save, **kwargs, ):
    from fun_os import U_dir, U_amp_plot_save
    kwargs['p_dir'] = 'PUMP'
    method = "DIFF"
    # %% 晶体内 o 光 折射率 分布

    if is_add_polarizer == 1:
        no_name = method + " - " + n_name + "o"
        folder_address = U_dir(no_name, is_save, **kwargs, )
        U_amp_plot_save(*args_U_amp_plot_save(folder_address, n1o, no_name),
                        **kwargs_U_amp_plot_save, **kwargs, )
    else:
        # no_name = method + " - " + n_name + "o"
        # folder_address = U_dir(no_name, is_save, **kwargs, )
        # U_amp_plot_save(*args_U_amp_plot_save(folder_address, n1_Vo, no_name),
        #                 **kwargs_U_amp_plot_save, **kwargs, )
        # %%  由于 V 到 o 的折射，和 H 到 o 的折射，均与 矢量光 的 相位部分 的 连续性 有关，即 遵循相同的 折射定律，所以 共用同一个 n 和 k 椭球
        n_Vo_name = method + " - " + n_name + "Vo"
        folder_address = U_dir(n_Vo_name, is_save, **kwargs, )
        U_amp_plot_save(*args_U_amp_plot_save(folder_address, n1_Vo, n_Vo_name),
                        **kwargs_U_amp_plot_save, **kwargs, )
        n_Ho_name = method + " - " + n_name + "Ho"
        folder_address = U_dir(n_Ho_name, is_save, **kwargs, )
        U_amp_plot_save(*args_U_amp_plot_save(folder_address, n1_Ho, n_Ho_name),
                        **kwargs_U_amp_plot_save, **kwargs, )

    # %% 晶体内 e 光 折射率 分布

    if is_add_polarizer == 1:
        ne_name = method + " - " + n_name + "e"
        folder_address = U_dir(ne_name, is_save, **kwargs, )
        U_amp_plot_save(*args_U_amp_plot_save(folder_address, n1e, ne_name),
                        **kwargs_U_amp_plot_save, **kwargs, )
    else:
        # ne_name = method + " - " + n_name + "e"
        # folder_address = U_dir(ne_name, is_save, **kwargs, )
        # U_amp_plot_save(*args_U_amp_plot_save(folder_address, n1_Ve, ne_name),
        #                 **kwargs_U_amp_plot_save, **kwargs, )
        # %%  由于 V 到 e 的折射，和 H 到 e 的折射，均与 矢量光 的 相位部分 的 连续性 有关，即 遵循相同的 折射定律，所以 共用同一个 n 和 k 椭球
        n_Ve_name = method + " - " + n_name + "Ve"
        folder_address = U_dir(n_Ve_name, is_save, **kwargs, )
        U_amp_plot_save(*args_U_amp_plot_save(folder_address, n1_Ve, n_Ve_name),
                        **kwargs_U_amp_plot_save, **kwargs, )
        n_He_name = method + " - " + n_name + "He"
        folder_address = U_dir(n_He_name, is_save, **kwargs, )
        U_amp_plot_save(*args_U_amp_plot_save(folder_address, n1_He, n_He_name),
                        **kwargs_U_amp_plot_save, **kwargs, )


# %%

def plot_n(n1, n_name, is_save,
           args_U_amp_plot_save,
           kwargs_U_amp_plot_save, **kwargs, ):
    from fun_os import U_dir, U_amp_plot_save
    if type(n1) == np.ndarray:
        folder_address = U_dir(n_name, is_save, **kwargs, )
        U_amp_plot_save(*args_U_amp_plot_save(folder_address, n1, n_name),
                        **kwargs_U_amp_plot_save, **kwargs, )


# %%

def gan_Gz_a(is_add_polarizer,
             Gz_o, Gz_e, E_uo, E_ue,
             Gz_Vo, Gz_Ve, E_u_Vo, E_u_Ve,
             Gz_Ho, Gz_He, E_u_Ho, E_u_He, **kwargs, ):
    if is_add_polarizer == 1:
        Gz_a = gan_g_eoa(Gz_o, Gz_e, E_uo, E_ue, **kwargs)
    else:
        Gz_Va = gan_g_eoa(Gz_Vo, Gz_Ve, E_u_Vo, E_u_Ve, **kwargs)
        Gz_Ha = gan_g_eoa(Gz_Ho, Gz_He, E_u_Ho, E_u_He, **kwargs)
        Gz_a = Gz_Va + Gz_Ha
    return Gz_a, Gz_Va, Gz_Ha


# %%

def plot_gan_Gz_a(is_add_polarizer,
                  g_a, g_V, g_H,
                  Gz_a, Gz_Va, Gz_Ha,
                  args_fGHU_plot_save,
                  is_Gz=0, **kwargs, ):
    if is_Gz == 1:
        part_z = "_z"
        is_end = 1
    else:
        part_z = ""
        is_end = 0

    if is_add_polarizer == 1:
        g_oea_vs_g_AST(Gz_a, g_a)
        fGHU_plot_save(*args_fGHU_plot_save, part_z="_oea" + part_z, is_end=is_end, **kwargs, )
    else:
        g_oea_vs_g_AST(Gz_Va, g_V)
        fGHU_plot_save(*args_fGHU_plot_save, part_z="_Voea" + part_z, **kwargs, )
        g_oea_vs_g_AST(Gz_Ha, g_H)
        fGHU_plot_save(*args_fGHU_plot_save, part_z="_Hoea" + part_z, **kwargs, )
        g_oea_vs_g_AST(Gz_a, Gz_a)
        fGHU_plot_save(*args_fGHU_plot_save, part_z="_oea" + part_z, is_end=is_end, **kwargs, )
        # 无法与 任何 previous 比较（不同赛道），就与自己比较


# %%

def cal_GU_oe_energy_add(is_add_polarizer, Gz_o, Gz_e,
                         Gz_Vo, Gz_Ho, Gz_Ve, Gz_He):
    from fun_linear import ifft2
    if is_add_polarizer == 1:
        G_oe_energy_add = np.abs(Gz_o) ** 2 + np.abs(Gz_e) ** 2  # 远场 平方和
        # H_energy = G_oe_energy_add / g_oe_energy_add
        U_oe_energy_add = np.abs(ifft2(Gz_o)) ** 2 + np.abs(ifft2(Gz_e)) ** 2  # 近场 平方和
    else:
        G_oe_energy_add = np.abs(Gz_Vo) ** 2 + np.abs(Gz_Ho) ** 2 + \
                          np.abs(Gz_Ve) ** 2 + np.abs(Gz_He) ** 2  # 远场 平方和
        U_oe_energy_add = np.abs(ifft2(Gz_Vo)) ** 2 + np.abs(ifft2(Gz_Ho)) ** 2 + \
                          np.abs(ifft2(Gz_Ve)) ** 2 + np.abs(ifft2(Gz_He)) ** 2  # 近场 平方和
    return G_oe_energy_add, U_oe_energy_add


# %%

def plot_GU_oe_energy_add(G_oe_energy_add, U_oe_energy_add,
                          is_save, is_print,
                          args_U_amp_plot_save,
                          kwargs_U_amp_plot_save,
                          z=0, is_end=0, **kwargs, ):
    from fun_os import U_dir, U_amp_plot_save, U_energy_print
    kwargs['p_dir'] = 'GU_oe_energy_add_XY'
    # %%

    if z == 0:
        G_oe_energy_add_name = Get("method") + ' - ' + "g" + Get("ray") + "_oe_energy_add"
    else:
        G_oe_energy_add_name = Get("method") + ' - ' + "G" + Get("ray") + "_oe_z_energy_add"
    folder_address = U_dir(G_oe_energy_add_name, is_save, **kwargs, )
    U_amp_plot_save(*args_U_amp_plot_save(folder_address, G_oe_energy_add, G_oe_energy_add_name),
                    **kwargs_U_amp_plot_save, z=z, **kwargs, )

    # %%
    # if z0 == 0:
    #     H_energy_name = G_oe_energy_add_name.replace(" g", " H")
    # else:
    #     H_energy_name = G_oe_energy_add_name.replace(" G", " H")
    # folder_address = U_dir(H_energy_name, is_save, **kwargs, )
    # U_amp_plot_save(*args_U_amp_plot_save(folder_address, H_energy, H_energy_name),
    #                 **kwargs_U_amp_plot_save, z=z0, **kwargs, )

    # %%
    if z == 0:
        U_oe_energy_add_name = G_oe_energy_add_name.replace(" g", " U")
    else:
        U_oe_energy_add_name = G_oe_energy_add_name.replace(" G", " U")
    U_energy_print(U_oe_energy_add ** 0.5, U_oe_energy_add_name, is_print,
                   z=z, is_end=is_end, **kwargs, )
    folder_address = U_dir(U_oe_energy_add_name, is_save, **kwargs, )
    U_amp_plot_save(*args_U_amp_plot_save(folder_address, U_oe_energy_add, U_oe_energy_add_name),
                    **kwargs_U_amp_plot_save, z=z, **kwargs, )


# %%

def init_locals(Str):
    Strs = Str.split(",")
    return [0] * len(Strs)  # 产生 一个 长为 len(args)，值全为 0 的 列表，然后 解包 给外面 赋值


# %%

def gan_gpnkE_VHoe_xyzinc_AST(is_birefringence_deduced, is_air,
                              is_add_polarizer, is_HOPS,
                              is_save, is_print, n_name,
                              g_shift, U_0, U2_0, polar_2,  # 防重名 polar_2
                              args_init_AST, args_U_amp_plot_save,
                              kwargs_init_AST, kwargs_U_amp_plot_save,
                              is_plot_n=1, **kwargs):
    g_p, p_p, g_V, g_H, p_V, p_H, \
    n1_inc, n1, k1_inc, k1, k1_z, k1_xy, E1_u, \
    n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
    n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue, \
    n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
    n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
    n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
    n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He = \
        init_locals("g_p, p_p, g_V, g_H, p_V, p_H, \
            n1_inc, n1, k1_inc, k1, k1_z, k1_xy, E1_u, \
            n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
            n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue, \
            n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
            n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
            n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
            n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He")
    # 主要是 is_add_polarizer 和 def 导致的，有些变量 没声明，却在 def 的 形参中 出现了，以致于 实参 在用到时 报错
    # 其实就是 把 pycharm 所提示的 “可能在赋值前引用” 的 局部变量 先赋好值

    if is_birefringence_deduced == 1:  # 考虑 偏振态 的 条件；is_air == 1 时 也可以 有偏振态，与是否 所处介质 无关
        # %% 起偏

        # if is_HOPS == 0:
        #     g_p, p_p = gan_gp_p(g_shift, **kwargs)
        # elif is_HOPS >= 1:  # 起偏器 不再有用：因为 已经能模拟所有矢量光了，何必再 塌缩为 线偏，降 2 个维度 呢？emm，打脸了
        #     if type(is_HOPS) != int:  # 如果 is_HOPS 不是 0，又不是整数，则再给 双泵浦 安排上 起偏，即 投影到 polarizer，但这样类似 单泵浦
        #         g_Hp, p_p = gan_gp_p(g_H, **kwargs)
        #         g_Vp, p_p = gan_gp_p(g_V, **kwargs)
        #         g_p = g_Hp + g_Vp
        #     else:
        #         p_H = gan_p_g(0, **kwargs)  # 对应 H 方向 的 偏振矢量，给 g_H 用
        #         p_V = gan_p_g(90, **kwargs)  # 对应 H 方向 的 偏振矢量，给 g_H 用

        if is_add_polarizer == 1:
            g_p, p_p = Gan_gp_p(is_HOPS, g_shift,
                                U_0, U2_0, polar_2, **kwargs)
        else:
            g_V, g_H, p_V, p_H = Gan_gp_VH(is_HOPS, U_0, U2_0, polar_2, **kwargs)

        # %% 空气中，偏振状态 与 入射方向 无关/独立，因此 无论 theta_x 怎么取，U 中所有点 偏振状态 均为 V，且 g 中 所有点的 偏振状态也 均为 V
        # 但晶体中，折射后的 偏振状态 与 g 中各点 kx,ky 对应的 入射方向 就有关了，因此得 在倒空间中 投影操作，且每个点都 分别考虑。
        if is_add_polarizer == 1:
            if "polar2" in kwargs: kwargs.pop("polar2")
            n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
            n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue \
                = gan_nkgE_oe(g_p, p_p, is_print,
                              args_init_AST, kwargs_init_AST, **kwargs)
        else:
            if "polar2" in kwargs: kwargs.pop("polar2")
            n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
            n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
            n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
            n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He = \
                gan_nkgE_VHoe(g_V, p_V, g_H, p_H, is_print,
                              args_init_AST, kwargs_init_AST, **kwargs)

        # %% 晶体内 oe 光 折射率 分布

        if is_plot_n == 1:
            plot_n_VHoe(n_name, is_save,
                        is_add_polarizer,
                        n1o, n1_Vo, n1_Ho,
                        n1e, n1_Ve, n1_He,
                        args_U_amp_plot_save,
                        kwargs_U_amp_plot_save, **kwargs, )

    else:  # 这个是 电脑 or 图片 坐标系 下的： kx 向右 为正，ky 向下 为正
        # n1_inc, n1, k1_inc, k1, k1_z, k1_xy = \
        #     init_AST(*args_init_AST,
        #              **kwargs_init_AST, **kwargs)

        n1_inc, n1, k1_inc, k1, k1_z, k1_xy, g_p, E1_u = \
            init_AST_12oe(*args_init_AST, is_print,  # p_ray=kwargs.get("polar", "e"), 或不加（即 p_ray=""），表示 无双折射
                          **kwargs_init_AST, **kwargs)
        # print(k1_xy[:, :, 0][0])  # 这个是 电脑 or 图片 坐标系 下的： x 向右 为正，y 向下 为正
        # print(k1_xy[:, :, 1][:, 0])  # 这个是 电脑 or 图片 坐标系 下的： x 向右 为正，y 向下 为正

        # %% 绘制 折射率 分布

        if is_plot_n == 1:
            plot_n(n1, n_name, is_save,
                   args_U_amp_plot_save,
                   kwargs_U_amp_plot_save, **kwargs, )

    E1_u = kwargs.get("E1_u", E1_u)  # 允许 被外界传入的 同名 关键字参数 覆盖
    g_o, E_uo = kwargs.get("g_o", g_o), kwargs.get("E_uo", E_uo)
    g_e, E_ue = kwargs.get("g_e", g_e), kwargs.get("E_ue", E_ue)
    g_Vo, E_u_Vo = kwargs.get("g_Vo", g_Vo), kwargs.get("E_u_Vo", E_u_Vo)
    g_Ve, E_u_Ve = kwargs.get("g_Ve", g_Ve), kwargs.get("E_u_Ve", E_u_Ve)
    g_Ho, E_u_Ho = kwargs.get("g_Ho", g_Ho), kwargs.get("E_u_Ho", E_u_Ho)
    g_He, E_u_He = kwargs.get("g_He", g_He), kwargs.get("E_u_He", E_u_He)

    return g_p, p_p, g_V, g_H, p_V, p_H, \
           n1_inc, n1, k1_inc, k1, k1_z, k1_xy, E1_u, \
           n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
           n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue, \
           n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
           n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
           n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
           n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He


# %%

def AST(U_name="",
        img_full_name="Grating.png",
        is_phase_only=0,
        # %%
        z_pump=0,
        is_LG=0, is_Gauss=0, is_OAM=0,
        l=0, p=0,
        theta_x=0, theta_y=0,
        # %%
        is_random_phase=0,
        is_H_l=0, is_H_theta=0, is_H_random_phase=0,
        # %%
        U_size=1, w0=0.3,
        z0=1,
        # %%
        lam1=0.8, is_air_pump=0, is_air=0, T=25,
        # %%
        is_save=0, is_save_txt=0, dpi=100,
        # %%
        cmap_2d='viridis',
        # %%
        ticks_num=6, is_contourf=0,
        is_title_on=1, is_axes_on=1, is_mm=1,
        # %%
        fontsize=9,
        font={'family': 'serif',
              'style': 'normal',  # 'normal', 'italic', 'oblique'
              'weight': 'normal',
              'color': 'black',  # 'black','gray','darkred'
              },
        # %%
        is_colorbar_on=1, is_energy=0,
        # %%
        is_print=1,
        # %%
        **kwargs, ):
    # %%

    if_image_Add_black_border(U_name, img_full_name,
                              __name__ == "__main__", is_print, **kwargs, )

    # %%
    is_HOPS = kwargs.get("is_HOPS_AST", 0)
    is_twin_pump_degenerate = int(is_HOPS >= 1)  # is_HOPS == 0.x 的情况 仍是单泵浦
    is_single_pump_birefringence = int(0 <= is_HOPS < 1 and kwargs.get("polar", "e") in "VvHhRrLl")
    is_birefringence_deduced = int(is_twin_pump_degenerate == 1 or is_single_pump_birefringence == 1)  # 等价于 is_HOPS > 0
    is_add_polarizer = int(is_HOPS > 0 and type(is_HOPS) != int)  # 等价于 is_birefringence_deduced == 1 and ...
    is_add_analyzer = int(type(kwargs.get("phi_a", 0)) != str)
    # %%
    U2_name = kwargs.get("U2_name", U_name)
    img2_full_name = kwargs.get("img2_full_name", img_full_name)
    is_phase_only_2 = kwargs.get("is_phase_only_2", is_phase_only)
    # %%
    z_pump2 = kwargs.get("z_pump2", z_pump)
    is_LG_2 = kwargs.get("is_LG_2", is_LG)
    is_Gauss_2 = kwargs.get("is_Gauss_2", is_Gauss)
    is_OAM_2 = kwargs.get("is_OAM_2", is_OAM)
    # %%
    l2 = kwargs.get("l2", l)
    p2 = kwargs.get("p2", p)
    theta2_x = kwargs.get("theta2_x", theta_x) if is_HOPS == 2 else theta_x  # 只有是 2 时，才能自由设定 theta2_x
    theta2_y = kwargs.get("theta2_y", theta_y) if is_HOPS == 2 else theta_y  # 只有是 2 时，才能自由设定 theta2_y
    # %%
    is_random_phase_2 = kwargs.get("is_random_phase_2", is_random_phase)
    is_H_l2 = kwargs.get("is_H_l2", is_H_l)
    is_H_theta2 = kwargs.get("is_H_theta2", is_H_theta)
    is_H_random_phase_2 = kwargs.get("is_H_random_phase_2", is_H_random_phase)
    # %%
    w0_2 = kwargs.get("w0_2", w0)
    # lam2 = kwargs.get("lam2", lam1)
    lam2 = lam1
    is_air_pump2 = kwargs.get("is_air_pump2", is_air_pump)
    T2 = kwargs.get("T2", T)
    polar2 = kwargs.get("polar2", 'H')
    # %%
    if is_twin_pump_degenerate == 1:
        # %%
        pump2_keys = kwargs["pump2_keys"]
        # %%
        [kwargs.pop(key) for key in kwargs["pump2_keys"]]  # 及时清理 kwargs ，尽量 保持 其干净
        kwargs.pop("pump2_keys")  # 这个有点意思， "pump2_keys" 这个键本身 也会被删除。

    # %%

    info = "AST_线性角谱"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # kwargs['ray'] = init_GLV_rmw(U_name, "~", "", "AST", **kwargs)
    init_GLV_rmw(U_name, "l", "AST", "", **kwargs)

    # %% 确定 波长（得在 所有用 lam1 的 函数 之前）

    lam1, n_name = define_lam_n_AST(lam1, **kwargs)

    # %%

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, \
    U_0, g_shift = pump_pic_or_U(U_name,
                                 img_full_name,
                                 is_phase_only,
                                 # %%
                                 z_pump,
                                 is_LG, is_Gauss, is_OAM,
                                 l, p,
                                 theta_x, theta_y,
                                 # %%
                                 is_random_phase,
                                 is_H_l, is_H_theta, is_H_random_phase,
                                 # %%
                                 U_size, w0,
                                 # %%
                                 lam1, is_air_pump, T,
                                 # %%
                                 is_save, is_save_txt, dpi,
                                 cmap_2d,
                                 # %%
                                 ticks_num, is_contourf,
                                 is_title_on, is_axes_on, is_mm,
                                 # %%
                                 fontsize, font,
                                 # %%
                                 is_colorbar_on, is_energy,
                                 # %%
                                 is_print,
                                 # %%
                                 ray_pump='1', **kwargs, )

    # %%

    if is_twin_pump_degenerate == 1:
        from fun_pump import pump_pic_or_U2
        U2_0, g2 = pump_pic_or_U2(U2_name,
                                  img2_full_name,
                                  is_phase_only_2,
                                  # %%
                                  z_pump2,
                                  is_LG_2, is_Gauss_2, is_OAM_2,
                                  l2, p2,
                                  theta2_x, theta2_y,
                                  # %%
                                  is_random_phase_2,
                                  is_H_l2, is_H_theta2, is_H_random_phase_2,
                                  # %%
                                  U_size, w0_2,
                                  # %%
                                  lam2, is_air_pump, T,
                                  polar2,
                                  # %%
                                  is_save, is_save_txt, dpi,
                                  # %%
                                  ticks_num, is_contourf,
                                  is_title_on, is_axes_on, is_mm,
                                  # %%
                                  fontsize, font,
                                  # %%
                                  is_colorbar_on, is_energy,
                                  # %%
                                  is_print,
                                  # %%
                                  ray_pump='2', **kwargs, )
    else:
        U2_0, g2 = 0, 0  # 之后总会 引用到，所以这里 先在 locals() 里加上

    # %%

    if "U" in kwargs:  # 防止对 U_amp_plot_save 造成影响
        kwargs.pop("U")

    # %% 确定 公有参数

    args_init_AST = \
        [Ix, Iy, size_PerPixel,
         lam1, is_air, T,
         theta_x, theta_y, ]
    kwargs_init_AST = {"is_air_pump": is_air_pump, "gp": g_shift, }

    def args_U_amp_plot_save(folder_address, U, U_name):
        return [U, U_name,
                [], folder_address,
                Get("img_name_extension"), is_save_txt,
                # %%
                size_PerPixel, dpi, Get("size_fig"),  # is_save = 1 - is_bulk 改为 不储存，因为 反正 都储存了
                # %%
                cmap_2d, ticks_num, is_contourf,
                is_title_on, is_axes_on, is_mm,
                fontsize, font,
                # %%
                is_colorbar_on, is_save,
                1, 0, 1, 0, ]  # 折射率分布差别很小，而 is_self_colorbar = 0 只看前 3 位小数的差异，因此用自动 colorbar。

    kwargs_U_amp_plot_save = {"suffix": ""}

    args_fGHU_plot_save = \
        [0,  # 默认 全自动 is_auto = 1
         img_name_extension, is_print,
         # %%
         [], 1, size_PerPixel,
         is_save, is_save_txt, dpi, size_fig,
         # %%
         "b", cmap_2d,
         ticks_num, is_contourf,
         is_title_on, is_axes_on, is_mm,
         fontsize, font,
         # %%
         is_colorbar_on, is_energy,
         z0, ]

    plot_group_AST = kwargs.get("plot_group_AST", "")

    g_a, g_Va, g_Ha, \
    Gz_o, Gz_Vo, Gz_Ho, \
    Gz_e, Gz_Ve, Gz_He, \
    Gz_a, Gz_Va, Gz_Ha = \
        init_locals("g_a, g_Va, g_Ha, \
                    Gz_o, Gz_Vo, Gz_Ho, \
                    Gz_e, Gz_Ve, Gz_He, \
                    Gz_a, Gz_Va, Gz_Ha")
    # 主要是 is_add_polarizer 和 def 导致的，有些变量 没声明，却在 def 的 形参中 出现了，以致于 实参 在用到时 报错
    # 其实就是 把 pycharm 所提示的 “可能在赋值前引用” 的 局部变量 先赋好值

    # %% 折射

    g_p, p_p, g_V, g_H, p_V, p_H, \
    n1_inc, n1, k1_inc, k1, k1_z, k1_xy, E1_u, \
    n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, \
    n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue, \
    n1_Vo_inc, n1_Vo, k1_Vo_inc, k1_Vo, k1_Vo_z, k1_Vo_xy, g_Vo, E_u_Vo, \
    n1_Ve_inc, n1_Ve, k1_Ve_inc, k1_Ve, k1_Ve_z, k1_Ve_xy, g_Ve, E_u_Ve, \
    n1_Ho_inc, n1_Ho, k1_Ho_inc, k1_Ho, k1_Ho_z, k1_Ho_xy, g_Ho, E_u_Ho, \
    n1_He_inc, n1_He, k1_He_inc, k1_He, k1_He_z, k1_He_xy, g_He, E_u_He = \
        gan_gpnkE_VHoe_xyzinc_AST(is_birefringence_deduced, is_air,
                                  is_add_polarizer, is_HOPS,
                                  is_save, is_print, n_name,
                                  g_shift, U_0, U2_0, polar2,
                                  args_init_AST, args_U_amp_plot_save,
                                  kwargs_init_AST, kwargs_U_amp_plot_save,
                                  is_plot_n=1, **kwargs)

    if is_birefringence_deduced == 1:  # 考虑 偏振态 的 条件；is_air == 1 时 也可以 有偏振态，与是否 所处介质 无关
        # %% 衍射前（前端面 但 晶体内），g_o，绘图

        if "o" in plot_group_AST or "f" in plot_group_AST:
            if is_add_polarizer == 1:
                g_oea_vs_g_AST(g_o, g_p)
                fGHU_plot_save(*args_fGHU_plot_save, part_z="_o", **kwargs, )
            else:
                g_oea_vs_g_AST(g_Vo, g_V)
                fGHU_plot_save(*args_fGHU_plot_save, part_z="_Vo", **kwargs, )
                g_oea_vs_g_AST(g_Ho, g_H)
                fGHU_plot_save(*args_fGHU_plot_save, part_z="_Ho", **kwargs, )

        # %% 衍射前（前端面 但 晶体内），g_e，绘图

        if "e" in plot_group_AST or "f" in plot_group_AST:
            if is_add_polarizer == 1:
                g_oea_vs_g_AST(g_e, g_p)
                fGHU_plot_save(*args_fGHU_plot_save, part_z="_e", **kwargs, )
            else:
                g_oea_vs_g_AST(g_Ve, g_V)
                fGHU_plot_save(*args_fGHU_plot_save, part_z="_Ve", **kwargs, )
                g_oea_vs_g_AST(g_He, g_H)
                fGHU_plot_save(*args_fGHU_plot_save, part_z="_He", **kwargs, )

        # %% 衍射前（前端面 但 晶体内），并通过 检偏器 a（或 不加偏振片，只看 光强）后，绘图

        if is_add_analyzer == 1:
            g_a, g_Va, g_Ha = gan_Gz_a(is_add_polarizer,
                                       g_o, g_e, E_uo, E_ue,
                                       g_Vo, g_Ve, E_u_Vo, E_u_Ve,
                                       g_Ho, g_He, E_u_Ho, E_u_He, **kwargs, )

            if "m" in plot_group_AST or "f" in plot_group_AST:
                plot_gan_Gz_a(is_add_polarizer,
                              g_a, g_V, g_H,
                              g_a, g_Va, g_Ha,
                              args_fGHU_plot_save,
                              is_Gz=0, **kwargs, )
        else:
            if "m" in plot_group_AST or "f" in plot_group_AST:
                # 如果 传进来的 phi_a 不是数字，则说明 没加 偏振片，则 正交线偏 直接叠加后，gUH 的 相位 就 没用了；只有 gU 的 能量分布 才有用
                # 并且 二者的 的 复场 和 能量，根本不满足 傅立叶变换对 的 关系；
                g_oe_energy_add, U_oe_energy_add = \
                    cal_GU_oe_energy_add(is_add_polarizer, g_o, g_e,
                                         g_Vo, g_Ho, g_Ve, g_He)

                # %%
                plot_GU_oe_energy_add(g_oe_energy_add, U_oe_energy_add,
                                      is_save, is_print,
                                      args_U_amp_plot_save,
                                      kwargs_U_amp_plot_save,
                                      **kwargs, )

        # %% 衍射后（晶体内 后端面），o 光 绘图

        if is_add_polarizer == 1:
            Gz_o = end_AST(z0, size_PerPixel,
                           g_o, k1o_z)
            if "o" in plot_group_AST or "b" in plot_group_AST:
                fGHU_plot_save(*args_fGHU_plot_save, part_z="_o_z", **kwargs, )
        else:
            Gz_Vo = end_AST(z0, size_PerPixel,
                            g_Vo, k1_Vo_z)
            if "o" in plot_group_AST or "b" in plot_group_AST:
                fGHU_plot_save(*args_fGHU_plot_save, part_z="_Vo_z", **kwargs, )
            Gz_Ho = end_AST(z0, size_PerPixel,
                            g_Ho, k1_Ho_z)
            if "o" in plot_group_AST or "b" in plot_group_AST:
                fGHU_plot_save(*args_fGHU_plot_save, part_z="_Ho_z", **kwargs, )

        # %% 衍射后（晶体内 后端面），e 光 绘图

        if is_add_polarizer == 1:
            Gz_e = end_AST(z0, size_PerPixel,
                           g_e, k1e_z)
            if "e" in plot_group_AST or "b" in plot_group_AST:
                fGHU_plot_save(*args_fGHU_plot_save, part_z="_e_z", **kwargs, )
        else:
            Gz_Ve = end_AST(z0, size_PerPixel,
                            g_Ve, k1_Ve_z)
            if "e" in plot_group_AST or "b" in plot_group_AST:
                fGHU_plot_save(*args_fGHU_plot_save, part_z="_Ve_z", **kwargs, )
            Gz_He = end_AST(z0, size_PerPixel,
                            g_He, k1_He_z)
            if "e" in plot_group_AST or "b" in plot_group_AST:
                fGHU_plot_save(*args_fGHU_plot_save, part_z="_He_z", **kwargs, )

        # %% 衍射后（晶体内 后端面），并通过 检偏器 a（或 不加偏振片，只看 光强）后，再绘图

        if is_add_analyzer == 1:
            Gz_a, Gz_Va, Gz_Ha = gan_Gz_a(is_add_polarizer,
                                          Gz_o, Gz_e, E_uo, E_ue,
                                          Gz_Vo, Gz_Ve, E_u_Vo, E_u_Ve,
                                          Gz_Ho, Gz_He, E_u_Ho, E_u_He, **kwargs, )

            if "m" in plot_group_AST or "b" in plot_group_AST \
                    or __name__ != "__main__" or "r" in plot_group_AST:  # 必然要 plot 的（也不一定），就不设条件了
                # 加 2 条：如果 该程序 被别的程序 调用，则必 plot 这最后的结果；或者 "r" 即 result 在 plot_group_AST 中
                plot_gan_Gz_a(is_add_polarizer,
                              g_a, g_V, g_H,
                              Gz_a, Gz_Va, Gz_Ha,
                              args_fGHU_plot_save,
                              is_Gz=1, **kwargs, )

        else:
            if "m" in plot_group_AST or "b" in plot_group_AST or "r" in plot_group_AST:  # 必然要 plot 的（也不一定），就不设条件了
                # 如果 传进来的 phi_a 不是数字，则说明 没加 偏振片，则 正交线偏 oe 直接叠加后，gUH 的 相位 就 没用了；只有 gU 的 能量分布 才有用
                # 并且 二者的 的 复场 和 能量，根本不满足 傅立叶变换对 的 关系；
                G_oe_energy_add, U_oe_energy_add = \
                    cal_GU_oe_energy_add(is_add_polarizer, Gz_o, Gz_e,
                                         Gz_Vo, Gz_Ho, Gz_Ve, Gz_He)

                # %%
                plot_GU_oe_energy_add(G_oe_energy_add, U_oe_energy_add,
                                      is_save, is_print,
                                      args_U_amp_plot_save,
                                      kwargs_U_amp_plot_save,
                                      z=z0, is_end=1, **kwargs, )

        return Gz_o, Gz_e, E_uo, E_ue, \
               Gz_Vo, Gz_Ve, E_u_Vo, E_u_Ve, \
               Gz_Ho, Gz_He, E_u_Ho, E_u_He, \
               Get("ray"), Get("method_and_way"), fkey("U")  # 加不加 起偏 或 检偏 器，都可能 会有 VH 两个分量，所以 将 return 写在外面

    else:
        # %% 后续绘图

        end_AST(z0, size_PerPixel,
                g_shift, k1_z, )

        fGHU_plot_save(*args_fGHU_plot_save, is_end=1, **kwargs, )

        return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U")


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",
         "img_full_name": "lena1.png",
         "U_pixels_x": 300, "U_pixels_y": 300,
         "is_phase_only": 0,
         # %%
         "z_pump": -5,
         "is_LG": 1, "is_Gauss": 1, "is_OAM": 1,
         "l": 50, "p": 0,
         "theta_x": 0, "theta_y": 0,
         # %%
         "is_random_phase": 0,
         "is_H_l": 0, "is_H_theta": 0, "is_H_random_phase": 0,
         # %%
         "U_size": 2, "w0": 0.05,
         "z0": 10,
         # %%
         "lam1": 1.064, "is_air_pump": 1, "is_air": 2, "T": 25,
         # %%  控制 单双泵浦 和 绘图方式："is_HOPS": 0 代表 无双折射，即 "is_linear_birefringence": 0
         "is_HOPS_AST": 1,  # 0.x 代表 单泵浦，1.x 代表 高阶庞加莱球，2.x 代表 最广义情况：2 个 线偏 标量场 叠加；这些都是在 左手系下，且都是 线偏基
         "Theta": 0, "Phi": 0,  # 是否 采用 高阶加莱球、若采用，请给出 极角 和 方位角
         # 是否 使用 起偏器（"is_HOPS": 整数 即不使用）、若使用，请给出 其相对于 H (水平 x) 方向的 逆时针 转角 phi_p
         "phi_p": "45", "phi_a": "45",  # 是否 使用 检偏器（"phi_a": str 则不使用）、若使用，请给出 其相对于 H (水平 x) 方向的 逆时针 转角 phi_a
         "plot_group_AST": "r",  # m 代表 oe 的 mix，o,e 代表 ~，fb 代表 frontface / backface
         # %%
         "is_save": 0, "is_no_data_save": 0,
         "is_save_txt": 0, "dpi": 100,
         # %%
         "cmap_2d": 'viridis',
         # %%
         "ticks_num": 6, "is_contourf": 0,
         "is_title_on": 1, "is_axes_on": 1, "is_mm": 1,
         # %%
         "fontsize": 10.0,
         "font": {'family': 'serif',
                  'style': 'normal',  # 'normal', 'italic', 'oblique'
                  'weight': 'normal',
                  'color': 'black',  # 'black','gray','darkred'
                  },
         # %%
         "is_colorbar_on": 1, "is_energy": 1,
         # %%
         "is_print": 1,
         # %% 该程序 作为 主入口时 -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "border_percentage": 0.1, "is_end": -1,
         # %%
         "theta_z": 90, "phi_z": 90, "phi_c": 23.7,
         # KTP 50 度 ：deff 最高： 90, ~, 24.3，（24.3 - 2002, 25.3 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.9 - 2000）
         # KTP 25 度 ：deff 最高： 90, ~, 23.7，（23.7 - 2002, 24.8 - 2000）
         #                1994 ：68.8, ~, 90，（68.8 - 2002, 68.7 - 2000）
         # LN 25 度 ：90, ~, ~
         "polar": "R", "ray": "1",
         }

    if kwargs.get("is_HOPS_AST", 0) >= 1:  # 如果 is_HOPS >= 1，则 默认 双泵浦 is_twin_pumps == 1
        pump2_kwargs = {
            "U2_name": "",
            "img2_full_name": "spaceship.png",
            "is_phase_only_2": 0,
            # %%
            "z_pump2": -5,
            "is_LG_2": 1, "is_Gauss_2": 1, "is_OAM_2": 1,
            "l2": -50, "p2": 0,
            "theta2_x": 0, "theta2_y": 0,
            # %%
            "is_random_phase_2": 0,
            "is_H_l2": 0, "is_H_theta2": 0, "is_H_random_phase_2": 0,
            # %%
            "w0_2": 0.05,
            # %%
            "lam2": 1.064, "is_air_pump2": 1, "T2": 25,
            "polar2": 'L',
            # 有双泵浦，则必然考虑偏振、起偏，和检偏，且原 "polar2": 'e'、 "polar": "e" 已再不起作用
            # 取而代之的是，既然原 "polar": "e" 不再 work 但还存在，就不能浪费 它的存在，让其 重新规定 第一束光
            # 偏振方向 为 "VHRL" 中的一个，而不再规定其 极化方向 为 “oe” 中的一个；这里 第二束 泵浦的 偏振方向 默认与之 正交，因而可以 不用填写
            # 但仍然可以 规定第 2 个泵浦 为其他偏振，比如 2 个 同向 线偏叠加，也就是 2 个图叠加，或者 一个 线偏基，另一个 圆偏基
        }
        pump2_kwargs.update({"pump2_keys": list(pump2_kwargs.keys())})
        # Object of type dict_keys is not JSON serializable，所以 得转为 list
        kwargs.update(pump2_kwargs)

    kwargs = init_GLV_DICT(**kwargs)
    AST(**kwargs)
