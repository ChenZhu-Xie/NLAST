# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
from fun_img_Resize import if_image_Add_black_border
from fun_global_var import init_GLV_DICT, tree_print, init_GLV_rmw, end_AST, g_oea_vs_g_AST, \
    Get, fget, fkey, fGHU_plot_save
from fun_pump import pump_pic_or_U
from fun_linear import init_AST, init_AST_pro, ifft2

np.seterr(divide='ignore', invalid='ignore')


# %%

def define_lam1(lam1, **kwargs):
    n_name = "n"
    if "lam3" in kwargs:
        lam1 = kwargs["lam3"]
        n_name += "3"
    elif "h" in Get("ray"):  # 如果 ray 中含有 倍频 标识符
        lam1 = lam1 / 2
        n_name += "2"
    else:
        n_name += "1"
    return lam1, n_name


# %%

def gan_p_xyp(**kwargs):
    p_x, p_y = np.array([1, 0, 0]), np.array([0, 1, 0])  # 生成 3 维 笛卡尔坐标系 下，线偏振基 的 单位矢量，与 p_p 同类
    if type(kwargs.get("phi_p", 0)) == str:
        phi = float(kwargs["phi_p"]) / 180 * np.pi
    else:
        phi = kwargs.get("phi_p", 0) / 180 * np.pi
    # phi = np.pi / 2 - phi  # x上y右z里 的 右手系 转换成 x右y上z里 笛卡尔 左手 坐标系
    p_ux, p_uy = np.cos(phi), np.sin(phi)  # 笛卡尔坐标系 下的 x,y 向 复振幅 占比
    # 基于 上述 规定的 线偏振方向 的 单位矢量，所以 理应先有 基矢，再有这个比例，即 p_ux, p_uy = np.dot(p_p, p_x), np.dot(p_p, p_y)
    p_p = np.array([p_ux, p_uy, 0])
    return p_p, p_x, p_y


def gan_gp_p(g_shift, **kwargs):  # polarizer
    p_p, p_x, p_y = gan_p_xyp(**kwargs)
    if type(kwargs.get("phi_p", 0)) == str:  # 如果 是 str，则认为 不加偏振片，但初始 线偏振 从 p_y 变为 p_p 方向。
        g_p = g_shift
    else:
        g_p = g_shift * np.dot(p_p, p_y)
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
    g_a = g_oe * np.dot(E_u, p_a)
    # 不能是 p_a * E_u，得是 E_u * p_a，因为 E_u 的 最末维度 是 2，而 p_a 的 第一个维度 也是 2
    return g_a


# %%

def gan_g_eoa(g_o, g_e, E_uo, E_ue, **kwargs):
    g_oa = gan_gp_a(g_o, E_uo, **kwargs)
    g_ea = gan_gp_a(g_e, E_ue, **kwargs)
    g_a = g_ea + g_oa
    return g_a


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

    info = "AST_线性角谱"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # kwargs['ray'] = init_GLV_rmw(U_name, "~", "", "AST", **kwargs)
    init_GLV_rmw(U_name, "l", "AST", "", **kwargs)

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

    if "U" in kwargs:  # 防止对 U_amp_plot_save 造成影响
        kwargs.pop("U")

    # %% 确定 波长

    lam1, n_name = define_lam1(lam1, **kwargs)

    # %% 确定 公有参数

    args_init_AST = \
        [Ix, Iy, size_PerPixel,
         lam1, is_air, T,
         theta_x, theta_y, ]
    kwargs_init_AST = {"is_air_pump": is_air_pump, "gp": g_shift, }

    def args_U_amp_plot_save(folder_address, U, U_name):
        return [folder_address,
                # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
                U, U_name,
                Get("img_name_extension"),
                is_save_txt,
                # %%
                [], 1, size_PerPixel,
                0, dpi, Get("size_fig"),  # is_save = 1 - is_bulk 改为 不储存，因为 反正 都储存了
                # %%
                cmap_2d, ticks_num, is_contourf,
                is_title_on, is_axes_on, is_mm, 0,  # 1, 1 或 0, 0
                fontsize, font,
                # %%
                1, is_colorbar_on, 0, ]  # 折射率分布差别很小，而 is_self_colorbar = 0 只看前 3 位小数的差异，因此用自动 colorbar。

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

    # %% 折射

    from fun_os import U_dir, U_amp_plot_save, U_energy_print
    if kwargs.get("is_linear_birefringence", 0) == 1:
        # %% 起偏

        g_p, p_p = gan_gp_p(g_shift, **kwargs)

        # %% 空气中，偏振状态 与 入射方向 无关/独立，因此 无论 theta_x 怎么取，U 中所有点 偏振状态 均为 V，且 g 中 所有点的 偏振状态也 均为 V
        # 但晶体中，折射后的 偏振状态 与 g 中各点 kx,ky 对应的 入射方向 就有关了，因此得 在倒空间中 投影操作，且每个点都 分别考虑。

        kwargs["polar"] = "o"
        n1o_inc, n1o, k1o_inc, k1o, k1o_z, k1o_xy, g_o, E_uo, PG_o_vz = \
            init_AST_pro(*args_init_AST, g_p, p_p,
                         is_print,
                         is_end2=-1,
                         **kwargs_init_AST, **kwargs)

        # %%  晶体 abc 坐标系 -x y z 下的 kxy 网格上 各点的 k 单位矢量： kx 向 左 为正，ky 向 上 为正
        kwargs["polar"] = "e"
        n1e_inc, n1e, k1e_inc, k1e, k1e_z, k1e_xy, g_e, E_ue, PG_e_vz = \
            init_AST_pro(*args_init_AST, g_p, p_p,
                         is_print,
                         add_level=1, is_end2=1,
                         **kwargs_init_AST, **kwargs)

        # %% 衍射前（前端面 但 晶体内），g_o，绘图

        g_oea_vs_g_AST(g_o, g_shift)

        fGHU_plot_save(*args_fGHU_plot_save, part_z="_o", **kwargs, )

        # %% 衍射前（前端面 但 晶体内），g_e，绘图

        g_oea_vs_g_AST(g_e, g_shift)

        fGHU_plot_save(*args_fGHU_plot_save, part_z="_e", **kwargs, )

        # %% 衍射前（前端面 但 晶体内），并通过 检偏器 a（或 不加偏振片，只看 光强）后，绘图

        if type(kwargs.get("phi_a", 0)) == str:
            # 如果 传进来的 phi_a 不是数字，则说明 没加 偏振片，则 正交线偏 直接叠加后，gUH 的 相位 就 没用了；只有 gU 的 能量分布 才有用
            # 并且 二者的 的 复场 和 能量，根本不满足 傅立叶变换对 的 关系；
            g_oe_energy_add = np.abs(g_o) ** 2 + np.abs(g_e) ** 2  # 远场 平方和
            # H_energy = g_oe_energy_add / np.abs(g_shift) ** 2
            U_oe_energy_add = np.abs(ifft2(g_o)) ** 2 + np.abs(ifft2(g_e)) ** 2  # 近场 平方和

            # %%
            g_oe_energy_add_name = Get("method") + ' - ' + "g" + Get("ray") + "_oe_energy_add"
            folder_address = U_dir(g_oe_energy_add_name, is_save, **kwargs, )
            U_amp_plot_save(*args_U_amp_plot_save(folder_address, g_oe_energy_add, g_oe_energy_add_name),
                            **kwargs_U_amp_plot_save, **kwargs, )

            # %%
            # H_energy_name = g_oe_energy_add_name.replace(" g", " H")
            # folder_address = U_dir(H_energy_name, is_save, **kwargs, )
            # U_amp_plot_save(*args_U_amp_plot_save(folder_address, H_energy, H_energy_name),
            #                 **kwargs_U_amp_plot_save, **kwargs, )

            # %%
            U_oe_energy_add_name = g_oe_energy_add_name.replace(" g", " U")
            U_energy_print(U_oe_energy_add ** 0.5, U_oe_energy_add_name, is_print,
                           z=z0, **kwargs, )
            folder_address = U_dir(U_oe_energy_add_name, is_save, **kwargs, )
            U_amp_plot_save(*args_U_amp_plot_save(folder_address, U_oe_energy_add, U_oe_energy_add_name),
                            **kwargs_U_amp_plot_save, **kwargs, )
        else:
            g_a = gan_g_eoa(g_o, g_e, E_uo, E_ue, **kwargs)
            g_oea_vs_g_AST(g_a, g_shift)

            fGHU_plot_save(*args_fGHU_plot_save, part_z="_oea", **kwargs, )

        # %% 晶体内 o 光 折射率 分布

        if is_air != 1:
            no_name = n_name + "o"
            folder_address = U_dir(no_name, is_save, **kwargs, )
            U_amp_plot_save(*args_U_amp_plot_save(folder_address, n1o, no_name),
                            **kwargs_U_amp_plot_save, **kwargs, )

        # %% 晶体内 e 光 折射率 分布

        if is_air != 1:
            ne_name = n_name + "e"
            folder_address = U_dir(ne_name, is_save, **kwargs, )
            U_amp_plot_save(*args_U_amp_plot_save(folder_address, n1e, ne_name),
                            **kwargs_U_amp_plot_save, **kwargs, )

        # %% 衍射后（晶体内 后端面），o 光 绘图

        Gz_o = end_AST(z0, size_PerPixel,
                       g_o, k1o_z, PG_o_vz)

        fGHU_plot_save(*args_fGHU_plot_save, part_z="_o_z", **kwargs, )

        # %% 衍射后（晶体内 后端面），e 光 绘图

        Gz_e = end_AST(z0, size_PerPixel,
                       g_e, k1e_z, PG_e_vz)

        fGHU_plot_save(*args_fGHU_plot_save, part_z="_e_z", **kwargs, )

        # %% 衍射后（晶体内 后端面），并通过 检偏器 a（或 不加偏振片，只看 光强）后，再绘图

        if type(kwargs.get("phi_a", 0)) == str:
            # 如果 传进来的 phi_a 不是数字，则说明 没加 偏振片，则 正交线偏 oe 直接叠加后，gUH 的 相位 就 没用了；只有 gU 的 能量分布 才有用
            # 并且 二者的 的 复场 和 能量，根本不满足 傅立叶变换对 的 关系；
            G_oe_energy_add = np.abs(Gz_o) ** 2 + np.abs(Gz_e) ** 2  # 远场 平方和
            # H_energy = G_oe_energy_add / g_oe_energy_add
            U_oe_energy_add = np.abs(ifft2(Gz_o)) ** 2 + np.abs(ifft2(Gz_e)) ** 2  # 近场 平方和

            # %%
            G_oe_energy_add_name = Get("method") + ' - ' + "G" + Get("ray") + "_oe_z_energy_add"
            folder_address = U_dir(G_oe_energy_add_name, is_save, **kwargs, )
            U_amp_plot_save(*args_U_amp_plot_save(folder_address, G_oe_energy_add, G_oe_energy_add_name),
                            **kwargs_U_amp_plot_save, z=z0, **kwargs, )

            # %%
            # H_energy_name = G_oe_energy_add_name.replace(" G", " H")
            # folder_address = U_dir(H_energy_name, is_save, **kwargs, )
            # U_amp_plot_save(*args_U_amp_plot_save(folder_address, H_energy, H_energy_name),
            #                 **kwargs_U_amp_plot_save, z=z0, **kwargs, )

            # %%
            U_oe_energy_add_name = G_oe_energy_add_name.replace(" G", " U")
            U_energy_print(U_oe_energy_add ** 0.5, U_oe_energy_add_name, is_print,
                           z=z0, is_end=1, **kwargs, )
            folder_address = U_dir(U_oe_energy_add_name, is_save, **kwargs, )
            U_amp_plot_save(*args_U_amp_plot_save(folder_address, U_oe_energy_add, U_oe_energy_add_name),
                            **kwargs_U_amp_plot_save, z=z0, **kwargs, )

        else:
            Gz_a = gan_g_eoa(Gz_o, Gz_e, E_uo, E_ue, **kwargs)
            g_oea_vs_g_AST(Gz_a, g_a)

            fGHU_plot_save(*args_fGHU_plot_save, part_z="_oea_z", is_end=1, **kwargs, )

            return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U")

    else:  # 这个是 电脑 or 图片 坐标系 下的： kx 向右 为正，ky 向下 为正
        n1_inc, n1, k1_inc, k1, k1_z, k1_xy = init_AST(*args_init_AST,
                                                       **kwargs_init_AST, **kwargs)
        # print(k1_xy[:, :, 0][0])  # 这个是 电脑 or 图片 坐标系 下的： x 向右 为正，y 向下 为正
        # print(k1_xy[:, :, 1][:, 0])  # 这个是 电脑 or 图片 坐标系 下的： x 向右 为正，y 向下 为正

        # %% 绘制 折射率 分布

        if is_air != 1:
            folder_address = U_dir(n_name, is_save, **kwargs, )
            U_amp_plot_save(*args_U_amp_plot_save(folder_address, n1, n_name),
                            **kwargs_U_amp_plot_save, **kwargs, )

        # %% 后续绘图

        end_AST(z0, size_PerPixel,
                g_shift, k1_z, )

        fGHU_plot_save(*args_fGHU_plot_save, is_end=1, **kwargs, )

        return fget("U"), fget("G"), Get("ray"), Get("method_and_way"), fkey("U")


if __name__ == '__main__':
    kwargs = \
        {"U_name": "",
         "img_full_name": "Grating.png",
         "U_pixels_x": 300, "U_pixels_y": 300,
         "is_phase_only": 0,
         # %%
         "z_pump": 0,
         "is_LG": 1, "is_Gauss": 1, "is_OAM": 1,
         "l": 50, "p": 0,
         "theta_x": 0, "theta_y": 0,
         # %%
         "is_random_phase": 0,
         "is_H_l": 0, "is_H_theta": 0, "is_H_random_phase": 0,
         # %%
         "U_size": 2, "w0": 0.04,
         "z0": 10,
         # %%
         "lam1": 1.064, "is_air_pump": 1, "is_air": 2, "T": 25,
         # %%  是否 考虑 双折射（采用 普通加莱球 + 琼斯矩阵 振幅比例 和 相位延迟 的 方案，自由度 还不如 2 个 VH 标量场 叠加 这 2 个 2 维数组 的 叠加）
         # 一个是 mn + 2，另一个是 mn * 2；然而用 2 个 VH 标量场 叠加，与这里只算 1 个 标量场 并 投影到 polarizer 的基底，没什么区别，只是最后 再复数 加起来 即可。
         "is_linear_birefringence": 1,  # 这里默认 生成的 标量场的 线偏振 是 V 即 // y 的，但晶轴 不一定 // y，然后 先向 起偏器 投影，再向 晶轴 投影，最后向 检偏器 投影。
         # 是否 使用 起偏器 polarizer（0 即不使用）、若使用，请给出 其 透光方向 相对于 V (竖直 y) 方向（也即 实验室坐标系 的 +y）的 顺时针 转角 phi_p
         "phi_p": "90", "phi_a": 30,  # 是否 使用 检偏器、若使用，请给出 其相对于 V (竖直 y) 方向的 顺时针 转角 phi_a
         # %%
         "is_save": 0, "is_no_data_save": 0,
         "is_save_txt": 0, "dpi": 100,
         # %%
         "cmap_2d": 'viridis',
         # %%
         "ticks_num": 6, "is_contourf": 0,
         "is_title_on": 1, "is_axes_on": 1, "is_mm": 1,
         # %%
         "fontsize": 10,
         "font": {'family': 'serif',
                  'style': 'normal',  # 'normal', 'italic', 'oblique'
                  'weight': 'normal',
                  'color': 'black',  # 'black','gray','darkred'
                  },
         # %%
         "is_colorbar_on": 1, "is_energy": 0,
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
         "polar": "e", "ray": "1",
         }

    kwargs = init_GLV_DICT(**kwargs)
    AST(**kwargs)

    # AST(U_name="",
    #     img_full_name="Grating.png",
    #     is_phase_only=0,
    #     # %%
    #     z_pump=0,
    #     is_LG=0, is_Gauss=0, is_OAM=0,
    #     l=0, p=0,
    #     theta_x=0, theta_y=0,
    #     # %%
    #     is_random_phase=0,
    #     is_H_l=0, is_H_theta=0, is_H_random_phase=0,
    #     # %%
    #     U_size=1, w0=0.1,
    #     z0=1,
    #     # %%
    #     lam1=0.8, is_air_pump=0, is_air=0, T=25,
    #     # %%
    #     is_save=1, is_save_txt=0, dpi=100,
    #     # %%
    #     cmap_2d='viridis',
    #     # %%
    #     ticks_num=6, is_contourf=0,
    #     is_title_on=1, is_axes_on=1, is_mm=1,
    #     # %%
    #     fontsize=9,
    #     font={'family': 'serif',
    #           'style': 'normal',  # 'normal', 'italic', 'oblique'
    #           'weight': 'normal',
    #           'color': 'black',  # 'black','gray','darkred'
    #           },
    #     # %%
    #     is_colorbar_on=1, is_energy=0,
    #     # %%
    #     is_print=1,
    #     # %%
    #     root_dir=r'',
    #     border_percentage=0.1, ray="1", is_end=-1, )
