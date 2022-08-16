# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

from fun_global_var import init_GLV_DICT
from a2_AST_EVV import AST_EVV

if __name__ == '__main__':
    kwargs = \
        {"U_name": "",
         "img_full_name": "lena1.png",
         "U_pixels_x": 500, "U_pixels_y": 500,
         "is_phase_only": 0,
         # %%
         "z_pump": -5,
         "is_LG": 1, "is_Gauss": 1, "is_OAM": 1,
         "l": -50, "p": 0,
         "theta_x": 0, "theta_y": 0,
         # %%
         "is_random_phase": 0,
         "is_H_l": 0, "is_H_theta": 0, "is_H_random_phase": 0,
         # %%
         "U_size": 1, "w0": 0.05,
         "z0": 10,
         # %%  控制 单双泵浦 和 绘图方式："is_HOPS": 0 代表 无双折射，即 "is_linear_birefringence": 0
         "is_HOPS_AST": 1,  # 0.x 代表 单泵浦，1.x 代表 高阶庞加莱球，2.x 代表 最广义情况：2 个 线偏 标量场 叠加；这些都是在 左手系下，且都是 线偏基
         "Theta": 0, "Phi": -45,  # 是否 采用 高阶加莱球、若采用，请给出 极角 和 方位角
         # 是否 使用 起偏器（"is_HOPS": 整数 即不使用）、若使用，请给出 其相对于 H (水平 x) 方向的 逆时针 转角 phi_p
         "phi_p": "45", "phi_a": "45",  # 是否 使用 检偏器（"phi_a": str 则不使用）、若使用，请给出 其相对于 H (水平 x) 方向的 逆时针 转角 phi_a
         "plot_group_AST": "r",  # m 代表 oe 的 mix，o,e 代表 ~，fb 代表 frontface / backface
         # %%
         "lam1": 1.064, "is_air_pump": 1, "is_air": 2, "T": 25,
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
         "is_colorbar_on": 1, "is_colorbar_log": -1,
         "is_energy": 1,
         # %%
         "is_print": 1,
         # %% 该程序 独有 -------------------------------
         "is_EVV_SSI": 0, "is_stored": 0, "sheets_stored_num": 10,
         # %%
         "sample": 1, "cmap_3d": 'rainbow',
         "elev": 10, "azim": -65, "alpha": 2,
         # %%
         "is_plot_EVV": 1, "is_plot_3d_XYz": 0, "is_plot_selective": 0,
         "X": 0, "Y": 0, "is_plot_YZ_XZ": 0, "is_plot_3d_XYZ": 0,
         # %%
         "plot_group": "Ua", "is_animated": 1,
         "loop": 0, "duration": 0.033, "fps": 5,
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
         "plot_center": 1,
         }

    if kwargs.get("is_HOPS_AST", 0) >= 1:  # 如果 is_HOPS >= 1，则 默认 双泵浦 is_twin_pumps == 1
        pump2_kwargs = {
            "U2_name": "",
            "img2_full_name": "spaceship.png",
            "is_phase_only_2": 0,
            # %%
            "z_pump2": -5,
            "is_LG_2": 1, "is_Gauss_2": 1, "is_OAM_2": 1,
            "l2": 50, "p2": 0,
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

    Gz_o, Gz_e, E_uo, E_ue, \
    Gz_Vo, Gz_Ve, E_u_Vo, E_u_Ve, \
    Gz_Ho, Gz_He, E_u_Ho, E_u_He, \
    ray, method_and_way, U_key = AST_EVV(**kwargs)

    # %%  再次衍射 z0

    import copy

    if kwargs.get("is_end", -1) == 0:  # 如果 上一个 is_EVV 的 kwargs["is_end"] == -1，则不 再次衍射 z0
        kwargs_AST = copy.deepcopy(kwargs)
        kwargs_AST.update({"z0": 5, "is_air": 1, "ray": ray, "is_end": -1, })
        kwargs_AST.update({"g_o": Gz_o, "g_e": Gz_e, "E_uo": E_uo, "E_ue": E_ue,
                           "g_Vo": Gz_Vo, "g_Ve": Gz_Ve, "E_u_Vo": E_u_Vo, "E_u_Ve": E_u_Ve,
                           "g_Ho": Gz_Ho, "g_He": Gz_He, "E_u_Ho": E_u_Ho, "E_u_He": E_u_He, })

        Gz_o, Gz_e, E_uo, E_ue, \
        Gz_Vo, Gz_Ve, E_u_Vo, E_u_Ve, \
        Gz_Ho, Gz_He, E_u_Ho, E_u_He, \
        ray, method_and_way, U_key = AST_EVV(**kwargs_AST)
