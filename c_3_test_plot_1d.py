# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
from scipy.io import loadmat
from fun_os import img_squared_bordered_Read, U_save, attr_get, get_Data_new_root_dir, \
    U_energy_plot, U_error_energy_plot_save, U_twin_energy_error_plot_save, U_twin_error_energy_plot_save
from fun_global_var import init_GLV_DICT, Get, tree_print


# %%

def plot_1D_test(test_target=3, is_energy_normalized=0,
                 Data_Seq=0,
                 img_full_name="lena1.png",
                 is_phase_only=0,
                 # %%
                 U_NonZero_size=0.9,
                 # %%
                 is_save=0, is_save_txt=0, dpi=100,
                 # %%
                 color_1d='b', color_1d2='r',
                 # %%
                 sample=1, ticks_num=6, is_print=1,
                 is_title_on=1, is_axes_on=1, is_mm=1,
                 # %%
                 fontsize=9,
                 font={'family': 'serif',
                       'style': 'normal',  # 'normal', 'italic', 'oblique'
                       'weight': 'normal',
                       'color': 'black',  # 'black','gray','darkred'
                       },
                 # %%
                 **kwargs, ):
    # %%

    if test_target == 0:
        test_func = "U_energy_plot"
    elif test_target == 1:
        test_func = "U_error_energy_plot_save"
    elif test_target == 2:
        test_func = "U_twin_energy_error_plot_save"
    elif test_target == 3:
        test_func = "U_twin_error_energy_plot_save"

    info = "plot_1d 测试 —— " + test_func
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=1) + info)
    # 没有 treeprint 会没有 Set("f_f")，导致 z 之后被 format 成 0.0。。。
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # %% 分析 all_data_info.txt

    new_root_dir, folder_new_address, U_new_address = get_Data_new_root_dir(Data_Seq)
    # U_new_address 没用，可用 *_ 代替其解包

    # %% 分析 data_info.txt

    txt_address = folder_new_address + "\\" + "data_info.txt"
    with open(txt_address, "r") as txt:
        lines = txt.readlines()

    ugHGU_list, U_name_list, U_address_list, U_list, z_list, U_name_no_suffix_list = [], [], [], [], [], []
    is_end = [0] * (len(lines) - 1)
    is_end.append(-1)
    for i in range(len(lines)):
        line = lines[i]
        line = line[:-1]  # 把 每一行的 换行 去掉

        ugHGU = attr_get(line, "ugHGU")
        U_name = attr_get(line, "U_name")

        U_address = attr_get(line, "U_address")
        root_dir = attr_get(line, "root_dir")
        U_new_address = U_address.replace(root_dir, new_root_dir)  # 用新的 root_dir 去覆盖 旧的 root_dir
        U = np.loadtxt(U_new_address, dtype=np.float64()) if is_save_txt == 1 else loadmat(U_new_address)[ugHGU]
        # print(U)
        z = float(attr_get(line, "z_str"))
        U_name_no_suffix = attr_get(line, "U_name_no_suffix")

        is_print and print(tree_print(is_end[i]) + "U_name = {}".format(U_name))

        ugHGU_list.append(ugHGU)
        U_name_list.append(U_name)
        U_address_list.append(U_new_address)
        U_list.append(U if is_save_txt == 1 else U[0])  # savemat 会使 1维 数组 变成 2维，也就是 会在外面 多加个 []
        z_list.append(z)
        U_name_no_suffix_list.append(U_name_no_suffix)

    # %% 绘图

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, U = \
        img_squared_bordered_Read(img_full_name,
                                  U_NonZero_size, dpi,
                                  is_phase_only)

    if test_func == "U_energy_plot":
        suffix = "_energy"
        U_energy_plot(folder_new_address,
                      U_list[1], U_name_no_suffix + suffix,
                      img_name_extension,
                      # %%
                      U_list[2], sample, size_PerPixel,
                      is_save, dpi, Get("size_fig_x"), Get("size_fig_y"),
                      color_1d, ticks_num,
                      is_title_on, is_axes_on, is_mm,
                      fontsize, font,
                      # %%
                      z=z, )
        U_save(U_list[1], U_name_no_suffix + suffix, folder_new_address,
               is_save, is_save_txt,
               z=z, suffix=suffix, **kwargs, )
    elif test_func == "U_error_energy_plot_save":
        U_error_energy_plot_save(U_list[0], U_list[1], U_name_no_suffix,
                                 img_name_extension, is_save_txt,
                                 # %%
                                 U_list[2], U_list[3], sample, size_PerPixel,
                                 is_save, dpi, Get("size_fig_x"), Get("size_fig_y"),
                                 # %%
                                 color_1d, color_1d2,
                                 ticks_num, is_title_on, is_axes_on, is_mm,
                                 fontsize, font,  # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                                 # %%
                                 z, **kwargs, )
    elif test_func == "U_twin_energy_error_plot_save":
        U_twin_energy_error_plot_save(U_list[0], U_list[1], U_name_no_suffix,
                                      img_name_extension, is_save_txt,
                                      # %%
                                      U_list[2], U_list[3], sample, size_PerPixel,
                                      is_save, dpi, Get("size_fig_x"), Get("size_fig_y"),
                                      # %%
                                      color_1d, color_1d2,
                                      ticks_num, is_title_on, is_axes_on, is_mm,
                                      fontsize, font,
                                      # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                                      # %%
                                      z,
                                      # %%
                                      is_energy_normalized=is_energy_normalized, **kwargs, )
    elif test_func == "U_twin_error_energy_plot_save":
        U_twin_error_energy_plot_save(U_list[0], U_list[1], U_list[2], U_name_no_suffix,
                                      img_name_extension, is_save_txt,
                                      # %%
                                      U_list[3], U_list[4], sample, size_PerPixel,
                                      is_save, dpi, Get("size_fig_x"), Get("size_fig_y"),
                                      # %%
                                      color_1d, color_1d2,
                                      ticks_num, is_title_on, is_axes_on, is_mm,
                                      fontsize, font,
                                      # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                                      # %%
                                      z,
                                      # %%
                                      is_energy_normalized=is_energy_normalized, **kwargs, )


if __name__ == '__main__':
    kwargs = \
        {"test_target": 3,
         "Data_Seq": 24,
         "img_full_name": "lena1.png",
         "is_phase_only": 0,
         # %%
         "U_NonZero_size": 0.9,
         # %%
         "is_save": 0, "is_save_txt": 0, "dpi": 100,
         # %%
         "color_1d": 'b', "color_1d2": 'r',
         # %%
         "sample": 1, "ticks_num": 7, "is_print": 1,
         "is_title_on": 1, "is_axes_on": 1, "is_mm": 1,
         # %%
         "fontsize": 9,
         "font": {'family': 'serif',
               'style': 'normal',  # 'normal', 'italic', 'oblique'
               'weight': 'normal',
               'color': 'black',  # 'black','gray','darkred'
               },
         "is_energy_normalized": 2,
         # %%
         "kwargs_seq": 0, "root_dir": r'',
         "is_end": -1,
         "size_fig_x_scale": 10, "size_fig_y_scale": 2,
         "ax_yscale": 'linear', }

    kwargs = init_GLV_DICT(**kwargs)
    plot_1D_test(**kwargs)

    # plot_1D_test(test_target=3,
    #              Data_Seq=24,
    #              img_full_name="lena1.png",
    #              is_phase_only=0,
    #              # %%
    #              U_NonZero_size=0.9,
    #              # %%
    #              is_save=0, is_save_txt=0, dpi=100,
    #              # %%
    #              color_1d='b', color_1d2='r',
    #              # %%
    #              sample=1, ticks_num=7, is_print=1,
    #              is_title_on=1, is_axes_on=1, is_mm=1,
    #              # %%
    #              fontsize=9,
    #              font={'family': 'serif',
    #                    'style': 'normal',  # 'normal', 'italic', 'oblique'
    #                    'weight': 'normal',
    #                    'color': 'black',  # 'black','gray','darkred'
    #                    },
    #              is_energy_normalized=2,
    #              # %%
    #              root_dir=r'', is_end=-1,
    #              size_fig_x_scale=10, size_fig_y_scale=2,
    #              ax_yscale='linear', )
