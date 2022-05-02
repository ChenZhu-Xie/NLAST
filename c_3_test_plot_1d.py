# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import numpy as np
from scipy.io import loadmat
from fun_os import get_desktop, img_squared_bordered_Read, U_save, get_Data_address, \
    U_energy_plot, U_error_energy_plot_save, U_twin_energy_error_plot_save, U_twin_error_energy_plot_save
from fun_global_var import tree_print


# %%

def plot_test(test_target=3, is_energy_normalized=0,
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
    if test_target == 0:
        test_func = "U_energy_plot"
    elif test_target == 1:
        test_func = "U_error_energy_plot_save"
    elif test_target == 2:
        test_func = "U_twin_energy_error_plot_save"
    elif test_target == 3:
        test_func = "U_twin_error_energy_plot_save"

    info = "plot_1d 测试 —— " + test_func
    is_print and print(tree_print(kwargs.get("is_end", -1), add_level=1) + info)
    # 没有 treeprint 会没有 Set("f_f")，导致 z 之后被 format 成 0.0。。。
    kwargs.pop("is_end", None); kwargs.pop("add_level", None) # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # %% 分析 data_dir_names.txt

    str_list = get_Data_address(Data_Seq)
    folder_address = str_list[1]

    # %% 分析 data_names.txt

    txt_address = folder_address + "\\" + "data_names.txt"
    with open(txt_address, "r") as txt:
        lines = txt.readlines()

    ugHGU_list, U_name_list, U_address_list, U_list, z_list, U_name_no_suffix_list = [], [], [], [], [], []
    is_end = [0] * (len(lines) - 1)
    is_end.append(-1)
    for i in range(len(lines)):
        line = lines[i]
        line = line[:-1] # 把 每一行的 换行 去掉

        ugHGU = line.split(' ; ')[1]
        U_name = line.split(' ; ')[2]
        U_address = line.split(' ; ')[3]
        U = np.loadtxt(U_address, dtype=np.float64()) if is_save_txt == 1 else loadmat(U_address)[ugHGU]
        z = float(line.split(' ; ')[4])
        U_name_no_suffix = line.split(' ; ')[5]

        is_print and print(tree_print(is_end[i]) + "U_name = {}".format(U_name))

        ugHGU_list.append(ugHGU)
        U_name_list.append(U_name)
        U_address_list.append(U_address)
        U_list.append(U if is_save_txt == 1 else U[0]) # savemat 会使 1维 数组 变成 2维，也就是 会在外面 多加个 []
        z_list.append(z)
        U_name_no_suffix_list.append(U_name_no_suffix)

    # %% 绘图

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, U = \
        img_squared_bordered_Read(img_full_name,
                                  U_NonZero_size, dpi,
                                  is_phase_only)

    size_fig_x, size_fig_y = size_fig * kwargs.get("size_fig_x_scale", 10), size_fig * kwargs.get("size_fig_y_scale", 1)
    p_dir = "7. GU_error"
    if test_func == "U_energy_plot":
        suffix = "_energy"
        U_energy_plot(folder_address,
                      U_list[1], U_name_no_suffix + suffix,
                      img_name_extension,
                      # %%
                      U_list[2], sample, size_PerPixel,
                      is_save, dpi, size_fig_x, size_fig_y,
                      color_1d, ticks_num,
                      is_title_on, is_axes_on, is_mm,
                      fontsize, font,
                      # %%
                      z=z, )
        U_save(U_list[1], U_name_no_suffix + suffix, folder_address,
               is_save, is_save_txt,
               z=z, suffix=suffix, **kwargs, )
    elif test_func == "U_error_energy_plot_save":
        U_error_energy_plot_save(U_list[0], U_list[1], U_name_no_suffix,
                                 img_name_extension, is_save_txt,
                                 # %%
                                 U_list[2], U_list[3], sample, size_PerPixel,
                                 is_save, dpi, size_fig_x, size_fig_y,
                                 # %%
                                 color_1d, color_1d2,
                                 ticks_num, is_title_on, is_axes_on, is_mm,
                                 fontsize, font,  # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                                 # %%
                                 z,
                                 # %%
                                 p_dir=p_dir, **kwargs, )
    elif test_func == "U_twin_energy_error_plot_save":
        U_twin_energy_error_plot_save(U_list[0], U_list[1], U_name_no_suffix,
                                      img_name_extension, is_save_txt,
                                      # %%
                                      U_list[2], U_list[3], sample, size_PerPixel,
                                      is_save, dpi, size_fig_x, size_fig_y,
                                      # %%
                                      color_1d, color_1d2,
                                      ticks_num, is_title_on, is_axes_on, is_mm,
                                      fontsize, font,
                                      # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                                      # %%
                                      z,
                                      # %%
                                      p_dir=p_dir, is_energy_normalized=is_energy_normalized, **kwargs, )
    elif test_func == "U_twin_error_energy_plot_save":
        U_twin_error_energy_plot_save(U_list[0], U_list[1], U_list[2], U_name_no_suffix,
                                      img_name_extension, is_save_txt,
                                      # %%
                                      U_list[3], U_list[4], sample, size_PerPixel,
                                      is_save, dpi, size_fig_x, size_fig_y,
                                      # %%
                                      color_1d, color_1d2,
                                      ticks_num, is_title_on, is_axes_on, is_mm,
                                      fontsize, font,
                                      # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                                      # %%
                                      z,
                                      # %%
                                      p_dir=p_dir, is_energy_normalized=is_energy_normalized, **kwargs, )

if __name__ == '__main__':
    plot_test(test_target=0, is_energy_normalized=2,
              Data_Seq=4,
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
              size_fig_x_scale=10, size_fig_y_scale=1,
              ax_yscale='linear', )
