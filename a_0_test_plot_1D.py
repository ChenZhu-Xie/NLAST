# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

from fun_os import img_squared_bordered_Read, get_Data_new_attrs, get_items_new_attr, \
    U_energy_plot_save, U_error_energy_plot_save, U_twin_energy_error_plot_save, U_twin_error_energy_plot_save
from fun_global_var import init_GLV_DICT, Get, tree_print


# %%

def plot_1D_test(plot_mode=3, is_energy_normalized=0,
                 Data_Seq=0,
                 img_full_name="lena1.png",
                 is_phase_only=0,
                 # %%
                 U_size=0.9,
                 # %%
                 is_save=0, is_save_txt=0, dpi=100,
                 # %%
                 color_1d='b', color_1d2='r',
                 # %%
                 sample=1, ticks_num=6,
                 is_title_on=1, is_axes_on=1, is_mm=1,
                 # %%
                 fontsize=9,
                 font={'family': 'serif',
                       'style': 'normal',  # 'normal', 'italic', 'oblique'
                       'weight': 'normal',
                       'color': 'black',  # 'black','gray','darkred'
                       },
                 # %%
                 is_print=1, **kwargs, ):
    # %%
    plot_func = get_Data_new_attrs(Data_Seq, "saver_name")[0]

    info = "plot_1d 测试 —— " + plot_func
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    # 没有 treeprint 会没有 Set("f_f")，导致 z 之后被 format 成 0.0。。。
    kwargs.pop("is_end", None);
    kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    # %% 分析 all_data_info.txt、data_info.txt

    folder_new_address, index, U_list, U_name_list, U_name_no_suffix_list, z_list = \
        get_items_new_attr(Data_Seq, is_save_txt, is_print, )

    # %% 绘图

    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, U = \
        img_squared_bordered_Read(img_full_name,
                                  U_size, dpi,
                                  is_phase_only, **kwargs, )

    if plot_func == "U_energy_plot_save":
        U_energy_plot_save(U_list[index+0], U_name_no_suffix_list[index+0],
                           img_name_extension,
                           is_save_txt,
                           # %%
                           U_list[index+1], sample, size_PerPixel,
                           is_save, dpi, Get("size_fig_x"), Get("size_fig_y"),
                           color_1d, ticks_num, is_title_on, is_axes_on, is_mm,
                           fontsize, font,  # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                           # %%
                           z_list[index+0], **kwargs, )
    elif plot_func == "U_error_energy_plot_save":
        U_error_energy_plot_save(U_list[index+0], U_list[index+1], U_list[index+2], U_name_no_suffix_list[index+0],
                                 img_name_extension, is_save_txt,
                                 # %%
                                 U_list[index+3], U_list[index+4], sample, size_PerPixel,
                                 is_save, dpi, Get("size_fig_x"), Get("size_fig_y"),
                                 # %%
                                 color_1d, color_1d2,
                                 ticks_num, is_title_on, is_axes_on, is_mm,
                                 fontsize, font,  # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                                 # %%
                                 z_list[index+0], **kwargs, )
    elif plot_func == "U_twin_energy_error_plot_save":
        U_twin_energy_error_plot_save(U_list[index+0], U_list[index+1], U_name_no_suffix_list[index+0],
                                      img_name_extension, is_save_txt,
                                      # %%
                                      U_list[index+2], U_list[index+3], sample, size_PerPixel,
                                      is_save, dpi, Get("size_fig_x"), Get("size_fig_y"),
                                      # %%
                                      color_1d, color_1d2,
                                      ticks_num, is_title_on, is_axes_on, is_mm,
                                      fontsize, font,
                                      # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                                      # %%
                                      z_list[index+0],
                                      # %%
                                      is_energy_normalized=is_energy_normalized, **kwargs, )
    elif plot_func == "U_twin_error_energy_plot_save":
        U_twin_error_energy_plot_save(U_list[index+0], U_list[index+1], U_list[index+2], U_name_no_suffix_list[index+0],
                                      img_name_extension, is_save_txt,
                                      # %%
                                      U_list[index+3], U_list[index+4], sample, size_PerPixel,
                                      is_save, dpi, Get("size_fig_x"), Get("size_fig_y"),
                                      # %%
                                      color_1d, color_1d2,
                                      ticks_num, is_title_on, is_axes_on, is_mm,
                                      fontsize, font,
                                      # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
                                      # %%
                                      z_list[index+0],
                                      # %%
                                      is_energy_normalized=is_energy_normalized, **kwargs, )

if __name__ == '__main__':
    kwargs = \
        {"plot_mode": -1,
         "Data_Seq": 460,
         "img_full_name": "lena1.png",
         "U_pixels_x": 300, "U_pixels_y": 300,
         "is_phase_only": 0,
         # %%
         "U_size": 1,
         # %%
         "is_save": 0, "is_no_data_save": 0,
         "is_save_txt": 0, "dpi": 100,
         # %%
         "color_1d": 'b', "color_1d2": 'r',
         # %%
         "sample": 1, "ticks_num": 6,
         "is_title_on": 1, "is_axes_on": 1, "is_mm": 1,
         # %%
         "fontsize": 10.0,
         "font": {'family': 'serif',
               'style': 'normal',  # 'normal', 'italic', 'oblique'
               'weight': 'normal',
               'color': 'black',  # 'black','gray','darkred'
               },
         "is_print": 1,
         "is_energy_normalized": 2,
         # %% 该程序 作为 主入口时 -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "is_end": -1,
         # %%
         "size_fig_x_scale": 10, "size_fig_y_scale": 2,
         "ax_yscale": 'linear', "xticklabels_rotate": 45, }

    kwargs = init_GLV_DICT(**kwargs)
    plot_1D_test(**kwargs)

    # plot_1D_test(plot_mode=3,
    #              Data_Seq=24,
    #              img_full_name="lena1.png",
    #              is_phase_only=0,
    #              # %%
    #              U_size=0.9,
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
