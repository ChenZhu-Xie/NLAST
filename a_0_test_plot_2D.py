# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

from fun_os import img_squared_bordered_Read, get_Data_new_attrs, get_items_new_attr,  \
    U_plot_save, U_amp_plot_save, U_amps_z_plot_save, U_phases_z_plot_save, U_slices_plot_save, U_selects_plot_save
from fun_global_var import init_GLV_DICT, tree_print

# %%

def plot_2D_test(test_target=3,
                 Data_Seq=0,
                 img_full_name="lena1.png",
                 is_phase_only=0,
                 # %%
                 U_NonZero_size=0.9,
                 # %%
                 is_save=0, is_save_txt=0, dpi=100,
                 is_show_structure_face=0, is_print=1,
                 # %%
                 cmap_2d='viridis',
                 # %%
                 sample=1, ticks_num=6, is_contourf=0,
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
                 is_animated=1,
                 loop=0, duration=0.033, fps=5,
                 # %%
                 **kwargs, ):
    # %%
    if test_target == 0:
        test_func = "U_plot_save"
    elif test_target == 0.1:
        test_func = "U_amp_plot_save"
    elif test_target == 1:
        test_func = "U_selects_plot_save"
    elif test_target == 1.1:
        test_func = "U_amps_z_plot_save"
    elif test_target == 1.2:
        test_func = "U_phases_z_plot_save"
    elif test_target == 2:
        test_func = "U_slices_plot_save"
    else:
        test_func = get_Data_new_attrs(Data_Seq, "saver_name")[0]

    info = "plot_2d 测试 —— " + test_func
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
                                  U_NonZero_size, dpi,
                                  is_phase_only)

    if test_func == "U_plot_save":
        # 如果函数的参数 只有 U_list[0] 一个，则不按 folder 来读，而直接读条目；
        # 但这样 便 意味着 不会有 多条属性（否则 不止传入 1 个 U_list data）? 并不，其实还可以 往下读！
        # 不要往上继续找 相同 saver_name 的 第一个，因为 那仍可能是 上一次 的结果，而不是这一次的结果
        U_plot_save(U_list[index], U_name_list[index], 0, # 不 print energy
                    img_name_extension,
                    # %%
                    size_PerPixel,
                    is_save, is_save_txt, dpi, size_fig,
                    # %%
                    cmap_2d, ticks_num, is_contourf,
                    is_title_on, is_axes_on, is_mm,
                    fontsize, font,
                    # %%
                    is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                    # %%                          何况 一般默认 is_self_colorbar = 1...
                    z=z_list[index], **kwargs, )
    elif test_func == "U_amp_plot_save":
        U_amp_plot_save(folder_new_address,
                        # 因为 要返回的话，太多了；返回一个 又没啥意义，而且 返回了 基本也用不上
                        U_list[index], U_name_list[index],
                        img_name_extension,
                        is_save_txt,
                        # %%
                        [], 1, size_PerPixel,
                        is_save, dpi, size_fig,
                        # %%
                        cmap_2d, ticks_num, is_contourf,
                        is_title_on, is_axes_on, is_mm, 0,
                        fontsize, font,
                        # %%
                        0, is_colorbar_on, 0, # 大部分 U_amp_plot_save 是没有 vmax,vmin 的，所以 将就大部分 使用情况
                        # %%
                        suffix="", **kwargs, )
    elif test_func == "U_selects_plot_save":
        U_selects_plot_save(folder_new_address,
                            U_list[index+0], U_name_list[index+0],
                            U_list[index+1], U_name_list[index+1],
                            U_list[index+2], U_name_list[index+2],
                            U_list[index+3], U_name_list[index+3],
                            img_name_extension,
                            is_save_txt,
                            # %%
                            sample, size_PerPixel,
                            is_save, dpi, size_fig,
                            # %%
                            cmap_2d, ticks_num, is_contourf,
                            is_title_on, is_axes_on, is_mm,
                            fontsize, font,
                            # %%
                            is_colorbar_on, is_energy, is_show_structure_face,
                            # %%
                            z_list[index+0], z_list[index+1], z_list[index+2], z_list[index+3],
                            # %%
                            is_no_data_save=kwargs.get("is_no_data_save", 0), )
    elif test_func == "U_amps_z_plot_save":
        U_amps_z_plot_save(folder_new_address,
                          U_list[index+0], U_name_list[index+0], # 这个倒是可以用 U_name_no_suffix
                          img_name_extension,
                          is_save_txt,
                          # %%
                          sample, size_PerPixel,
                          is_save, dpi, size_fig,
                          # %%
                          cmap_2d, ticks_num, is_contourf,
                          is_title_on, is_axes_on, is_mm,
                          fontsize, font,
                          # %%
                          is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                          # %%                          何况 一般默认 is_self_colorbar = 1...
                          U_list[index+1], is_animated,
                          duration, fps, loop,
                          z_list[index+0], **kwargs, )  # 传 z 是为了 储存时，给 G_stored 命名
    elif test_func == "U_phases_z_plot_save":
        U_phases_z_plot_save(folder_new_address,
                            U_list[index+0], U_name_list[index+0], # 这个倒是可以用 U_name_no_suffix
                            img_name_extension,
                            is_save_txt,
                            # %%
                            sample, size_PerPixel,
                            is_save, dpi, size_fig,
                            # %%
                            cmap_2d, ticks_num, is_contourf,
                            is_title_on, is_axes_on, is_mm,
                            fontsize, font,
                            # %%
                            is_colorbar_on,  # 默认无法 外界设置 vmax 和 vmin，默认 自动统一 colorbar
                            # %%
                            U_list[index+1], is_animated,
                            duration, fps, loop,
                            z_list[index+0], **kwargs, )
    elif test_func == "U_slices_plot_save":
        U_slices_plot_save(folder_new_address,
                           U_list[index+0], U_name_list[index+0],
                           U_list[index+1], U_name_list[index+1],
                           img_name_extension,
                           is_save_txt,
                           # %%
                           U_list[index+2], sample, size_PerPixel,
                           is_save, dpi, size_fig,
                           # %%
                           cmap_2d, ticks_num, is_contourf,
                           is_title_on, is_axes_on, is_mm,
                           fontsize, font,
                           # %%
                           is_colorbar_on, is_energy,
                           # %%
                           z_list[index+0], z_list[index+1],
                           # %%
                           is_no_data_save=kwargs.get("is_no_data_save", 0), )
    

if __name__ == '__main__':
    kwargs = \
        {"test_target": -1, # 自动化了，不用填这个参数了
         "Data_Seq": 16,
         "img_full_name": "lena1.png",
         "is_phase_only": 0,
         # %%
         "U_NonZero_size": 0.9,
         # %%
         "is_save": 1, "is_save_txt": 0, "dpi": 100,
         "is_show_structure_face": 1, "is_print": 1,
         # %%
         "cmap_2d": 'viridis',
         # %%
         "sample": 1, "ticks_num": 7, "is_contourf": 0,
         "is_title_on": 1, "is_axes_on": 1, "is_mm": 1,
         # %%
         "fontsize": 9,
         "font": {'family': 'serif',
                  'style': 'normal',  # 'normal', 'italic', 'oblique'
                  'weight': 'normal',
                  'color': 'black',  # 'black','gray','darkred'
                  },
         # %%
         "is_colorbar_on": 1, "is_energy": 0,
         # %%
         "is_animated": -1,
         "loop": 0, "duration": 0.033, "fps": 5,
         # %%
         "kwargs_seq": 0, "root_dir": r'1',
         "is_end": -1, }

    kwargs = init_GLV_DICT(**kwargs)
    plot_2D_test(**kwargs)

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
