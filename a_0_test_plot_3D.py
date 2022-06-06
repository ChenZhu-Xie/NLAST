# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

from fun_os import img_squared_bordered_Read, get_Data_new_attrs, get_items_new_attr, \
    U_amp_plot_save_3d_XYz, U_phase_plot_save_3d_XYz, U_amp_plot_save_3d_XYZ, U_phase_plot_save_3d_XYZ
from fun_global_var import init_GLV_DICT, tree_print


# %%

def plot_3D_test(test_target=3,
                 Data_Seq=0,
                 img_full_name="lena1.png",
                 is_phase_only=0,
                 # %%
                 U_NonZero_size=0.9,
                 # %%
                 is_save=0, is_save_txt=0, dpi=100,
                 is_show_structure_face=0, is_print=1,
                 # %%
                 cmap_3d='rainbow',
                 elev=10, azim=-65, alpha=2,
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
                 is_colorbar_on=1, is_energy=0,
                 # %%
                 **kwargs, ):
    # %%
    if test_target == 1.1:
        test_func = "U_amp_plot_save_3d_XYz"
    elif test_target == 1.2:
        test_func = "U_phase_plot_save_3d_XYz"
    elif test_target == 2.1:
        test_func = "U_amp_plot_save_3d_XYZ"
    elif test_target == 2.2:
        test_func = "U_phase_plot_save_3d_XYZ"
    else:
        test_func = get_Data_new_attrs(Data_Seq, "saver_name")[0]

    info = "plot_3d 测试 —— " + test_func
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

    if test_func == "U_amp_plot_save_3d_XYz":
        U_amp_plot_save_3d_XYz(folder_new_address,
                               U_list[index+0], U_name_list[index+0],
                               img_name_extension,
                               is_save_txt,
                               # %%
                               sample, size_PerPixel,
                               is_save, dpi, size_fig,
                               elev, azim, alpha,
                               # %%
                               cmap_3d, ticks_num,
                               is_title_on, is_axes_on, is_mm,
                               fontsize, font,
                               # %%
                               is_colorbar_on, is_energy,
                               # %%
                               U_list[index+1], U_list[index+2],
                               is_no_data_save=kwargs.get("is_no_data_save", 0), **kwargs, )
    elif test_func == "U_phase_plot_save_3d_XYz":
        U_phase_plot_save_3d_XYz(folder_new_address,
                                 U_list[index+0], U_name_list[index+0],
                                 img_name_extension,
                                 is_save_txt,
                                 # %%
                                 sample, size_PerPixel,
                                 is_save, dpi, size_fig,
                                 elev, azim, alpha,
                                 # %%
                                 cmap_3d, ticks_num,
                                 is_title_on, is_axes_on, is_mm,
                                 fontsize, font,
                                 # %%
                                 is_colorbar_on,
                                 # %%
                                 U_list[index+1], U_list[index+2],
                                 is_no_data_save=kwargs.get("is_no_data_save", 0), **kwargs, )
    elif test_func == "U_amp_plot_save_3d_XYZ":
        U_amp_plot_save_3d_XYZ(folder_new_address,
                               U_name_no_suffix_list[index+0],
                               U_list[index+0][0], U_list[index+0][1],
                               U_list[index+0][2], U_list[index+0][3],
                               U_list[index+0][4], U_list[index+0][5],
                               U_list[index+1][0], U_list[index+1][1],
                               U_list[index+1][2], U_list[index+1][3],
                               U_list[index+1][4], U_list[index+1][5],
                               img_name_extension,
                               is_save_txt,
                               # %%
                               sample, size_PerPixel,
                               is_save, dpi, size_fig,
                               elev, azim, alpha,
                               # %%
                               cmap_3d, ticks_num,
                               is_title_on, is_axes_on, is_mm,
                               fontsize, font,
                               # %%
                               is_colorbar_on, is_energy, is_show_structure_face,
                               # %%
                               U_list[index+2], z=U_list[index+2][-1],
                               is_no_data_save=kwargs.get("is_no_data_save", 0),
                               # %%
                               vmax=U_list[index+3][0],
                               vmin=U_list[index+3][1], )
    elif test_func == "U_phase_plot_save_3d_XYZ":
        U_phase_plot_save_3d_XYZ(folder_new_address,
                                 U_name_no_suffix_list[index + 0],
                                 U_list[index + 0][0], U_list[index + 0][1],
                                 U_list[index + 0][2], U_list[index + 0][3],
                                 U_list[index + 0][4], U_list[index + 0][5],
                                 U_list[index + 1][0], U_list[index + 1][1],
                                 U_list[index + 1][2], U_list[index + 1][3],
                                 U_list[index + 1][4], U_list[index + 1][5],
                                 img_name_extension,
                                 is_save_txt,
                                 # %%
                                 sample, size_PerPixel,
                                 is_save, dpi, size_fig,
                                 elev, azim, alpha,
                                 # %%
                                 cmap_3d, ticks_num,
                                 is_title_on, is_axes_on, is_mm,
                                 fontsize, font,
                                 # %%
                                 is_colorbar_on, is_show_structure_face,
                                 # %%
                                 U_list[index + 2], z=U_list[index + 2][-1],
                                 is_no_data_save=kwargs.get("is_no_data_save", 0),
                                 # %%
                                 vmax=U_list[index + 3][0],
                                 vmin=U_list[index + 3][1], )


if __name__ == '__main__':
    kwargs = \
        {"test_target": -1,  # 自动化了，不用填这个参数了
         "Data_Seq": 24,
         "img_full_name": "lena1.png",
         "is_phase_only": 0,
         # %%
         "U_NonZero_size": 0.9,
         # %%
         "is_save": 0, "is_save_txt": 0, "dpi": 100,
         "is_show_structure_face": 1, "is_print": 1,
         # %%
         "cmap_3d": 'rainbow',
         "elev": 10, "azim": -65, "alpha": 2,
         # %%
         "sample": 1, "ticks_num": 7, "is_contourf": 0,
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
         # %% 该程序 作为 主入口时 -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "is_end": -1, }

    kwargs = init_GLV_DICT(**kwargs)
    plot_3D_test(**kwargs)

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
