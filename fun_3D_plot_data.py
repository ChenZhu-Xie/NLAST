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

def plot_data_3D(plot_mode=3,
                 Data_Seq=0,
                 img_full_name="lena1.png",
                 is_phase_only=0,
                 # %%
                 U_size=0.9,
                 # %%
                 is_save=0,
                 is_save_txt=0, dpi=100,
                 # %%
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
                 is_colorbar_on=1, is_colorbar_log=0,
                 is_energy=0,
                 # %%
                 **kwargs, ):
    # %%
    plot_func = get_Data_new_attrs(Data_Seq, "saver_name")[0]

    info = "plot_3d 测试 —— " + plot_func
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

    share_args_3D = [folder_new_address,
                     img_name_extension, is_save_txt,
                     # %%
                     sample, size_PerPixel,
                     is_save, dpi, size_fig,
                     elev, azim, alpha,
                     # %%
                     cmap_3d, ticks_num,
                     is_title_on, is_axes_on, is_mm,
                     fontsize, font,
                     # %%
                     is_colorbar_on, is_energy, ]

    if "plot_save_3d_XYz" in plot_func:
        args_plot_save_3d_XYz = [U_list[index + 0], U_name_no_suffix_list[index + 0],
                                 # 这个倒是可以用 U_name_no_suffix，否则会双倍 suffix，而不覆盖之前的，生成新 .mat
                                 U_list[index + 2], ]
    elif "plot_save_3d_XYZ" in plot_func:
        args_plot_save_3d_XYZ = [U_name_no_suffix_list[index + 0],
                                 # 这个倒是可以用 U_name_no_suffix，否则会双倍 suffix，而不覆盖之前的，生成新 .mat
                                 U_list[index + 0][0], U_list[index + 0][1],
                                 U_list[index + 0][2], U_list[index + 0][3],
                                 U_list[index + 0][4], U_list[index + 0][5],
                                 U_list[index + 1][0], U_list[index + 1][1],
                                 U_list[index + 1][2], U_list[index + 1][3],
                                 U_list[index + 1][4], U_list[index + 1][5],
                                 U_list[index + 2], ]

        if is_colorbar_log == -1:
            v_kwargs = {}
        else:
            v_kwargs = {
                "vmax": U_list[index + 3][0],
                "vmin": U_list[index + 3][1],
            }

    if plot_func == "U_amp_plot_save_3d_XYz":
        U_amp_plot_save_3d_XYz(*args_plot_save_3d_XYz, *share_args_3D,
                               # %%
                               U_list[index + 1],
                               is_colorbar_log=is_colorbar_log,
                               **kwargs, )
    elif plot_func == "U_phase_plot_save_3d_XYz":
        U_phase_plot_save_3d_XYz(*args_plot_save_3d_XYz, *share_args_3D[:-1],
                                 # %%
                                 U_list[index + 1],
                                 is_colorbar_log=is_colorbar_log,
                                 **kwargs, )
    elif plot_func == "U_amp_plot_save_3d_XYZ":
        U_amp_plot_save_3d_XYZ(*args_plot_save_3d_XYZ,
                               *share_args_3D, is_show_structure_face,
                               # %%
                               z=U_list[index + 2][-1],
                               **v_kwargs, **kwargs, )
    elif plot_func == "U_phase_plot_save_3d_XYZ":
        U_phase_plot_save_3d_XYZ(*args_plot_save_3d_XYZ,
                                 *share_args_3D[:-1], is_show_structure_face,
                                 # %%
                                 z=U_list[index + 2][-1],
                                 **v_kwargs, **kwargs, )


if __name__ == '__main__':
    kwargs = \
        {"plot_mode": -1,
         "Data_Seq": 24,
         "img_full_name": "lena1.png",
         "U_pixels_x": 300, "U_pixels_y": 300,
         "is_phase_only": 0,
         # %%
         "U_size": 0.9,
         # %%
         "is_save": 0, "is_no_data_save": 0,
         "is_save_txt": 0, "dpi": 100,
         # %%
         "is_show_structure_face": 1, "is_print": 1,
         # %%
         "cmap_3d": 'rainbow',
         "elev": 10, "azim": -65, "alpha": 2,
         # %%
         "sample": 1, "ticks_num": 7, "is_contourf": 0,
         "is_title_on": 1, "is_axes_on": 1, "is_mm": 1,
         # %%
         "fontsize": 10.0,
         "font": {'family': 'serif',
                  'style': 'normal',  # 'normal', 'italic', 'oblique'
                  'weight': 'normal',
                  'color': 'black',  # 'black','gray','darkred'
                  },
         # %%
         "is_colorbar_on": 1, "is_colorbar_log": 0,
         "is_energy": 0,
         # %% 该程序 作为 主入口时 -------------------------------
         "kwargs_seq": 0, "root_dir": r'1',
         "is_end": -1, }

    kwargs = init_GLV_DICT(**kwargs)
    plot_data_3D(**kwargs)

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
