# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

from fun_os import U_error_energy_plot_save
from fun_os import img_squared_bordered_Read


# %%

def plot_test_U_error_energy_plot_save(folder_address="",
                                       img_full_name="lena1.png",
                                       is_phase_only=0,
                                       # %%
                                       U_NonZero_size=0.5,
                                       # %%
                                       is_save=0, is_save_txt=0, dpi=100,
                                       # %%
                                       color_1d='b', color_1d2='r',
                                       # %%
                                       sample=2, ticks_num=6,
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
    img_name, img_name_extension, img_squared, \
    size_PerPixel, size_fig, Ix, Iy, U = \
        img_squared_bordered_Read(img_full_name,
                                  U_NonZero_size, dpi,
                                  is_phase_only)

    txt_address = folder_address + "\\" + "data_names.txt"
    with open(txt_address, "r+") as txt:
        lines = txt.readlines() # 注意是 readlines 不是 readline，否则 只读了 一行，而不是 所有行 构成的 列表
        # lines = lines[:-1] # 把最后的 换行 去掉（不用去了，每个 \n 包含在上一行了）
        for line in lines:
            line = line[:-1]
            print(line)
            # write(ugHGU + ' ; ' + U_name + ' ; ' + U_address + "\n")

    # U_error_energy_plot_save(G_energy, G_error_energy, fkey("G"),
    #                          img_name_extension, is_save_txt,
    #                          # %%
    #                          array_dkQ, array_Tz, sample, size_PerPixel,
    #                          is_save, dpi, size_fig * 10, size_fig,
    #                          # %%
    #                          color_1d, color_1d2,
    #                          ticks_num, is_title_on, is_axes_on, is_mm,
    #                          fontsize, font,  # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
    #                          # %%
    #                          L0_Crystal, **kwargs, )
    #
    # U_error_energy_plot_save(U_energy, U_error_energy, fkey("U"),
    #                          img_name_extension, is_save_txt,
    #                          # %%
    #                          array_dkQ, array_Tz, sample, size_PerPixel,
    #                          is_save, dpi, size_fig * 10, size_fig,
    #                          # %%
    #                          color_1d, color_1d2,
    #                          ticks_num, is_title_on, is_axes_on, is_mm,
    #                          fontsize, font,  # 默认无法 外界设置，只能 自动设置 y 轴 max 和 min 了（不是 但 类似 colorbar），还有 is_energy
    #                          # %%
    #                          L0_Crystal, **kwargs, )


plot_test_U_error_energy_plot_save(folder_address=r"D:\CtoD\Default\桌面\7. GU_error\5. G_h_2.66mm_energy_sync & error",
                                   img_full_name="lena1.png",
                                   is_phase_only=0,
                                   # %%
                                   U_NonZero_size=0.5,
                                   # %%
                                   is_save=0, is_save_txt=0, dpi=100,
                                   # %%
                                   color_1d='b', color_1d2='r',
                                   # %%
                                   sample=2, ticks_num=6,
                                   is_title_on=1, is_axes_on=1, is_mm=1,
                                   # %%
                                   fontsize=9,
                                   font={'family': 'serif',
                                         'style': 'normal',  # 'normal', 'italic', 'oblique'
                                         'weight': 'normal',
                                         'color': 'black',  # 'black','gray','darkred'
                                         },
                                   # %%
                                   )
