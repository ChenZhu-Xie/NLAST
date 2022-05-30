# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

import numpy as np
from fun_os import split_parts, U_plot_save, U_error_plot_save, U_plot, U_energy_print, U_custom_print
from fun_global_var import Get, fkey, tree_print

#%%

def U_compare(U, U_0, U_0_title, z,
              #%%
              img_name_extension, size_PerPixel, size_fig, 
              # %%
              is_save=0, is_save_txt=0, dpi=100, 
              #%%
              cmap_2d = 'viridis', 
              #%%
              ticks_num = 6, is_contourf = 0, 
              is_title_on = 1, is_axes_on = 1, is_mm = 1,
              #%%
              fontsize = 9, 
              font = {'family': 'serif',
                      'style': 'normal', # 'normal', 'italic', 'oblique'
                      'weight': 'normal',
                      'color': 'black', # 'black','gray','darkred'
                      }, 
              #%%
              is_colorbar_on = 1, is_energy = 1,
              #%%
              is_amp_relative = 1, is_print = 2, **kwargs, ):
    kwargs['p_dir'] = 'GU_error_2d'
    #%%
    U_name_no_seq, method_and_way, Part_2, ugHGU, ray_seq = split_parts(U_0_title)

    info = ugHGU + "_" + str(float(Get('f_f') % z)) + "mm" + "_对比"
    is_print and print(tree_print(kwargs.get("is_end", 0), add_level=2) + info)
    kwargs.pop("is_end", None); kwargs.pop("add_level", None)  # 该 def 子分支 后续默认 is_end = 0，如果 kwargs 还会被 继续使用 的话。

    #%%
    # 画一下 两个待比较的 对象，并 print 一下 能量情况

    U_energy_print(U_0, U_0_title, is_print,
                   z=z, )

    U_plot("",
           U_0, U_0_title,
           img_name_extension,
           is_save_txt,
           # %%
           1, size_PerPixel, # sample = 1
           0, dpi, size_fig, # 不 save
           cmap_2d, ticks_num, is_contourf,
           is_title_on, is_axes_on, is_mm,
           fontsize, font,
           is_colorbar_on, is_energy,  # 自己 colorbar
           # %%
           z=z, is_no_data_save=1, )

    # %%
    U_title = kwargs["U_title"] if "U_title" in kwargs else fkey(ugHGU)

    U_energy_print(U, U_title, is_print,
                   z=z, )

    U_plot("",
           U, U_title,
           img_name_extension,
           is_save_txt,
           # %%
           1, size_PerPixel, # sample = 1
           0, dpi, size_fig, # 不 save
           cmap_2d, ticks_num, is_contourf,
           is_title_on, is_axes_on, is_mm,
           fontsize, font,
           is_colorbar_on, is_energy,  # 自己 colorbar
           # %%
           z=z, is_no_data_save=1, )

    # %% 归一化，查看 相对分布 的 大小

    if is_save == 2:
        is_save = 1

    if is_amp_relative == 1: # 归一化
        # print(np.max(np.abs(U)), np.max(np.abs(U_0)))
        U_norm = U/np.max(np.abs(U)) if np.max(np.abs(U)) != 0 else U
        U_0_norm = U_0 / np.max(np.abs(U_0)) if np.max(np.abs(U_0)) != 0 else U_0
    else:
        U_norm = U
        U_0_norm = U_0

    # %%
    # 对比 U 与 U_0 的 绝对误差 1

    info = ugHGU + "_先误差_后取模或相位"
    is_print and print(tree_print(add_level=2) + info)

    U_error = U_norm - U_0_norm
    U_error_name = U_title + "_error"
    folder_address = U_plot_save(U_error, U_error_name, is_print,
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
                                 z=z, is_end=1, **kwargs, )

    #%%

    folder_address, U_amp_error_energy = U_error_plot_save(U_norm, U_0_norm, ugHGU, is_print,
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
                                                          z=z, **kwargs, )

    U_0_norm_energy = np.sum(np.abs(U_0_norm) ** 2)
    # print(U_amp_error_energy)
    U_error_energy = U_amp_error_energy / U_0_norm_energy
    # print(U_error_energy)
    U_custom_print(U_error_energy, U_title, "relative_error", is_print,
                   z=z, is_end=1)

    # U_custom_print(U_energy_error, U_title, "relative_error", is_print,
    #                z=z, )
    # U_custom_print(U_energy_error / U_0_energy, U_title, "error_coefficient", is_print,
    #                z=z, is_end=1)

    U_energy = np.sum(np.abs(U) ** 2)
    U_0_energy = np.sum(np.abs(U_0) ** 2)
    return (U_energy, U_0_energy, U_error_energy)

    # #%%
    # # 对比 U 与 U_0 的 绝对误差 的 相对误差
    #
    # U_error_relative = (U - U_0) / U_0
    #
    # # 扔掉 amp 偏离 amp 均值 3 倍于 总体 标准差 以外 的 数据，保留 剩下的 3 倍 以内的 数据。
    # U_error_relative = U_Drop_n_sigma(U_error_relative, 3, is_energy)
    #
    # U_error_relative_amp = np.abs(U_error_relative)
    # U_error_relative_phase = np.angle(U_error_relative)
    #
    # if is_print == 1:
    #     print("Compare - U_{}mm_error_relative.total_amp = {}".format(z, np.sum(U_error_relative_amp)))
    #     print("Compare - U_{}mm_error_relative.total_energy = {}".format(z, np.sum(U_error_relative_amp**2)))
    #     print("Compare - U_{}mm_error_relative.rsd = {}".format(z, np.std(U_error_relative_amp) / np.mean(U_error_relative_amp) ))
    #
    # if is_save == 1:
    #     if not os.path.isdir("6. U_" + str(float('%.2g' % z)) + "mm" + "_error_relative"):
    #         os.makedirs("6. U_" + str(float('%.2g' % z)) + "mm" + "_error_relative")
    #
    # #%%
    # #绘图：U_error_relative_amp
    #
    # U_error_relative_amp_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "\\" + "6.1. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_amp" + " = " + "U_substract_U_0" + "_abs" + img_name_extension
    #
    # plot_2d([], 1, size_PerPixel,
    #         U_error_relative_amp, U_error_relative_amp_address, "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_amp",
    #         is_save, dpi, size_fig,
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
    #         fontsize, font,
    #         1, is_colorbar_on, is_energy, vmax, vmin)
    #
    # #%%
    # #绘图：U_error_relative_phase
    #
    # U_error_relative_phase_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "\\" + "6.2. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_phase" + " = " + "U_substract_U_0" + "_angle" + img_name_extension
    #
    # plot_2d([], 1, size_PerPixel,
    #         U_error_relative_phase, U_error_relative_phase_address, "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_phase",
    #         is_save, dpi, size_fig,
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
    #         fontsize, font,
    #         1, is_colorbar_on, 0, vmax, vmin)
    #
    # #%%
    # # 储存 U_error_relative 到 txt 文件
    #
    # U_error_relative_full_name = "6. Compare - U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + (is_save_txt and ".txt" or ".mat")
    # if is_save == 1:
    #     U_error_relative_txt_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "\\" + U_error_relative_full_name
    #     np.savetxt(U_error_relative_txt_address, U_error_relative) if is_save_txt else savemat(U_error_relative_txt_address, {"U_error_relative":U_error_relative})
    #
    #     #%%
    #     #再次绘图：U_error_relative_amp
    #
    #     U_error_relative_amp_address = "6.1. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_amp" + " = " + "U_substract_U_0" + "_abs" + img_name_extension
    #
    #     plot_2d([], 1, size_PerPixel,
    #             U_error_relative_amp, U_error_relative_amp_address, "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_amp",
    #             is_save, dpi, size_fig,
    #             cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
    #             fontsize, font,
    #             1, is_colorbar_on, is_energy, vmax, vmin)
    #
    #     #再次绘图：U_error_relative_phase
    #
    #     U_error_relative_phase_address = "6.2. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_phase" + " = " + "U_substract_U_0" + "_angle" + img_name_extension
    #
    #     plot_2d([], 1, size_PerPixel,
    #             U_error_relative_phase, U_error_relative_phase_address, "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_phase",
    #             is_save, dpi, size_fig,
    #             cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
    #             fontsize, font,
    #             1, is_colorbar_on, 0, vmax, vmin)
    #
    # #%%
    # # 储存 U_error_relative 到 txt 文件
    #
    # # if is_save == 1:
    # np.savetxt(U_error_relative_full_name, U_error_relative) if is_save_txt else savemat(U_error_relative_full_name, {ugHGU:U_error_relative})
    #
    # #%%
    # # 对比 U 与 U_0 的 相对误差
    #
    # U_relative_error = U / U_0
    #
    # U_relative_error_amp = np.abs(U_relative_error)
    # U_relative_error_phase = np.angle(U_relative_error)
    #
    # if is_print == 1:
    #     print("Compare - U_{}mm_relative_error.total_amp = {}".format(z, np.sum(U_relative_error_amp)))
    #     print("Compare - U_{}mm_relative_error.total_energy = {}".format(z, np.sum(U_relative_error_amp**2)))
    #     print("Compare - U_{}mm_relative_error.rsd = {}".format(z, np.std(U_relative_error_amp) / np.mean(U_relative_error_amp) ))
    #
    # if is_save == 1:
    #     if not os.path.isdir("6. U_" + str(float('%.2g' % z)) + "mm" + "_relative_error"):
    #         os.makedirs("6. U_" + str(float('%.2g' % z)) + "mm" + "_relative_error")
    #
    # #%%
    # #绘图：U_relative_error_amp
    #
    # U_relative_error_amp_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "\\" + "6.1. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_amp" + " = " + "U_substract_U_0" + "_abs" + img_name_extension
    #
    # plot_2d([], 1, size_PerPixel,
    #         U_relative_error_amp, U_relative_error_amp_address, "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_amp",
    #         is_save, dpi, size_fig,
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
    #         fontsize, font,
    #         1, is_colorbar_on, is_energy, vmax, vmin)
    #
    # #%%
    # #绘图：U_relative_error_phase
    #
    # U_relative_error_phase_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "\\" + "6.2. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_phase" + " = " + "U_substract_U_0" + "_angle" + img_name_extension
    #
    # plot_2d([], 1, size_PerPixel,
    #         U_relative_error_phase, U_relative_error_phase_address, "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_phase",
    #         is_save, dpi, size_fig,
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
    #         fontsize, font,
    #         1, is_colorbar_on, 0, vmax, vmin)
    #
    # #%%
    # # 储存 U_relative_error 到 txt 文件
    #
    # U_relative_error_full_name = "6. Compare - U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + (is_save_txt and ".txt" or ".mat")
    # if is_save == 1:
    #     U_relative_error_txt_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "\\" + U_relative_error_full_name
    #     np.savetxt(U_relative_error_txt_address, U_relative_error) if is_save_txt else savemat(U_relative_error_txt_address, {"U_relative_error":U_relative_error})
    #
    #     #%%
    #     #再次绘图：U_relative_error_amp
    #
    #     U_relative_error_amp_address = "6.1. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_amp" + " = " + "U_substract_U_0" + "_abs" + img_name_extension
    #
    #     plot_2d([], 1, size_PerPixel,
    #             U_relative_error_amp, U_relative_error_amp_address, "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_amp",
    #             is_save, dpi, size_fig,
    #             cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
    #             fontsize, font,
    #             1, is_colorbar_on, is_energy, vmax, vmin)
    #
    #     #再次绘图：U_relative_error_phase
    #
    #     U_relative_error_phase_address = "6.2. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_phase" + " = " + "U_substract_U_0" + "_angle" + img_name_extension
    #
    #     plot_2d([], 1, size_PerPixel,
    #             U_relative_error_phase, U_relative_error_phase_address, "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_phase",
    #             is_save, dpi, size_fig,
    #             cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0,
    #             fontsize, font,
    #             1, is_colorbar_on, 0, vmax, vmin)
    #
    # #%%
    # # 储存 U_relative_error 到 txt 文件
    #
    # # if is_save == 1:
    # np.savetxt(U_relative_error_full_name, U_relative_error) if is_save_txt else savemat(U_relative_error_full_name, {ugHGU:U_relative_error})
    