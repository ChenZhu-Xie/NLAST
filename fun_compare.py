# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

import os
import numpy as np
from scipy.io import savemat
from fun_plot import plot_2d
from fun_statistics import U_Drop_n_sigma

#%%

def U_compare(U, U_0, z, 
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
              is_colorbar_on = 1,
              is_energy = 1, vmax = 1, vmin = 0, 
              #%%
              is_print = 0 ):
    
    #%%
    # 对比 U 与 U_0 的 绝对误差 1
    U_error = U - U_0
    
    U_error_amp = np.abs(U_error)
    U_error_phase = np.angle(U_error)
    
    if is_print == 1:
        print("Compare - U_{}mm_error.total_amp = {}".format(z, np.sum(U_error_amp)))
        print("Compare - U_{}mm_error.total_energy = {}".format(z, np.sum(U_error_amp**2)))
        print("Compare - U_{}mm_error.rsd = {}".format(z, np.std(U_error_amp) / np.mean(U_error_amp) ))

    if is_save == 1:
        if not os.path.isdir("6. U_" + str(float('%.2g' % z)) + "mm" + "_error"):
            os.makedirs("6. U_" + str(float('%.2g' % z)) + "mm" + "_error")

    #%%
    #绘图：U_error_amp

    U_error_amp_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_error" + "\\" + "6.1. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_error" + "_amp" + " = " + "U_substract_U_0" + "_abs" + img_name_extension

    plot_2d([], 1, size_PerPixel, 
            U_error_amp, U_error_amp_address, "U_" + str(float('%.2g' % z)) + "mm" + "_error" + "_amp", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    #%%
    #绘图：U_error_phase

    U_error_phase_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_error" + "\\" + "6.2. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_error" + "_phase" + " = " + "U_substract_U_0" + "_angle" + img_name_extension

    plot_2d([], 1, size_PerPixel, 
            U_error_phase, U_error_phase_address, "U_" + str(float('%.2g' % z)) + "mm" + "_error" + "_phase", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)
    
    #%%
    # 储存 U_error 到 txt 文件

    U_error_full_name = "6. Compare - U_" + str(float('%.2g' % z)) + "mm" + "_error" + (is_save_txt and ".txt" or ".mat")
    if is_save == 1:
        U_error_txt_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_error" + "\\" + U_error_full_name
        np.savetxt(U_error_txt_address, U_error) if is_save_txt else savemat(U_error_txt_address, {"U":U_error})

        #%%
        #再次绘图：U_error_amp
    
        U_error_amp_address = "6.1. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_error" + "_amp" + " = " + "U_substract_U_0" + "_abs" + img_name_extension
    
        plot_2d([], 1, size_PerPixel, 
                U_error_amp, U_error_amp_address, "U_" + str(float('%.2g' % z)) + "mm" + "_error" + "_amp", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf,
                is_title_on, is_axes_on, is_mm, 0,
                fontsize, font,
                1, is_colorbar_on, is_energy, vmax, vmin)
    
        #再次绘图：U_error_phase
    
        U_error_phase_address = "6.2. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_error" + "_phase" + " = " + "U_substract_U_0" + "_angle" + img_name_extension
    
        plot_2d([], 1, size_PerPixel, 
                U_error_phase, U_error_phase_address, "U_" + str(float('%.2g' % z)) + "mm" + "_error" + "_phase", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf,
                is_title_on, is_axes_on, is_mm, 0,
                fontsize, font,
                1, is_colorbar_on, 0, vmax, vmin)

    #%%
    # 储存 U_error 到 txt 文件

    # if is_save == 1:
    np.savetxt(U_error_full_name, U_error) if is_save_txt else savemat(U_error_full_name, {"U":U_error})
    
    #%%
    # 对比 U 与 U_0 的 绝对误差 2
    
    U_amp_error = np.abs(U) - np.abs(U_0)
    U_phase_error = np.abs(U) - np.angle(U_0)
    
    if is_print == 1:
        print("Compare - U_{}mm_amp_error.total_amp = {}".format(z, np.sum(U_amp_error)))
        print("Compare - U_{}mm_amp_error.total_energy = {}".format(z, np.sum(U_amp_error**2)))
        print("Compare - U_{}mm_amp_error.rsd = {}".format(z, np.std(U_amp_error) / np.mean(U_amp_error) ))
    
    # print("Compare - U_{}mm_phase_error.total_phase = {}".format(z, np.sum(U_phase_error)))
    # print("Compare - U_{}mm_phase_error.total_energy = {}".format(z, np.sum(U_phase_error**2)))
    # print("Compare - U_{}mm_phase_error.rsd = {}".format(z, np.std(U_phase_error) / np.mean(U_phase_error) ))

    if is_save == 1:
        if not os.path.isdir("6. U_" + str(float('%.2g' % z)) + "mm" + "_error"):
            os.makedirs("6. U_" + str(float('%.2g' % z)) + "mm" + "_error")

    #%%
    #绘图：U_amp_error

    U_amp_error_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_error" + "\\" + "6.1. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_amp_error" + " = " + "U_abs__substract__U_0_abs" + img_name_extension

    plot_2d([], 1, size_PerPixel, 
            U_amp_error, U_amp_error_address, "U_" + str(float('%.2g' % z)) + "mm" + "_amp_error", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            1, is_colorbar_on, is_energy, vmax, vmin)

    #%%
    #绘图：U_phase_error

    U_phase_error_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_error" + "\\" + "6.2. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_phase_error" + " = " + "U_angle__substract__U_0_angle" + img_name_extension

    plot_2d([], 1, size_PerPixel, 
            U_phase_error, U_phase_error_address, "U_" + str(float('%.2g' % z)) + "mm" + "_phase_error", 
            is_save, dpi, size_fig,  
            cmap_2d, ticks_num, is_contourf,
            is_title_on, is_axes_on, is_mm, 0,
            fontsize, font,
            1, is_colorbar_on, 0, vmax, vmin)

    if is_save == 1:
        #%%
        #再次绘图：U_amp_error
    
        U_amp_error_address = "6.1. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_amp_error" + " = " + "U_abs__substract__U_0_abs" + img_name_extension
    
        plot_2d([], 1, size_PerPixel, 
                U_amp_error, U_amp_error_address, "U_" + str(float('%.2g' % z)) + "mm" + "_amp_error", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf,
                is_title_on, is_axes_on, is_mm, 0,
                fontsize, font,
                1, is_colorbar_on, is_energy, vmax, vmin)
    
        #再次绘图：U_phase_error
    
        U_phase_error_address = "6.2. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_phase_error" + " = " + "U_angle__substract__U_0_angle" + img_name_extension
    
        plot_2d([], 1, size_PerPixel, 
                U_phase_error, U_phase_error_address, "U_" + str(float('%.2g' % z)) + "mm" + "_phase_error", 
                is_save, dpi, size_fig,  
                cmap_2d, ticks_num, is_contourf,
                is_title_on, is_axes_on, is_mm, 0,
                fontsize, font,
                1, is_colorbar_on, 0, vmax, vmin)
    
    # #%%
    # # 对比 U 与 U_0 的 绝对误差 的 相对误差
    
    # U_error_relative = (U - U_0) / U_0
    
    # # 扔掉 amp 偏离 amp 均值 3 倍于 总体 标准差 以外 的 数据，保留 剩下的 3 倍 以内的 数据。
    # U_error_relative = U_Drop_n_sigma(U_error_relative, 3, is_energy)
    
    # U_error_relative_amp = np.abs(U_error_relative)
    # U_error_relative_phase = np.angle(U_error_relative)
    
    # if is_print == 1:
    #     print("Compare - U_{}mm_error_relative.total_amp = {}".format(z, np.sum(U_error_relative_amp)))
    #     print("Compare - U_{}mm_error_relative.total_energy = {}".format(z, np.sum(U_error_relative_amp**2)))
    #     print("Compare - U_{}mm_error_relative.rsd = {}".format(z, np.std(U_error_relative_amp) / np.mean(U_error_relative_amp) ))

    # if is_save == 1:
    #     if not os.path.isdir("6. U_" + str(float('%.2g' % z)) + "mm" + "_error_relative"):
    #         os.makedirs("6. U_" + str(float('%.2g' % z)) + "mm" + "_error_relative")

    # #%%
    # #绘图：U_error_relative_amp

    # U_error_relative_amp_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "\\" + "6.1. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_amp" + " = " + "U_substract_U_0" + "_abs" + img_name_extension

    # plot_2d([], 1, size_PerPixel, 
    #         U_error_relative_amp, U_error_relative_amp_address, "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_amp", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         1, is_colorbar_on, is_energy, vmax, vmin)

    # #%%
    # #绘图：U_error_relative_phase

    # U_error_relative_phase_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "\\" + "6.2. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_phase" + " = " + "U_substract_U_0" + "_angle" + img_name_extension

    # plot_2d([], 1, size_PerPixel, 
    #         U_error_relative_phase, U_error_relative_phase_address, "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_phase", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         1, is_colorbar_on, 0, vmax, vmin)
    
    # #%%
    # # 储存 U_error_relative 到 txt 文件

    # U_error_relative_full_name = "6. Compare - U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + (is_save_txt and ".txt" or ".mat")
    # if is_save == 1:
    #     U_error_relative_txt_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "\\" + U_error_relative_full_name
    #     np.savetxt(U_error_relative_txt_address, U_error_relative) if is_save_txt else savemat(U_error_relative_txt_address, {"U_error_relative":U_error_relative})

    #     #%%
    #     #再次绘图：U_error_relative_amp
    
    #     U_error_relative_amp_address = "6.1. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_amp" + " = " + "U_substract_U_0" + "_abs" + img_name_extension
    
    #     plot_2d([], 1, size_PerPixel, 
    #             U_error_relative_amp, U_error_relative_amp_address, "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_amp", 
    #             is_save, dpi, size_fig,  
    #             cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #             fontsize, font,
    #             1, is_colorbar_on, is_energy, vmax, vmin)
    
    #     #再次绘图：U_error_relative_phase
    
    #     U_error_relative_phase_address = "6.2. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_phase" + " = " + "U_substract_U_0" + "_angle" + img_name_extension
    
    #     plot_2d([], 1, size_PerPixel, 
    #             U_error_relative_phase, U_error_relative_phase_address, "U_" + str(float('%.2g' % z)) + "mm" + "_error_relative" + "_phase", 
    #             is_save, dpi, size_fig,  
    #             cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #             fontsize, font,
    #             1, is_colorbar_on, 0, vmax, vmin)

    # #%%
    # # 储存 U_error_relative 到 txt 文件

    # # if is_save == 1:
    # np.savetxt(U_error_relative_full_name, U_error_relative) if is_save_txt else savemat(U_error_relative_full_name, {"U":U_error_relative})
    
    # #%%
    # # 对比 U 与 U_0 的 相对误差
    
    # U_relative_error = U / U_0
    
    # U_relative_error_amp = np.abs(U_relative_error)
    # U_relative_error_phase = np.angle(U_relative_error)
    
    # if is_print == 1:
    #     print("Compare - U_{}mm_relative_error.total_amp = {}".format(z, np.sum(U_relative_error_amp)))
    #     print("Compare - U_{}mm_relative_error.total_energy = {}".format(z, np.sum(U_relative_error_amp**2)))
    #     print("Compare - U_{}mm_relative_error.rsd = {}".format(z, np.std(U_relative_error_amp) / np.mean(U_relative_error_amp) ))

    # if is_save == 1:
    #     if not os.path.isdir("6. U_" + str(float('%.2g' % z)) + "mm" + "_relative_error"):
    #         os.makedirs("6. U_" + str(float('%.2g' % z)) + "mm" + "_relative_error")

    # #%%
    # #绘图：U_relative_error_amp

    # U_relative_error_amp_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "\\" + "6.1. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_amp" + " = " + "U_substract_U_0" + "_abs" + img_name_extension

    # plot_2d([], 1, size_PerPixel, 
    #         U_relative_error_amp, U_relative_error_amp_address, "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_amp", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         1, is_colorbar_on, is_energy, vmax, vmin)

    # #%%
    # #绘图：U_relative_error_phase

    # U_relative_error_phase_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "\\" + "6.2. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_phase" + " = " + "U_substract_U_0" + "_angle" + img_name_extension

    # plot_2d([], 1, size_PerPixel, 
    #         U_relative_error_phase, U_relative_error_phase_address, "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_phase", 
    #         is_save, dpi, size_fig,  
    #         cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #         fontsize, font,
    #         1, is_colorbar_on, 0, vmax, vmin)
    
    # #%%
    # # 储存 U_relative_error 到 txt 文件

    # U_relative_error_full_name = "6. Compare - U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + (is_save_txt and ".txt" or ".mat")
    # if is_save == 1:
    #     U_relative_error_txt_address = "6. U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "\\" + U_relative_error_full_name
    #     np.savetxt(U_relative_error_txt_address, U_relative_error) if is_save_txt else savemat(U_relative_error_txt_address, {"U_relative_error":U_relative_error})

    #     #%%
    #     #再次绘图：U_relative_error_amp
    
    #     U_relative_error_amp_address = "6.1. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_amp" + " = " + "U_substract_U_0" + "_abs" + img_name_extension
    
    #     plot_2d([], 1, size_PerPixel, 
    #             U_relative_error_amp, U_relative_error_amp_address, "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_amp", 
    #             is_save, dpi, size_fig,  
    #             cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #             fontsize, font,
    #             1, is_colorbar_on, is_energy, vmax, vmin)
    
    #     #再次绘图：U_relative_error_phase
    
    #     U_relative_error_phase_address = "6.2. Compare - " + "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_phase" + " = " + "U_substract_U_0" + "_angle" + img_name_extension
    
    #     plot_2d([], 1, size_PerPixel, 
    #             U_relative_error_phase, U_relative_error_phase_address, "U_" + str(float('%.2g' % z)) + "mm" + "_relative_error" + "_phase", 
    #             is_save, dpi, size_fig,  
    #             cmap_2d, ticks_num, is_contourf, is_title_on, is_axes_on, is_mm, 0, 
    #             fontsize, font,
    #             1, is_colorbar_on, 0, vmax, vmin)

    # #%%
    # # 储存 U_relative_error 到 txt 文件

    # # if is_save == 1:
    # np.savetxt(U_relative_error_full_name, U_relative_error) if is_save_txt else savemat(U_relative_error_full_name, {"U":U_relative_error})
    