# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 00:42:22 2022

@author: Xcz
"""

import imageio as imgio
from PIL import Image

def imgs2gif_imgio(img_paths, gif_address, # loop = 0 代表 循环播放, 1 只播放 1 次
             duration=None, fps=None, loop=0, ): # 如果传入了 fps，则可 over write duration
    if fps: duration = 1/fps
    imgs = [imgio.imread(str(img_path)) for img_path in img_paths]
    imgio.mimsave(gif_address, imgs, "gif", duration=duration, loop=loop)

def imgs2gif_PIL(img_paths, gif_address, # loop = 0 代表 循环播放, 1 只播放 1 次
             duration=None, fps=None, loop=0, ): # 如果传入了 fps，则可 over write duration
    if fps: duration = 1/fps
    duration *= 1000
    imgs = [Image.open(str(img_path)) for img_path in img_paths]
    imgs[0].save(gif_address, save_all=True, append_images=imgs, duration=duration, loop=loop)