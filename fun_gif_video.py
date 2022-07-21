# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 00:42:22 2022

@author: Xcz
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fun_thread import noop, my_thread


# %%
def imgs2gif_imgio(img_paths, gif_address,  # loop = 0 代表 循环播放, 1 只播放 1 次
                   imgs_list, is_save,
                   duration=None, fps=None, loop=0, ):  # 如果传入了 fps，则可 over write duration
    import imageio as imgio
    # %%
    if fps: duration = 1 / fps
    if is_save == 1:
        imgs = [imgio.imread(str(img_path)) for img_path in img_paths]
    elif is_save == -1:
        imgs = imgs_list
    imgio.mimsave(gif_address, imgs, "gif", duration=duration, loop=loop)


# %%
def imgs2gif_PIL(img_paths, gif_address,  # loop = 0 代表 循环播放, 1 只播放 1 次
                 imgs_list, is_save,
                 duration=None, fps=None, loop=0, ):  # 如果传入了 fps，则可 over write duration
    from PIL import Image
    # %%
    if fps: duration = 1 / fps
    duration *= 1000
    if is_save == 1:
        imgs = [Image.open(str(img_path)) for img_path in img_paths]
    elif is_save == -1:
        imgs = [Image.fromarray(img) for img in imgs_list]
    imgs[0].save(gif_address, save_all=True, append_images=imgs, duration=duration, loop=loop)


# %%
def imgs2gif_art(img_paths, gif_address, dpi,  # loop = 0 代表 循环播放, 1 只播放 1 次
                 imgs_list, is_save,
                 duration=None, fps=None, loop=0, percentage=0, ):  # 如果传入了 fps，则可 over write duration

    percentage /= 100
    dpi *= (1 + percentage)
    img = plt.imread(img_paths[0])
    size_fig_x = img.shape[1] / dpi
    size_fig_y = img.shape[0] / dpi
    # %%
    fig = plt.figure(figsize=(size_fig_x, size_fig_y), dpi=dpi)  # 还不好设置画布大小。。。因为已经加了 colorbar、标题 等等 了
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # %%
    ax1 = fig.add_subplot(111, label="1")  # plt.subplots(4, 3, figsize=(8, 4), tight_layout=True)
    # 还得创建画布，否则 报 <Figure size 640x486 with 0 Axes>
    ax1.axis('off')
    ax1.margins(0, 0)
    # if is_save == 1:
    # %%

    global ims
    ims = []

    '''单线程'''
    # for k in range(len(img_paths)):
    #     # img = cv2.imread(img_paths[k], cv2.IMREAD_UNCHANGED) - 无法读取 中文路径图片
    #     img = cv2.imdecode(np.fromfile(img_paths[k],dtype=np.uint8),cv2.IMREAD_UNCHANGED) # 保留 BGR + alpha 通道 3维 * 4通道
    #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    #     im = plt.imshow(img, animated=True)
    #     # fig.savefig(".png", transparent=True, pad_inches=0)
    #     ims.append([im])

    '''多线程 begin'''

    def fun1(for_th, fors_num, *arg, **kwargs, ):
        # img = cv2.imdecode(np.fromfile(img_paths[for_th], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # 保留 BGR + alpha 通道 3维 * 4通道
        # img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        # img = Image.open(img_paths[for_th])
        # img = np.array(img)
        if is_save == 1:
            img = plt.imread(img_paths[for_th])
        else:
            img = imgs_list[for_th]
        # print(img)
        im = ax1.imshow(img, animated=True)
        return im

    def fun2(for_th, fors_num, im, *args, **kwargs, ):
        global ims
        ims.append([im])

    my_thread(10, len(img_paths),
              fun1, fun2, noop,
              is_ordered=1, is_print=0, )
    '''多线程 end'''
    # elif is_save == -1:
    #     ims = [[im] for im in imgs_list]

    if fps: duration = 1 / fps
    repeat_delay = 0  # 下一个循环 过几秒 才开始
    ani = animation.ArtistAnimation(fig, ims, interval=duration * 1000, blit=True,
                                    repeat=(loop == 0), repeat_delay=repeat_delay * 1000)
    ani.save(gif_address)
