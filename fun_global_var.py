import numpy as np
import math
from fun_os import set_ray, GHU_plot_save, U_SSI_plot
from fun_linear import ifft2, Cal_n, Cal_kz
from fun_statistics import U_Drop_n_sigma

def init_GLV_DICT(U1_name, ray, way, method, **kwargs):
    global GLOBALS_DICT
    GLOBALS_DICT = {}

    ray = set_ray(U1_name, ray, **kwargs)
    set("ray", ray)
    set("way", way)
    set("method", method)

    return ray

def set(key, value): # 第一次设置值的时候用 set，以后可直接用 get(key) = ~
    try:
        GLOBALS_DICT[key] = value
        return True  # 创建 key-value 成功
    except KeyError:
        return False  # 创建 key-value 失败

def get(key):
    try:
        return GLOBALS_DICT[key]  # 取 value 成功，返回 value
    except KeyError:
        return False  # 取 value 失败

# %%

def fkey(key):  # get_format_key
    return key + get("ray") + "_z" + get("way")
def fset(key, value):  # set_format_key_value
    return set(fkey(key), value)
def fget(key):  # get_format_key_value
    return get(fkey(key))

# %%

def ekey(key):  # get_format_key
    return key + get("ray") + "_z_energy"
def eset(key, value):  # set_format_key_value
    return set(ekey(key), value)
def eget(key):  # get_format_key_value
    return get(ekey(key))

# %%

def dkey(key):  # get_format_key
    return key + get("ray") + "_zdz"
def dset(key, value):  # set_format_key_value
    return set(dkey(key), value)
def dget(key):  # get_format_key_value
    return get(dkey(key))

# %%

def skey(key):  # get_format_key
    return key + get("ray") + "_z_stored"
def sset(key, value):  # set_format_key_value
    return set(skey(key), value)
def sget(key):  # get_format_key_value
    return get(skey(key))

# %%

def YZkey(key):  # get_format_key
    return key + get("ray") + "_YZ"
def YZset(key, value):  # set_format_key_value
    return set(YZkey(key), value)
def YZget(key):  # get_format_key_value
    return get(YZkey(key))

# %%

def XZkey(key):  # get_format_key
    return key + get("ray") + "_XZ"
def XZset(key, value):  # set_format_key_value
    return set(XZkey(key), value)
def XZget(key):  # get_format_key_value
    return get(XZkey(key))

# %%

def key1(key):  # get_format_key
    return key + get("ray") + "_sec1"
def set1(key, value):  # set_format_key_value
    return set(key1(key), value)
def get1(key):  # get_format_key_value
    return get(key1(key))

# %%

def key2(key):  # get_format_key
    return key + get("ray") + "_sec2"
def set2(key, value):  # set_format_key_value
    return set(key2(key), value)
def get2(key):  # get_format_key_value
    return get(key2(key))

# %%

def keyf(key):  # get_format_key
    return key + get("ray") + "_front"
def setf(key, value):  # set_format_key_value
    return set(keyf(key), value)
def getf(key):  # get_format_key_value
    return get(keyf(key))

# %%

def keye(key):  # get_format_key
    return key + get("ray") + "_end"
def sete(key, value):  # set_format_key_value
    return set(keye(key), value)
def gete(key):  # get_format_key_value
    return get(keye(key))

# %%

def init_SSI(g1_shift, U1_0,
             is_energy_evolution_on, is_stored,
             sheets_num, sheets_stored_num,
             X, Y, Iz, size_PerPixel, ):

    Ix, Iy = U1_0.shape
    # set("Ix", Ix)
    # set("Iy", Iy)

    set("is_energy_evolution_on", is_energy_evolution_on)
    set("is_stored", is_stored)
    set("sheets_num", sheets_num)
    set("sheets_stored_num", sheets_stored_num)

    set("X", X)
    set("Y", Y)
    set("th_X", Iy // 2 + int(X / size_PerPixel))
    set("th_Y", Ix // 2 - int(Y / size_PerPixel))

    #%%

    dset("G", g1_shift)
    dset("U", U1_0)

    if is_energy_evolution_on == 1:

        eset("G", np.zeros((sheets_num + 1), dtype=np.float64()))
        eset("U", np.zeros((sheets_num + 1), dtype=np.float64()))
        eget("G")[0] = np.sum(np.abs(dget("G")) ** 2)
        eget("U")[0] = np.sum(np.abs(dget("U")) ** 2)

    if is_stored == 1:

        sheet_th_stored = np.zeros(int(sheets_stored_num + 1), dtype=np.int64())
        iz_stored = np.zeros(int(sheets_stored_num + 1), dtype=np.float64())
        z_stored = np.zeros(int(sheets_stored_num + 1), dtype=np.float64())

        sheet_th_stored[sheets_stored_num] = sheets_num
        iz_stored[sheets_stored_num] = Iz
        z_stored[sheets_stored_num] = Iz * size_PerPixel

        set("sheet_th_stored", sheet_th_stored) # 设置成全局变量
        set("iz_stored", iz_stored) # 方便之后在这个 py 和 主 py 文件里直接调用
        set("z_stored", z_stored) # 懒得搞 返回 和 获取 这仨了

        sset("G", np.zeros((Ix, Iy, int(sheets_stored_num + 1)), dtype=np.complex128()))
        sset("U", np.zeros((Ix, Iy, int(sheets_stored_num + 1)), dtype=np.complex128()))

        # 小写的 x,y 表示 电脑中 矩阵坐标系，大写 X,Y 表示 笛卡尔坐标系
        YZset("G", np.zeros((Ix, sheets_num + 1), dtype=np.complex128()))
        XZset("G", np.zeros((Iy, sheets_num + 1), dtype=np.complex128()))
        YZset("U", np.zeros((Ix, sheets_num + 1), dtype=np.complex128()))
        XZset("U", np.zeros((Iy, sheets_num + 1), dtype=np.complex128()))

        setf("G", np.zeros((Ix, Iy), dtype=np.complex128()))
        setf("U", np.zeros((Ix, Iy), dtype=np.complex128()))
        sete("G", np.zeros((Ix, Iy), dtype=np.complex128()))
        sete("U", np.zeros((Ix, Iy), dtype=np.complex128()))
        set1("G", np.zeros((Ix, Iy), dtype=np.complex128()))
        set1("U", np.zeros((Ix, Iy), dtype=np.complex128()))
        set2("G", np.zeros((Ix, Iy), dtype=np.complex128()))
        set2("U", np.zeros((Ix, Iy), dtype=np.complex128()))

# %%

def fun3(for_th, fors_num, G1_z_plus_dz_shift, *args, **kwargs, ):

    U1_z_plus_dz = ifft2(G1_z_plus_dz_shift)

    if get("is_energy_evolution_on") == 1:
        eget("G")[for_th + 1] = np.sum(np.abs(G1_z_plus_dz_shift) ** 2)
        eget("U")[for_th + 1] = np.sum(np.abs(U1_z_plus_dz) ** 2)

    if get("is_stored") == 1:

        # 小写的 x,y 表示 电脑中 矩阵坐标系，大写 X,Y 表示 笛卡尔坐标系
        YZget("G")[:, for_th] = G1_z_plus_dz_shift[:, get("th_X")]
        # X 增加，则 从 G1_z_shift 中 读取的 列 向右移，也就是 YZ 面向 列 增加的方向（G1_z_shift 的 右侧）移动
        XZget("G")[:, for_th] = G1_z_plus_dz_shift[get("th_Y"), :]
        # Y 增加，则 从 G1_z_shift 中 读取的 行 向上移，也就是 XZ 面向 行 减小的方向（G1_z_shift 的 上侧）移动
        YZget("U")[:, for_th] = U1_z_plus_dz[:, get("th_X")]
        XZget("U")[:, for_th] = U1_z_plus_dz[get("th_Y"), :]

        # %%

        if np.mod(for_th, get("sheets_num") // get("sheets_stored_num")) == 0:
            # 如果 for_th 是 get("sheets_num") // get("sheets_stored_num") 的 整数倍（包括零），则 储存之
            get("sheet_th_stored")[int(for_th // (get("sheets_num") // get("sheets_stored_num")))] = for_th + 1
            get("iz_stored")[int(for_th // (get("sheets_num") // get("sheets_stored_num")))] = get("izj")[for_th + 1]
            get("z_stored")[int(for_th // (get("sheets_num") // get("sheets_stored_num")))] = get("zj")[for_th + 1]
            sget("G")[:, :, int(for_th // (get("sheets_num") // get("sheets_stored_num")))] = G1_z_plus_dz_shift
            # 储存的 第一层，实际上不是 G1_0，而是 G1_dz
            sget("U")[:, :, int(for_th // (get("sheets_num") // get("sheets_stored_num")))] = U1_z_plus_dz
            # 储存的 第一层，实际上不是 U1_0，而是 U1_dz

        if for_th == get("sheet_th_frontface"):  # 如果 for_th 是 sheet_th_frontface，则把结构 前端面 场分布 储存起来，对应的是 zj[sheets_num_frontface]
            setf("G", G1_z_plus_dz_shift)
            setf("U", U1_z_plus_dz)
        if for_th == get("sheet_th_endface"):  # 如果 for_th 是 sheet_th_endface，则把结构 后端面 场分布 储存起来，对应的是 zj[sheets_num_endface]
            sete("G", G1_z_plus_dz_shift)
            sete("U", U1_z_plus_dz)
        if for_th == get("sheet_th_sec1"):  # 如果 for_th 是 想要观察的 第一个面 前面那一层的 层序数，则 将储存之于 该层 前面那一层的 后端面（毕竟 算出来的是 z + dz） 分布中
            set1("G", G1_z_plus_dz_shift)  # 对应的是 zj[sheets_num_sec1]
            set1("U", U1_z_plus_dz)
        if for_th == get("sheet_th_sec2"):  # 如果 for_th 是 想要观察的 第二个面 前面那一层的 层序数，则 将储存之于 该层 前面那一层的 后端面（毕竟 算出来的是 z + dz） 分布中
            set2("G", G1_z_plus_dz_shift)  # 对应的是 zj[sheets_num_sec2]
            set2("U", U1_z_plus_dz)

# %%

def end_AST(z0, size_PerPixel,
            g1_shift, k1_z, ):

    iz = z0 / size_PerPixel
    fset("H", np.power(math.e, k1_z * iz * 1j))
    fset("G", g1_shift * fget("H"))
    fset("U", ifft2(fget("G")))

# %%

def end_SSI(g1_shift, is_energy, n_sigma = 3, ):
    fset("G", dget("G"))
    fset("H", fget("G") / np.max(np.abs(fget("G"))) / (g1_shift / np.max(np.abs(g1_shift))))
    if n_sigma > 0:
        # 扔掉 amp 偏离 amp 均值 3 倍于 总体 标准差 以外 的 数据，保留 剩下的 3 倍 以内的 数据。
        fset("H", U_Drop_n_sigma(fget("H"), n_sigma, is_energy))
    fset("U", ifft2(fget("G")))

# %%

def fGHU_plot_save(U1_name, is_energy_evolution_on,  # 默认 全自动 is_auto = 1
                   img_name_extension,
                   # %%
                   zj, sample, size_PerPixel,
                   is_save, is_save_txt, dpi, size_fig,
                   # %%
                   color_1d, cmap_2d,
                   ticks_num, is_contourf,
                   is_title_on, is_axes_on, is_mm,
                   fontsize, font,
                   # %%
                   is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                   # %%                          何况 一般默认 is_self_colorbar = 1...
                   z, ):

    GHU_plot_save(U1_name, is_energy_evolution_on,  # 默认 全自动 is_auto = 1
                  fget("G"), fkey("G"), get("method"),  # 这边 要省事 免代入 的话，得确保 提前 传入 ray,way,method 三个参数
                  eget("G"),
                  fget("H"), fkey("H"),  # 以及 传入 GHU 这三个 小东西
                  fget("U"), fkey("U"),
                  eget("U"),
                  img_name_extension,
                  # %%
                  zj, sample, size_PerPixel,
                  is_save, is_save_txt, dpi, size_fig,
                  # %%
                  color_1d, cmap_2d,
                  ticks_num, is_contourf,
                  is_title_on, is_axes_on, is_mm,
                  fontsize, font,
                  # %%
                  is_colorbar_on, is_energy,  # 默认无法 外界设置 vmax 和 vmin，因为 同时画 振幅 和 相位 得 传入 2*2 个 v
                  # %%                          何况 一般默认 is_self_colorbar = 1...
                  z, )

# %%

def fU_SSI_plot(U1_name, folder_address,
               th_f, th_e,
               img_name_extension,
               # %%
               sample, size_PerPixel,
               is_save, dpi, size_fig,
               elev, azim, alpha,
               # %%
               cmap_2d, cmap_3d,
               ticks_num, is_contourf,
               is_title_on, is_axes_on, is_mm,
               fontsize, font,
               # %%
               is_colorbar_on, is_energy, is_show_structure_face,
               # %%
               z_1, z_2,
               z_f, z_e, z, ):

    if get("is_stored") == 1:

        sget("G")[:, :, get("sheets_stored_num")] = fget("G")  # 储存的 第一层，实际上不是 G1_0，而是 G1_dz
        sget("U")[:, :, get("sheets_stored_num")] = fget("U")  # 储存的 第一层，实际上不是 U1_0，而是 U1_dz

        # 小写的 x,y 表示 电脑中 矩阵坐标系，大写 X,Y 表示 笛卡尔坐标系
        YZget("G")[:, get("sheets_num")] = fget("G")[:, get("th_X")]
        XZget("G")[:, get("sheets_num")] = fget("G")[get("th_Y"), :]
        YZget("U")[:, get("sheets_num")] = fget("U")[:, get("th_X")]
        XZget("U")[:, get("sheets_num")] = fget("U")[get("th_Y"), :]

        U_SSI_plot(U1_name, folder_address,
                   sget("G"), fkey("G"), get("method"),
                   sget("U"), fkey("U"),
                   YZget("G"), XZget("G"),
                   YZget("U"), XZget("U"),
                   get1("G"), get2("G"),
                   getf("G"), gete("G"),
                   get1("U"), get2("U"),
                   getf("U"), gete("U"),
                   get("th_X"), get("th_Y"),
                   get("sheet_th_sec1"), get("sheet_th_sec2"),
                   th_f, th_e,
                   img_name_extension,
                   # %%
                   sample, size_PerPixel,
                   is_save, dpi, size_fig,
                   elev, azim, alpha,
                   # %%
                   cmap_2d, cmap_3d,
                   ticks_num, is_contourf,
                   is_title_on, is_axes_on, is_mm,
                   fontsize, font,
                   # %%
                   is_colorbar_on, is_energy, is_show_structure_face,
                   # %%
                   get("X"), get("Y"), # 这个也可 顺便 设成 global 的，懒得搞
                   z_1, z_2,
                   z_f, z_e,
                   get("zj"), get("z_stored"), z, )
