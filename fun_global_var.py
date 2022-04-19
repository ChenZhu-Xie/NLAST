# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:09:04 2021

@author: Xcz
"""

# %%

import inspect
import math
import numpy as np
from fun_linear import fft2, ifft2
from fun_statistics import U_Drop_n_sigma

#%%

global GLV_init_times
GLV_init_times = 0

def init_GLV():
    global GLV_init_times
    if GLV_init_times == 0: # 只在第一次初始化的时候，才初始化
        global GLOBALS_DICT
        GLOBALS_DICT = {}
    GLV_init_times += 1

#%%

def Set(key, value):  # 第一次设置值的时候用 Set，以后可直接用 Get(key) = ~
    init_GLV()
    try:  # 为了不与 set 集合 重名
        GLOBALS_DICT[key] = value
        return True  # 创建 key-value 成功
    except KeyError:
        return False  # 创建 key-value 失败


def Get(key):
    init_GLV()
    try:
        return GLOBALS_DICT[key]  # 取 value 成功，返回 value
    except KeyError:
        return False  # 取 value 失败

#%%

def init_GLV_tree_print():

    #%%
    # tree = "tree_"
    #
    # Set(tree + "0", ".")
    # Set(tree + "1", "├── ")
    # Set(tree + "2", "└── ")
    # Set(tree + ".", "|    ")
    #
    # def tree_set(tree, level):
    #     Set(tree + level, Get(tree + ".") * len(re.findall('\.', level)) + Get(tree + level.split('.')[-1]))
    #     # len(re.findall('\.', level)) 就是 level 中，点的 个数，其实就等于 下面 for 循环中的 i...
    #
    # sub_levels_num = 4 # 子层级的 层数
    # for i in [i+1 for i in range(sub_levels_num)]: # i = 1, 2, 3, ..., sub_levels_num
    #     tree_set(tree, "." * i + "1")
    #     tree_set(tree, "." * i + "2")

    #%%

    # Set("tree_print", [])
    #
    # Get("tree_print").append([]) # [[]]
    # Get("tree_print")[-1].append(".") # [["."]]
    # Get("tree_print")[-1].append("├── ") # [[".", "├── "]]
    # Get("tree_print")[-1].append("|    ") # [[".", "├── ", "|    "]]
    # Get("tree_print")[-1].append("└── ")  # [[".", "├── ", "|    ", "└── "]]
    # Get("tree_print")[-1].append("    ")  # [[".", "├── ", "|    ", "└── ", "    "]]
    #
    # sub_levels_num = 4
    # for i in [i + 1 for i in range(sub_levels_num)]:  # i = 1, 2, 3, ..., sub_levels_num
    #     Get("tree_print").append([])  # [[".", "├── ", "└── ", "|    "], []]
    #     Get("tree_print")[-1].append(Get("tree_print")[0][2] * i + Get("tree_print")[0][0])  # ["|    " * i + "."]
    #     Get("tree_print")[-1].append(Get("tree_print")[0][2] * i + Get("tree_print")[0][1])  # ["|    " * i + "├── "]
    #     Get("tree_print")[-1].append(Get("tree_print")[0][2] * i + Get("tree_print")[0][-2])  # ["|    " * i + "└── "]

    #%%
    init_GLV()
    if "tree_print" not in GLOBALS_DICT:
        Set("tree_print", [])
        Get("tree_print").append(".")     # ["."]
        Get("tree_print").append("├── ")  # [".", "├── "]
        Get("tree_print").append("|    ") # [".", "├── ", "|    "]
        Get("tree_print").append("└── ")  # [".", "├── ", "|    ", "└── "]
        Get("tree_print").append("    ")  # [".", "├── ", "|    ", "└── ", "    "]

        print(Get("tree_print")[0])

        suffix_1 = "_who_called_tree_print"
        suffix_2 = "_whose_is_end_equals_to_1"

        Set("ex_dir" + suffix_1, "dir_of_.")
        Set("level_print", 0)
        Set("next_level", 0)
        Set("ex_is_end", 0)
        Set("dirs" + suffix_2, [[]])

def set_tag_tree_print(level, is_end, ):
    if abs(is_end) == 1: # 如果 第 i 层 的 is_end 的 模 = 1，则 该层用 "    "
        # is_end 可以为 -1，表示 最末一个 最靠外的 层级
        Set("tree_print_" + str(level), Get("tree_print")[-1])
    else: # 如果 第 i 层 的 is_end != 1，则 该层用 "|    "
        Set("tree_print_" + str(level), Get("tree_print")[2])

def get_tags_tree_print(level, ):
    ex_levels_tags = "" # 查看该层 之前 所有层的 tags (is_end)
    for l in range(level):
        ex_levels_tags += Get("tree_print_" + str(l)) if Get("tree_print_" + str(l)) else Get("tree_print")[2]
        # 如果 Dict 里没 "tree_print_" + str(level) 这个名字，则 return False，则 该层 默认 用 "|    "
        # 如果有，则返回的值为 "    " 或 "|    " 而 if 这两个东西 恒为 True
    return ex_levels_tags

def info_tree_print(level, is_end=0, ):
    # 默认 is_end 为 0，即 该层不是 同类层 peer_levels 的 最后一层（与它有无 子层 sub_levels 没关系），对应 该层用 "├── "
    # 即使 is_end = 1，该层也可能有 子层 sub_levels，只需要 是同一个 py 文件里，第 2 次 print 的 就行。
    set_tag_tree_print(level, is_end, ) # 设置的是 该层的 tag，可以不传 is_end 进来
    ex_levels_tags = get_tags_tree_print(level, ) # 获取的是 该层以前的层的 tags，所以 该层的 tag 对 ex_levels_tags 不起作用
    return ex_levels_tags + (Get("tree_print")[-2] if abs(is_end)==1 else Get("tree_print")[1]) # 默认用  "├── "，否则用 "└── "
    # is_end 可以为 -1，表示 最末一个 最靠外的 层级

def tree_print(is_end=0, add_level=0): # 默认 is_end = 0 ，即 默认 该层不是 同类层的 最后一层
    # is_end > 0 后，不再新增 同级层（但仍可能 往下 继续 深入子层级），但该层 完事后，返回上一个 print_level，即 level -= 1，直到 is_end 为 0
    init_GLV_tree_print()

    list_1d = [list_1d[3] for list_1d in inspect.stack()]  # 取 inspect.stack() 每一行的第 3 列，凑成一个 1 维 行 list
    list_1d = list_1d[1:] # 删掉 list 中的 第一个 元素，因为 它总是 tree_print
    dir = ''.join(list_1d) # 把该 全员字符串的 1 维 行 list 每个元素 加起来，凑成一个 字符串，作为 dir 路径

    ex_level = Get("level_print")
    suffix_1 = "_who_called_tree_print"
    suffix_2 = "_whose_is_end_equals_to_1"
    if Get("ex_dir" + suffix_1) in dir: # 如果 上一个 调用 tree_print 的 方法 的 路径，是 这次 调用的 子集，说明 是 横着(平行/同级) 或 往下(子层)走的
        # 如果 路径 没变 或 扩增，即使 上一个 print 说是该层最后一个同级，其旗下 也可能有 子层级 sub_levels，且 其子级 的 第 2 次 同级 print 也会 level + 1
        # 如果 平着 或 往下走，是第一次重复（第 2 次 dir 无非交集），则产生了 一个新 level 即 level + 1，否则 第 2 次及以上的重复，表示 回到 相同的 py 文件，则 level 不变
        if Get("ex_dir" + suffix_1) == dir: # 如果 dir 没变
            if not Get(dir + "_" + suffix_1): # 且只重复 1 次，则产生新 level，否则不产生
                Set("level_print", ex_level + 1)  # 等 新开一个 py 文件 的 第 2 个 print， 才 print_level + 1
                Set(dir + "_" + suffix_1, True) # 表示已经 重复过 1 次了：但，同一个完全相同的 dir，在没 退出 本分支之前，只能让 level + 1 一次。
            elif Get("ex_is_end") == -1: # 允许在同一 def 里 往回跳 level，主要是 配合 add_level=1：先 level 加 1，再 level 减 1
                Set("level_print", ex_level - 1) # 但如果用 is_end=1 会 is_end 记录数 + 1，导致多向前缩进，所以得 is_end=-1 不记录。
                # 缩进多了就考虑用 is_end = -1，这个亦真亦假的 is_end
        elif (Get("ex_dir" + suffix_1) != dir) and Get("next_level") == 1: # 专门针对 新开的 子 dir 的 第一个 tree_print 提升 层级
            Set("level_print", ex_level + 1) # 其第 2 个同级 仍是可 level + 1 的，所以没有 Set(dir + "_" + suffix_1, True)
        # ex_is_end = 0 # 推测 上一个 dir 的 is_end 为 0；
        # 但这个不一定：几乎 无法预测 上级目录，也就是 每个 def 里 第一个 print 是否是该 level 的 最后一个 同级。
        # 无论 is_end 是多少，即无论是否 还有同级，其 dir 都可能有 子分支；所以这里 从“有子分支” 进来，并不能判断 上一个 ex_dir 的 is_end 的值
        # ex_dir 有子分支 ≠≠> ex_dir 的 is_end = 0
        # 这似乎需要 上上级 dir 给出暗示，那反正都要 传参的，不如将就这里的传 is_end，工作量 是一样的。
        # 每个 def 里，第一个 tree_print 的 is_end 是否为 1，需要给。
        # 如果 def 里只有 1 个 tree_print，那这个 tree_print 肯定 is_end = 1，但可能有子分支，因而可能往下走，所以往下走 的 is_end 不一定为 0。
    elif Get("ex_is_end") != 0: # 如果 新路径 不再包含 旧路径（有交集 但有 非交集：即 分叉了） 且 上一个 tree_print 的 is_end 不是 0：若是 0 则还有同级，则 level 不变，啥也不做。
        # if len(Get("dirs" + suffix_2)[-1]) > 0: # 上一个 tree_print 的 is_end 不是 0，则肯定 len(Get("dirs" + suffix_2)) 不为零
        # 且 上一个 print 说后面 没有 同级 peer_levels（ 上一个 print 的 is_end > 0）；这个判断其实可以没有，如果 is_end 只取 0 或 1 的话。
        # print(len(Get("dirs" + suffix_2)[-1])) # 多缩进了的话，启用这个
        Set("level_print", ex_level - len(Get("dirs" + suffix_2)[-1])) # level 回跳到 上一层，甚至上几层，取决于 之前声明了多少次 is_end = 1
        for DIR in Get("dirs" + suffix_2)[-1]: # 退出 上一的 dir 的 分支 之后，如果再重入那个 module（分支），是可以 继续 level + 1 的
            Set(DIR + "_" + suffix_1, False) # 所以 退出之后，得清空 之前所有 is_end = 1 但往下（子 tree） 走了 而 没来得及回跳 的 dir 的 不可重入 标记
            # 这里的 DIR 要是写成 dir，会使 下面的 Set("ex_dir" + suffix_1, dir) 中的 dir 是这里的最后一个 DIR 值
        Set("dirs" + suffix_2, Get("dirs" + suffix_2)[:-1]) # False 全 Set 之后，把最近邻 is_end = 1 的 dirs 的 储存器 删了（而不是 清空）
        # else: # 如果 名称 改变了，且 上一个 print 默认后面 还有 同级 peer_levels（ 上一个 print 的 is_end = 0），则 level 不变
        # ex_is_end = 1 # 推测 上一个 ex_dir 的 is_end 为 1，不然怎么会 退出 其分支，另开一路 分叉树？
        # 还真可以：比如从 args_SHG 中的 Cal_lc_SHG 里返回 并跳到 Cal_GxGyGz 内时
        # 延迟 一步 打印 则可 省略每个 def 里 最后一个 tree_print 里填 is_end 值的操作
        # ex_dir 没有 子分支 ==> ex_dir 的 is_end = 1，但 ex_dir 有子分支 ≠≠> ex_dir 的 is_end = 0
        # 每个 def 里，最后一个 tree_print 的 is_end 必为 1，但 第一个 tree_print 的 is_end 是否为 1，需要给；中间的其他 tree_print 的 is_end 默认为 0 。
        # 如果 def 里只有 1 个 tree_print，它既是 最后一个，也是第一个，但默认 它是最后一个。
    if len(Get("dirs" + suffix_2)) == 0: # 如果 储存了 is_end=1 的 dirs 是 单层中括号 []，给里面加个 子[]，方便后面的 往 子[] 里加东西
        Set("dirs" + suffix_2, [[]])
    if Get("ex_is_end") == 0 and Get("level_print") == ex_level + 1: # 如果 上一层/个 还有 同级，且 level + 1 了
        # print("find") # 该分隔符判断，得在 下面的 is_end = 1 判断 之前：先看前一个的 is_end 判断 是否分隔，再看自己的 is_end，判断是否 is_end 数 + 1
        if Get("dirs" + suffix_2)[-1] != []: # 且最末 没有 空容器，则最末 另起一个 空容器（加上分隔符，分开）
            Get("dirs" + suffix_2).append([]) # 不能每次都减掉 积累的所有 is_end，而是只跳到 上一个 is_end=0 隔开的地方（与最末一个 is_end=1 之间的 is_end=1 们）
    if is_end == 1: # is_end 可以为 -1，表示 最末一个 最靠外的 层级
        # 此时 必须 不让 is_end 积累数加 1，否则 会多一次 shift + tab 前向缩进；但又得用 "└── ", "    " 来显示其和其子层级。
        Get("dirs" + suffix_2)[-1].append(dir) # 只给最末一个容器里加 is_end=0 的 dir
    # print(is_end, Get("ex_is_end"), ";", Get("level_print"), ex_level)

    Set("next_level", 0)  # 及时 回归 子 dir 的 第一个 tree_print 默认不 level + 1 的原则（浪费了 强制缩进 的 机会，是不给补的，所以在 所有 if 外
    # 如果 在同一个 dir 里，想在第一次 于第 2 个 tree_print 做出子层后，再做出子层，则可考虑 添加 add_level = 1
    if add_level == 1: # 原本不加 变加（同一个 def 第 3 个 及以后的 tree_print），但得第 2 个及以后的 tree_print 就调用
        Set(dir + "_" + suffix_1, False) # 注意，add_level = 1 后，得下次 调用该 tree_print，level 才会加 1，也有滞后性
    elif add_level == 2: # 使接下来 子 dir 的第一个 tree_print 的 level 加 1（跨文件 强制提升层级）
        Set("next_level", 1) # add_level > 0 均是 强制缩进，< 0 即 克制自己
        Set(dir + "_" + suffix_1, True) # 并且 set 自己该 dir 完成了 level + 1 使命，后续 默认 若该 dir 重复，则 不加 层级了
    elif add_level == -1: # 原本加 变不加（同一个 def 第 2 个 tree_print），但得第 1 个 tree_print 就调用
        Set(dir + "_" + suffix_1, True)  # 注意，add_level = -1 后，得下次 调用该 tree_print，level 才会不加 1，也有滞后性

    # 最后才设置这些“上一次”的东西
    Set("ex_dir" + suffix_1, dir)  # 把这次调用该 print 的 路径，覆盖 之前的，方便 inform 下次是否应该 level + 1
    Set("ex_is_end", is_end)  # 储存 is_end，方便下次用 这一次的
    return info_tree_print(Get("level_print"), is_end, )

#%%

def init_GLV_rmw(U_name, ray_new, method, way, **kwargs): # kwargs 里面已经有个 ray 的键了
    from fun_os import set_ray
    ray_set = set_ray(U_name, ray_new, **kwargs)
    Set("ray", ray_set)
    Set("method", method)
    Set("way", way) #  method 肯定有，但 way 不一定有
    return ray_set

# %%

def init_accu(key, init_value=0): # 设定一个 全局变量累加器（名称、初值 默认为 0），用于不可重入的计数，或只能进行一次的过程。
    init_GLV()
    if key not in GLOBALS_DICT:
        Set(key, init_value)
    else:
        Set(key, Get(key) + 1)
    return Get(key)

def init_tkey(key):  # get_format_key：每次调用 tset 的时候，创建一个名为 key_th 的变量，且每次创建的 th 不同
    name = key + "_" + "th" # 给 th 取一个与 key 本身 有关的键，这样 对不同名的 key，都可 对他们 独立 累加 其 th
    return key + "_" + str(init_accu(name)) # 每次产生一个新的变量名

def init_tset(key, value):  # set_format_key_value：每次都 初始化一个 新的 键-值对
    name = init_tkey(key) # 初始化一个新的 key_th 名
    return Set(name, value), name # 名字也要输出出去，且最好作为局部变量（钥匙），以便之后 直接用 Set(name, value), Get(name) 即可

# %%

def fkey(key):  # get_format_key
    return Get("method") + ("_" + Get("way") if Get("way") != "" else "") + " - " + key + Get("ray") + "_z"
# 不然会 多出 一条杠 _

def fset(key, value):  # set_format_key_value
    return Set(fkey(key), value)


def fget(key):  # get_format_key_value
    return Get(fkey(key))


# %%

def ekey(key):  # get_format_key
    return Get("method") + ("_" + Get("way") if Get("way") != "" else "") + " - " + key + Get("ray") + "_z_energy"


def eset(key, value):  # set_format_key_value
    return Set(ekey(key), value)


def eget(key):  # get_format_key_value
    return Get(ekey(key))


# %%

def dkey(key):  # get_format_key
    return Get("method") + ("_" + Get("way") if Get("way") != "" else "") + " - " + key + Get("ray") + "_zdz"


def dset(key, value):  # set_format_key_value
    return Set(dkey(key), value)


def dget(key):  # get_format_key_value
    return Get(dkey(key))


# %%

def skey(key):  # get_format_key
    return Get("method") + ("_" + Get("way") if Get("way") != "" else "") + " - " + key + Get("ray") + "_z_stored"


def sset(key, value):  # set_format_key_value
    return Set(skey(key), value)


def sget(key):  # get_format_key_value
    return Get(skey(key))


# %%

def YZkey(key):  # get_format_key
    return Get("method") + ("_" + Get("way") if Get("way") != "" else "") + " - " + key + Get("ray") + "_YZ"


def YZset(key, value):  # set_format_key_value
    return Set(YZkey(key), value)


def YZget(key):  # get_format_key_value
    return Get(YZkey(key))


# %%

def XZkey(key):  # get_format_key
    return Get("method") + ("_" + Get("way") if Get("way") != "" else "") + " - " + key + Get("ray") + "_XZ"


def XZset(key, value):  # set_format_key_value
    return Set(XZkey(key), value)


def XZget(key):  # get_format_key_value
    return Get(XZkey(key))


# %%

def key1(key):  # get_format_key
    return Get("method") + ("_" + Get("way") if Get("way") != "" else "") + " - " + key + Get("ray") + "_sec1"


def set1(key, value):  # set_format_key_value
    return Set(key1(key), value)


def get1(key):  # get_format_key_value
    return Get(key1(key))


# %%

def key2(key):  # get_format_key
    return Get("method") + ("_" + Get("way") if Get("way") != "" else "") + " - " + key + Get("ray") + "_sec2"


def set2(key, value):  # set_format_key_value
    return Set(key2(key), value)


def get2(key):  # get_format_key_value
    return Get(key2(key))


# %%

def keyf(key):  # get_format_key
    return Get("method") + ("_" + Get("way") if Get("way") != "" else "") + " - " + key + Get("ray") + "_front"


def setf(key, value):  # set_format_key_value
    return Set(keyf(key), value)


def getf(key):  # get_format_key_value
    return Get(keyf(key))


# %%

def keye(key):  # get_format_key
    return Get("method") + ("_" + Get("way") if Get("way") != "" else "") + " - " + key + Get("ray") + "_end"


def sete(key, value):  # set_format_key_value
    return Set(keye(key), value)


def gete(key):  # get_format_key_value
    return Get(keye(key))


# %%

def init_SSI(g_shift, U_0,
             is_energy_evolution_on, is_stored,
             sheets_num, sheets_stored_num,
             X, Y, Iz, size_PerPixel, ):
    Ix, Iy = U_0.shape
    # Set("Ix", Ix)
    # Set("Iy", Iy)

    Set("is_energy_evolution_on", is_energy_evolution_on)
    Set("is_stored", is_stored)
    Set("sheets_num", sheets_num)
    Set("sheets_stored_num", sheets_stored_num)

    Set("X", X)
    Set("Y", Y)
    Set("th_X", Iy // 2 + int(X / size_PerPixel))
    Set("th_Y", Ix // 2 - int(Y / size_PerPixel))

    # %%

    if Get("method") == "nLA":
        dset("G", g_shift)  # 仅为了 B_2_nLA_SSI 而存在
        dset("U", U_0)  # 仅为了 B_2_nLA_SSI 而存在
    else:
        dset("G", np.zeros((Ix, Iy), dtype=np.float64()))  # 初始化 正确的 矩阵维度 Ix, Iy 和 类型 np.complex128()
        dset("U", np.zeros((Ix, Iy), dtype=np.float64())) # 初始化 正确的 矩阵维度 Ix, Iy 和 类型 np.complex128()

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

        Set("sheet_th_stored", sheet_th_stored)  # 设置成全局变量
        Set("iz_stored", iz_stored)  # 方便之后在这个 py 和 主 py 文件里直接调用
        Set("z_stored", z_stored)  # 懒得搞 返回 和 获取 这仨了

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


def init_EVV(g_shift, U_0,
             is_energy_evolution_on, is_stored,
             sheets_num, sheets_stored_num,
             Iz, size_PerPixel, ):
    Ix, Iy = U_0.shape

    Set("is_energy_evolution_on", is_energy_evolution_on)
    Set("is_stored", is_stored)
    Set("sheets_num", sheets_num)  # sheets_num 相关的 是多余的，尽管如此，也选择了 保留着
    Set("sheets_stored_num", sheets_stored_num)

    # %%

    if Get("method") == "nLA":
        dset("G", g_shift)  # 仅为了 B_2_nLA_SSI 而存在
        dset("U", U_0)  # 仅为了 B_2_nLA_SSI 而存在
    else:
        dset("G", np.zeros((Ix, Iy), dtype=np.float64()))  # 初始化 正确的 矩阵维度 Ix, Iy 和 类型 np.complex128()
        dset("U", np.zeros((Ix, Iy), dtype=np.float64()))  # 初始化 正确的 矩阵维度 Ix, Iy 和 类型 np.complex128()

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

        Set("sheet_th_stored", sheet_th_stored)  # 设置成全局变量
        Set("iz_stored", iz_stored)  # 方便之后在这个 py 和 主 py 文件里直接调用
        Set("z_stored", z_stored)  # 懒得搞 返回 和 获取 这仨了

        sset("G", np.zeros((Ix, Iy, int(sheets_stored_num + 1)), dtype=np.complex128()))
        sset("U", np.zeros((Ix, Iy, int(sheets_stored_num + 1)), dtype=np.complex128()))


# %%

def fun3(for_th, fors_num, G_zdz, *args, **kwargs, ):
    if "is_U" in kwargs and kwargs["is_U"] == 1:
        U_zdz = G_zdz
        G_zdz = fft2(U_zdz)
    else:
        U_zdz = ifft2(G_zdz)

    if Get("is_energy_evolution_on") == 1:
        eget("G")[for_th + 1] = np.sum(np.abs(G_zdz) ** 2)
        # print(eget("G")[for_th + 1])
        eget("U")[for_th + 1] = np.sum(np.abs(U_zdz) ** 2)
        # print(eget("U")[for_th + 1])

    if Get("is_stored") == 1:

        # 小写的 x,y 表示 电脑中 矩阵坐标系，大写 X,Y 表示 笛卡尔坐标系
        YZget("G")[:, for_th] = G_zdz[:, Get("th_X")]
        # X 增加，则 从 G1_z_shift 中 读取的 列 向右移，也就是 YZ 面向 列 增加的方向（G1_z_shift 的 右侧）移动
        XZget("G")[:, for_th] = G_zdz[Get("th_Y"), :]
        # Y 增加，则 从 G1_z_shift 中 读取的 行 向上移，也就是 XZ 面向 行 减小的方向（G1_z_shift 的 上侧）移动
        YZget("U")[:, for_th] = U_zdz[:, Get("th_X")]
        XZget("U")[:, for_th] = U_zdz[Get("th_Y"), :]

        # %%

        if np.mod(for_th, Get("sheets_num") // Get("sheets_stored_num")) == 0:
            # 如果 for_th 是 Get("sheets_num") // Get("sheets_stored_num") 的 整数倍（包括零），则 储存之
            Get("sheet_th_stored")[int(for_th // (Get("sheets_num") // Get("sheets_stored_num")))] = for_th + 1
            Get("iz_stored")[int(for_th // (Get("sheets_num") // Get("sheets_stored_num")))] = Get("izj")[for_th + 1]
            Get("z_stored")[int(for_th // (Get("sheets_num") // Get("sheets_stored_num")))] = Get("zj")[for_th + 1]
            sget("G")[:, :, int(for_th // (Get("sheets_num") // Get("sheets_stored_num")))] = G_zdz
            # 储存的 第一层，实际上不是 G1_0，而是 G1_dz
            sget("U")[:, :, int(for_th // (Get("sheets_num") // Get("sheets_stored_num")))] = U_zdz
            # 储存的 第一层，实际上不是 U_0，而是 U_dz

        if for_th == Get(
                "sheet_th_frontface"):  # 如果 for_th 是 sheet_th_frontface，则把结构 前端面 场分布 储存起来，对应的是 zj[sheets_num_frontface]
            setf("G", G_zdz)
            setf("U", U_zdz)
        if for_th == Get(
                "sheet_th_endface"):  # 如果 for_th 是 sheet_th_endface，则把结构 后端面 场分布 储存起来，对应的是 zj[sheets_num_endface]
            sete("G", G_zdz)
            sete("U", U_zdz)
        if for_th == Get(
                "sheet_th_sec1"):  # 如果 for_th 是 想要观察的 第一个面 前面那一层的 层序数，则 将储存之于 该层 前面那一层的 后端面（毕竟 算出来的是 z + dz） 分布中
            set1("G", G_zdz)  # 对应的是 zj[sheets_num_sec1]
            set1("U", U_zdz)
        if for_th == Get(
                "sheet_th_sec2"):  # 如果 for_th 是 想要观察的 第二个面 前面那一层的 层序数，则 将储存之于 该层 前面那一层的 后端面（毕竟 算出来的是 z + dz） 分布中
            set2("G", G_zdz)  # 对应的是 zj[sheets_num_sec2]
            set2("U", U_zdz)


def Fun3(for_th, fors_num, G_zdz, *args, **kwargs, ):
    U_zdz = ifft2(G_zdz)

    if Get("is_energy_evolution_on") == 1:
        eget("G")[for_th] = np.sum(np.abs(G_zdz) ** 2)
        eget("U")[for_th] = np.sum(np.abs(U_zdz) ** 2)

    if Get("is_stored") == 1:
        Get("sheet_th_stored")[for_th] = for_th
        Get("iz_stored")[for_th] = Get("izj")[for_th]
        Get("z_stored")[for_th] = Get("zj")[for_th]
        sget("G")[:, :, for_th] = G_zdz
        sget("U")[:, :, for_th] = U_zdz


# %%

def end_AST(z0, size_PerPixel,
            g_shift, k1_z, ):
    iz = z0 / size_PerPixel
    fset("H", np.power(math.e, k1_z * iz * 1j))
    fset("G", g_shift * fget("H"))
    fset("U", ifft2(fget("G")))

# %%

def end_STD(U1_z, g_shift,
            is_energy, n_sigma=3, ): # standard
    fset("U", U1_z)
    fset("G", fft2(fget("U")))
    fset("H", fget("G") / np.max(np.abs(fget("G"))) / (g_shift / np.max(np.abs(g_shift))))
    if n_sigma > 0:
        # 扔掉 amp 偏离 amp 均值 3 倍于 总体 标准差 以外 的 数据，保留 剩下的 3 倍 以内的 数据。
        fset("H", U_Drop_n_sigma(fget("H"), n_sigma, is_energy))

# %%

def end_SSI(g_shift, is_energy, n_sigma=3, **kwargs, ):
    if "is_U" in kwargs and kwargs["is_U"] == 1:
        fset("U", dget("U"))
        fset("G", fft2(fget("U")))
    else:
        fset("G", dget("G"))
        fset("U", ifft2(fget("G")))
    fset("H", fget("G") / np.max(np.abs(fget("G"))) / (g_shift / np.max(np.abs(g_shift))))
    if n_sigma > 0:
        # 扔掉 amp 偏离 amp 均值 3 倍于 总体 标准差 以外 的 数据，保留 剩下的 3 倍 以内的 数据。
        fset("H", U_Drop_n_sigma(fget("H"), n_sigma, is_energy))


# %%

def fGHU_plot_save(is_energy_evolution_on,  # 默认 全自动 is_auto = 1
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
                   z, **kwargs, ):
    from fun_os import GHU_plot_save

    GHU_plot_save(fget("G"), fkey("G"), is_energy_evolution_on,  # 这边 要省事 免代入 的话，得确保 提前 传入 ray,way,method 三个参数
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
                  z, **kwargs, )


# %%

def fU_SSI_plot(th_f, th_e,
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
                plot_group, is_animated,
                loop, duration, fps,
                # %%
                is_plot_3d_XYz, is_plot_selective,
                is_plot_YZ_XZ, is_plot_3d_XYZ,
                # %%
                z_1, z_2,
                z_f, z_e, z, ):
    from fun_os import U_SSI_plot
    if Get("is_stored") == 1:
        sget("G")[:, :, Get("sheets_stored_num")] = fget("G")  # 储存的 第一层，实际上不是 G1_0，而是 G1_dz
        sget("U")[:, :, Get("sheets_stored_num")] = fget("U")  # 储存的 第一层，实际上不是 U_0，而是 U_dz

        # 小写的 x,y 表示 电脑中 矩阵坐标系，大写 X,Y 表示 笛卡尔坐标系
        YZget("G")[:, Get("sheets_num")] = fget("G")[:, Get("th_X")]
        XZget("G")[:, Get("sheets_num")] = fget("G")[Get("th_Y"), :]
        YZget("U")[:, Get("sheets_num")] = fget("U")[:, Get("th_X")]
        XZget("U")[:, Get("sheets_num")] = fget("U")[Get("th_Y"), :]

        U_SSI_plot(sget("G"), fkey("G"),
                   sget("U"), fkey("U"),
                   YZget("G"), XZget("G"),
                   YZget("U"), XZget("U"),
                   get1("G"), get2("G"),
                   getf("G"), gete("G"),
                   get1("U"), get2("U"),
                   getf("U"), gete("U"),
                   Get("th_X"), Get("th_Y"),
                   Get("sheet_th_sec1"), Get("sheet_th_sec2"),
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
                   plot_group, is_animated,
                   loop, duration, fps,
                   # %%
                   is_plot_3d_XYz, is_plot_selective,
                   is_plot_YZ_XZ, is_plot_3d_XYZ,
                   # %%
                   Get("X"), Get("Y"),  # 这个也可 顺便 设成 global 的，懒得搞
                   z_1, z_2,
                   z_f, z_e,
                   Get("zj"), Get("z_stored"), z, )


def fU_EVV_plot(img_name_extension,
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
                is_colorbar_on, is_energy,
                # %%
                plot_group, is_animated,
                loop, duration, fps,
                # %%
                is_plot_3d_XYz,
                # %%
                z, ):
    from fun_os import U_EVV_plot
    if Get("is_stored") == 1:
        sget("G")[:, :, Get("sheets_stored_num")] = fget("G")  # 储存的 第一层，实际上不是 G1_0，而是 G1_dz
        sget("U")[:, :, Get("sheets_stored_num")] = fget("U")  # 储存的 第一层，实际上不是 U_0，而是 U_dz

        U_EVV_plot(sget("G"), fkey("G"),
                   sget("U"), fkey("U"),
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
                   is_colorbar_on, is_energy,
                   # %%
                   plot_group, is_animated,
                   loop, duration, fps,
                   # %%
                   is_plot_3d_XYz,
                   # %%
                   Get("zj"), Get("z_stored"), z, )
