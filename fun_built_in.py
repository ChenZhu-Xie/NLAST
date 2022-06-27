# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:11 2021

@author: Xcz
"""

#%%
'''
既然 python 你要这么为难我，我就直接让 函数返回的对象 丢弃其外层的 圆括号，并加方括号，
使得方括号内 没有更多的方括号或圆括号，并且最外层的方括号不消失，
这样无论如何都可以解包，且只需要解一次包，就直达内层
'''

def var_or_tuple_to_list(var_or_tuple):

    if type(var_or_tuple) == tuple: 
        var_or_tuple = list(var_or_tuple)
    else:
        var_or_tuple = [var_or_tuple]
        
    return var_or_tuple

def get_var_name(var):
    
    # 这里的 name 就是 key，但可能 一个 var 对应多个 key，比如 a,b,c 都指向 同一块 内存区域，多 key 对 1 值，比较鸡肋
    for name, value in globals().items():
        if value is var:
            return name

# b = 45
# a = 45
# print(get_var_name(a))

def str_to_float(Str):
    try:
        float(Str)
        return float(Str)
    except:
        return Str