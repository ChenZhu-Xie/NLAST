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