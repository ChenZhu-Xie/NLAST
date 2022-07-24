# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:30:27 2022

@author: Xcz
"""

import math
from typing import Union

import os
from fun_global_var import init_GLV_DICT, Get


import pyecharts.options as opts
from pyecharts.charts import Surface3D

"""
Gallery 使用 pyecharts 1.1.0
参考地址: https://echarts.apache.org/examples/editor.html?c=surface-wave&gl=1
源代码：https://gallery.pyecharts.org/#/Surface3D/surface_wave

目前无法实现的功能:

1、暂时无法设置光滑表面 wireframe
2、暂时无法把 visualmap 进行隐藏
"""


def float_range(start: int, end: int, step: Union[int, float], round_number: int = 2):
    """
    浮点数 range
    :param start: 起始值
    :param end: 结束值
    :param step: 步长
    :param round_number: 精度
    :return: 返回一个 list
    """
    temp = []
    while True:
        if start < end:
            temp.append(round(start, round_number))
            start += step
        else:
            break
    return temp


def surface3d_data():
    for y in float_range(-3, 3, 0.05):
        for x in float_range(-3, 3, 0.05):
            z = math.sin(x ** 2 + y ** 2) * x / 3.14
            yield [x, y, z]

def surface3d_data2():
    for y in float_range(-3, 3, 0.05):
        for x in float_range(-3, 3, 0.05):
            z = math.sin(x ** 2 + y ** 2) * x
            yield [x, y, z]

print(list(surface3d_data()))

init_GLV_DICT()
html_address = Get("root_dir") + '\\' + "surface_wave" + ".html"

(
    Surface3D(init_opts=opts.InitOpts(width="1600px", height="800px")).add()
    .add(
        series_name="",
        shading="color",
        data=list(surface3d_data()),
        xaxis3d_opts=opts.Axis3DOpts(type_="value"),
        yaxis3d_opts=opts.Axis3DOpts(type_="value"),
        zaxis3d_opts=opts.Axis3DOpts(type_="value"),
        grid3d_opts=opts.Grid3DOpts(width=100, height=40, depth=100),
    )
    # .add(
    #     series_name="",
    #     shading="color",
    #     data=list(surface3d_data2()),
    #     xaxis3d_opts=opts.Axis3DOpts(type_="value"),
    #     yaxis3d_opts=opts.Axis3DOpts(type_="value"),
    #     zaxis3d_opts=opts.Axis3DOpts(type_="value"),
    #     grid3d_opts=opts.Grid3DOpts(width=100, height=40, depth=100),
    # )
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
            dimension=2,
            max_=1,
            min_=-1,
            range_color=[
                "#313695",
                "#4575b4",
                "#74add1",
                "#abd9e9",
                "#e0f3f8",
                "#ffffbf",
                "#fee090",
                "#fdae61",
                "#f46d43",
                "#d73027",
                "#a50026",
            ],
        )
    )
    .render(html_address)
)

os.startfile(html_address)