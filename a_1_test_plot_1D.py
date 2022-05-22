# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:30:27 2022

@author: Xcz
"""

import random
import datetime

from pyecharts.charts import *
from pyecharts.components import Table
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from pyecharts.globals import CurrentConfig
CurrentConfig.ONLINE_HOST = "https://cdn.kesci.com/lib/pyecharts_assets/"

import os
from fun_global_var import init_GLV_DICT, Get

init_GLV_DICT()
title = "test"
html_address = Get("root_dir") + '\\' + title + ".html"

x_data = ['Apple', 'Huawei', 'Xiaomi', 'Oppo', 'Vivo', 'Meizu']
y_data = [123, 153, 89, 107, 98, 23]


bar = (Bar()
       .add_xaxis(x_data)
       .add_yaxis('', y_data)
      )

bar.render(html_address)
os.startfile(html_address)