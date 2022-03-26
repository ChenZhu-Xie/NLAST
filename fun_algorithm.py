# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

#%%

def find_nearest(array, goal):
    index = np.abs(array - goal).argmin()
    return index, array.flat[index]