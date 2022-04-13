# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import re
import numpy as np
import math

#%%

def find_nearest(array, goal):
    index = np.abs(array - goal).argmin()
    return index, array.flat[index]

#%%

def num_of_decimal_places(f): # float
    return len(str(f).split(".")[1]) if "." in str(f) else 0

def the_effective_nums(f): # float
    f_str = str(f)
    first_non_zero = re.findall('[1-9]', f_str)[0]
    pos_non_zero = f_str.find(first_non_zero)
    return int(f_str[pos_non_zero:])

def gcd_of_float(f): # float
    molecule = the_effective_nums(f)
    denominator = 10 ** num_of_decimal_places(f)
    gcd = math.gcd(molecule, denominator)
    return gcd / denominator

#%%