#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 11:34:04 2025

@author: franziskaz

determine spatial scale of spectral cameras by using GREGOR/HiFI images
"""

import imageio.v3 as iio
import numpy as np

# Step 1: load reconstructed THEMIS context (630 nm) and HiFI TiO (705 nm) --> difference of 5% in terms of size

    
themis = iio.imread('/home/franziskaz/projects/themis/images/context-2025-07-05/deppng/z01_20250705_11022611_3.5s3k5ip100fr_decpsf.png')

# Step 2: Find 