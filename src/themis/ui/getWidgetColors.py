#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 16:09:00 2023

@author: zeuner
"""

# colorsystem.py is the full list of colors that can be used to easily create themes.

from PyQt5 import QtGui

class Gray:
    B0 = '#000000' #black
    B10 = '#19232D'#dark-gray-blue
    B20 = '#293544'
    B30 = '#37414F'
    B40 = '#455364'
    B50 = '#54687A'
    B60 = '#60798B'
    B70 = '#788D9C'
    B80 = '#9DA9B5'
    B90 = '#ACB1B6'
    B100 = '#B9BDC1'
    B110 = '#C9CDD0'
    B120 = '#CED1D4'
    B130 = '#E0E1E3'
    B140 = '#FAFAFA'
    B150 = '#FFFFFF'


class Blue:
    B0 = '#000000'
    B10 = '#062647'
    B20 = '#26486B'
    B30 = '#375A7F'
    B40 = '#346792'
    B50 = '#1A72BB'
    B60 = '#057DCE'
    B70 = '#259AE9'
    B80 = '#37AEFE'
    B90 = '#73C7FF'
    B100 = '#9FCBFF'
    B110 = '#C2DFFA'
    B120 = '#CEE8FF'
    B130 = '#DAEDFF'
    B140 = '#F5FAFF'
    B150 = '##FFFFFF'
    
class getWidgetColors():
    
    #backgrounds
    BG_DARK = QtGui.QColor(Gray.B0) 
    BG_NORMAL = QtGui.QColor(Gray.B10)
    BG_LIGHT = QtGui.QColor(Gray.B20)
    
    #fonts
    FG_DARK = QtGui.QColor(Gray.B110)
    FG_NORMAL = QtGui.QColor(Gray.B130)
    FG_LIGHT = QtGui.QColor(Gray.B80)
    
    #highlight
    HG_DARK = QtGui.QColor(Blue.B80)
    HG_NORMAL = QtGui.QColor(Blue.B20)
    HG_LIGHT = QtGui.QColor(Blue.B40)