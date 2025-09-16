#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:57:34 2020

@author: franziskaz
style sheet

make sure that you installed:
    
    - for fonts: conda install -c conda-forge mscorefonts (after that: rm ~/.cache/matplotlib -rf)


"""




def style_aa(zoom):

 params = {'backend': 'pdf',
          'axes.titlesize':9*zoom,
          'axes.labelsize': 9*zoom,
          'axes.linewidth': zoom*0.5,
          'xtick.major.size': zoom*3,
          'ytick.major.size': zoom*4,
          'xtick.major.width': zoom*0.5,
          'ytick.major.width': zoom*0.5,
          'axes.edgecolor': 'black',
          'grid.color': 'black',
          'font.size': 10*zoom,
          'legend.fontsize': 8*zoom,
          'font.family': 'serif',
          'font.serif': 'DejaVu Sans',  #'font.serif': 'DejaVu Sans',
          'mathtext.fontset' : 'cm', 
          'xtick.labelsize': 8*zoom,
          'ytick.labelsize': 8*zoom,       
          'lines.dash_joinstyle': 'round',
          'lines.dash_capstyle': 'round',
          'lines.solid_joinstyle': 'round',
          'lines.solid_capstyle': 'round',
          'lines.markersize': 4*zoom,
          'lines.linewidth': 0.6*zoom,
          'text.usetex': False}
 
 my_dpi = 300
 golden_mean = 1/1.61         # Aesthetic ratio
 fig_width_in = 6.75*zoom  # Get this from https://www.overleaf.com/project/61094147597423658d92716b
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean
 fig_size_double = [fig_width, fig_height]
 
 fig_width_in = fig_width_in/2.  # Get this from https://www.overleaf.com/project/61094147597423658d92716b
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean
 fig_size_single = [fig_width, fig_height]
 
 return(my_dpi, fig_size_single, fig_size_double, params)

def style_aa_small(zoom):

 params = {'backend': 'pdf',
          'axes.titlesize':9*zoom,
          'axes.labelsize': 9*zoom,
          'axes.linewidth': zoom*0.5,
          'xtick.major.size': zoom*3,
          'ytick.major.size': zoom*4,
          'xtick.major.width': zoom*0.5,
          'ytick.major.width': zoom*0.5,
          'axes.edgecolor': 'black',
          'grid.color': 'black',
          'font.size': 10*zoom,
          'legend.fontsize': 8*zoom,
          'font.family': 'serif',
          'font.serif': 'Nimbus Roman No9 L',  #'font.serif': 'DejaVu Sans',
          'mathtext.fontset' : 'cm', 
          'xtick.labelsize': 8*zoom,
          'ytick.labelsize': 8*zoom,       
          'lines.dash_joinstyle': 'round',
          'lines.dash_capstyle': 'round',
          'lines.solid_joinstyle': 'round',
          'lines.solid_capstyle': 'round',
          'lines.markersize': 4*zoom,
          'lines.linewidth': 0.6*zoom,
          'text.usetex': False}
 return(params)


def style_normal(zoom):

 params = {'backend': 'pdf',
           'axes.titlesize':12*zoom,
           'axes.labelsize': 12*zoom,
           'axes.linewidth': zoom*0.5,
           'xtick.major.size': zoom*3,
           'ytick.major.size': zoom*4,
           'xtick.major.width': zoom*0.5,
           'ytick.major.width': zoom*0.5,
           'axes.edgecolor': 'black',
           'grid.color': 'black',
           'font.size': 11*zoom,
           'legend.fontsize': 11*zoom,
           'font.family': 'serif',
           'font.serif': 'Nimbus Roman No9 L',#'Nimbus Roman No9 L',  #'font.serif': 'DejaVu Sans',
           'mathtext.fontset' : 'cm', 
           'xtick.labelsize': 11*zoom,
           'ytick.labelsize': 11*zoom,       
           'lines.dash_joinstyle': 'round',
           'lines.dash_capstyle': 'round',
           'lines.solid_joinstyle': 'round',
           'lines.solid_capstyle': 'round',
           'lines.markersize': 4*zoom,
           'lines.linewidth': 1*zoom,
          'text.usetex': False}
 
 my_dpi = 300
 golden_mean = 1/1.61         # Aesthetic ratio
 fig_width_in = 6.75*zoom  
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean
 fig_size_double = [fig_width, fig_height]
 
 fig_width_in = fig_width_in/2. 
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean
 fig_size_single = [fig_width, fig_height]
 
 return(my_dpi, fig_size_single, fig_size_double, params)
 

def style_apj(zoom):

 params = {'backend': 'pdf',
          'axes.titlesize':10*zoom,
          'axes.labelsize': 10*zoom,
          'axes.linewidth': zoom*0.5,
          'axes.edgecolor': 'black',
          'xtick.major.size': zoom*3,
          'ytick.major.size': zoom*4,
          'xtick.major.width': zoom*0.5,
          'ytick.major.width': zoom*0.5,
          'grid.color': 'black',
          'grid.alpha': 0.1,
          'font.size': 11*zoom,
          'legend.fontsize': 8*zoom,
          'legend.framealpha': 0.7,
          'font.family': 'serif',
          'font.serif': 'Nimbus Roman No9 L',  #'font.serif': 'Nimbus Roman No9 L',
          'mathtext.fontset' : 'cm', 
          'xtick.labelsize': 10*zoom,
          'ytick.labelsize': 10*zoom,
          'lines.dash_joinstyle': 'round',
          'lines.dash_capstyle': 'round',
          'lines.solid_joinstyle': 'round',
          'lines.solid_capstyle': 'round',
          'lines.markersize': 1*zoom,
          'lines.linewidth': 1*zoom,
          'text.usetex': False}
 
 my_dpi = 300
 golden_mean = 1/1.61         # Aesthetic ratio
 fig_width_in = 6.75*zoom  # Get this from https://www.overleaf.com/project/61094147597423658d92716b
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean
 fig_size_double = [fig_width, fig_height]
 
 fig_width_in = fig_width_in/2.  # Get this from https://www.overleaf.com/project/61094147597423658d92716b
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean
 fig_size_single = [fig_width, fig_height]
 
 return(my_dpi, fig_size_single, fig_size_double, params)

def style_apj_legacy(zoom):
    
 params = {'backend': 'pdf',
          'axes.titlesize':8*zoom,
          'axes.labelsize': 9*zoom,
          'axes.linewidth': zoom*0.5,
          'axes.edgecolor': 'black',
          'grid.color': 'black',
          'xtick.major.size': zoom*3,
          'ytick.major.size': zoom*4,
          'xtick.major.width': zoom*0.5,
          'ytick.major.width': zoom*0.5,
          'grid.color': 'black',
          'grid.alpha': 0.1,
          'font.size': 9*zoom,
          'legend.fontsize': 8*zoom,
          'legend.framealpha': 0.7,
          'font.family': 'serif',
          'font.serif': 'DejaVu Sans',  #'font.serif': 'Nimbus Roman No9 L',
          'mathtext.fontset' : 'cm', 
          'xtick.labelsize': 9*zoom,
          'ytick.labelsize': 9*zoom,
          'lines.dash_joinstyle': 'round',
          'lines.dash_capstyle': 'round',
          'lines.solid_joinstyle': 'round',
          'lines.solid_capstyle': 'round',
          'lines.markersize': 0.5*zoom,
          'lines.linewidth': 0.6*zoom,

          'text.usetex': False}
 
 my_dpi = 200
 golden_mean = 1/1.61         # Aesthetic ratio
 fig_width_in = 6.75*zoom  # Get this from https://www.overleaf.com/project/61094147597423658d92716b
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean
 fig_size_double = [fig_width, fig_height]
 
 fig_width_in = fig_width_in/2.  # Get this from https://www.overleaf.com/project/61094147597423658d92716b
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean*2.
 fig_size_single = [fig_width, fig_height]
 
 return(my_dpi, fig_size_single, fig_size_double, params)



def style_spie(zoom): #https://www.spiedigitallibrary.org/journals/journal-of-astronomical-telescopes-instruments-and-systems/author-guidelines?SSO=1

 params = {'backend': 'pdf',
    'axes.titlesize':9*zoom,
    'axes.labelsize': 9*zoom,
    'axes.linewidth': zoom*0.5,
    'xtick.major.size': zoom*3,
    'ytick.major.size': zoom*4,
    'xtick.major.width': zoom*0.5,
    'ytick.major.width': zoom*0.5,
    'axes.edgecolor': 'black',
    'grid.color': 'black',
    'grid.alpha': 0.2,
    'grid.linestyle': 'dotted',
    'font.size': 10*zoom,
    'legend.fontsize': 8*zoom,
    'font.family': 'serif',
    'font.serif': 'Nimbus Roman No9 L',  #'font.serif': 'DejaVu Sans',
    'mathtext.fontset' : 'cm', 
    'xtick.labelsize': 8*zoom,
    'ytick.labelsize': 8*zoom,       
    'lines.dash_joinstyle': 'round',
    'lines.dash_capstyle': 'round',
    'lines.solid_joinstyle': 'round',
    'lines.solid_capstyle': 'round',
    'lines.markersize': 4*zoom,
    'lines.linewidth': 0.6*zoom,
    'text.usetex': False}
 
 my_dpi = 300
 golden_mean = 1/1.61         # Aesthetic ratio
 fig_width_in = 6.75*zoom  # Get this from https://www.overleaf.com/project/61094147597423658d92716b
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean
 fig_size_double = [fig_width, fig_height]
 
 fig_width_in = fig_width_in/2.  # Get this from https://www.overleaf.com/project/61094147597423658d92716b
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean
 fig_size_single = [fig_width, fig_height]
 
 return(my_dpi, fig_size_single, fig_size_double, params)


def style_interactive(zoom):

 params = {'backend': 'pdf',
    'axes.titlesize':9*zoom,
    'axes.labelsize': 9*zoom,
    'axes.linewidth': zoom*0.5,
    'xtick.major.size': zoom*3,
    'ytick.major.size': zoom*4,
    'xtick.major.width': zoom*0.5,
    'ytick.major.width': zoom*0.5,
    'axes.edgecolor': 'black',
    'grid.color': 'black',
    'grid.alpha': 0.2,
    'grid.linestyle': 'dotted',
    'font.size': 10*zoom,
    'legend.fontsize': 8*zoom,
    'font.family': 'serif',
    'font.serif': 'DejaVu Sans', 
    'mathtext.fontset' : 'cm', 
    'xtick.labelsize': 8*zoom,
    'ytick.labelsize': 8*zoom,       
    'lines.dash_joinstyle': 'round',
    'lines.dash_capstyle': 'round',
    'lines.solid_joinstyle': 'round',
    'lines.solid_capstyle': 'round',
    'lines.markersize': 4*zoom,
    'lines.linewidth': 0.8*zoom,
    'text.usetex': False}
 
 my_dpi = 200
 golden_mean = 1/1.61         # Aesthetic ratio
 fig_width_in = 15*zoom  # Get 
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean
 fig_size_double = [fig_width, fig_height]
 
 fig_width_in = fig_width_in/2.  # Get this from 
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean
 fig_size_single = [fig_width, fig_height]
 
 return(my_dpi, fig_size_single, fig_size_double, params)


def style_gant(zoom):
   

 params = {'backend': 'pdf',
    'axes.titlesize':9*zoom,
    'axes.labelsize': 9*zoom,
    'axes.linewidth': zoom*0.5,
    'xtick.major.size': zoom*3,
    'ytick.major.size': zoom*4,
    'xtick.major.width': zoom*0.5,
    'ytick.major.width': zoom*0.5,
    'axes.edgecolor': 'black',
    'grid.color': 'black',
    'grid.alpha': 0.1,
    'grid.linestyle': 'dotted',
    'font.size': 8*zoom,
    'legend.fontsize': 7*zoom,
    'legend.framealpha': 0.7,
    'font.family': 'serif',
    'font.serif': 'Nimbus Roman No9 L',  #'font.serif': 'DejaVu Sans',
    'mathtext.fontset' : 'cm', 
    'xtick.labelsize': 7*zoom,
    'xtick.bottom': False,
    'xtick.top' : False,
    'ytick.labelsize': 7*zoom,     
    'ytick.left' : False,
    'lines.dash_joinstyle': 'round',
    'lines.dash_capstyle': 'round',
    'lines.solid_joinstyle': 'round',
    'lines.solid_capstyle': 'round',
    'lines.markersize': 0.5*zoom,
    'lines.linewidth': 0.6*zoom,
    'text.usetex': False}
 
 my_dpi = 300
 golden_mean = 1/1.61         # Aesthetic ratio
 fig_width_in = 6.7*zoom  # Get this from word document
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean
 fig_size_double = [fig_width, fig_height]
 
 fig_width_in = fig_width_in/2.  
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean
 fig_size_single = [fig_width, fig_height]
 
 return(my_dpi, fig_size_single, fig_size_double, params)

def style_gant_milic(zoom):
   

 params = {'backend': 'pdf',
    'axes.titlesize':10*zoom,
    'axes.labelsize': 10*zoom,
    'axes.linewidth': zoom*0.5,
    'xtick.major.size': zoom*3,
    'ytick.major.size': zoom*4,
    'xtick.major.width': zoom*0.5,
    'ytick.major.width': zoom*0.5,
    'axes.edgecolor': 'black',
    'grid.color': 'black',
    'grid.alpha': 0.1,
    'grid.linestyle': 'dotted',
    'font.size': 10*zoom,
    'legend.fontsize': 9*zoom,
    'legend.framealpha': 0.7,
    'font.family': 'serif',
    'font.serif': 'Nimbus Roman No9 L',  #'font.serif': 'DejaVu Sans',
    'mathtext.fontset' : 'cm', 
    'xtick.labelsize': 9*zoom,
    'xtick.bottom': False,
    'xtick.top' : False,
    'ytick.labelsize': 9*zoom,     
    'ytick.left' : False,
    'lines.dash_joinstyle': 'round',
    'lines.dash_capstyle': 'round',
    'lines.solid_joinstyle': 'round',
    'lines.solid_capstyle': 'round',
    'lines.markersize': 0.5*zoom,
    'lines.linewidth': 0.6*zoom,
    'text.usetex': False}
 
 my_dpi = 300
 golden_mean = 1/1.61         # Aesthetic ratio
 fig_width_in = 6.7*zoom  # Get this from word document
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean
 fig_size_double = [fig_width, fig_height]
 
 fig_width_in = fig_width_in/2.  
 fig_width = fig_width_in  # width in inches
 fig_height = fig_width*golden_mean
 fig_size_single = [fig_width, fig_height]
 
 return(my_dpi, fig_size_single, fig_size_double, params)