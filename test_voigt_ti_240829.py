#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: franziskaz

plots the ZIMPOL I, Q,U V image from GREGOR campaign

ver. 06.01.2025

"""


import numpy as np

from astropy.io import fits
from matplotlib import gridspec
import matplotlib.pyplot as plt
from style_all import style_aa

from matplotlib.ticker import (NullFormatter,
                               MaxNLocator,
                               MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)

import syn_tools as syn
import line_tools as lt
import z3extra as z3 

from scipy.special import voigt_profile
from lmfit.models import VoigtModel, ConstantModel , SkewedVoigtModel

###################
# Definitions
###################

# reshaping
def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

# Definitions


def gaus(x, a, x0, sigma, offset):
    return offset+a*np.exp(-(x-x0)**2/(2*sigma**2))


def gaus_der(x, a, x0, sigma):
    return a*(x-x0)*np.exp(-(x-x0)**2/(2*sigma**2))

def read_data(directory_sp, sp_files, sp_roi):
    
    sif_sp = fits.open(directory_sp+sp_files[0]+'.fits')
    sp_header=sif_sp[0].header
    sample=np.array([float(sp_header['XSCALE']),float(sp_header['XSTART'])])


    data=np.array(sif_sp[0].data)
    data=data[:,:,sp_roi[0]:sp_roi[1],sp_roi[2]:sp_roi[3]]

    # dimensions: time, space, spectrum
    si=data[:,0,:,:]
    si=si/si.max()
    sq=data[:,1,:,:]
    sq=100*sq
    su=data[:,2,:,:]
    su=100*su
    sv=data[:,3,:,:]
    sv=100*sv
    
    sif_sp.close()
    
    return(si,sq,su,sv,sample)
##%%

def initialize_parameter(params, ti_center_wl):
   
    params['A_amplitude'].value= -0.09
    params['A_center'].value = ti_center_wl
    params['A_sigma'].value= 0.03
    params['A_gamma'].value= 1
    #params['A_skew'].value= 1
    
    # params['A_parasite_amplitude'].value= -0.07
    # params['A_parasite_center'].value = 4536.60
    # params['A_parasite_sigma'].value= 0.03
    # params['A_parasite_gamma'].value= 1
    
    params['B_amplitude'].value= -0.09
    params['B_center'].value = 4536.42
    params['B_sigma'].value= 0.03
    params['B_gamma'].value= 1
    
    params['C_amplitude'].value= -0.09
    params['C_center'].value = 4536.20
    params['C_sigma'].value= 0.03
    params['C_gamma'].value= 1
    
    params['D_amplitude'].value= -0.05
    params['D_center'].value = 4536.3
    params['D_sigma'].value= 0.03
    params['D_gamma'].value= 1
    
    params['c_c'].value= 1.

    return(params)

class line():
    
    def __init__(self, wvl, profile, center, width):
        
        self.wvl = wvl
        
        self.profile = profile

# fit blue and red part of profile with two different models
# voigt model has four Parameters: amplitude, center, sigma, and gamma, skewed:  skew
# skew is not helping too much..
        voigtA = VoigtModel(prefix='A_')  # Ti I
        pars = voigtA.make_params()
        # voigtAParasite = VoigtModel(prefix='A_parasite_')  # Ti I
        # pars.update(voigtAParasite.make_params())

        voigtB = VoigtModel(prefix='B_')
        pars.update(voigtB.make_params())
        voigtC = VoigtModel(prefix='C_')
        pars.update(voigtC.make_params())
        voigtD = VoigtModel(prefix='D_')
        pars.update(voigtD.make_params())

        const = ConstantModel(prefix='c_')
        pars.update(const.make_params())

        pars = initialize_parameter(pars, center)

        # build model, but only fit it for specific wavelength without red wing of T
        model_blue = voigtA  + voigtB + voigtC +  voigtD + const #+ voigtAParasite

        # build model, but only fit it for specific wavelength in red wing of T
        model_red = voigtA + const #+ voigtAParasite

        # find wavelength points which belong to the line
        idx_left = np.argmin(abs(self.wvl - (center-width/2.)))
        idx_right = np.argmin(abs(self.wvl - (center+width/2.)))

        # find center of the line to cut into blue and red part
        self.idx_center= np.argmin(self.profile[idx_left:idx_right])+2  # should be improved by fitting
        
        # adding 2 more pixel to get the minimum right
        pixel_overlap = 2
        self.profile_red = self.profile[idx_left+self.idx_center-pixel_overlap:]
        self.wvl_red = self.wvl[idx_left+self.idx_center-pixel_overlap:]

        self.profile_blue = profile[:idx_left+self.idx_center+pixel_overlap]
        self.wvl_blue = self.wvl[:idx_left+self.idx_center+pixel_overlap]
        
       
        self.out_blue = model_blue.fit(self.profile_blue, pars, x=self.wvl_blue)
        
       
        components_blue = self.out_blue.eval_components(x=self.wvl_blue)
        
        self.component_ti_blue =  components_blue['A_']+components_blue['c_']    
        self.out_red = model_red.fit(self.profile_red, pars, x=self.wvl_red )
        
        # for the red part of the profile to extract Ti, I need to construct a new profile
        
        components_blue_full = self.out_blue.eval_components(x=self.wvl)
        self.new_profile = self.profile - (components_blue_full['B_']+components_blue_full['C_']  +components_blue_full['D_'])
        self.out_red_ti = model_red.fit(self.new_profile[idx_left+self.idx_center-pixel_overlap:], pars, x=self.wvl_red ) # only red part of new profile
        components_red = self.out_red_ti.eval_components(x=self.wvl_red)
        self.component_ti_red =  components_red['A_']+components_red['c_']
        
        

###########################
#
# Font

zoom=1 # zoom factor
my_dpi, fig_size_single, fig_size, params=style_aa(zoom)
plt.style.use('default')
plt.rcParams.update(params)
nullfmt=NullFormatter()

intmeth = 'none' # imshow integration method

# -----------------------------

obs_date='240829'

directory_sp='/home/franziskaz/data/gregor/ti_2024/'+obs_date+'/reduced/'

measurements=['4530_m6_temp', '4530_m5_temp'] # no fits ending

sp_files=[measurements[0]]

xstart=160 # in pixel
xend = 280 #in pixel

# -----------------------------

sp_roi=[0,140,xstart,xend] # x and y sampling of spectrograph, 300-800

si,sq,su,sv,sample= read_data(directory_sp, sp_files, sp_roi)
image_sz=np.array(si.shape)

wavelengthscale=np.arange(sample[1]+xstart*sample[0],\
               xstart*sample[0]+sample[1]+sample[0]*image_sz[2],sample[0]) # the grid to which your data corresponds
    
if wavelengthscale.shape[0] > si.shape[2]:
    wavelengthscale = wavelengthscale[:-1]
if wavelengthscale.shape[0] < si.shape[2]:
    print('Error')
# -----------------------------
 
ti_center_wl = 4536.55 # Ti
cont_center_wl = 4537.25

spec_region_width=0.1 # /2 +/- center_wl

####################
# plotting parameter
##################

i_gscale=[0.2,0.95]# greyscale for si
lin_gscale1=[-0.03,0.03] # grayscale for linear polarization
lin_gscale2=[-0.03,0.03] # grayscale for linear polarization
circ_gscale=[-0.1,0.1] # grayscale for linear polarization

line_color1 = 'red'
line_color2 = 'lightseagreen'
line_color3 = 'blue'
line_color4 = 'indigo'

# data processing

idx_continuum=np.argmin(abs(wavelengthscale-4536.7)) #continuum
intergranule = line( wavelengthscale, si[0,np.argmin(si[0,:,idx_continuum]),  :], ti_center_wl,  spec_region_width)
granule = line( wavelengthscale, si[0,np.argmax(si[0,:,idx_continuum]),  :], ti_center_wl,  spec_region_width)


# components_intergranule = out_intergranule.eval_components(x=wavelengthscale)
# components_granule = out_granule.eval_components(x=wavelengthscale)

#%%
  # --------------------------------------------------------
  # PLOT STARTS
  # -------------------------------------------------------
f=plt.figure(num=1) 
f.set_size_inches(fig_size,forward=True)
plt.clf()

gs = gridspec.GridSpec(2,1)#, width_ratios=[1,1,1,1] , height_ratios=[1,1,1,1] )
gs.update(top=0.94, bottom=0.11,left=0.09, right=0.9, wspace=0.15, hspace=0.2)

  #--------------------------------------------------------

ax=f.add_subplot(gs[0,0])

ax.get_yaxis().set_visible(True)
 

  # ------------- Stokes I image
  
im=ax.imshow(si[0,:], cmap='gray',origin='lower',\
              interpolation=intmeth,vmin=i_gscale[0], vmax=i_gscale[1], 
              extent=[wavelengthscale[0],wavelengthscale[-1],0,image_sz[1]], aspect='auto')#  
pos=ax.get_position()
cbaxes = f.add_axes([pos.x1, pos.y0, zoom*0.01*(pos.x1-pos.x0), (pos.y1-pos.y0)])
cb = plt.colorbar(im,orientation='vertical',cax=cbaxes,ticks=[i_gscale[0],i_gscale[1]])
cb.set_label('$I/I_c$ [a.u.]')


ax.axvline(x=ti_center_wl-spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted')
ax.axvline(x=ti_center_wl+spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted')


# ax.axvline(x=cont_center_wl-spec_region_width/2., color=line_color3, linewidth=1, linestyle='dotted')
# ax.axvline(x=cont_center_wl+spec_region_width/2., color=line_color3, linewidth=1, linestyle='dotted')
ax.set_title('ZIMPOL@GREGOR, 2024-08-29, $\mu=0.6$')
ax.set_ylabel(r'X along slit [pixel]')
ax.get_xaxis().set_visible(False)

  # ------------- Stokes I spectrum
  
ax1=f.add_subplot(gs[1,0])  

ax1.plot( granule.wvl, granule.profile, color=line_color4, marker='+',linestyle='None', label = 'Granule') 
ax1.plot( granule.wvl_blue[:-2], granule.out_blue.best_fit[:-2], '-', color='blue', label='best fit granule blue')
ax1.plot( granule.wvl_red[2:], granule.out_red.best_fit[2:], '-', color='red', label='best fit granule red')
ax1.plot( granule.wvl_blue[:-2], granule.component_ti_blue[:-2], '-', marker='.', color='blue', label='best fit granule, Ti')
ax1.plot( granule.wvl_red[2:], granule.component_ti_red[2:],'-',  marker='.', color='red')
# ax1.plot(wavelengthscale, components_granule_red['A_']+components_granule['c_'], '-', color='blue', label='best fit granule, Ti')
# ax1.plot(wavelengthscale, components_granule_blue['A_']+components_granule['c_'], '-', color='magenta', label='best fit granule, Ti')



ax1.plot(intergranule.wvl, intergranule.profile, color=line_color2, marker='+',linestyle='None', label = 'Intergranule') 
ax1.plot( intergranule.wvl_blue[:-2], intergranule.out_blue.best_fit[:-2], '-', color='blue', label='best fit intergranule blue')
ax1.plot( intergranule.wvl_red[2:], intergranule.out_red.best_fit[2:], '-', color='red', label='best fit intergranule red')
ax1.plot( intergranule.wvl_blue[:-2], intergranule.component_ti_blue[:-2], '-', marker='.', color='green', label='best fit intergranule, Ti')
ax1.plot( intergranule.wvl_red[2:], intergranule.component_ti_red[2:],'-',  marker='.', color='magenta')
# ax1.plot(wavelengthscale, si[0,:].mean(axis=0), color='black', label = 'Spatial mean') 

ax.axhline(y=np.argmin(si[0,:,idx_continuum]), color=line_color2, linewidth=1, linestyle='dotted')
ax.axhline(y=np.argmax(si[0,:,idx_continuum]), color=line_color4, linewidth=1, linestyle='dotted')

ax1.axvline(x=ti_center_wl-spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted')
ax1.axvline(x=ti_center_wl+spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted', label='Ti I')
ax1.legend()


ax1.set_xlabel(r'Wavelength $[\AA]$')  
ax1.set_ylabel(r'Counts [a.u.]')


plt.savefig('/home/franziskaz/figures/ti_gregor_2024/'+obs_date+'_'+measurements[0]+'_test_voigt.png'
          , dpi=my_dpi, metadata={'Software': __file__})




