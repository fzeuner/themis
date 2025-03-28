#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:24:27 2025

@author: zeuner
"""

import themis_tools as tt
import themis_display_scan as display
import themis_display_v as display_v
import themis_datasets as dst

import matplotlib.pyplot as plt
import numpy as np

import gc

def plot_pca_comp(data, transformed, pca_components):

    f = plt.figure(num=0)

    gs = f.add_gridspec(1, pca_components+1)  # len(file_name))#, width_ratios=[5,5,5,5,1])
    gs.update(left=0.07, right=0.99, top=0.99,
          bottom=0.05, wspace=0.2, hspace=0.15)
    f.clf()

    ax = f.add_subplot(gs[0, 0])

    for i in range(data.shape[1]):

        ax.plot(  data[:,i] )

         
    for p in range(pca_components):
        ax = f.add_subplot(gs[0, p+1])
        ax.plot( transformed[:,p]/ transformed[:,0].max(), label='PCA {} comp., norm.'.format(p+1),)

        ax.legend(loc='best')
    f.show()

def plot_pca_weights(pca, pca_components,shape=1):
    f = plt.figure(num=1)
    
    gs = f.add_gridspec(pca_components, 1)  
    gs.update(left=0.01, right=0.99, top=0.99,
              bottom=0.05, wspace=0.1, hspace=0.15)
    f.clf()
    
    for p in range(pca_components):
     ax = f.add_subplot(gs[p, 0])

     if shape == 1:
        ax.plot( np.array(pca.components_[p,:]))
     else:
         im = ax.imshow( np.array(pca.components_[p,:]).reshape(shape))
         cb=f.colorbar(im,ax=ax, orientation="vertical")
         cb.ax.minorticks_on()
         cb.set_label('Weight PCA {}'.format(p+1))
         
def plot_pca_compare(data, data_wpca, points=[35,6]):
    f = plt.figure(num=10)
    
    gs = f.add_gridspec(len(points), 3)  
    gs.update(left=0.01, right=0.99, top=0.99,
              bottom=0.05, wspace=0.1, hspace=0.1)
    f.clf()
    
    for p, i in enumerate(points):
     ax = f.add_subplot(gs[p, 0])


     im = ax.imshow( data[i,3,:]-data_wpca[i,3,:])
     cb=f.colorbar(im,ax=ax, orientation="vertical")
     cb.ax.minorticks_on()
     cb.set_label('Difference in wavelength point {}'.format(i))
     
     for p, i in enumerate(points):
      ax = f.add_subplot(gs[p, 1])


      im = ax.imshow( data[i,3,:])
      cb=f.colorbar(im,ax=ax, orientation="vertical")
      cb.ax.minorticks_on()
      cb.set_label('original data')
      
      for p, i in enumerate(points):
       ax = f.add_subplot(gs[p, 2])

       im = ax.imshow( data_wpca[i,3,:])
       cb=f.colorbar(im,ax=ax, orientation="vertical")
       cb.ax.minorticks_on()
       cb.set_label('pca data')



#%%
#----------------------------------------------------------------------
"""
MAIN: 
   
"""
#---------------------------------------------------------------

# initialize files and folder object

ff = tt.ff() 

gc.collect()
data, header = tt.read_v_file(ff.v_file, ff.i_file)

reduced_data, xlam, ff_p =  tt.process_data_for_inversion(data,  scan=0, ff_p =ff,  
                                             wl_calibration = False,
                                             binning =  1, # scan binning
                                             mask = False,
                                             pca = 0,
                                             continuum = True,  # correct continuum: intensity/I_c, V - V_c
                                             cut=True, 
                                             test = False, debug = False, save = True) #only one scan at a time

#%%
###################
# PCA spectra - will be calculated on reduced data for inversion
###################

#initialize PCA with first principal components
n_components=4


#prepare data - only Stokes V
data_line_1, mean_1=tt.prep_data_for_pca(reduced_data[int(reduced_data.shape[0]/2):,3, :])
data_line_2, mean_2=tt.prep_data_for_pca(reduced_data[:int(reduced_data.shape[0]/2),3, :])



pca_line_1, transformed_line_1, inverted_line_1 = tt.run_pca_s(data_line_1, n_components)

pca_line_2, transformed_line_2, inverted_line_2 = tt.run_pca_s(data_line_2, n_components)
 
#%%
# display pca
#plot_pca_comp(data_line_1, transformed_line_1, n_components)
plot_pca_comp(data_line_2, transformed_line_2, n_components)
#%%
#plot_pca_weights(pca_line_1, n_components, shape=(reduced_data.shape[2],reduced_data.shape[3] ))
plot_pca_weights(pca_line_2, n_components, shape=(reduced_data.shape[2],reduced_data.shape[3] ))
#%%
X_pca_2 = pca_line_2.transform(data_line_2)
X_reconstructed = pca_line_2.inverse_transform(X_pca_2)
#%%
plt.plot(X_reconstructed[:,1131], label = 'PCA reconstructed, {} components'.format(n_components))
plt.plot(data_line_2[:,1131], label = 'original data')

plt.plot(10*X_reconstructed[:,300], linestyle = 'dashed',label = 'PCA reconstructed, {} components x 10, high noise pixel'.format(n_components))
plt.plot(10*data_line_2[:,300],  linestyle = 'dashed', label = 'original data x 10, high noise pixel')
plt.legend()

#%%
gc.collect()
data, header = tt.read_v_file(ff.v_file, ff.i_file)
reduced_data_pca, xlam, ff_p =  tt.process_data_for_inversion(data,  scan=0, ff_p =ff,  
                                             wl_calibration = False,
                                             binning =  1, # scan binning
                                             mask = False,
                                             pca = 4,
                                             continuum = True,  # correct continuum: intensity/I_c, V - V_c
                                             cut=True, 
                                             test = False, debug = False, save = True) #only one scan at a time

#%%
# difference between data with and without PCA
plot_pca_compare(reduced_data, reduced_data_pca)

