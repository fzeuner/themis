import matplotlib.pyplot as plt
from astropy.io import fits 
import numpy as np
from themis.ui.style_themis import style_aa
from matplotlib.ticker import (NullFormatter)

from scipy.signal import convolve2d
from scipy.io import readsav
from scipy.signal import medfilt
from tqdm import tqdm

from scipy.interpolate import interp1d
from scipy.signal import windows
from scipy import ndimage
from scipy.ndimage import shift

from skimage.registration import phase_cross_correlation

from matplotlib import gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import gc
import warnings

from pathlib import Path
import imreg_dft as ird# pip install git+https://github.com/matejak/imreg_dft.git
from themis.core import themis_data_reduction as tdr
from themis.core import data_classes as dct



def process_data_for_inversion(data,  scan=0, ff_p =ff,  
                                             binning_x =  1, # scan pos, scan, x along slit, wavelength
                                             pca = 0,  # define number of pca components
                                             continuum = True,  # correct continuum: intensity/I_c, V - V_c
                                             cut=True, 
                                             test = False, save = True): #only one scan at a time
    

    if binning == 1:
       print("process_dataset: Processing data \n"+ff_p.v_file)
    else:
       print("process_dataset: Processing data \n"+ff_p.v_file+' \n with binning '+ str(binning_x)+'scans')
    print('----------------------')
    
    # figure out if it is a single scan
    single_scan=False
    if len(data.i.shape) == 3: 
       single_scan = True
       print('process_dataset: There is only one scan in the data. \n')
    if not single_scan:   
       # mask tellurics
        for tel in range(2):
         data.i[:,:,:,dst.tellurics[tel][0]:dst.tellurics[tel][1]] = -2*np.max(data.i)
         data.v[:,:,:,dst.tellurics[tel][0]:dst.tellurics[tel][1]] = 0.0 # 

        
        i_c = data.i[:,:,20:-20,dst.continuum[0]:dst.continuum[1]].mean(axis=3)
        i_c = i_c.max() 
        
        if continuum:
            data.i /= i_c
            for i in range(data.v.shape[0]):
               for m in range(data.v.shape[1]):
                 for n in range(data.v.shape[2]):
                     data.v[i,m,:,n] -= data.v[i,m,n,dst.continuum[0]:dst.continuum[1]].mean()
        
        if cut:
               data.i = data.i[dst.yroi[0]:dst.yroi[1],:,dst.roi[0]:dst.roi[1],dst.sroi[0]:dst.sroi[1]]
               data.v = data.v[dst.yroi[0]:dst.yroi[1],:,dst.roi[0]:dst.roi[1],dst.sroi[0]:dst.sroi[1]]

               if continuum:
                    data.i /= data.i.max()    
                            
        reduced_data = np.zeros((data.i.shape[3], 4, data.i.shape[0], data.i.shape[2] ) ) 
        # stokes, wvl, x, y
        xlam = 1000*(np.arange( data.i.shape[3]) - dst.line_core[0]+dst.sroi[0])*dst.spectral_sampling # to be used in the wavelength grid for SIR
        
        print('According to the parameters in themis_datasets:'+'\n'+\
              'start wavelength [mA]: '+str(xlam[0]) +'\n' + \
              'wl step [mA]: '+str(1000*dst.spectral_sampling)+'\n' + \
               'end wavelength [mA]: '+str(xlam[-1]) +'\n'  )
        # if wl_calibration:
        #     wl_reference, ref_spectrum, _ = read_fts5(6299, 6305)
        #     spectrum = data.i.mean(axis=(0,1,2))
        #     spectrum/=spectrum.max()
        #     xlam = z3ccspectrum(spectrum,  wl_reference, ref_spectrum, FACL=0.8, FACH=1.5, FACS=0.005, 
        #                            CUT=[10,20], DERIV=None, CONT=1.01, SHOW=2)
            
        for i in tqdm(range(reduced_data.shape[0])):
            for m in range(reduced_data.shape[2]):
              reduced_data[i,0, m,:] = data.i[m,scan,:,i]
              
              reduced_data[i,3,m,:] = data.v[m,scan,:,i]
              
        if pca > 0:
            data_line_1, mean_1=prep_data_for_pca(reduced_data[int(reduced_data.shape[0]/2):,3, :])
            data_line_2, mean_2=prep_data_for_pca(reduced_data[:int(reduced_data.shape[0]/2),3, :])
            
            pca_line_1, transformed_line_1, inverted_line_1 = run_pca_s(data_line_1, pca)
            pca_line_2, transformed_line_2, inverted_line_2 = run_pca_s(data_line_2, pca)
            
            X_pca_1 = pca_line_1.transform(data_line_1)
            X_reconstructed_1 = pca_line_1.inverse_transform(X_pca_1) + mean_1
            
            X_pca_2 = pca_line_2.transform(data_line_2)
            X_reconstructed_2 = pca_line_2.inverse_transform(X_pca_2) + mean_2

            reduced_data[int(reduced_data.shape[0]/2):,3,:] = X_reconstructed_1.reshape((int(reduced_data.shape[0]/2),reduced_data.shape[2],reduced_data.shape[3]))
            reduced_data[:int(reduced_data.shape[0]/2),3,:] = X_reconstructed_2.reshape((int(reduced_data.shape[0]/2),reduced_data.shape[2],reduced_data.shape[3]))
  
        if save:
            
            save_inv_input_data(reduced_data, xlam,ff_p, scan, overwrite = True)
        
    return(reduced_data, xlam, ff_p)    
    

def process_pixel_for_inversion(data,  x=0, y=0, scan=0, ff_p =ff,  
                                             pca = 0,  # define number of pca components
                                             continuum = True,  # correct continuum: intensity/I_c, V - V_c
                                             cut=True, 
                                             test = False, save = True, dir_= '', ext = ''): #only one scan at a time
    

    print("process_pixel: Processing data \n"+ff_p.v_file)

    print('----------------------')
    
    
    # figure out if it is a single scan
    single_scan=False
    if len(data.i.shape) == 3: 
       single_scan = True
       print('process_dataset: There is only one scan in the data. \n')
    if not single_scan:   
       # mask tellurics
        for tel in range(2):
         data.i[:,:,:,dst.tellurics[tel][0]:dst.tellurics[tel][1]] = -2*np.max(data.i)
         data.v[:,:,:,dst.tellurics[tel][0]:dst.tellurics[tel][1]] = 0.0 # 

        
        i_c = data.i[:,:,20:-20,dst.continuum[0]:dst.continuum[1]].mean(axis=3)
        i_c = i_c.max()
        
        if continuum:
            data.i /= i_c
            for i in range(data.v.shape[0]):
               for m in range(data.v.shape[1]):
                 for n in range(data.v.shape[2]):
                     data.v[i,m,:,n] -= data.v[i,m,n,dst.continuum[0]:dst.continuum[1]].mean()
        
        if cut:
               data.i = data.i[:,:,dst.roi[0]:dst.roi[1],dst.sroi[0]:dst.sroi[1]]
               data.v = data.v[:,:,dst.roi[0]:dst.roi[1],dst.sroi[0]:dst.sroi[1]]
        

        if continuum:
                    data.i /= data.i.max()    
                            
        reduced_data = np.zeros((data.i.shape[3], 4, data.i.shape[0], data.i.shape[2] ) ) 
        # stokes, wvl, x, y
        xlam = 1000*(np.arange( data.i.shape[3]) - dst.line_core[0]+dst.sroi[0])*dst.spectral_sampling # to be used in the wavelength grid for SIR
        
        print('According to the parameters in themis_datasets:'+'\n'+\
              'start wavelength [mA]: '+str(xlam[0]) +'\n' + \
              'wl step [mA]: '+str(1000*dst.spectral_sampling)+'\n' + \
               'end wavelength [mA]: '+str(xlam[-1]) +'\n'  )
        # if wl_calibration:
        #     wl_reference, ref_spectrum, _ = read_fts5(6299, 6305)
        #     spectrum = data.i.mean(axis=(0,1,2))
        #     spectrum/=spectrum.max()
        #     xlam = z3ccspectrum(spectrum,  wl_reference, ref_spectrum, FACL=0.8, FACH=1.5, FACS=0.005, 
        #                            CUT=[10,20], DERIV=None, CONT=1.01, SHOW=2)
            
        for i in range(reduced_data.shape[0]):
            for m in range(reduced_data.shape[2]):
              reduced_data[i,0, m,:] = data.i[m,scan,:,i]
              
              reduced_data[i,3,m,:] = data.v[m,scan,:,i]
              
        if pca > 0:
            data_line_1, mean_1=prep_data_for_pca(reduced_data[int(reduced_data.shape[0]/2):,3, :])
            data_line_2, mean_2=prep_data_for_pca(reduced_data[:int(reduced_data.shape[0]/2),3, :])
            
            pca_line_1, transformed_line_1, inverted_line_1 = run_pca_s(data_line_1, pca)
            pca_line_2, transformed_line_2, inverted_line_2 = run_pca_s(data_line_2, pca)
            
            X_pca_1 = pca_line_1.transform(data_line_1)
            X_reconstructed_1 = pca_line_1.inverse_transform(X_pca_1) + mean_1
            
            X_pca_2 = pca_line_2.transform(data_line_2)
            X_reconstructed_2 = pca_line_2.inverse_transform(X_pca_2) + mean_2

            reduced_data[int(reduced_data.shape[0]/2):,3,:] = X_reconstructed_1.reshape((int(reduced_data.shape[0]/2),reduced_data.shape[2],reduced_data.shape[3]))
            reduced_data[:int(reduced_data.shape[0]/2),3,:] = X_reconstructed_2.reshape((int(reduced_data.shape[0]/2),reduced_data.shape[2],reduced_data.shape[3]))
        
        single_profile = reduced_data[:,:,x,y]
        
        if save:
            
            
            file_name="profile_"+format(scan,'02')+'_'+format(x,'03')+'_'+format(y,'03')+ext+".per"
            print('Save single pixel SIR input file:\n')
            print(file_name)
            print('Save wavelength.grid file\n')
            
            writepro(ff_p.inversion_dir+'/'+dir_+'/'+file_name,np.zeros(xlam.shape)+dst.line_idx,xlam,\
                          single_profile[:,0], single_profile[:,1], single_profile[:,2], single_profile[:,3])  
            make_grid_file(ff_p.inversion_dir+'/'+dir_,dst.line_idx,xlam)
            print('----------------')
            print('Important: you still need to add the blend line 2 if you would like to invert it in the wavelength.grid file!')
            print('---------------------------------------')
    return(single_profile, xlam, ff_p)    

def make_grid_file(file_dir,line_idx,xlam):
    """
    Create .grid file for SIR

    Parameters
    ----------
    first : number of lines
    second : line index based on the standart LINE file excluding blends
    third : array with min wavelenght
    fourth: array with wavelenght step
    fifth: array with max wavelenght
    """
    file_grid=open(file_dir+'/wavelength.grid','w')
    file_grid.write('IMPORTANT: a) All items must be separated by commas.\n')
    file_grid.write('           b) The first six characters of the last line\n')
    file_grid.write('              in the header (if any) must contain the symbol ---\n')
    file_grid.write('\n')
    file_grid.write('Line and blends indices   :   Initial lambda     Step     Final lambda\n')
    file_grid.write('(in this order)                    (mA)          (mA)         (mA)\n')
    file_grid.write('-----------------------------------------------------------------------\n')
   
    file_grid.write(str(line_idx)+'                      :       '+'{:>4.3f}'.format(xlam[0])+',      '+str(xlam[-1]-xlam[-2])+',        '+'{:>4.3f}'.format(xlam[-1])+'\n')
    file_grid.close()


def add_noise_to_profiles(file, noise): # gaussian noise
    
    line_ind, wvlen, StkI, StkQ, StkU, StkV = readpro(file)
    noise_v = np.random.normal(0.0, noise, StkV.shape[0])
    noise_q = np.random.normal(0.0, noise, StkQ.shape[0])
    noise_u = np.random.normal(0.0, noise, StkU.shape[0])

    print('max SNR Q: '+str(max(StkQ)/noise))
    print('max SNR U: '+str(max(StkU)/noise))
    print('max SNR V: '+str(max(StkV)/noise))
    
    filename_without_ext = os.path.splitext(os.path.basename(file))[0]
    dirname, fname = os.path.split(file)
    
    writepro(dirname+'/'+filename_without_ext+'_n.per',line_ind, wvlen, StkI, StkQ+noise_q, StkU+noise_u, StkV+noise_v)
        
    return(0)

def write_wht_file(xlam, directory):
    file=open(directory+'profiles.wht','w')
    for x in xlam:
       file.write(str(1)+'   {:>4.3f}'.format(x)+'    {:>1.3f}'.format(1)+'   {:>1.3f}'.format(0)+'    {:>1.3f}'.format(0)+'   {:>1.3f}'.format(1)+'\n')
   # correction_second_line = (dst.line_core[1]-dst.line_core[0])*dst.spectral_sampling
   # for x in xlam:
   #    file.write(str(2)+'   {:>4.3f}'.format(x+correction_second_line)+'    {:>1.3f}'.format(1)+'   {:>1.3f}'.format(0)+'    {:>1.3f}'.format(0)+'   {:>1.3f}'.format(1)+'\n')

    file.close()
    
    print('.wht file written')