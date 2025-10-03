import matplotlib.pyplot as plt
from themis.ui.style_themis import style_aa
from matplotlib.ticker import (NullFormatter)
import numpy as np
import matplotlib.gridspec as gridspec
from scipy import windows


def plot_power_spectrum(data, zoom=1): # data: y, x

    print(r'Power spectrum')
    image=1.*data
    # create figure
    intmeth = 'none' # imshow integration method
    my_dpi, fig_size_single, fig_size, params=style_aa(zoom)
    plt.style.use('default')
    plt.rcParams.update(params)
    nullfmt=NullFormatter()
    
    fig=plt.figure(num=32) 
    fig.clf()
    fig.set_size_inches([fig_size_single[0], 1.5*fig_size_single[1]],forward=True)

    gs = gridspec.GridSpec(2,1, height_ratios=[0.38,1] )#width_ratios=[0.25,0.25,0.25,0.25] , height_ratios=[1,1,1,1] )
    gs.update(top=0.96, bottom=0.1,left=0.13, right=0.99, wspace=0.05, hspace=0.26)
         
         
    axs=fig.add_subplot(gs[1,0]) 
    ax0=fig.add_subplot(gs[0,0]) 
    
    window_x=windows.hann(image.shape[1]) 
    window_y=windows.hann(image.shape[0]) 
    for x in range(image.shape[0]):
        image[x,:]*=window_x
    for y in range(image.shape[1]):
        image[:,y]*=window_y
        

    fs = np.fft.fftn(image)
    fs_abs=(np.abs(np.fft.fftshift(fs))**2)
    # fs_sz=np.shape(fs_abs)
    # axs[0].imshow(np.log(fs_abs/np.amax(fs_abs)), cmap='jet', interpolation=intmeth, aspect='auto')
   
    y,x = np.indices((fs_abs.shape)) # first determine radii of all pixels
    center=[int(np.shape(fs_abs)[0]/2),int(np.shape(fs_abs)[1]/2)]
    r = np.sqrt((x-center[1])**2+(y-center[0])**2)    

    # radius of the image.
    r_max = int(np.floor(np.max(r)))

    # ring_brightness, radius = np.histogram(r, weights=fs_abs, bins=r_max*2)
    # ring_size, radius_1 = np.histogram(r, bins=r_max*2)
    # #plt.plot(radius[1:], ring_brightness/ring_size)
    # liney,= axs[1].plot(radius[1:]/np.max(radius[1:])*1./(2.*dst.pixel[dst.line]), 100*ring_brightness/ring_brightness.max(), 
    #                     color='black', marker='.', linestyle='None') #   
    # axs[1].set_xlabel("1/arcsec")
    # axs[1].set_ylim(0.0000001,0.1)
    # axs[1].set_xlim(0.2,10.1)
    # axs[1].set_ylabel('Power [a.u]')
    # axs[1].set_yscale('log')
    
    image_2=1.*data
    fs_ = np.zeros((image_2.shape[0],int(image_2.shape[1]/2)))
    
    freqs = (np.fft.fftfreq(image_2.shape[1])* 1./(dst.pixel[dst.line]))[:int(image_2.shape[1]/2)]

    contrast=np.zeros(image_2.shape[0])
    for i in range(image_2.shape[0]):
        i_m=np.mean(image_2[i,:])
      #  max_da=np.mean(data[0,i,550,np.argsort(data[0,i,550,:])][-100:-5])
        
        contrast[i]=100.*np.std(image_2[i,:])/i_m
        
    print(r'Mean rms contrast [%]: '+str(np.mean(contrast)))
    ax0.plot(contrast, color='black')
    ax0.set_ylabel(r'RMS contrast [%]', labelpad=0)
    ax0.set_xlabel("Scan position", labelpad=-0.8)
    ax0.axhline(y=contrast.mean(), color='black',
               linestyle="dotted", alpha=0.8, linewidth=0.8*zoom)
    for te in range(image_2.shape[0]):
     # if contrast[te] > contrast.mean():  
        image_2[te,:]*=window_x
        fs=np.fft.fftn(image_2[te,:])
        fs_abs=np.roll((np.abs(np.fft.fftshift(fs))**2)[int(image_2.shape[1]/2):],-int(image_2.shape[1]/2))
        fs_[te,:]=fs_abs/fs_abs.max()
    
    fs_=np.mean(fs_, axis=0)
    # use only non-zero parts    
    fs_/=fs_.mean()
    #idx_nonzero=np.where(fs > 10e-11)
    
   # freq_axis=(radius[1:]/np.max(radius[1:])*1./(2.*dst.pixel[dst.line]))[idx_nonzero]
    freq_axis=freqs#[idx_nonzero]
  
    liney,= axs.plot(freq_axis,fs_,  color='black', marker='.', linestyle='None') #   
    
    n_filtered = 111 # needs to be odd!!!
    medi_filtered = medfilt(fs_,n_filtered)
    
    
    axs.plot(freq_axis[int((n_filtered+1)):-int((n_filtered+1))],
                medi_filtered[int((n_filtered+1)):-int((n_filtered+1))],
                        color='blue',linestyle='solid', linewidth=2*zoom ) #   
    
    # gradient=np.gradient(medi_filtered[int((n_filtered+1)):-int((n_filtered+1))])
    # axs[2].plot((radius[1:]/np.max(radius[1:])*1./(2.*dst.pixel[dst.line]))[int((n_filtered+1)):-int((n_filtered+1))],
    #             gradient,
    #                     color='red',linestyle='solid' )
    
    axs.axvline(x=1/0.58, color='red',
               linestyle="dashed", alpha=0.8, linewidth=0.8*zoom)
    axs.axhline(y=2e-7, color='black',
               linestyle="dashed", alpha=0.8, linewidth=0.8*zoom)
    axs.set_ylabel('Power [a.u]', labelpad=-1)
    axs.set_yscale('log')
    axs.set_ylim(0.00000001,0.01)
    axs.set_xlim(0.1,2.1)
    axs.set_xlabel(r"arcsec$^{-1}$",labelpad=-1)
    #fig.show()

    return(fig, fs_)