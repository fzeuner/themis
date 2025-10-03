import matplotlib.pyplot as plt
from themis.ui.style_themis import style_aa
from matplotlib.ticker import (NullFormatter)
import matplotlib.colors as mplcolor
import numpy as np
import matplotlib.gridspec as gridspec


def add_axis(fig,ax,data, wl,label, pixel, xvis='no', yvis='no',title=None):
    
   ax.cla()
   intmeth = 'none' # imshow integration method
   dmin,dmax=np.percentile(data.flatten(),(0.5,99.5))
   mn=np.mean(data)
   fct=np.mean(data)/mn
   
   image_sz=np.array(data.shape)
   
   extent=[0,image_sz[1]*pixel,0,image_sz[0]*pixel]

   if wl.shape[0] == image_sz[0]:
       extent[2]=wl[0]
       extent[3]=wl[-1]
   if title != None:
       ax.set_title(title)
   
   im=ax.imshow(data, cmap='gray',origin='lower',\
   interpolation=intmeth,clim=[fct*dmin,fct*dmax], 
   extent=extent, aspect='auto')#
       
   #Plot color bar   
   cb=fig.colorbar(im,ax=ax, orientation="vertical")
   cb.ax.minorticks_on()
   cb.set_label(label)
   
   if xvis == 'no':
       ax.get_xaxis().set_visible(False)
   else:
       ax.get_xaxis().set_visible(True)
       if xvis != 'yes':         
           ax.set_xlabel(xvis)  
   if yvis == 'no':
       ax.get_yaxis().set_visible(False)
   else:
       ax.get_yaxis().set_visible(True)
       if yvis != 'yes':         
           ax.set_ylabel(yvis)  
       
   return(ax)


def add_scan_axis(fig,ax,data, label, stokes =False, clim=[0,0], xvis='no', yvis='no', x_offset=0, title=None, total=False):
    
   if stokes:  
    if total:
        cdict = {'red':   ((0.0, 0.0, 0.0), (0.25, 1.0, 1.0), (0.75,1.0,1.0), (1.0,1.0,1.0)),
                'green': ((0.0, 0.0, 0.0),  (0.25, 1.0, 1.0), (0.75,0.0,0.0), (1.0,1.0,1.0)),
                'blue':  ((0.0, 1.0, 1.0),  (0.25, 0.0, 0.0), (0.75,0.0,0.0), (1.0,1.0,1.0))}
    else:

     cdict = {'red':  ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.5, 1.0, 1.0), (0.75,1.0,1.0)  , (1.0,1.0,1.0)),
             'green': ((0.0, 1.0, 1.0), (0.25, 0.0, 0.0), (0.5, 1.0, 1.0), (0.75,0.75,0.75), (1.0,0.0,1.0)),
             'blue':  ((0.0, 0.0, 0.0), (0.25, 1.0, 1.0), (0.5, 1.0, 1.0), (0.75,0.0,0.0)  , (1.0,0.0,0.0))}

    p_cmap = mplcolor.LinearSegmentedColormap('p_colormap',cdict,256)    
    
   else: 
       p_cmap='gray'
    
   ax.cla()
   intmeth = 'none' # imshow integration method
   dmin,dmax=np.percentile(data.flatten(),(0.5,99.5))
   mn=np.mean(data)
   fct=np.mean(data)/mn
   
   image_sz=np.array(data.shape)
   
   extent=[x_offset*dst.pixel[dst.line],(x_offset+image_sz[1])*dst.pixel[dst.line],0,image_sz[0]*dst.slitwidth]

   if title != None:
       ax.set_title(title)
   
   clim_plot=[fct*dmin,fct*dmax]
   if abs(clim[0] - clim[1]) > 10e-8:
       clim_plot=clim
   im=ax.imshow(data, cmap=p_cmap,origin='lower',\
                    interpolation=intmeth,clim=clim_plot, 
                    extent=extent, aspect='auto')#
       
   #Plot color bar   
   cb=fig.colorbar(im,ax=ax, orientation="vertical")
   cb.ax.minorticks_on()
   cb.set_label(label)
   
   if xvis == 'no':
       ax.get_xaxis().set_visible(False)
   else:
       ax.get_xaxis().set_visible(True)
       if xvis != 'yes':         
           ax.set_xlabel(xvis, labelpad=-0.8)  
   if yvis == 'no':
       ax.get_yaxis().set_visible(False)
   else:
       ax.get_yaxis().set_visible(True)
       if yvis != 'yes':         
           ax.set_ylabel(yvis, labelpad=-0.8)  
       
   return(ax)

def add_axis_contour(fig, ax, data, data2, threshold, zoom, wl,label, pixel, xvis='no', yvis='no',title=None):
    
   ax.cla()
   intmeth = 'none' # imshow integration method
   dmin,dmax=np.percentile(data.flatten(),(3.,97.5))
   mn=np.mean(data)
   fct=np.mean(data)/mn

   
   image_sz=np.array(data.shape)
   linewidth_contour=2.*zoom
   alpha_value=0.7
   
   extent=[0,image_sz[1]*pixel,0,image_sz[0]*pixel]

   if wl.shape[0] == image_sz[0]:
       extent[2]=wl[0]
       extent[3]=wl[-1]
   if title != None:
       ax.set_title(title)
   
   im=ax.imshow(data, cmap='gray',origin='lower',\
   interpolation=intmeth,clim=[fct*dmin,fct*dmax], 
   extent=extent, aspect='auto')#
       
   #Plot color bar   
   cb=fig.colorbar(im,ax=ax, orientation="vertical")
   cb.ax.minorticks_on()
   cb.set_label(label)
   
   if xvis == 'no':
       ax.get_xaxis().set_visible(False)
   else:
       ax.get_xaxis().set_visible(True)
       if xvis != 'yes':         
           ax.set_xlabel(xvis)  
   if yvis == 'no':
       ax.get_yaxis().set_visible(False)
   else:
       ax.get_yaxis().set_visible(True)
       if yvis != 'yes':         
           ax.set_ylabel(yvis)  
            
   
   # for u, contour in enumerate(contours):
   #      #ax.plot(contour[:,1],contour[:,0],linewidth=linewidth_contour, color='k', alpha=alpha_value)
   #      ax.plot([0,2000],[0,2000],linewidth=linewidth_contour, color='k', alpha=alpha_value)
   #      #mask=sm.grid_points_in_poly([subfield[1]-subfield[0],subfield[3]-subfield[2]], contour)
   ax.contour(data2, threshold, colors='magenta', origin='lower', extent=extent, linewidths= linewidth_contour, alpha=alpha_value)
   return(ax)


 

def plot_stokes_simple(data, wl, pixel=1,zoom=1, fnum=0, label_up = False, title=None, save_figure=[False,'None',__file__]):
    
    plt.close()
    my_dpi, fig_size_single, fig_size, params=style_aa(zoom)
    plt.style.use('default')
    plt.rcParams.update(params)
    nullfmt=NullFormatter()
    
    fig, axs = plt.subplots(4,constrained_layout=True, num=3)
    fig.set_size_inches(fig_size,forward=True)
    
    if label_up:
        stokes_str=[r'$I$',r'$Q$ [$I_c$]',r'$U$ [$I_c$]',r'$V$ [$I_c$]']
    else:
        stokes_str=[r'$I$',r'$Q/I$ [$\%$]',r'$U/I$ [$\%$]',r'$V/I$ [$\%$]']
    titles=[title, None, None, None]
    yvis=[r'Wavelength $[\mathrm{\AA}]$', r'Wavelength $[\mathrm{\AA}]$', r'Wavelength $[\mathrm{\AA}]$', r'Wavelength $[\mathrm{\AA}]$']
    
    if pixel == 1:
     xvis=[r'no', r'no',r'no', r'pixel']
    else:
     xvis=[r'no', r'no',r'no', r'arcsec'] 
     
    
    for i,s in enumerate(stokes_str):
        ax=add_axis(fig,axs[i],data[i,:], wl,s, pixel,xvis=xvis[i], yvis=yvis[i], title=titles[i])
        
    fig.show()
    
    if save_figure[0]:
        
       fig.savefig(save_figure[1], 
              dpi=my_dpi, metadata={'Software': save_figure[2]})
    return(fig, axs)

def plot_scan_images(data, wl_idx, zoom=2, fnum=0, binning= [1,1,1], clim=[0,0], vlim=[0,0], title=None, save_figure=[False,'None',__file__]):
    # binning is in spatially in scan direction, spatially along slit, spectrally
    
    # print('plotted data is binned to' )
    my_dpi, fig_size_single, fig_size, params=style_aa(zoom)
    plt.style.use('default')
    plt.rcParams.update(params)
    nullfmt=NullFormatter()
    
    fig = plt.figure(num=fnum, layout="constrained")
    fig.set_size_inches([fig_size_single[0],1.3*fig_size_single[1]],forward=True)
    fig.clf()
    
    gs = gridspec.GridSpec(4,1, figure=fig)#width_ratios=[1,0.25,1,0.25])# , height_ratios=[1,1,1,1] )
    gs.update(top=0.99, bottom=0.14,left=0.11, right=0.95, wspace=0.17, hspace=0.12)
    
    
    stokes_str=[r'$I$',r'$Q/I$ [$\%$]',r'$U/I$ [$\%$]',r'$V/I$ [$\%$]']
    titles=[title, None, None, None]
    yvis=[r'arcsec', r'arcsec', r'arcsec', r'arcsec']
    
    xvis=[r'no', r'no',r'no', r'arcsec'] 
    
    data_plot= np.copy(data)
    
    # wavelength binning
    shifts=[0,1,-1,2,-2,3,-3,4,-4,5,-5]
    
    if binning[2]>1:
       data_plot=0.* np.copy(data)
       for wv in range(binning[2]):
          data_plot += np.roll(data, shifts[wv], axis=2) 
       data_plot/=binning[2]
    data_plot=data_plot[:,:,wl_idx,:]
    kernel = np.ones((binning[0],binning[1]))/(binning[0]*binning[1])
    
    
    for i,s in enumerate(stokes_str):
        
        axs = fig.add_subplot(gs[i, 0])
        
        if i > 0:
            stokes=True
        else:
            stokes=False
            
        # spatial binning
        if binning[0] > 1 or binning[1]> 1:
               data_plot_dummy = convolve2d(data_plot[i,:], kernel, 'valid')
        else:
            data_plot_dummy = data_plot[i,:]
        if i == 1 or i == 2:
            clim_plot = clim
        else:
            clim_plot=[0,0]
            if i == 3:
                clim_plot=vlim
        ax=add_scan_axis(fig,axs,data_plot_dummy,s, xvis=xvis[i], yvis=yvis[i],  clim=clim_plot,title=titles[i], stokes=stokes)
        ax.set_yticks([0,0.5,1])
    
    fig.show()
    
    if save_figure[0]:
        
       fig.savefig(save_figure[1], 
              dpi=my_dpi, metadata={'Software': save_figure[2]})
       
      # plot intensity spectrum with vertical line indicating the wavelength point plotted  
       
    fig2 = plt.figure(num=44,  layout="constrained")
    fig2.set_size_inches(fig_size_single,forward=True)
    fig2.clf()

    gs2 = gridspec.GridSpec(1,1, figure=fig2 )#width_ratios=[1,0.25,1,0.25])# , height_ratios=[1,1,1,1] )
    gs2.update(top=0.95, bottom=0.12,left=0.08, right=0.99, wspace=0.17, hspace=0.05)
    
    ax2 = fig2.add_subplot(gs2[0, 0]) # 
    ax2.cla()
    
    intensity_spectrum = data[0,0,:,0]
    ax2.plot(intensity_spectrum)
    ax2.vlines(wl_idx, ymin=intensity_spectrum.min(), ymax=intensity_spectrum.max(), color='red', linewidth=2)
    fig2.show() 
    return(fig)