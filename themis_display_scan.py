#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:56:31 2024

@author: franziskaz
"""

"""

WARNING: if the qdarkstyle is used, there is some bug in Dock and VerticalLabel (look at the files in the folder.)
"""

import numpy as np
import matplotlib.colors as mplcolor

import pyqtgraph as pg

from pyqtgraph.Qt import QtWidgets
import datasets as dst
from pyqtgraph.Qt import QtCore
import qdarkstyle
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from getWidgetColors import getWidgetColors

import themis_tools as tt


# import pyqtgraph.examples
# pyqtgraph.examples.run()



class StokesImageWindow((QtWidgets.QWidget)):
    
    positionChanged = QtCore.pyqtSignal(float)
    
    def __init__(self, xlam, data, win_spectrum, win_image_spectrum,stokes=0, coordinates = [0,0,0,0],parent=None):
        super().__init__(parent)   
        
        self.data=data[stokes, :, :, :, :] # state, scan pos, scan, x, wvl
        # print(self.data.shape)
        self.n_x_pixel = self.data[:,0,:, 0].shape[1]
        self.n_y_pixel = self.data[:,0,:, 0].shape[0]
        
        self.xlam = xlam
        self.wavelength_pos = 0
        self.scan = 0
        # Set up the main layout for this widget
        layout = QtWidgets.QHBoxLayout(self)

        # Create a GraphicsLayoutWidget to hold the image 
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground(getWidgetColors.BG_NORMAL)
        # Create and configure an ImageItem to display the image
        self.image_item = pg.ImageItem()
        self.image_item.setImage(self.data[:,self.scan, :, self.wavelength_pos]) # initial image
        
        self.plotItem = self.graphics_widget.addPlot(row = 0, col=0,colspan=1,rowspan=4)      # add PlotItem to the main GraphicsLayoutWidget
        self.plotItem.invertY(False)              # orient y axis to run bottom-to-top
        self.plotItem.setDefaultPadding(0.0)      # plot without padding data range
        
            
            # show full frame, label tick marks at top and left sides, with some extra space for labels:
        self.plotItem.showAxes( True, showValues=(True, True, False, False), size=20 )

            # define major tick marks and labels:
         
        # x_ticks=np.arange(self.n_x_pixel, step = int(self.n_x_pixel/8))  # eigtht ticks min   
        # y_ticks=np.arange(self.n_y_pixel, step = int( self.n_y_pixel/3))  # three ticks min    
        # if coordinates[1]-coordinates[0] < 0.00001:        
            
        #     self.plotItem.getAxis('top').setTicks([[(x_axis_tick, '{:.1f}'.format(x_axis_tick*dst.pixel[dst.line])) for x_axis_tick in x_ticks]])
        # else: 
        #     x_ticks=np.arange(coordinates[1], start=coordinates[0],step = int(self.n_x_pixel*dst.pixel[dst.line]/8))  # eight ticks min 
        #     self.plotItem.getAxis('top').setTicks([[(x_axis_tick, '{:.1f}'.format(x_axis_tick)) for x_axis_tick in x_ticks]])
        
        # if coordinates[3]-coordinates[2] < 0.00001:      
        #     self.plotItem.getAxis('left').setTicks([[(y_axis_tick, '{:.1f}'.format(y_axis_tick*dst.slitwidth)) for y_axis_tick in y_ticks]])
        # else: 
        #     y_ticks=np.arange(coordinates[3], start=coordinates[2],step = int(self.n_y_pixel*dst.slitwidth/3))  # three ticks min 
        #     self.plotItem.getAxis('left').setTicks([[(y_axis_tick, '{:.1f}'.format(y_axis_tick)) for y_axis_tick in y_ticks]])
        
        self.plotItem.getAxis('left').setWidth(50)
        self.plotItem.getAxis('bottom').setHeight(15)
        
        
        #  -------- COLORMAP IS NOT YET WORKING      
        # if 0 == stokes:
        #   colorMap = pg.colormap.getFromMatplotlib("grey")     # choose color map
         
        # else:
        #    colorMap=p_cmap 
           
        # # Get colormap data as RGBA lookup table values and scale to 0-255 for pyqtgraph   
        # lut = (colorMap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
         
        # # # Apply the colormap
        # self.image_item.setLookupTable(lut)  # Apply the custom colormap
       
        # self.image_item.setLevels([data[stokes, :, 0, :].min(), data[stokes, :, 0, :].max()])  # Set the levels to the data range
        #  --------
       
        self.plotItem.addItem(self.image_item)    # display 
        pg.setConfigOptions(imageAxisOrder="row-major")
        
        self.plotItem.getAxis('top').enableAutoSIPrefix(False)  # x-axis  disable auto scaling of unit
        self.plotItem.getAxis('bottom').enableAutoSIPrefix(False)  # x-axis  disable auto scaling of unit
        self.plotItem.getAxis('left').enableAutoSIPrefix(False)    # y-axis  disable auto scaling of unit
        self.plotItem.setLabel("bottom", text="x", units="pixel")
        self.plotItem.setLabel("left", text="y", units="scan pos")

        # Add an InfiniteLine for crosshair functionality
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.vLine.setPen('blue', width=2)
        self.plotItem.addItem(self.vLine, ignoreBounds=True)
        
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.hLine.setPen('blue', width=2)
        self.plotItem.addItem(self.hLine, ignoreBounds=True)

        # Add the GraphicsLayoutWidget to the layout
        layout.addWidget(self.graphics_widget)

        # Create a HistogramLUTWidget for intensity scaling
        self.histogram = pg.HistogramLUTWidget()
        self.histogram.setImageItem(self.image_item)
        self.histogram.setBackground(getWidgetColors.BG_NORMAL)
        layout.addWidget(self.histogram)

        # Set the main layout for this widget
        self.setLayout(layout)
        
        # Hold reference to the target image widget (for updating its image)
        self.win_spectrum = win_spectrum
        self.win_image_spectrum = win_image_spectrum
        
        # self.xscale=np.arange(self.n_x_pixel*dst.pixel[dst.line], step = 1.*dst.pixel[dst.line])
        # self.yscale=np.arange(self.n_y_pixel*dst.slitwidth, step = 1.*dst.slitwidth)
        
        self.xscale=np.arange(self.n_x_pixel, step = 1)
        self.yscale=np.arange(self.n_y_pixel, step = 1)
        
        self.plotItem.scene().sigMouseMoved.connect(self.mouseMoved)
        self.plotItem.scene().sigMouseClicked.connect(self.mouseClicked)
        
        # Variables to track crosshair lock state and position
        self.crosshair_locked = False

        self.label = pg.LabelItem(justify='left')
        self.graphics_widget.addItem(self.label,row=5, col=0,colspan=1,rowspan=1)
        self.update_label()
    
    def update_label(self):
        
        xpos, ypos= self.vLine.value(), self.hLine.value()
        index_x = np.abs(self.xscale - xpos).argmin()
        index_y = np.abs(self.yscale - ypos).argmin()
        self.label.setText(
               # Find the closest index in xpos to the mouse 
                "x={:.1f}".format(xpos*dst.pixel[dst.line]) +
                " y={:.1f} ".format(ypos*dst.slitwidth) +
                "z={:.5f}".format(self.data[index_y,self.scan,index_x, self.wavelength_pos ] ),     
            size='6pt')
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

    def mouseMoved(self, pos):
            """Handle the mouse moved event over the plot area."""
            if not self.crosshair_locked:   
             if self.plotItem.sceneBoundingRect().contains(pos):
                # Map the scene position to plot coordinates (wavelength scale)
                mousePoint = self.plotItem.vb.mapSceneToView(pos)
                xpos, ypos = mousePoint.x(), mousePoint.y()

                # Find the closest index in xpos to the mouse 
                index_x = np.abs(self.xscale - xpos).argmin()
                index_y = np.abs(self.yscale - ypos).argmin()
             
                if 0 <= index_x < self.n_x_pixel and 0 <= index_y < self.n_y_pixel:
                    self.update_vline(xpos)
                    self.update_hline(ypos)
                    self.win_spectrum.update_plot_data(self.data[index_y, self.scan, index_x, :])
                    self.win_image_spectrum.update_plot_data(self.data[index_y, self.scan, :, :])
                    self.update_label()
                                   
    def mouseClicked(self, event):
        if event.double():
            # On double-click, fix crosshair at current position
            mouse_point = self.plotItem.vb.mapSceneToView(event.scenePos())
            self.update_vline(mouse_point.x())
            self.update_hline(mouse_point.y())
            self.crosshair_locked = not self.crosshair_locked  # Toggle lock state
       
    def update_image(self):
         self.image_item.setImage(self.data[:,self.scan,:,self.wavelength_pos]) #
         self.update_label()
         
    def update_vline(self, xpos):
         self.vLine.setPos(xpos)
    def update_hline(self, vpos):
        self.hLine.setPos(vpos)
        
class StokesSpectrumWindow((QtWidgets.QWidget)):
    def __init__(self, xlam, data, stokes=0, coordinates = [0,0,0,0], parent=None):
        super().__init__(parent)

        # Set up the main layout for this widget
        layout = QtWidgets.QHBoxLayout(self)

        # Create a GraphicsLayoutWidget to hold the image 
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground(getWidgetColors.BG_NORMAL)
        # Create and configure an ImageItem to display the image
        
        self.xlam = xlam
        
        self.plot = self.graphics_widget.addPlot(row = 0, col=0)      # add PlotItem to the main GraphicsLayoutWidget
        
        self.plot_data = data[stokes, 0, 0, 0, :]
        
        self.plot.plot(xlam, self.plot_data)
        self.plot.invertY(False)              # orient y axis to run bottom-to-top
        self.plot.setDefaultPadding(0.0)      # plot without padding data range
        
            # show full frame, label tick marks at top and left sides, with some extra space for labels:
        self.plot.showAxes( True, showValues=(True, True, False, False), size=15 )

         # Configure x-axis ticks
        x_ticks = xlam[0] + np.arange(xlam.shape[0], step=int(xlam.shape[0] / 10))
        self.plot.getAxis('top').setTicks([[(tick, f'{tick:.1f}') for tick in x_ticks]])
        self.plot.getAxis('top').enableAutoSIPrefix(False)
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.plot.setLabel("bottom", text="Wavelength", units="A")
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        # self.plot.getAxis('left').setWidth(40)
                
        # pg.setConfigOptions(imageAxisOrder="row-major")
        
        # self.plotItem.setLabel("bottom", text="x", units="arcsec")
        # self.plotItem.setLabel("left", text="y", units="arcsec")

        # # Add the GraphicsLayoutWidget to the layout
        layout.addWidget(self.graphics_widget)

        # # Set the main layout for this widget
        self.setLayout(layout)
        
        
    def update_plot_data(self, new_data):
            self.plot.clear()
            self.plot.plot(self.xlam, new_data)

class StokesSpectrumImageWindow(QtWidgets.QWidget):
    def __init__(self, xlam, data, stokes=0, coordinates=[0, 0, 0, 0], parent=None):
        super().__init__(parent)

        # Extract specific data for the stokes parameter
        self.data = data[stokes, :, :, :, :]
        self.n_x_pixel = self.data.shape[2]
        self.n_y_pixel = self.data.shape[3]
        
        self.xlam = xlam
        
        # Set up the main layout
        layout = QtWidgets.QHBoxLayout(self)
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground(getWidgetColors.BG_NORMAL)
        self.image_item = pg.ImageItem()
        self.image_item.setImage(self.data[0, 0, :, :])  # Initial image display
        
        # Configure the plot and add ImageItem
        self.plotItem = self.graphics_widget.addPlot(row=0, col=0)
        self.plotItem.addItem(self.image_item)
        self.plotItem.invertY(False)
        self.plotItem.setDefaultPadding(0.0)
        self.plotItem.showAxes(True, showValues=(True, True, False, False), size=20)
        
      
        # Add the GraphicsLayoutWidget and HistogramLUTWidget to layout
        layout.addWidget(self.graphics_widget)
        self.histogram = pg.HistogramLUTWidget()
        self.histogram.setImageItem(self.image_item)
        self.histogram.setBackground(getWidgetColors.BG_NORMAL)
        layout.addWidget(self.histogram)
        
        self.setLayout(layout)

    def update_plot_data(self, new_data):
        self.image_item.clear()
        self.image_item.setImage(new_data)     


    
class InteractiveSpectrumWidget(QtWidgets.QWidget):
    def __init__(self, xlam, data, wins,stokes=0,parent=None):
        super().__init__(parent)

        # Initialize the GraphicsLayoutWidget for plotting
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground(getWidgetColors.BG_NORMAL)
        self.xlam = xlam
        # Create a plot within the GraphicsLayoutWidget
        self.plot = self.plot_widget.addPlot(row=1, col=0)
        
        # Plot the initial data
        self.data = data
        self.plot_data = data[stokes, :, :, :].mean(axis=(0,1, 2))
        self.plot.plot(xlam, self.plot_data)
        
        # Configure x-axis ticks
        x_ticks = xlam[0] + np.arange(xlam.shape[0], step=int(xlam.shape[0] / 10))
        self.plot.getAxis('top').setTicks([[(tick, f'{tick:.1f}') for tick in x_ticks]])
        self.plot.getAxis('top').enableAutoSIPrefix(False)
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.plot.setLabel("bottom", text="Wavelength", units="A")
        
        # Add an InfiniteLine for crosshair functionality
        self.vLine = pg.InfiniteLine(angle=90, movable=True)
        self.vLine.setPen('blue', width=2)

        self.plot.addItem(self.vLine, ignoreBounds=True)

        # Hold reference to the target image widget (for updating its image)
        self.wins = wins
       

        # Layout and add the plot widget to this container
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        # Set up mouse move event
        self.plot.scene().sigMouseMoved.connect(self.mouseMoved)
        self.plot.scene().sigMouseClicked.connect(self.mouseClicked)
   
            # Variables to track crosshair lock state and position
        self.crosshair_locked = False
    def mouseMoved(self, pos):
        """Handle the mouse moved event over the plot area."""
        if not self.crosshair_locked:   
         if self.plot.sceneBoundingRect().contains(pos):
            # Map the scene position to plot coordinates (wavelength scale)
            mousePoint = self.plot.vb.mapSceneToView(pos)
            wavelength = mousePoint.x()

            # Find the closest index in xlam to the mouse wavelength
            index = np.abs(self.xlam - wavelength).argmin()
         
            if 0 <= index < self.plot_data.shape[0]:
                # Update position of the vertical line
                self.vLine.setPos(self.xlam[index])
                # Update the image in the external win_i_image widget
                for  win in self.wins:
                   win.wavelength_pos = index
                   win.update_image()

                
    def mouseClicked(self, event):
            if event.double():
                # On double-click, fix crosshair at current position
                mouse_point = self.plot.vb.mapSceneToView(event.scenePos())
                self.vLine.setPos(mouse_point.x())
                self.crosshair_locked = not self.crosshair_locked  # Toggle lock state
                
                
class InteractiveScanPosWidget(QtWidgets.QWidget):
    def __init__(self, xlam, data, wins,stokes=0,parent=None):
        super().__init__(parent)

        # Initialize the GraphicsLayoutWidget for plotting
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground(getWidgetColors.BG_NORMAL)
        
        # Create a plot within the GraphicsLayoutWidget
        self.plot = self.plot_widget.addPlot(row=1, col=0)
        
        self.scan = 0

        
        # Plot the initial data
        self.plot_data = data[stokes, :, :, :].mean(axis=(0,2, 3))
        self.scan_pos = np.arange(self.plot_data.shape[0])
        self.plot.plot(self.scan_pos, self.plot_data, symbol='o')
        
        # Configure x-axis ticks
        x_ticks = np.arange(self.plot_data.shape[0])
        self.plot.getAxis('top').setTicks([[(tick, f'{tick:.1f}') for tick in x_ticks]])
        self.plot.getAxis('top').enableAutoSIPrefix(False)
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.plot.setLabel("bottom", text="Scan number", units="")
        
        # Add an InfiniteLine for crosshair functionality
        self.vLine = pg.InfiniteLine(angle=90, movable=True)
        self.vLine.setPen('blue', width=2)

        self.plot.addItem(self.vLine, ignoreBounds=True)

        # Hold reference to the target image widget (for updating its image)
        self.wins = wins

        # Layout and add the plot widget to this container
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        # Set up mouse move event
        self.plot.scene().sigMouseMoved.connect(self.mouseMoved)
        self.plot.scene().sigMouseClicked.connect(self.mouseClicked)
   
            # Variables to track crosshair lock state and position
        self.crosshair_locked = False
    def mouseMoved(self, pos):
        """Handle the mouse moved event over the plot area."""
        if not self.crosshair_locked:   
         if self.plot.sceneBoundingRect().contains(pos):
            # Map the scene position to plot coordinates (wavelength scale)
            mousePoint = self.plot.vb.mapSceneToView(pos)
            self.scan = np.abs(self.scan_pos - mousePoint.x()).argmin()
         
            if 0 <= self.scan < self.plot_data.shape[0]:
                # Update position of the vertical line
                self.vLine.setPos(self.scan)
                # Update the image in the external win_i_image widget
                for  win in self.wins:
                   win.scan = self.scan
                   win.update_image()

                
    def mouseClicked(self, event):
            if event.double():
                # On double-click, fix crosshair at current position
                mouse_point = self.plot.vb.mapSceneToView(event.scenePos())
                self.vLine.setPos(mouse_point.x())
                self.crosshair_locked = not self.crosshair_locked  # Toggle lock state

def display_scan_data(data, xlam, coordinates=[0,1,0,1], title='Example'): # 4, X, wl, Y
   
    # Initialize the application
    app = pg.mkQApp("THEMIS Data")
    win = QtWidgets.QMainWindow()
    area = DockArea()
    win.setCentralWidget(area)
    win.resize(1600,850)
    win.setWindowTitle(title)
    
    # if len(np.array(data).shape) < 4:
    #     data=np.zeros((4,10,500, 500))
    #     xlam=np.arange(data.shape[2])
    #     print('Data has the wrong dimensions!')
    

    ## Create docks, place them into the window one at a time.
    ## Note that size arguments are only a suggestion; docks will still have to
    ## fill the entire dock area and obey the limits of their internal widgets.
    d_scan_ui_state0 = Dock("Scan upper image, state 0", size=(300, 300))     ## give this dock the minimum possible size
    d_scan_li_state0 = Dock("Scan lower image, state 0", size=(300, 300))     ## give this dock the minimum possible size
    
    d_scan_ui_state1 = Dock("Scan upper image, state 1", size=(300, 300))     ## give this dock the minimum possible size
    d_scan_li_state1 = Dock("Scan lower image, state 1", size=(300, 300))     ## give this dock the minimum possible size
    
    # -----------------
    
    d_spectrum_ui_state0 = Dock("Upper spectrum, state 0", size=(300,300))
    d_spectrum_li_state0 = Dock("Lower spectrum, state 0", size=(300,300))
    
    
    d_spectrum_image_ui_state0 = Dock("Upper spectrum image, state 0", size=(300,300))
    d_spectrum_image_li_state0 = Dock("Lower spectrum image, state 0", size=(300,300))
    
    d_spectrum_ui_state1 = Dock("Upper spectrum, state 1", size=(300,300))
    d_spectrum_li_state1 = Dock("Lower spectrum, state 1", size=(300,300))
    
    
    d_spectrum_image_ui_state1 = Dock("Upper spectrum image, state 1", size=(300,300))
    d_spectrum_image_li_state1 = Dock("Lower spectrum image, state 1", size=(300,300))

    # ------------
    
    d_spectrum_ui = Dock("Upper full spectrum", size=(20,170))
    d_spectrum_li = Dock("Lower full spectrum", size=(20,170))
    
    d_scan_ui = Dock("Upper scan", size=(30,100))
    d_scan_li = Dock("Lower scan", size=(30,100))
   
    # ------------------

    
    area.addDock(d_scan_ui_state0, 'left')      ##  
    area.addDock(d_scan_li_state0, 'right')      ## 

    area.addDock(d_scan_ui_state1, 'bottom', d_scan_ui_state0)      ## 
    area.addDock(d_scan_li_state1, 'bottom', d_scan_li_state0)      ## 
    
    area.addDock(d_spectrum_ui, 'bottom', d_scan_ui_state1 )      ## 
    area.addDock(d_spectrum_li, 'bottom', d_scan_li_state1)      ## 
    
    area.addDock(d_scan_ui, 'bottom', d_spectrum_ui)      ## 
    area.addDock(d_scan_li, 'bottom', d_spectrum_li)      ## 
       
    area.addDock(d_spectrum_ui_state0, 'right', d_scan_ui_state0)      ##  
    area.addDock(d_spectrum_li_state0, 'right', d_scan_li_state0)      ##
  
    
    area.addDock(d_spectrum_image_ui_state0, 'above', d_spectrum_ui_state0)       ## tab
    area.addDock(d_spectrum_image_li_state0, 'above', d_spectrum_li_state0) 
    
    area.addDock(d_spectrum_ui_state1, 'right', d_scan_ui_state1)      ##  
    area.addDock(d_spectrum_li_state1, 'right', d_scan_li_state1)      ##
  
    
    area.addDock(d_spectrum_image_ui_state1, 'above', d_spectrum_ui_state1)       ## tab
    area.addDock(d_spectrum_image_li_state1, 'above', d_spectrum_li_state1) 

    
    # area.addDock(d10, 'bottom')## place at the bottom  
    # area.addDock(d21, 'right')
    # area.addDock(d20, 'above', d21)      ## tab
    
    # d9.label.setMinimumWidth(200)  # Set minimum width for the first dock's label
    # d11.label.setMinimumWidth(200)  # Set minimum width for the second dock's label
    # d13.label.setMinimumWidth(200)  # Set minimum width for the third dock's label
    
    ## Add widgets into each dock
# --------------------------------------------------------        
    ## Stokes I spectrum and spectrum image dock setup

    win_spectrum_ui_state0 = StokesSpectrumWindow(xlam, data.ui, stokes=0)  # Create a GraphicsLayoutWidget
    
    # Add the GraphicsLayoutWidget to the dock
    d_spectrum_ui_state0.addWidget(win_spectrum_ui_state0)
    
    win_image_spectrum_ui_state0 = StokesSpectrumImageWindow(xlam, data.ui, stokes=0)
    d_spectrum_image_ui_state0.addWidget(win_image_spectrum_ui_state0)
    
    
    win_spectrum_ui_state1 = StokesSpectrumWindow(xlam, data.ui, stokes=1)  # Create a GraphicsLayoutWidget
    
    # Add the GraphicsLayoutWidget to the dock
    d_spectrum_ui_state1.addWidget(win_spectrum_ui_state1)
    
    win_image_spectrum_ui_state1 = StokesSpectrumImageWindow(xlam, data.ui, stokes=1)
    d_spectrum_image_ui_state1.addWidget(win_image_spectrum_ui_state1)
    
    
    win_spectrum_li_state0 = StokesSpectrumWindow(xlam, data.li, stokes=0)  # Create a GraphicsLayoutWidget
    
    # Add the GraphicsLayoutWidget to the dock
    d_spectrum_li_state0.addWidget(win_spectrum_li_state0)
    
    win_image_spectrum_li_state0 = StokesSpectrumImageWindow(xlam, data.li, stokes=0)
    d_spectrum_image_li_state0.addWidget(win_image_spectrum_li_state0)
    
    
    win_spectrum_li_state1 = StokesSpectrumWindow(xlam, data.li, stokes=1)  # Create a GraphicsLayoutWidget
    
    # Add the GraphicsLayoutWidget to the dock
    d_spectrum_li_state1.addWidget(win_spectrum_li_state1)
    
    win_image_spectrum_li_state1 = StokesSpectrumImageWindow(xlam, data.li, stokes=1)
    d_spectrum_image_li_state1.addWidget(win_image_spectrum_li_state1)
    
    
# --------------------------------------------------------    
    
    ## First state scan image dock setup   
    
    win_image_ui_state0 = StokesImageWindow(xlam, data.ui, win_spectrum_ui_state0, win_image_spectrum_ui_state0,stokes=0)  # Create a GraphicsLayoutWidget
    
    # # Add the GraphicsLayoutWidget to the dock
    d_scan_ui_state0.addWidget(win_image_ui_state0)
    
    win_image_li_state0 = StokesImageWindow(xlam, data.li, win_spectrum_li_state0, win_image_spectrum_li_state0,stokes=0)  # Create a GraphicsLayoutWidget
    
    # # Add the GraphicsLayoutWidget to the dock
    d_scan_li_state0.addWidget(win_image_li_state0)
    
    win_image_ui_state1 = StokesImageWindow(xlam, data.ui, win_spectrum_ui_state1, win_image_spectrum_ui_state1,stokes=1)  # Create a GraphicsLayoutWidget
    
    # # Add the GraphicsLayoutWidget to the dock
    d_scan_ui_state1.addWidget(win_image_ui_state1)
    
    win_image_li_state1 = StokesImageWindow(xlam, data.li, win_spectrum_li_state1, win_image_spectrum_li_state1,stokes=1)  # Create a GraphicsLayoutWidget
    
    # # Add the GraphicsLayoutWidget to the dock
    d_scan_li_state1.addWidget(win_image_li_state1)

# --------------------------------------------------------  
    ## full spectrum dock 

    spectrum_widget_ui = InteractiveSpectrumWidget(xlam,  data.ui, [win_image_ui_state0,win_image_ui_state1])
    # # Add the GraphicsLayoutWidget to the dock
    d_spectrum_ui.addWidget(spectrum_widget_ui)
    
    
    ## scan dock
    scan_widget_ui = InteractiveScanPosWidget(xlam,  data.ui, [win_image_ui_state0,win_image_ui_state1])
    # # Add the GraphicsLayoutWidget to the dock
    d_scan_ui.addWidget(scan_widget_ui)
    
    spectrum_widget_li = InteractiveSpectrumWidget(xlam,  data.li, [win_image_li_state0,win_image_li_state1])
    # # Add the GraphicsLayoutWidget to the dock
    d_spectrum_li.addWidget(spectrum_widget_li)
    
    
    ## scan dock
    scan_widget_li = InteractiveScanPosWidget(xlam,  data.li, [win_image_li_state0,win_image_li_state1])
    # # Add the GraphicsLayoutWidget to the dock
    d_scan_li.addWidget(scan_widget_li)
    
 

# --------------------------------------------------------  


    win.show()
    dark_stylesheet=qdarkstyle.load_stylesheet_from_environment(is_pyqtgraph=True)
    app.setStyleSheet(dark_stylesheet)

    app.exec_()

if __name__ == '__main__':
    #create numpy arrays
    #make the numbers large to show that the range shows data from 10000 to all the way 0
    data = tt.l1_data()
    data.ui = np.random.random(size=(2,41,16,256, 512))   # stokes, scan position, wavelength, spatial along slit
    data.li = 1.*data.ui
    xlam = np.arange(data.ui.shape[4])
    display_scan_data(data, xlam,  title = 'Test data')
