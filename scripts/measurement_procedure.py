# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import TimeDelta

import sunpy.data.sample
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a


date="2025-07-05"

# Download and load HMI map
temp_dir = '/data/mdata/projects/themis/data/rdata/2025-07-05/context'
hmi_file = os.path.join(temp_dir, 'hmi_20250705.fits')
os.makedirs(temp_dir, exist_ok=True)

if os.path.exists(hmi_file):
    hmi_map = sunpy.map.Map(hmi_file)
else:
    print(f"Downloading HMI continuum map for 2025-07-05...")
    result = Fido.search(a.Time('2025-07-05T00:00:00', '2025-07-05T23:59:59'),
                         a.Instrument('hmi'),
                         a.Physobs('intensity'),
                         a.Sample(24*u.hour))
    if len(result[0]) == 0:
        # Try magnetogram if intensity not available
        result = Fido.search(a.Time('2025-07-05T00:00:00', '2025-07-05T23:59:59'),
                             a.Instrument('hmi'),
                             a.Physobs('magnetic_field'),
                             a.Sample(24*u.hour))
    downloaded_files = Fido.fetch(result[0][0], path=temp_dir)
    hmi_map = sunpy.map.Map(downloaded_files[0])
    # Save to our designated location
    hmi_map.save(hmi_file, overwrite=True)
    print(f"HMI map saved to {hmi_file}")

# Fix HMI coordinate system issue using sunpy documentation solution
hmi_rotated = hmi_map.rotate(order=3)


def plot_measurement_positions(ax, hmi_rotated):
    """Plot HMI map with measurement position boxes at specific latitudes.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis with WCS projection for the HMI map
    hmi_rotated : sunpy map
        Rotated HMI map to plot
    """
    hmi_rotated.plot(axes=ax, clip_interval=(1, 99.99)*u.percent)
    ax.set_title(f'HMI continuum map for 2025-07-05')
    hmi_rotated.draw_grid(axes=ax)
    
    # Parameters for measurement boxes
    box_height = 196.61 * u.arcsec  # north-south direction, 2028*0.0095
    box_width = 42.3 * u.arcsec     # east-west direction, 141*0.3
    latitudes_list = [0, 30, -30, 40, -40, 45, -45, 55, -55]  # degrees
    
    # Define distinct colors for each box
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink']
    
    def draw_box_at_latitude(latitude_deg, label, color, fontsize=8, fontweight='normal'):
        """Draw a box centered at disk center (longitude 0) at given latitude with label."""
        # Calculate center position at disk center (x=0, longitude=0)
        x_center = 0 * u.arcsec
        
        # Convert latitude to heliographic coordinates
        # For a point at disk center (longitude=0), the y-coordinate in arcsec
        # corresponds directly to the latitude
        from sunpy.coordinates import HeliographicStonyhurst
        
        # Create a coordinate at the specified latitude at disk center (longitude=0)
        hgs_coord = HeliographicStonyhurst(0*u.deg, latitude_deg*u.deg, 
                                           obstime=hmi_rotated.date)
        # Transform to Helioprojective frame as seen by the observer
        hpc_coord = hgs_coord.transform_to(hmi_rotated.coordinate_frame)
        y_center = hpc_coord.Ty
        
        # Calculate box corners (height is north-south, width is east-west)
        y_min = y_center - box_height / 2
        y_max = y_center + box_height / 2
        x_min = x_center - box_width / 2
        x_max = x_center + box_width / 2
        
        # Create box corners
        box_x = [x_min, x_max, x_max, x_min, x_min]
        box_y = [y_min, y_min, y_max, y_max, y_min]
        box_coords = SkyCoord(box_x, box_y, frame=hmi_rotated.coordinate_frame)
        ax.plot_coord(box_coords, color=color, linewidth=2, alpha=0.8)
        
        # Calculate label position (to the right of the box)
        label_offset = 50 * u.arcsec
        text_x = x_max.value + label_offset.value
        text_y = y_center.value
        
        # Add label using pixel coordinates
        text_coord = SkyCoord(text_x * u.arcsec, text_y * u.arcsec, frame=hmi_rotated.coordinate_frame)
        pixel_x, pixel_y = hmi_rotated.world_to_pixel(text_coord)
        ax.annotate(label, xy=(pixel_x.value, pixel_y.value),
                    xycoords='data', color=color, fontsize=fontsize, fontweight=fontweight,
                    verticalalignment='center')
    
    # Draw boxes at each latitude at disk center
    for i, lat in enumerate(latitudes_list):
        if lat >= 0:
            label = f'{lat}°N' if lat > 0 else '0°'
        else:
            label = f'{abs(lat)}°S'
        draw_box_at_latitude(lat, label, colors[i])




def main():
    """Main function to plot the HMI map with measurement positions."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=hmi_rotated)
    plot_measurement_positions(ax, hmi_rotated)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()




