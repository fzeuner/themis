#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 14:30:44 2025

@author: franziskaz
"""

import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, amplitude, mean, std_dev):
    """Generate a Gaussian function."""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))


def main():
    # Generate x values
    x = np.linspace(-10, 10, 1000)
    
    # Parameters for two Gaussians with variable shifts
    amplitude1, amplitude2 = -1.0, 1.0
    mean1, mean2 = -3.0, 3.0  # Variable shifts
    std_dev1, std_dev2 = 1.0, 1.0
    
    # Generate two Gaussians
    gauss1 = gaussian(x-3, amplitude1, mean1, std_dev1)
    gauss2 = gaussian(x+3, amplitude2, mean2, std_dev1)
    
    # Calculate addition and difference
    addition = gauss1 + gauss2
    difference = gauss1 - gauss2
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('Gaussian Functions: Addition and Difference', fontsize=16)
    
    # Plot individual Gaussians
    axes[0, 0].plot(x, gauss1, 'b-', label=f'Gaussian 1 (μ={mean1})', linewidth=2)
    axes[0, 0].plot(x, gauss2, 'r-', label=f'Gaussian 2 (μ={mean2})', linewidth=2)
    axes[0, 0].set_title('Individual Gaussians')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot addition
    axes[0, 1].plot(x, gauss1, 'b--', alpha=0.5, label='Gaussian 1')
    axes[0, 1].plot(x, gauss2, 'r--', alpha=0.5, label='Gaussian 2')
    axes[0, 1].plot(x, addition, 'g-', label='Addition', linewidth=2)
    axes[0, 1].set_title('Addition of Gaussians')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot difference
    axes[1, 0].plot(x, gauss1, 'b--', alpha=0.5, label='Gaussian 1')
    axes[1, 0].plot(x, gauss2, 'r--', alpha=0.5, label='Gaussian 2')
    axes[1, 0].plot(x, difference, 'm-', label='Difference', linewidth=2)
    axes[1, 0].set_title('Difference of Gaussians')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot all together for comparison
    axes[1, 1].plot(x, gauss1, 'b-', alpha=0.7, label='Gaussian 1')
    axes[1, 1].plot(x, gauss2, 'r-', alpha=0.7, label='Gaussian 2')
    axes[1, 1].plot(x, addition, 'g-', label='Addition', linewidth=2)
    axes[1, 1].plot(x, difference, 'm-', label='Difference', linewidth=2)
    axes[1, 1].set_title('All Functions Combined')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gaussian_addition_difference.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'gaussian_addition_difference.png'")


if __name__ == "__main__":
    main()

