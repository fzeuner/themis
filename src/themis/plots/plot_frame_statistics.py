#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting helper functions for FramesSet and CycleSet objects.

These functions provide convenient plotting utilities for analyzing frame statistics
from THEMIS data. They support both FramesSet (simple indexed frames) and CycleSet
(frames with polarization state information).
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional, List, Tuple
from themis.core.data_classes import FramesSet, CycleSet


def plot_frame_statistics(frame_set: Union[FramesSet, CycleSet],
                           stat: str = 'mean',
                           ax: Optional[plt.Axes] = None,
                           title: Optional[str] = None,
                           **plot_kwargs) -> plt.Axes:
    """
    Plot statistical measures (mean, median, std, etc.) for upper and lower frames.
    
    Parameters
    ----------
    frame_set : FramesSet or CycleSet
        The frame data to plot
    stat : str, optional
        Statistical measure: 'mean', 'median', 'std', 'min', 'max', 'percentile_95', etc.
        Default is 'mean'
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    title : str, optional
        Plot title
    **plot_kwargs : dict
        Additional keyword arguments passed to plt.plot()
    
    Returns
    -------
    plt.Axes
        The axes object used for plotting
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define stat function
    stat_funcs = {
        'mean': lambda x: np.mean(x, axis=(1, 2)),
        'median': lambda x: np.median(x, axis=(1, 2)),
        'std': lambda x: np.std(x, axis=(1, 2)),
        'min': lambda x: np.min(x, axis=(1, 2)),
        'max': lambda x: np.max(x, axis=(1, 2)),
        'percentile_95': lambda x: np.percentile(x, 95, axis=(1, 2)),
        'percentile_5': lambda x: np.percentile(x, 5, axis=(1, 2)),
    }
    
    if stat not in stat_funcs:
        raise ValueError(f"Unknown stat '{stat}'. Choose from: {list(stat_funcs.keys())}")
    
    stat_func = stat_funcs[stat]
    
    if isinstance(frame_set, CycleSet):
        pol_states = sorted(set(key[0] for key in frame_set.frames.keys()))
        
        for pol_state in pol_states:
            state_subset = frame_set.get_state(pol_state)
            
            upper_stat = stat_func(state_subset.stack_all('upper'))
            ax.plot(upper_stat, label=f'{pol_state} upper', **plot_kwargs)
            
            lower_stat = stat_func(state_subset.stack_all('lower'))
            ax.plot(lower_stat, label=f'{pol_state} lower', linestyle='--', **plot_kwargs)
    
    elif isinstance(frame_set, FramesSet):
        upper_stat = stat_func(frame_set.stack_all('upper'))
        ax.plot(upper_stat, label='upper', **plot_kwargs)
        
        lower_stat = stat_func(frame_set.stack_all('lower'))
        ax.plot(lower_stat, label='lower', **plot_kwargs)
    
    else:
        raise TypeError(f"Expected FramesSet or CycleSet, got {type(frame_set)}")
    
    ax.set_xlabel('Frame Index')
    ax.set_ylabel(f'{stat.capitalize()} Pixel Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Frame {stat.capitalize()} Statistics')
    
    return ax


def plot_state_comparison(cycle_set: CycleSet,
                          frame_position: str = 'upper',
                          stat: str = 'mean',
                          ax: Optional[plt.Axes] = None,
                          title: Optional[str] = None,
                          **plot_kwargs) -> plt.Axes:
    """
    Compare statistics across different polarization states for a single frame position.
    
    Parameters
    ----------
    cycle_set : CycleSet
        The frame data to plot (must be CycleSet)
    frame_position : str, optional
        Frame position: 'upper' or 'lower'. Default is 'upper'
    stat : str, optional
        Statistical measure to plot. Default is 'mean'
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    title : str, optional
        Plot title
    **plot_kwargs : dict
        Additional keyword arguments passed to plt.plot()
    
    Returns
    -------
    plt.Axes
        The axes object used for plotting
    """
    if not isinstance(cycle_set, CycleSet):
        raise TypeError("plot_state_comparison requires a CycleSet object")
    
    if frame_position not in ('upper', 'lower'):
        raise ValueError(f"frame_position must be 'upper' or 'lower', got '{frame_position}'")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define stat function
    stat_funcs = {
        'mean': lambda x: np.mean(x, axis=(1, 2)),
        'median': lambda x: np.median(x, axis=(1, 2)),
        'std': lambda x: np.std(x, axis=(1, 2)),
        'min': lambda x: np.min(x, axis=(1, 2)),
        'max': lambda x: np.max(x, axis=(1, 2)),
    }
    
    if stat not in stat_funcs:
        raise ValueError(f"Unknown stat '{stat}'. Choose from: {list(stat_funcs.keys())}")
    
    stat_func = stat_funcs[stat]
    
    pol_states = sorted(set(key[0] for key in cycle_set.frames.keys()))
    
    for pol_state in pol_states:
        state_subset = cycle_set.get_state(pol_state)
        data_stat = stat_func(state_subset.stack_all(frame_position))
        ax.plot(data_stat, label=pol_state, **plot_kwargs)
    
    ax.set_xlabel('Frame Index')
    ax.set_ylabel(f'{stat.capitalize()} Pixel Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{frame_position.capitalize()} Frame: {stat.capitalize()} by Polarization State')
    
    return ax


def plot_frame_difference(frame_set: Union[FramesSet, CycleSet],
                         stat: str = 'mean',
                         ax: Optional[plt.Axes] = None,
                         title: Optional[str] = None,
                         **plot_kwargs) -> plt.Axes:
    """
    Plot the difference between upper and lower frame position statistics.
    
    Useful for identifying systematic differences or drift between frame positions.
    
    Parameters
    ----------
    frame_set : FramesSet or CycleSet
        The frame data to plot
    stat : str, optional
        Statistical measure to compare. Default is 'mean'
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    title : str, optional
        Plot title
    **plot_kwargs : dict
        Additional keyword arguments passed to plt.plot()
    
    Returns
    -------
    plt.Axes
        The axes object used for plotting
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    stat_funcs = {
        'mean': lambda x: np.mean(x, axis=(1, 2)),
        'median': lambda x: np.median(x, axis=(1, 2)),
        'std': lambda x: np.std(x, axis=(1, 2)),
    }
    
    if stat not in stat_funcs:
        raise ValueError(f"Unknown stat '{stat}'. Choose from: {list(stat_funcs.keys())}")
    
    stat_func = stat_funcs[stat]
    
    if isinstance(frame_set, CycleSet):
        pol_states = sorted(set(key[0] for key in frame_set.frames.keys()))
        
        for pol_state in pol_states:
            state_subset = frame_set.get_state(pol_state)
            
            upper_stat = stat_func(state_subset.stack_all('upper'))
            lower_stat = stat_func(state_subset.stack_all('lower'))
            difference = upper_stat - lower_stat
            
            ax.plot(difference, label=pol_state, **plot_kwargs)
    
    elif isinstance(frame_set, FramesSet):
        upper_stat = stat_func(frame_set.stack_all('upper'))
        lower_stat = stat_func(frame_set.stack_all('lower'))
        difference = upper_stat - lower_stat
        
        ax.plot(difference, **plot_kwargs)
    
    else:
        raise TypeError(f"Expected FramesSet or CycleSet, got {type(frame_set)}")
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel(f'{stat.capitalize()} Difference (Upper - Lower)')
    
    if isinstance(frame_set, CycleSet):
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Frame Position Difference: {stat.capitalize()}')
    
    return ax


def plot_frame_overview(frame_set: Union[FramesSet, CycleSet],
                        figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create a comprehensive overview plot with multiple statistics.
    
    Generates a multi-panel figure showing:
    - Mean values per frame position (and state if CycleSet)
    - Standard deviation
    - Min/Max range
    - Frame position difference
    
    Parameters
    ----------
    frame_set : FramesSet or CycleSet
        The frame data to plot
    figsize : tuple, optional
        Figure size (width, height). Default is (15, 10)
    
    Returns
    -------
    plt.Figure
        The figure object containing all subplots
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    plot_frame_statistics(frame_set, stat='mean', ax=axes[0], title='Mean pixel values')
    plot_frame_statistics(frame_set, stat='std', ax=axes[1], title='Standard Deviation')
    plot_frame_statistics(frame_set, stat='max', ax=axes[2], title='Max Pixel Values')
    plot_frame_difference(frame_set, stat='mean', ax=axes[3], title='Frame Position Difference (Mean)')
    
    plt.tight_layout()
    return fig
