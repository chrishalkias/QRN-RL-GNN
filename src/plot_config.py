# -*- coding: utf-8 -*-
# src/models.py

'''
Created Wed 07 May 2025
Contains the parameter update for pretties plots
'''

import matplotlib.pyplot as plt

def set():
    """Sets the parameters for matplotlib"""
    plt.rcParams.update({
        # --- Fonts ---
        "font.family": "serif",          # Use serif fonts (like Times New Roman)
        "font.serif": ["DejaVu Serif"],  # Built-in font that supports math symbols
        "mathtext.fontset": "dejavuserif",  # Math font matching main text
        "font.size": 11,                 # Base font size

        # --- Figure Layout ---
        "figure.figsize": (5.5, 4.0),    # Width, height in inches (similar to PRL)
        "figure.autolayout": False,       # Prevent label clipping
        "figure.dpi": 300,               # High resolution for papers
        
        # --- Axes ---
        "axes.labelsize": 8,            # Axis label size
        "axes.titlesize": 7,            # Title size
        "axes.linewidth": 0.8,           # Border thickness
        "axes.grid": False,              # Disable grid (common in physics papers)
        
        # --- Ticks ---
        "xtick.direction": "in",         # Ticks point inward
        "ytick.direction": "in",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 4,           # Tick length
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,        # Tick thickness
        "ytick.major.width": 0.8,
        
        # --- Lines ---
        "lines.linewidth": 1.2,          # Plot line thickness
        "lines.markersize": 5,           # Marker size
        
        # --- Legend ---
        "legend.fontsize": 9,
        "legend.frameon": False,         # No background box
        "legend.loc": "best",

        # --- Core Layout ---
        "figure.constrained_layout.use": True,  # Enable smarter layout engine
        "figure.constrained_layout.h_pad": 0.1,  # Horizontal padding (inches)
        "figure.constrained_layout.w_pad": 0.1,  # Vertical padding
        "figure.constrained_layout.hspace": 0.1, # Subplot spacing
        "figure.constrained_layout.wspace": 0.1,

        # --- Title/Label Protection ---
        "axes.titlelocation": "center",   # Prevents title drifting
        "axes.titlepad": 10,              # Space above title (points)
        "axes.labelpad": 5,               # Space for axis labels
        
        # --- Font Adjustments ---
        "font.size": 9,                  # Slightly smaller base size
        "axes.titlesize": 10,             # Explicit title size
        "axes.labelsize": 10,             # Axis label size
    })

