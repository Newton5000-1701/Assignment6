#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:03:19 2024

@author: isaacthompson
"""

import numpy as np
from matplotlib import pyplot as pp


dx = 0.001 #float(input('Please enter the x-spacing on grid (m): '))  
dy = 0.001 #float(input('Please enter the y-spacing on grid (m): '))  


x = np.arange(-0.1, 0.1, dx)
y = np.arange(-0.1, 0.1, dy)
X, Y = np.meshgrid(x, y)


q1 = 1 #float(input("Please enter the first charge (C): "))
q2 = -1 #float(input('Please enter the second charge (C): '))
e_0 = 8.8541 * 10**(-12)  # Vacuum permittivity in SI units
a1 = q1 / (4 * np.pi * e_0)
a2 = q2 / (4 * np.pi * e_0)

# Positions of charges
x1 = -0.05 #float(input('Please enter the x-position of charge 1 (m): '))
y1 = 0 #float(input('Please enter the y-position of charge 1 (m): '))
x2 = 0.05 #float(input('Please enter the x-position of charge 2 (m): '))
y2 = 0 #float(input('Please enter the y-position of charge 2 (m): '))

# Masking radius in meters (0.1 cm = 0.001 m)
mask_radius = 0.002

# Distance functions with masking to avoid singularities
def r1(X, Y, xp1, yp1):
    r = np.sqrt((X - xp1)**2 + (Y - yp1)**2)
    return np.where(r <= mask_radius, np.nan, r)


def r2(X, Y, xp2, yp2):
    r = np.sqrt((X - xp2)**2 + (Y - yp2)**2)
    return np.where(r <= mask_radius, np.nan, r)

# Potential functions
def V1(X, Y, xp1, yp1):
    return a1 / r1(X, Y, xp1, yp1)

def V2(X, Y, xp2, yp2):
    return a2 / r2(X, Y, xp2, yp2)

def V(X, Y, xp1, xp2, yp1, yp2):
    return V1(X, Y, xp1, yp1) + V2(X, Y, xp2, yp2)


V_total = V(X, Y, x1, x2, y1, y2)

def x_partial(f, X, Y, xp1, xp2, yp1, yp2, h):
    return (f(X + h, Y, xp1, xp2, yp1, yp2) - f(X - h, Y, xp1, xp2, yp1, yp2)) / (2 * h)

def y_partial(f, X, Y, xp1, xp2, yp1, yp2, H):
    return (f(X, Y + H, xp1, xp2, yp1, yp2) - f(X, Y - H, xp1, xp2, yp1, yp2)) / (2 * H)

#calculate the partials
E_x = -x_partial(V, X, Y, x1, x2, y1, y2, dx)
E_y = -y_partial(V, X, Y, x1, x2, y1, y2, dy)


# Plot electric potential with masked regions to avoid singularities
pp.figure(figsize=(8, 6))
pp.contourf(X, Y, V_total, cmap='inferno', levels=50)
pp.colorbar(label="Electric Potential (V)")
pp.xlabel("x (m)")
pp.ylabel("y (m)")
pp.title("Electric Potential of Two Charges with Masked Regions")
pp.savefig('potential.png')
pp.show()

E_magnitude = np.sqrt(E_x**2 + E_y**2)
E_x_norm = E_x / E_magnitude
E_y_norm = E_y / E_magnitude

# Create a mask for plotting only every nth point for reduced density
density_factor = 5  
mask = np.zeros_like(E_magnitude, dtype=bool)
mask[::density_factor, ::density_factor] = True  

# Apply mask to normalized field components for plotting field's directions
E_x_reduced = np.where(mask, E_x_norm, np.nan)  # Use NaN to ignore in quiver plot
E_y_reduced = np.where(mask, E_y_norm, np.nan)


#Plotting the direction with quiver and magnitude with contour

pp.figure(figsize=(8, 6))
contour = pp.contour(X, Y, E_magnitude, cmap='Spectral', levels=1000)
pp.colorbar(contour, label="Electric Field Magnitude (V/m)")
pp.quiver(X, Y, E_x_reduced, E_y_reduced, color='black', scale=35, width=0.003)
pp.xlabel("x (m)")
pp.ylabel("y (m)")
pp.title("Normalized Electric Field of Two Charges (Density-Reduced)")
pp.savefig('efield.png')
pp.show()

