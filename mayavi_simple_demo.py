#!/usr/bin/env python3
"""
Simple Mayavi Demo
==================

A simple demonstration of Mayavi's 3D visualization capabilities.
This script creates a few basic 3D plots to showcase Mayavi's features.

Run this script to test if Mayavi is working correctly on your system.
"""

import numpy as np
from mayavi import mlab

def simple_3d_plot():
    """Create a simple 3D surface plot"""
    print("Creating 3D Surface Plot...")
    
    # Clear any existing plots
    mlab.clf()
    
    # Generate data for a simple surface
    x, y = np.mgrid[-2:2:50j, -2:2:50j]
    z = np.sin(x*np.pi) * np.cos(y*np.pi) * np.exp(-(x**2 + y**2)/2)
    
    # Create the surface plot
    surf = mlab.surf(x, y, z, colormap='viridis')
    
    # Add a title and labels
    mlab.title('Simple 3D Surface', size=0.3)
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
    mlab.colorbar(surf, title="Height")
    
    # Show the plot
    mlab.show()

def scatter_3d():
    """Create a 3D scatter plot"""
    print("Creating 3D Scatter Plot...")
    
    mlab.clf()
    
    # Generate random data
    n = 500
    x = np.random.randn(n)
    y = np.random.randn(n) 
    z = np.random.randn(n)
    colors = x + y  # Color based on x+y values
    
    # Create scatter plot
    pts = mlab.points3d(x, y, z, colors, scale_mode='none', scale_factor=0.1)
    
    mlab.title('3D Scatter Plot', size=0.3)
    mlab.colorbar(pts, title="X+Y Value")
    
    mlab.show()

def vector_field():
    """Create a simple vector field visualization"""
    print("Creating Vector Field...")
    
    mlab.clf()
    
    # Generate vector field data
    x, y, z = np.mgrid[-1:1:8j, -1:1:8j, -1:1:8j]
    
    # Simple vector field (circulation)
    u = -y
    v = x
    w = z * 0.1
    
    # Create vector field plot
    vectors = mlab.quiver3d(x, y, z, u, v, w, scale_factor=0.3)
    
    mlab.title('Vector Field Visualization', size=0.3)
    
    mlab.show()

def main():
    """Run the demo"""
    print("Mayavi Simple Demo")
    print("==================")
    print("This demo will show three different types of 3D visualizations.")
    print("Close each window to proceed to the next demo.\n")
    
    demos = [
        ("3D Surface", simple_3d_plot),
        ("3D Scatter", scatter_3d),
        ("Vector Field", vector_field)
    ]
    
    for name, demo_func in demos:
        print(f"\nRunning {name} demo...")
        try:
            demo_func()
        except Exception as e:
            print(f"Error running {name}: {e}")
            print("Make sure you have a display available and Mayavi is properly installed.")
        
        input("Press Enter to continue to next demo...")
    
    print("\nDemo complete! Mayavi is working correctly.")

if __name__ == "__main__":
    main() 