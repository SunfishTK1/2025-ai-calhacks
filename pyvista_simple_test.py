#!/usr/bin/env python3
"""
Simple PyVista Test
===================

A simple test to verify PyVista is working correctly.
"""

import numpy as np
import pyvista as pv

def simple_test():
    """Create a simple 3D plot to test PyVista"""
    print("Testing PyVista...")
    
    # Create a simple sphere
    sphere = pv.Sphere(radius=2)
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add the sphere
    plotter.add_mesh(sphere, color='red', opacity=0.8)
    plotter.add_title('PyVista Test - Simple Sphere')
    
    # Show the plot
    print("Opening 3D window... Close the window when done.")
    plotter.show()
    
    print("PyVista test completed successfully!")

if __name__ == "__main__":
    simple_test() 