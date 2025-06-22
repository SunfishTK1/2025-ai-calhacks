#!/usr/bin/env python3
"""
PyVista 3D Visualization Demo
=============================

A comprehensive demonstration of PyVista's 3D visualization capabilities.
PyVista is often more reliable and easier to use than Mayavi.

Run this script to see various 3D visualization examples.
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

def demo_basic_surface():
    """Create a basic 3D surface plot"""
    print("Creating Basic 3D Surface...")
    
    # Generate surface data
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    z = np.sin(np.sqrt(x**2 + y**2))
    
    # Create a structured grid
    grid = pv.StructuredGrid(x, y, z)
    
    # Plot
    plotter = pv.Plotter()
    plotter.add_mesh(grid, colormap='viridis', show_edges=True)
    plotter.add_title('Basic 3D Surface')
    plotter.show()

def demo_scatter_3d():
    """Create a 3D scatter plot"""
    print("Creating 3D Scatter Plot...")
    
    # Generate random data
    n = 1000
    points = np.random.randn(n, 3)
    colors = np.linalg.norm(points, axis=1)
    
    # Create point cloud
    cloud = pv.PolyData(points)
    cloud['colors'] = colors
    
    # Plot
    plotter = pv.Plotter()
    plotter.add_mesh(cloud, scalars='colors', colormap='plasma', 
                     point_size=8, render_points_as_spheres=True)
    plotter.add_title('3D Scatter Plot')
    plotter.show()

def demo_parametric_surface():
    """Create parametric surfaces"""
    print("Creating Parametric Surfaces...")
    
    # Create torus
    torus = pv.ParametricTorus(ringradius=3, crosssectionalradius=1)
    
    # Create sphere
    sphere = pv.Sphere(radius=1.5, center=(6, 0, 0))
    
    # Plot both
    plotter = pv.Plotter()
    plotter.add_mesh(torus, color='red', opacity=0.8)
    plotter.add_mesh(sphere, color='blue', opacity=0.8)
    plotter.add_title('Parametric Surfaces: Torus and Sphere')
    plotter.show()

def demo_volume_rendering():
    """Create volume rendering"""
    print("Creating Volume Rendering...")
    
    # Create 3D data
    dims = (64, 64, 64)
    origin = (-2, -2, -2)
    spacing = (4/(dims[0]-1), 4/(dims[1]-1), 4/(dims[2]-1))
    
    # Create structured grid
    grid = pv.ImageData(dimensions=dims, origin=origin, spacing=spacing)
    
    # Generate scalar field
    x, y, z = grid.points.T
    scalars = np.sin(x*y*z) * np.exp(-(x**2 + y**2 + z**2)/4)
    grid['scalars'] = scalars
    
    # Volume rendering
    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars='scalars', cmap='viridis', opacity='sigmoid')
    plotter.add_title('Volume Rendering')
    plotter.show()

def demo_vector_field():
    """Create vector field visualization"""
    print("Creating Vector Field...")
    
    # Create grid
    x = np.arange(-5, 5, 1)
    y = np.arange(-5, 5, 1)
    z = np.arange(-5, 5, 1)
    x, y, z = np.meshgrid(x, y, z)
    
    # Create vectors
    u = -y
    v = x
    w = z * 0.1
    
    # Create vector field
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    vectors = np.column_stack((u.ravel(), v.ravel(), w.ravel()))
    
    grid = pv.PolyData(points)
    grid['vectors'] = vectors
    
    # Create arrows
    arrows = grid.glyph(orient='vectors', scale='vectors', factor=0.3)
    
    # Plot
    plotter = pv.Plotter()
    plotter.add_mesh(arrows, colormap='jet')
    plotter.add_title('Vector Field Visualization')
    plotter.show()

def demo_multiple_objects():
    """Create scene with multiple objects"""
    print("Creating Multi-Object Scene...")
    
    # Create various objects
    sphere = pv.Sphere(radius=1, center=(0, 0, 0))
    cube = pv.Cube(center=(3, 0, 0))
    cone = pv.Cone(center=(-3, 0, 0))
    cylinder = pv.Cylinder(center=(0, 3, 0))
    
    # Create surface
    x = np.arange(-2, 2, 0.1)
    y = np.arange(-2, 2, 0.1)
    x, y = np.meshgrid(x, y)
    z = np.sin(x) * np.cos(y) - 2
    surface = pv.StructuredGrid(x, y, z)
    
    # Plot all
    plotter = pv.Plotter()
    plotter.add_mesh(sphere, color='red', opacity=0.8)
    plotter.add_mesh(cube, color='blue', opacity=0.8)
    plotter.add_mesh(cone, color='green', opacity=0.8)
    plotter.add_mesh(cylinder, color='yellow', opacity=0.8)
    plotter.add_mesh(surface, colormap='viridis', opacity=0.6)
    plotter.add_title('Multi-Object 3D Scene')
    plotter.show()

def demo_contours():
    """Create contour plots"""
    print("Creating 3D Contours...")
    
    # Generate data
    x = np.arange(-5, 5, 0.2)
    y = np.arange(-5, 5, 0.2)
    z = np.arange(-5, 5, 0.2)
    x, y, z = np.meshgrid(x, y, z)
    
    # Create scalar field
    scalars = x**2 + y**2 + z**2
    
    # Create grid
    grid = pv.StructuredGrid(x, y, z)
    grid['scalars'] = scalars.ravel()
    
    # Create contours
    contours = grid.contour(isosurfaces=6)
    
    # Plot
    plotter = pv.Plotter()
    plotter.add_mesh(contours, colormap='coolwarm', opacity=0.7)
    plotter.add_title('3D Contour Surfaces')
    plotter.show()

def demo_molecular_structure():
    """Create molecular-like structure"""
    print("Creating Molecular Structure...")
    
    # Generate atomic positions
    n_atoms = 50
    positions = np.random.randn(n_atoms, 3) * 3
    
    # Create atoms as spheres
    atoms = pv.PolyData(positions)
    atom_spheres = atoms.glyph(geom=pv.Sphere(radius=0.2))
    
    # Create bonds (connect nearby atoms)
    bonds = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 2.0:  # bond threshold
                bond = pv.Line(positions[i], positions[j])
                bonds.append(bond)
    
    # Combine bonds
    if bonds:
        all_bonds = bonds[0]
        for bond in bonds[1:]:
            all_bonds = all_bonds + bond
    
    # Plot
    plotter = pv.Plotter()
    plotter.add_mesh(atom_spheres, color='red')
    if bonds:
        plotter.add_mesh(all_bonds, color='gray', line_width=3)
    plotter.add_title('Molecular Structure Visualization')
    plotter.show()

def demo_animation():
    """Create simple animation"""
    print("Creating Animation...")
    
    # Create initial sphere
    sphere = pv.Sphere(radius=1)
    
    # Create plotter
    plotter = pv.Plotter()
    actor = plotter.add_mesh(sphere, color='red')
    plotter.add_title('Animated Sphere')
    
    # Animation function
    def animate_sphere():
        for i in range(100):
            # Scale the sphere
            scale = 1 + 0.5 * np.sin(i * 0.1)
            sphere_scaled = sphere.scale([scale, scale, scale])
            
            # Update the mesh
            actor.mapper.SetInputData(sphere_scaled)
            plotter.render()
            
        plotter.close()
    
    # Start animation in a separate thread
    plotter.add_timer_event(animate_sphere, 1000)  # Start after 1 second
    plotter.show()

def demo_interactive_widgets():
    """Create interactive widgets demo"""
    print("Creating Interactive Widgets...")
    
    # Create initial mesh
    mesh = pv.Sphere(radius=2)
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add mesh
    actor = plotter.add_mesh(mesh, colormap='viridis')
    
    # Add interactive widgets
    def update_mesh(value):
        """Update mesh based on slider value"""
        new_mesh = pv.Sphere(radius=value)
        actor.mapper.SetInputData(new_mesh)
        plotter.render()
    
    # Add slider widget
    plotter.add_slider_widget(
        callback=update_mesh,
        rng=[0.5, 5.0],
        value=2.0,
        title="Sphere Radius",
        pointa=(0.1, 0.9),
        pointb=(0.4, 0.9)
    )
    
    plotter.add_title('Interactive Sphere with Slider')
    plotter.show()

def main():
    """Run all PyVista demos"""
    print("PyVista 3D Visualization Demo")
    print("=============================")
    print("Note: Each demo will open in a separate window.")
    print("Close the window to proceed to the next demo.\n")
    
    demos = [
        ("Basic 3D Surface", demo_basic_surface),
        ("3D Scatter Plot", demo_scatter_3d),
        ("Parametric Surfaces", demo_parametric_surface),
        ("Volume Rendering", demo_volume_rendering),
        ("Vector Field", demo_vector_field),
        ("Multi-Object Scene", demo_multiple_objects),
        ("3D Contours", demo_contours),
        ("Molecular Structure", demo_molecular_structure),
        ("Interactive Widgets", demo_interactive_widgets)
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n{i}. Running {name} Demo...")
        print("-" * (len(name) + 15))
        
        try:
            demo_func()
        except Exception as e:
            print(f"Error running {name}: {e}")
            print("Make sure PyVista is properly installed.")
        
        input(f"\nPress Enter to continue to next demo...")
    
    print("\nDemo complete! PyVista is working correctly.")

if __name__ == "__main__":
    main() 