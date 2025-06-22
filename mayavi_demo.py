import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt

def demo_3d_points():
    """Demo of 3D points with Mayavi"""
    print("3D Points Demo")
    
    # Generate random data
    n = 1000
    x = np.random.randn(n)
    y = np.random.randn(n)
    z = np.random.randn(n)
    s = np.random.randn(n)  # scalar data for coloring
    
    # Clear previous plots
    mlab.clf()
    
    # Create 3D points plot
    pts = mlab.points3d(x, y, z, s, scale_mode='none', scale_factor=0.05)
    
    # Customize the plot
    mlab.title('3D Points with Mayavi', size=0.2)
    mlab.colorbar(pts, title="Values", orientation="vertical")
    
    mlab.show()

def demo_3d_line():
    """Demo of 3D line plots"""
    print("3D Line Demo")
    
    mlab.clf()
    
    # Generate parametric curves
    t = np.linspace(0, 4*np.pi, 200)
    
    # Helix
    x1 = np.cos(t)
    y1 = np.sin(t)
    z1 = t
    
    # Decaying helix
    x2 = np.cos(t) * np.exp(-t/10)
    y2 = np.sin(t) * np.exp(-t/10)
    z2 = t
    
    # Plot lines
    mlab.plot3d(x1, y1, z1, color=(0, 0, 1), tube_radius=0.05)
    mlab.plot3d(x2, y2, z2, color=(1, 0, 0), tube_radius=0.05)
    
    mlab.title('3D Lines: Helix and Decaying Helix', size=0.2)
    mlab.show()

def demo_surface():
    """Demo of surface plots"""
    print("Surface Demo")
    
    mlab.clf()
    
    # Generate surface data
    x, y = np.mgrid[-3:3:100j, -3:3:100j]
    z = np.sin(np.sqrt(x**2 + y**2))
    
    # Create surface plot
    surf = mlab.surf(x, y, z, colormap='viridis')
    
    mlab.title('3D Surface Plot', size=0.2)
    mlab.colorbar(surf, title="Height", orientation="vertical")
    mlab.show()

def demo_mesh():
    """Demo of mesh plots"""
    print("Mesh Demo")
    
    mlab.clf()
    
    # Generate mesh data
    phi, theta = np.mgrid[0:np.pi:20j, 0:2*np.pi:20j]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # Create mesh plot (sphere)
    mesh = mlab.mesh(x, y, z, colormap='plasma')
    
    mlab.title('3D Mesh: Sphere', size=0.2)
    mlab.show()

def demo_contour3d():
    """Demo of 3D contour plots"""
    print("3D Contour Demo")
    
    mlab.clf()
    
    # Generate 3D scalar field
    x, y, z = np.mgrid[-5:5:64j, -5:5:64j, -5:5:64j]
    scalars = np.sin(x*y*z) / (x*y*z + 0.1)
    
    # Create 3D contour plot
    contour = mlab.contour3d(scalars, contours=8, transparent=True, colormap='coolwarm')
    
    mlab.title('3D Contour Plot', size=0.2)
    mlab.colorbar(contour, title="Values", orientation="vertical")
    mlab.show()

def demo_volume():
    """Demo of volume rendering"""
    print("Volume Rendering Demo")
    
    mlab.clf()
    
    # Generate 3D volume data
    x, y, z = np.mgrid[-10:10:64j, -10:10:64j, -10:10:64j]
    r = np.sqrt(x**2 + y**2 + z**2)
    volume_data = np.exp(-r**2/50) * np.sin(r)
    
    # Create volume plot
    vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(volume_data), 
                               vmin=0, vmax=0.8)
    
    mlab.title('Volume Rendering', size=0.2)
    mlab.show()

def demo_quiver3d():
    """Demo of 3D vector field"""
    print("3D Vector Field Demo")
    
    mlab.clf()
    
    # Generate vector field data
    x, y, z = np.mgrid[-2:3:10j, -2:3:10j, -2:3:10j]
    u = np.sin(np.pi*x) * np.cos(np.pi*z)
    v = -2*np.sin(np.pi*y) * np.cos(2*np.pi*z)
    w = np.cos(np.pi*x)*np.sin(np.pi*z) + np.cos(np.pi*y)*np.sin(2*np.pi*z)
    
    # Create vector field plot
    quiver = mlab.quiver3d(x, y, z, u, v, w, scale_factor=0.5, colormap='jet')
    
    mlab.title('3D Vector Field', size=0.2)
    mlab.colorbar(quiver, title="Magnitude", orientation="vertical")
    mlab.show()

def demo_pipeline():
    """Demo of Mayavi pipeline for advanced visualization"""
    print("Advanced Pipeline Demo")
    
    mlab.clf()
    
    # Generate data
    x, y, z = np.mgrid[-3:3:64j, -3:3:64j, -3:3:64j]
    scalars = x*x*0.5 + y*y + z*z*2.0
    
    # Create scalar field
    src = mlab.pipeline.scalar_field(scalars)
    
    # Add iso-surfaces at different levels
    iso1 = mlab.pipeline.iso_surface(src, contours=[scalars.min()+0.1*scalars.ptp()], 
                                     colormap='Reds', opacity=0.3)
    iso2 = mlab.pipeline.iso_surface(src, contours=[scalars.min()+0.5*scalars.ptp()], 
                                     colormap='Blues', opacity=0.3)
    iso3 = mlab.pipeline.iso_surface(src, contours=[scalars.min()+0.9*scalars.ptp()], 
                                     colormap='Greens', opacity=0.3)
    
    # Add a plane cut
    plane = mlab.pipeline.scalar_cut_plane(src, plane_orientation='z_axes')
    
    mlab.title('Advanced Pipeline: Multiple Iso-surfaces + Cut Plane', size=0.2)
    mlab.show()

def demo_parametric_surface():
    """Demo of parametric surfaces"""
    print("Parametric Surface Demo")
    
    mlab.clf()
    
    # Generate parametric surface (torus)
    phi, theta = np.mgrid[0:2*np.pi:40j, 0:2*np.pi:40j]
    R = 3  # major radius
    r = 1  # minor radius
    
    x = (R + r*np.cos(theta)) * np.cos(phi)
    y = (R + r*np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    
    # Color by height
    colors = z
    
    # Create parametric surface
    surf = mlab.mesh(x, y, z, scalars=colors, colormap='plasma')
    
    mlab.title('Parametric Surface: Torus', size=0.2)
    mlab.colorbar(surf, title="Height", orientation="vertical")
    mlab.show()

def demo_molecular_visualization():
    """Demo of molecular-like visualization"""
    print("Molecular Visualization Demo")
    
    mlab.clf()
    
    # Generate atomic positions (simple cubic lattice)
    n = 5
    x, y, z = np.mgrid[0:n:1, 0:n:1, 0:n:1]
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    
    # Add some random displacement
    x += np.random.randn(len(x)) * 0.1
    y += np.random.randn(len(y)) * 0.1
    z += np.random.randn(len(z)) * 0.1
    
    # Different atom types
    atom_types = np.random.randint(0, 3, len(x))
    
    # Plot atoms as spheres with different colors/sizes
    atoms = mlab.points3d(x, y, z, atom_types, scale_mode='none', 
                         scale_factor=0.3, colormap='Set1')
    
    # Add bonds (connect nearby atoms)
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2)
            if dist < 1.5:  # bond threshold
                mlab.plot3d([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 
                           color=(0.5, 0.5, 0.5), tube_radius=0.02)
    
    mlab.title('Molecular-like Visualization', size=0.2)
    mlab.show()

def demo_animations():
    """Demo of simple animation"""
    print("Animation Demo")
    
    mlab.clf()
    
    # Generate initial data
    t = np.linspace(0, 2*np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    z = np.zeros_like(t)
    
    # Create initial plot
    plot = mlab.plot3d(x, y, z, color=(1, 0, 0), tube_radius=0.05)
    
    mlab.title('Animated Helix', size=0.2)
    
    # Simple animation
    for phase in np.linspace(0, 4*np.pi, 50):
        z_new = 0.5 * np.sin(t + phase)
        plot.mlab_source.set(z=z_new)
        mlab.process_ui_events()  # Update the display
        
    mlab.show()

def main():
    """Run all Mayavi demos"""
    print("Mayavi 3D Visualization Demo")
    print("============================")
    print("Note: Each demo will open in a separate window.")
    print("Close the window to proceed to the next demo.\n")
    
    demos = [
        ("3D Points", demo_3d_points),
        ("3D Lines", demo_3d_line),
        ("Surface Plot", demo_surface),
        ("Mesh Plot", demo_mesh),
        ("3D Contours", demo_contour3d),
        ("Volume Rendering", demo_volume),
        ("Vector Field", demo_quiver3d),
        ("Advanced Pipeline", demo_pipeline),
        ("Parametric Surface", demo_parametric_surface),
        ("Molecular Visualization", demo_molecular_visualization),
        ("Simple Animation", demo_animations)
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n{i}. Running {name} Demo...")
        print("-" * (len(name) + 15))
        
        try:
            demo_func()
        except Exception as e:
            print(f"Error running {name}: {e}")
            print("Make sure Mayavi is properly installed and you have a display.")
        
        input(f"\nPress Enter to continue to next demo...")

if __name__ == "__main__":
    main() 