#!/usr/bin/env python3
"""
Plotly 3D Visualization Demo
============================

A comprehensive demonstration of Plotly's 3D visualization capabilities.
Plotly opens visualizations in your web browser, making it very reliable.

Run this script to see various 3D visualizations in your browser.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

def demo_3d_scatter():
    """Create a 3D scatter plot"""
    print("Creating 3D Scatter Plot...")
    
    # Generate random data
    n = 500
    x = np.random.randn(n)
    y = np.random.randn(n)
    z = np.random.randn(n)
    colors = x + y + z  # Color based on sum
    
    # Create scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            colorscale='Viridis',
            colorbar=dict(title="Values"),
            opacity=0.8
        )
    ))
    
    fig.update_layout(
        title='3D Scatter Plot with Plotly',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        )
    )
    
    fig.show()

def demo_3d_surface():
    """Create a 3D surface plot"""
    print("Creating 3D Surface Plot...")
    
    # Generate surface data
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    # Create surface plot
    fig = go.Figure(data=go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        colorbar=dict(title="Height")
    ))
    
    fig.update_layout(
        title='3D Surface Plot',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    fig.show()

def demo_3d_line():
    """Create 3D line plots"""
    print("Creating 3D Line Plot...")
    
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
    
    # Create figure
    fig = go.Figure()
    
    # Add helix
    fig.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode='lines',
        line=dict(color='blue', width=6),
        name='Helix'
    ))
    
    # Add decaying helix
    fig.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode='lines',
        line=dict(color='red', width=6),
        name='Decaying Helix'
    ))
    
    fig.update_layout(
        title='3D Line Plots: Helix and Decaying Helix',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    fig.show()

def demo_3d_mesh():
    """Create 3D mesh plot"""
    print("Creating 3D Mesh Plot...")
    
    # Generate sphere data
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2*np.pi, 20)
    phi, theta = np.meshgrid(phi, theta)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # Create mesh plot
    fig = go.Figure(data=go.Mesh3d(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        alphahull=5,
        opacity=0.8,
        color='lightblue'
    ))
    
    fig.update_layout(
        title='3D Mesh: Sphere',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    fig.show()

def demo_3d_volume():
    """Create volume plot"""
    print("Creating Volume Plot...")
    
    # Generate 3D data
    X, Y, Z = np.mgrid[-5:5:20j, -5:5:20j, -5:5:20j]
    values = np.sin(X*Y*Z) / (X*Y*Z + 0.1)
    
    # Create volume plot
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=0.1,
        isomax=0.8,
        opacity=0.1,
        surface_count=17,
        colorscale='RdYlBu'
    ))
    
    fig.update_layout(
        title='3D Volume Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    fig.show()

def demo_3d_cone():
    """Create 3D vector field with cones"""
    print("Creating 3D Vector Field...")
    
    # Generate vector field data
    x, y, z = np.mgrid[-2:2:8j, -2:2:8j, -2:2:8j]
    u = -y
    v = x
    w = z * 0.1
    
    # Create cone plot
    fig = go.Figure(data=go.Cone(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        u=u.flatten(),
        v=v.flatten(),
        w=w.flatten(),
        colorscale='Blues',
        sizemode="absolute",
        sizeref=0.5
    ))
    
    fig.update_layout(
        title='3D Vector Field (Cones)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))
        )
    )
    
    fig.show()

def demo_multiple_surfaces():
    """Create multiple surfaces in one plot"""
    print("Creating Multiple Surfaces...")
    
    # Generate data for multiple surfaces
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)
    
    Z1 = np.sin(X) * np.cos(Y) + 2
    Z2 = np.cos(X) * np.sin(Y)
    Z3 = -np.sin(X) * np.cos(Y) - 2
    
    fig = go.Figure()
    
    # Add first surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z1,
        colorscale='Reds',
        opacity=0.8,
        name='Surface 1'
    ))
    
    # Add second surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z2,
        colorscale='Blues',
        opacity=0.8,
        name='Surface 2'
    ))
    
    # Add third surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z3,
        colorscale='Greens',
        opacity=0.8,
        name='Surface 3'
    ))
    
    fig.update_layout(
        title='Multiple 3D Surfaces',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    fig.show()

def demo_parametric_surface():
    """Create parametric surface (torus)"""
    print("Creating Parametric Surface (Torus)...")
    
    # Generate torus data
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, 2*np.pi, 50)
    u, v = np.meshgrid(u, v)
    
    R = 3  # major radius
    r = 1  # minor radius
    
    x = (R + r*np.cos(v)) * np.cos(u)
    y = (R + r*np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    
    # Create surface
    fig = go.Figure(data=go.Surface(
        x=x, y=y, z=z,
        colorscale='Plasma',
        colorbar=dict(title="Height")
    ))
    
    fig.update_layout(
        title='Parametric Surface: Torus',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        )
    )
    
    fig.show()

def demo_subplots():
    """Create multiple 3D subplots"""
    print("Creating Multiple 3D Subplots...")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'scatter3d'}],
               [{'type': 'surface'}, {'type': 'scatter3d'}]],
        subplot_titles=('Surface 1', 'Scatter 1', 'Surface 2', 'Scatter 2')
    )
    
    # Data for surfaces
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    Z1 = X**2 + Y**2
    Z2 = np.sin(X) * np.cos(Y)
    
    # Data for scatter plots
    n = 100
    x_scatter = np.random.randn(n)
    y_scatter = np.random.randn(n)
    z_scatter = np.random.randn(n)
    
    # Add traces
    fig.add_trace(go.Surface(x=X, y=Y, z=Z1, colorscale='Viridis'), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=x_scatter, y=y_scatter, z=z_scatter, 
                              mode='markers', marker=dict(color='red')), row=1, col=2)
    fig.add_trace(go.Surface(x=X, y=Y, z=Z2, colorscale='Plasma'), row=2, col=1)
    fig.add_trace(go.Scatter3d(x=x_scatter*2, y=y_scatter*2, z=z_scatter*2, 
                              mode='markers', marker=dict(color='blue')), row=2, col=2)
    
    fig.update_layout(title='Multiple 3D Subplots')
    fig.show()

def demo_animated_surface():
    """Create animated 3D surface"""
    print("Creating Animated Surface...")
    
    # Generate data
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    
    # Create frames for animation
    frames = []
    for t in np.linspace(0, 2*np.pi, 30):
        Z = np.sin(np.sqrt(X**2 + Y**2) + t)
        frames.append(go.Frame(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')]))
    
    # Initial surface
    Z_init = np.sin(np.sqrt(X**2 + Y**2))
    
    fig = go.Figure(
        data=[go.Surface(x=X, y=Y, z=Z_init, colorscale='Viridis')],
        frames=frames
    )
    
    # Add play/pause buttons
    fig.update_layout(
        title='Animated 3D Surface',
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {'label': 'Play', 'method': 'animate', 'args': [None]},
                {'label': 'Pause', 'method': 'animate', 'args': [None, {'frame': {'duration': 0}}]}
            ]
        }],
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    fig.show()

def main():
    """Run all Plotly 3D demos"""
    print("Plotly 3D Visualization Demo")
    print("============================")
    print("Each demo will open in your web browser.")
    print("Close the browser tab to proceed to the next demo.\n")
    
    demos = [
        ("3D Scatter Plot", demo_3d_scatter),
        ("3D Surface Plot", demo_3d_surface),
        ("3D Line Plot", demo_3d_line),
        ("3D Mesh Plot", demo_3d_mesh),
        ("3D Volume Plot", demo_3d_volume),
        ("3D Vector Field", demo_3d_cone),
        ("Multiple Surfaces", demo_multiple_surfaces),
        ("Parametric Surface", demo_parametric_surface),
        ("Multiple 3D Subplots", demo_subplots),
        ("Animated Surface", demo_animated_surface)
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n{i}. Running {name} Demo...")
        print("-" * (len(name) + 15))
        
        try:
            demo_func()
        except Exception as e:
            print(f"Error running {name}: {e}")
        
        input(f"\nPress Enter to continue to next demo...")
    
    print("\nDemo complete! Plotly 3D visualizations work great in your browser.")

if __name__ == "__main__":
    main() 