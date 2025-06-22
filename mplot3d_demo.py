import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.patches as patches

def demo_3d_scatter():
    """Demo of 3D scatter plot"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate random data
    n = 100
    x = np.random.randn(n)
    y = np.random.randn(n)
    z = np.random.randn(n)
    colors = np.random.randn(n)
    
    # Create scatter plot
    scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=60, alpha=0.7)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Scatter Plot Demo')
    
    # Add colorbar
    plt.colorbar(scatter)
    plt.show()

def demo_3d_line():
    """Demo of 3D line plot"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate parametric curve data
    t = np.linspace(0, 4*np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    z = t
    
    # Create line plot
    ax.plot(x, y, z, 'b-', linewidth=2, label='Helix')
    
    # Add another curve
    x2 = np.cos(t) * np.exp(-t/10)
    y2 = np.sin(t) * np.exp(-t/10)
    z2 = t
    ax.plot(x2, y2, z2, 'r-', linewidth=2, label='Decaying Helix')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Line Plot Demo')
    ax.legend()
    plt.show()

def demo_3d_surface():
    """Demo of 3D surface plot"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate surface data
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Surface Plot Demo')
    
    # Add colorbar
    fig.colorbar(surf)
    plt.show()

def demo_3d_wireframe():
    """Demo of 3D wireframe plot"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate wireframe data
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)
    Z = X * np.exp(-X**2 - Y**2)
    
    # Create wireframe plot
    ax.plot_wireframe(X, Y, Z, color='blue', alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Wireframe Plot Demo')
    plt.show()

def demo_3d_contour():
    """Demo of 3D contour plot"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate contour data
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
    
    # Create 3D contour plot
    contours = ax.contour(X, Y, Z, levels=15, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Contour Plot Demo')
    plt.show()

def demo_3d_bar():
    """Demo of 3D bar plot"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate bar data
    xpos = np.arange(5)
    ypos = np.arange(4)
    xposM, yposM = np.meshgrid(xpos, ypos)
    
    xpos = xposM.ravel()
    ypos = yposM.ravel()
    zpos = np.zeros(20)
    
    dx = np.ones(20) * 0.5
    dy = np.ones(20) * 0.5
    dz = np.random.randint(1, 10, 20)
    
    colors = cm.rainbow(dz/float(max(dz)))
    
    # Create 3D bar plot
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Bar Plot Demo')
    plt.show()

def demo_parametric_surface():
    """Demo of parametric surface"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate parametric surface (torus)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 50)
    U, V = np.meshgrid(u, v)
    
    R = 3  # Major radius
    r = 1  # Minor radius
    
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    
    # Create parametric surface plot
    ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Parametric Surface Demo (Torus)')
    plt.show()

def demo_multiple_subplots():
    """Demo of multiple 3D subplots"""
    fig = plt.figure(figsize=(15, 10))
    
    # Subplot 1: Scatter
    ax1 = fig.add_subplot(221, projection='3d')
    n = 50
    x = np.random.randn(n)
    y = np.random.randn(n)
    z = np.random.randn(n)
    ax1.scatter(x, y, z, c='red', s=50)
    ax1.set_title('3D Scatter')
    
    # Subplot 2: Line
    ax2 = fig.add_subplot(222, projection='3d')
    t = np.linspace(0, 2*np.pi, 50)
    x = np.cos(t)
    y = np.sin(t)
    z = t
    ax2.plot(x, y, z, 'b-', linewidth=2)
    ax2.set_title('3D Line')
    
    # Subplot 3: Surface
    ax3 = fig.add_subplot(223, projection='3d')
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax3.set_title('3D Surface')
    
    # Subplot 4: Wireframe
    ax4 = fig.add_subplot(224, projection='3d')
    x = np.linspace(-2, 2, 15)
    y = np.linspace(-2, 2, 15)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    ax4.plot_wireframe(X, Y, Z, color='green')
    ax4.set_title('3D Wireframe')
    
    plt.tight_layout()
    plt.show()

def main():
    """Run all demos"""
    print("3D Plotting Demo with mpl_toolkits.mplot3d")
    print("==========================================")
    
    demos = [
        ("3D Scatter Plot", demo_3d_scatter),
        ("3D Line Plot", demo_3d_line),
        ("3D Surface Plot", demo_3d_surface),
        ("3D Wireframe Plot", demo_3d_wireframe),
        ("3D Contour Plot", demo_3d_contour),
        ("3D Bar Plot", demo_3d_bar),
        ("Parametric Surface (Torus)", demo_parametric_surface),
        ("Multiple 3D Subplots", demo_multiple_subplots)
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n{i}. {name}")
        print("-" * (len(name) + 3))
        
        try:
            demo_func()
        except Exception as e:
            print(f"Error running {name}: {e}")
        
        input("Press Enter to continue to next demo...")

if __name__ == "__main__":
    main() 