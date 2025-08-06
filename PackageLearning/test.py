import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# sns.set_style("ticks")
# sns.set_palette("muted")

def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(8, 6), dpi=300)
ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='plasma')
ax.plot_surface(X, Y, Z, cmap='coolwarm')
# ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title(r'3D Surface Plot of $f(x, y)=\sin(\sqrt{x^2+y^2})$')
plt.show()
