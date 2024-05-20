import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

period = 1.0  # период решетки
total_strokes = 800  # общее число штрихов
d = 5  # ширина щели

theta = np.linspace(-np.pi / 2, np.pi / 2, 100)  # угол дифракции
wavelength = np.linspace(400, 750, 1000)  # длина волны в нм
Theta, Wavelength = np.meshgrid(theta, wavelength)

Intensity = (np.sin(np.pi * period * np.sin(Theta) / Wavelength) ** 2 / \
             ((np.pi * period * np.sin(Theta) / Wavelength) ** 2)) * \
            np.sin(np.pi * total_strokes * d * np.sin(Theta) / Wavelength) ** 2 / \
            np.sin(np.pi * d * np.sin(Theta) / Wavelength) ** 2

Intensity_normalized = Intensity / np.max(Intensity)

cmap = LinearSegmentedColormap.from_list(
    "wavelength_colormap",
    [(0.0, "violet"), (0.15, "indigo"), (0.3, "blue"), (0.45, "cyan"),
     (0.6, "green"), (0.75, "yellow"), (0.9, "orange"), (1.0, "red")]
)


def get_rgba_colors(wavelength, cmap, alpha):
    norm_wavelength = (wavelength - 400) / (700 - 400)
    colors = cmap(norm_wavelength)
    colors[..., -1] = np.where((wavelength < 400) | (wavelength > 700), 0, alpha)
    return colors


fig = plt.figure(figsize=(12, 12))

ax1 = fig.add_subplot(111, projection='3d')
colors1 = get_rgba_colors(Wavelength, cmap, Intensity_normalized)
ax1.plot_surface(Theta, Wavelength, Intensity, facecolors=colors1)
ax1.set_xlabel('Угол дифракции')
ax1.set_ylabel('Длина волны, нм')
ax1.set_zlabel('Интенсивность')

ax1.set_box_aspect([1, 1, 1])

plt.savefig("graph1.png", dpi=500)
plt.show()

fig2 = plt.figure(figsize=(12, 6))

ax1 = fig2.add_subplot(121, projection='3d')
ax1.plot_surface(Theta, Wavelength, Intensity, cmap='inferno')
ax1.set_xlabel('Угол дифракции')
ax1.set_ylabel('Длина волны, нм')
ax1.set_zlabel('Интенсивность')

ax2 = fig2.add_subplot(122, projection='3d')
ax2.plot_surface(Theta, Wavelength[::-1], Intensity, cmap='inferno')
ax2.set_xlabel('Угол дифракции')
ax2.set_ylabel('Длина волны, нм')
ax2.set_zlabel('Интенсивность')

yticks = ax2.get_yticks()
ax2.set_yticklabels(np.flip(yticks))

plt.tight_layout()
plt.savefig("graph2.png", dpi=500)
plt.show()
