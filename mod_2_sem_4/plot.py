import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Задаем параметры дифракционной решетки
period = 1.0          # период решетки
total_strokes = 1000  # общее число штрихов

# Создаем координатную сетку
theta = np.linspace(-np.pi / 2, np.pi / 2, 100)     # угол дифракции
wavelength = np.linspace(400, 750, 2000)  # длина волны в нм

Theta, Wavelength = np.meshgrid(theta, wavelength)

# Расчет интенсивности с использованием формулы для дифракции
Intensity = np.sin(np.pi * total_strokes * np.sin(Theta) / Wavelength) ** 2 / \
            (np.sin(np.pi * np.sin(Theta) / Wavelength) ** 2)

# Нормализуем интенсивность для использования в качестве параметра alpha
Intensity_normalized = Intensity / np.max(Intensity)

# Создаем цветовую карту для видимого спектра с учетом прозрачности
cmap = LinearSegmentedColormap.from_list(
    "wavelength_colormap",
    [(0.0, "violet"), (0.15, "indigo"), (0.3, "blue"), (0.45, "cyan"),
     (0.6, "green"), (0.75, "yellow"), (0.9, "orange"), (1.0, "red")]
)


# Функция для получения RGBA цветов с учетом alpha и прозрачности для вне диапазона
def get_rgba_colors(wavelength, cmap, alpha):
    # Нормализуем длины волн в диапазоне 400-700 нм
    norm_wavelength = (wavelength - 400) / (700 - 400)
    # Применяем цветовую карту к нормализованным значениям длин волн
    colors = cmap(norm_wavelength)
    # Применяем прозрачность для значений вне диапазона
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
