import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from scipy.integrate import odeint


class ElectronMotionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.plot_canvas = None
        self.plot_layout = None
        self.plot_container = None
        self.n_edit = None
        self.Ic_edit = None
        self.U_edit = None
        self.Rk_edit = None
        self.Ra_edit = None
        self.D_edit = None
        self.charge_button = None
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Electron Motion Simulator')

        grid = QGridLayout()
        self.setLayout(grid)

        grid.addWidget(QLabel('Диаметр солиноида (D):'), 0, 0)
        self.D_edit = QLineEdit()
        grid.addWidget(self.D_edit, 0, 1)

        grid.addWidget(QLabel('Радиус анода (Ra):'), 1, 0)
        self.Ra_edit = QLineEdit()
        grid.addWidget(self.Ra_edit, 1, 1)

        grid.addWidget(QLabel('Радиус катода (Rk):'), 2, 0)
        self.Rk_edit = QLineEdit()
        grid.addWidget(self.Rk_edit, 2, 1)

        grid.addWidget(QLabel('Разность потенциалов (U):'), 3, 0)
        self.U_edit = QLineEdit()
        grid.addWidget(self.U_edit, 3, 1)

        grid.addWidget(QLabel('Ток в солиноиде (Ic):'), 4, 0)
        self.Ic_edit = QLineEdit()
        grid.addWidget(self.Ic_edit, 4, 1)

        grid.addWidget(QLabel('Количество витков на единицу длины (n):'), 5, 0)
        self.n_edit = QLineEdit()
        grid.addWidget(self.n_edit, 5, 1)

        simulate_button = QPushButton('График траектории электрона')
        simulate_button.clicked.connect(self.simulate)
        grid.addWidget(simulate_button, 6, 0)

        plot_button = QPushButton('График зависимости Ic от U')
        plot_button.clicked.connect(self.plot_Ic_U_diagram)
        grid.addWidget(plot_button, 6, 1)

        self.charge_button = QPushButton('Изменить знак заряда')
        self.charge_button.clicked.connect(self.change_charge_sign)
        grid.addWidget(self.charge_button, 7, 0, 1, 2)

        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_canvas = FigureCanvasQTAgg(Figure(figsize=(8, 6)))
        self.plot_layout.addWidget(self.plot_canvas)

        grid.addWidget(self.plot_container, 8, 0, 1, 2)

    def change_charge_sign(self):
        global e_charge
        e_charge = -e_charge
        self.charge_button.setText(
            'Изменение знака заряда: {} (для изменений снова нажмите на построение графиков)'.format(
                'Отрицательный' if e_charge > 0 else 'Положительный'
            )
        )

    def simulate(self):
        D = float(self.D_edit.text())
        Ra = float(self.Ra_edit.text())
        Rk = float(self.Rk_edit.text())
        U = float(self.U_edit.text())
        Ic = float(self.Ic_edit.text())
        n = int(self.n_edit.text())

        simulate_and_plot(self, n, U, Ic, Rk, stop_velocity=10)

    def plot_Ic_U_diagram(self):
        D = float(self.D_edit.text())
        Ra = float(self.Ra_edit.text())
        Rk = float(self.Rk_edit.text())
        n = int(self.n_edit.text())

        plot_Ic_U_diagram_with_circle(self, Ra, Rk, n)


def simulate_and_plot(self, n, U, Ic, Rk, steps=1000000, tmax=1e-6, stop_velocity=None):
    B = mu_0 * n * Ic  # Магнитное поле внутри соленоида

    # Начальные условия
    x0 = Rk
    y0 = 0
    v0 = np.sqrt(2 * np.abs(e_charge) * U / m_electron)  # Используем np.abs() для учета обоих знаков заряда
    r_orbit = orbit_radius(Ic, U, n)  # Вычисление радиуса орбиты

    conditions = [
        ((0.05, 0.2), 1e-5),
        ((0.005, 0.05), 1e-4),
        ((0.0005, 0.005), 1e-3),
        ((0.00005, 0.0005), 1e-2),
        ((0.000005, 0.00005), 1e-1),
        ((0.0000005, 0.000005), 1),
    ]

    for bounds, tmax_val in conditions:
        if any(low <= x < high for x in (U, Ic) for low, high in [bounds]):
            tmax = tmax_val
            break

    # Временной массив
    t = np.linspace(0, tmax, steps)

    # Решение дифференциальных уравнений
    sol = odeint(electron_motion, [x0, y0, v0, 0], t, args=(U, B, m_electron, e_charge))

    # Остановка моделирования по заданной скорости
    if stop_velocity is not None:
        idx = np.where(np.sqrt(sol[:, 2] ** 2 + sol[:, 3] ** 2) <= stop_velocity)[0]
        if len(idx) > 0:
            sol = sol[:idx[0] + 1]

    # Построение траектории
    self.plot_canvas.figure.clear()
    ax = self.plot_canvas.figure.add_subplot(111)
    ax.plot(sol[:, 0], sol[:, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Траектория электрона')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    self.plot_canvas.draw()

    return r_orbit


def find_required_current_positive(Ra, Rk, n, U):
    r_orbit = (Ra - Rk) / 2
    if (r_orbit * e_charge * n * mu_0) == 0:
        return

    Ic_required = (m_electron * np.sqrt(2 * (e_charge / m_electron) * U)) / (r_orbit * e_charge * n * mu_0)

    return Ic_required


def find_required_current_negative(Ra, Rk, n, U):
    r_orbit = (Ra - Rk) / 2
    if (r_orbit * -e_charge * n * mu_0) == 0:
        return

    Ic_required = (m_electron * np.sqrt(2 * (-e_charge / m_electron) * U)) / (r_orbit * -e_charge * n * mu_0)

    return Ic_required


def plot_Ic_U_diagram_with_circle(self, Ra, Rk, n):
    U_values = np.linspace(0, 1000, 100)  # Диапазон разности потенциалов

    # Создание списка для значений силы тока в соленоиде
    Ic_values = []

    for U in U_values:
        # Вычисление соответствующей силы тока Ic, чтобы электрон описывал окружность диаметром (Ra-Rk)
        if e_charge > 0:
            Ic_required = find_required_current_positive(Ra, Rk, n, U)
        else:
            Ic_required = find_required_current_negative(Ra, Rk, n, U)
        Ic_values.append(Ic_required)

    # Приведение значений к числовому типу
    Ic_values = np.array(Ic_values, dtype=np.float64)

    self.plot_canvas.figure.clear()

    ax = self.plot_canvas.figure.add_subplot(111)

    ax.plot(U_values, Ic_values, label='Ic', color='blue')

    ax.fill_between(U_values, Ic_values, color='red', alpha=0.3,
                    label='Электрон описывает окружность диаметром  (Ra-Rk)')

    ax.set_xlabel('Разность потенциалов (U)')
    ax.set_ylabel('Ток в солиноиде(Ic)')
    ax.set_title('Ic от U')
    ax.grid(True)
    ax.legend()

    self.plot_canvas.draw()


# Константы
e_charge = 1.60217662e-19  # элементарный заряд
m_electron = 9.10938356e-31  # масса электрона
mu_0 = 4 * np.pi * 1e-7  # магнитная проницаемость вакуума


# Функция для дифференциальных уравнений, описывающих движение электрона
def electron_motion(y, t, U, B, m, e):
    x, y, vx, vy = y
    dvxdt = (e / m) * vy * B
    dvydt = -(e / m) * vx * B - (e / m) * U * vy / np.sqrt(vx ** 2 + vy ** 2)
    return [vx, vy, dvxdt, dvydt]


# Функция для вычисления радиуса орбиты электрона
def orbit_radius(I, U, n):
    B = mu_0 * n * I
    if B == 0:
        return
    return (m_electron * 2 * U) / (e_charge * B ** 2)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ElectronMotionGUI()
    gui.show()
    sys.exit(app.exec_())