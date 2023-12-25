import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp
import numpy as np
from math import pi

# Общие константы
R_Earth = 6371000  # Радиус Земли
G_M_Earth = 6.674 * 5.9722 * 10 ** 13  # Гравитационная постоянная * масса Земли
geostationary_orbit_h = 35786000  # Высота геостационарной орбиты над экватором
M = 0.029  # Молярная масса воздуха,
R = 8.314  # Универсальная газовая постоянная

# Погодные условия на уровне моря
T0 = 19  # Температура , град Цельсия
P0 = 760  # Давление, мм ртутного столба

# Общие параметры ракеты
# Зависимость коэффициента лобового аэродинамического сопротивления от числа Маха
Cx = [[0, 0.165], [0.5, 0.149], [0.7, 0.175], [0.9, 0.255], [1, 0.304], [1.1, 0.36], [1.3, 0.484], [1.5, 0.5],
      [2, 0.51], [2.5, 0.502], [3, 0.5], [3.5, 0.485], [4, 0.463], [4.5, 0.458], [5, 0.447]]
S = 172  # Площадь наибольшего поперечного сечения (миделево сечения), м^2

# Время работы каждого этапа полета, c
T1 = 123  # I ступень
T2 = 218  # II ступень
T3 = 242  # III ступень
T4 = 500  # Автономный полет
T5 = 270  # 1-ый запуск РБ "Бриз-М"
T6 = 3000  # Автономный полет
T7 = 1080  # 2-ый запуск РБ "Бриз-М"
T8 = 20000  # Автономный полет
T9 = 1200  # 3-ый запуск РБ "Бриз-М"
T10 = 84000  # Автономный полет

# Массы ступеней вместе с топливом, кг
M1 = 458.9 * 1000  # I ступень
M2 = 168.3 * 1000  # II ступень
M3 = 46.562 * 1000  # III ступень
M4_1 = 6.565 * 1000  # РБ "Бриз-М" (1-й этап)
M4_2 = 5.871 * 1000  # РБ "Бриз-М" (2-ой этап)
M4_3 = 3.095 * 1000  # РБ "Бриз-М" (3-ий этап)
M5 = 2210  # Спутник "Экспресс-80"


# Метод для получения значения из таблицы коэффициентов лобового аэродинамического сопротивления
def get_cx(m):
    for i in range(len(Cx) - 1):
        if m == Cx[i][0]:
            return Cx[i][1]
        elif Cx[i][0] < m < Cx[i + 1][0]:
            return (Cx[i][1] + Cx[i + 1][1]) / 2
    return Cx[len(Cx) - 1][1]


# Далее идут методы, нацеленные на получение физических параметров
# Зависимость температуры от высоты (T в град Цельсия)
def get_temperature(h, T0):
    return max(h * (-0.0065) + T0, 4 - 273.15)


# Зависимость давления от высоты
def get_pressure(h, p0):
    return (p0 * 133.32) * np.exp(-(M * 9.81 * h) / (R * (get_temperature(h, T0) + 273.15)))


# Зависимость плотности воздуха от высоты
def get_density(h):
    T = get_temperature(h, T0) + 273.15
    P = get_pressure(h, P0)
    return 0 if h >= 50000 else (P * M) / (R * T)


# Зависимость скорости звука от высоты (T в Кельвинах)
def get_speed_of_sound(t):
    return 250 if t < 150 else np.sqrt(1.4 * R * t / M)


# Сила сопротивления воздуха
def get_resistance(r, phi, r_dot, phi_dot):
    return get_cx(((r_dot ** 2 + (r * phi_dot) ** 2) ** 0.5) / (
        get_speed_of_sound(get_temperature(r - R_Earth, T0) + 273.15))) * get_density(r - R_Earth) * (
            r_dot ** 2 + (r_dot * phi_dot) ** 2) * S / 2


# перевод градусов в радианы
def convert_to_rad(angle):
    return angle * pi / 180


# Следующие 3 метода описывают полёт ракеты - разгон с учётом атмосферы, разгон без учёта атмосферы, и автономный полёт
def acceleration_stage_atm(initial_conditions, T, F, sigma, M, k, beta_start, beta_end):
    beta_incr = convert_to_rad((beta_end - beta_start) / T)
    beta_start = convert_to_rad(beta_start)
    def right_part(t, y):
        y1, y2, y3, y4 = y
        return [
            y2,
            y1 * (y4 ** 2) - G_M_Earth / (y1 ** 2) + (np.cos(beta_start + beta_incr * t) / (M - k * t)) * (
                    (F + sigma * t) - get_resistance(y1, y3, y2, y4)),
            y4,
            (4000000 * np.sin(beta_start + beta_incr * t) * ((F + sigma * t) - get_resistance(y1, y3, y2, y4)) / (
                    M - k * t) - 2 * y2 * y1 * y4) / (y1 ** 2)
        ]

    t = np.array([i for i in range(0, T, 1)])
    solver = solve_ivp(right_part, [0, T], initial_conditions, method='RK45', dense_output=True)
    num_solution = solver.sol(t)
    return num_solution


def acceleration_stage(initial_conditions, T, F, M, k, beta_start, beta_end):
    beta_incr = convert_to_rad((beta_end - beta_start) / T)
    beta_start = convert_to_rad(beta_start)

    def right_part(t, y):
        y1, y2, y3, y4 = y
        return [
            y2,
            y1 * (y4 ** 2) - G_M_Earth / (y1 ** 2) + (np.cos(beta_start + beta_incr * t) / (M - k * t)) * F,
            y4,
            (4000000 * np.sin(beta_start + beta_incr * t) * F / (M - k * t) - 2 * y2 * y1 * y4) / (y1 ** 2)
        ]

    t = np.array([i for i in range(0, T, 1)])
    solver = solve_ivp(right_part, [0, T], initial_conditions, method='RK45', dense_output=True)
    num_solution = solver.sol(t)
    return num_solution


def autonomous_flight(initial_conditions, T):
    def right_part(t, y):
        y1, y2, y3, y4 = y
        return [
            y2,
            y1 * (y4 ** 2) - G_M_Earth / (y1 ** 2),
            y4,
            -2 * ((y4 * y2) / y1)
        ]

    t = np.array([i for i in range(0, T, 1)])
    solver = solve_ivp(right_part, [0, T], initial_conditions, method='RK45', dense_output=True)
    num_solution = solver.sol(t)
    return num_solution


# Подсчет траектории на всех этапах
def get_vessel_trajectory(start_pos):
    trajectory = []
    trajectory.append(
        acceleration_stage_atm(start_pos, T1, 10026 * 1000, 7983.5, M1 + M2 + M3 + M4_1 + M5, 3622, 0, 60))
    trajectory.append(
        acceleration_stage_atm(trajectory[-1][:, -1], T2, 2400 * 1000, 0, M2 + M3 + M4_1 + M5, 731.63, 60, 60))
    trajectory.append(acceleration_stage(trajectory[-1][:, -1], T3, 583 * 1000, M3 + M4_1 + M5, 180, 60, 60))
    trajectory.append(autonomous_flight(trajectory[-1][:, -1], T4))
    trajectory.append(acceleration_stage(trajectory[-1][:, -1], T5, 150 * 1000, M4_1 + M5, 2.57, 60, 80))
    trajectory.append(autonomous_flight(trajectory[-1][:, -1], T6))
    trajectory.append(acceleration_stage(trajectory[-1][:, -1], T7, 32.2 * 1000, M4_2 + M5, 2.57, 90, 90))
    trajectory.append(autonomous_flight(trajectory[-1][:, -1], T8))
    trajectory.append(acceleration_stage(trajectory[-1][:, -1], T9, 39.7 * 1000, M4_3 + M5, 2.57, 89.6, 89.6))
    trajectory.append(autonomous_flight(trajectory[-1][:, -1], T10))
    return trajectory


def join_flight_stages(trajectory):
    t = np.array([i for i in range(0, T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10, 1)])
    r = np.concatenate([stage[0, :] for stage in trajectory])
    r_dot = np.concatenate([stage[1, :] for stage in trajectory])
    phi = np.concatenate([stage[2, :] for stage in trajectory])
    phi_dot = np.concatenate([stage[3, :] for stage in trajectory])
    return t, np.array([r, r_dot, phi, phi_dot])


# Получение точек на границах этапов полёта
def get_stage_border_points(path, t):
    if t == 0:
        return np.array([])
    T = np.array([T1, T2, T3, T4, T5, T6, T7, T8, T9, T10])
    points = []
    for i in range(len(path) - 1):
        if np.sum(T[:i]) < t:
            points.append([path[i][0][-1], path[i][2][-1]])
    return np.array(points)


# График высоты
def draw_height(axis, trajectory):
    t, stages = join_flight_stages(trajectory)
    h = stages[0, :] - R_Earth
    axis.plot(t[:28000], h[:28000])
    axis.grid()


# График скорости
def draw_vessel_speed(axis, trajectory):
    t, stages = join_flight_stages(trajectory)
    v = np.sqrt(stages[1, :] ** 2 + (stages[0, :] * stages[3, :]) ** 2)
    axis.plot(t[:28000], v[:28000])
    axis.grid()


# График угловой скорости
def draw_angular_velocity(axis, trajectory):
    t, stages = join_flight_stages(trajectory)
    axis.plot(t[:28000], stages[3, :][:28000])
    axis.grid()


def draw_earth(axis):
    x = np.concatenate((np.arange(-R_Earth, R_Earth, 1000), np.array([R_Earth])))
    y1 = np.array([np.sqrt(R_Earth ** 2 - int(x_i) ** 2) for x_i in x])
    y2 = np.array([-np.sqrt(R_Earth ** 2 - int(x_i) ** 2) for x_i in x])

    axis.plot(x, y1, linewidth=2, color='green')
    axis.plot(x, y2, linewidth=2, color='green')


def draw_geostationary_orbit(axis):
    x = np.concatenate((np.arange(-(R_Earth + geostationary_orbit_h), R_Earth + geostationary_orbit_h, 1000),
                        np.array([R_Earth + geostationary_orbit_h])))
    y1 = np.array([np.sqrt((R_Earth + geostationary_orbit_h) ** 2 - int(x_i) ** 2) for x_i in x])
    y2 = np.array([-np.sqrt((R_Earth + geostationary_orbit_h) ** 2 - int(x_i) ** 2) for x_i in x])

    axis.plot(x, y1, linewidth=2, color='grey', linestyle='--')
    axis.plot(x, y2, linewidth=2, color='grey', linestyle='--')


# Траектрия пути
def draw_vessel_trajectory(axis, stages):
    line, = axis.plot(stages[0] * np.cos(stages[2]), stages[0] * np.sin(stages[2]), linewidth=1, color='blue')
    return line


def show_trajectory_plot(trajectory):
    fig, axis = plt.subplots(figsize=(8.2, 7))
    fig.subplots_adjust(left=0.25)
    time_axis = fig.add_axes([0.1, 0.25, 0.0225, 0.63])  # ось для размещения слайдера

    time_slider = Slider(
        ax=time_axis,
        label='время, с',
        valmin=0,
        valmax=T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10,
        valinit=T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10,
        valstep=1,
        orientation="vertical",
        color="blue",
    )

    # Эта функция будет вызываться при изменении значения времени через слайдер
    def update_time(val):
        line.set_xdata(stages[0][:time_slider.val] * np.cos(stages[2][:time_slider.val]))
        line.set_ydata(stages[0][:time_slider.val] * np.sin(stages[2][:time_slider.val]))
        points = get_stage_border_points(trajectory, time_slider.val)
        sc.set_array(points[:, 0] * np.cos(points[:, 1]))
        fig.canvas.draw_idle()

    time_slider.on_changed(update_time)

    draw_earth(axis)
    draw_geostationary_orbit(axis)
    _, stages = join_flight_stages(trajectory)
    line, = axis.plot(stages[0] * np.cos(stages[2]), stages[0] * np.sin(stages[2]), linewidth=1, color='blue')
    points = get_stage_border_points(trajectory, T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)
    sc = axis.scatter(points[:, 0] * np.cos(points[:, 1]), points[:, 0] * np.sin(points[:, 1]), color='blue', s=8,
                      alpha=0.7)

    plt.axis('off')
    plt.show()


def show_flight_parameter_plots(trajectory):
    fig, axs = plt.subplots(nrows=3, figsize=(8, 8))
    axs[0].set_xlabel('t, с')
    axs[0].set_ylabel('h, м')
    axs[1].set_xlabel('t, с')
    axs[1].set_ylabel('V, м/с')
    axs[2].set_xlabel('t, с')
    axs[2].set_ylabel('omega, рад/c')

    draw_height(axs[0], trajectory)
    draw_vessel_speed(axs[1], trajectory)
    draw_angular_velocity(axs[2], trajectory)
    plt.show()


trajectory = get_vessel_trajectory([R_Earth, 0, 0, 0])

show_trajectory_plot(trajectory)
show_flight_parameter_plots(trajectory)