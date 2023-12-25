import krpc
import matplotlib.pyplot as plt
import time


def get_magnitude(vector):
    return round((vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) ** 0.5, 4)


conn = krpc.connect(name="Hello, world")
# Получение объекта космического корабля
vessel = conn.space_center.active_vessel

# Массивы, в которых мы будем хранить полученные во время полёта данные
time_values = []
speed_values = []
altitude_values = []
angular_speed_values = []

# # Получение скорости корабля на протяжении полета
while True:

    altitude = vessel.flight().surface_altitude
    time = conn.space_center.ut
    speed = vessel.flight(vessel.orbit.body.reference_frame).speed
    angular_speed = get_magnitude(vessel.angular_velocity(vessel.orbit.body.reference_frame))

    altitude_values.append(altitude)
    speed_values.append(speed)
    time_values.append(time)
    angular_speed_values.append(angular_speed)

    print(angular_speed)

    # Проверка условия завершения сбора данных
    if altitude > 100000:
        break
    time.sleep(1)

fig, axs = plt.subplots(nrows=3, figsize=(8, 8))

# график высоты
axs[0].set_xlabel('t, с')
axs[0].set_ylabel('h, м')

axs[0].plot(time_values, altitude_values)

axs[1].set_xlabel('t, с')
axs[1].set_ylabel('V, м/с')

axs[1].plot(time_values, speed_values)

axs[2].set_xlabel('t, с')
axs[2].set_ylabel('omega, рад/с')

axs[2].plot(time_values, angular_speed_values)

plt.show()
