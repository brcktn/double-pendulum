import numpy as np
import matplotlib.pyplot as plt

from dynamics import DoublePendulum
from vizualize import animate_double_pendulum

pendulum = DoublePendulum(state0=np.array([0.0, np.pi, np.pi, 0.1, 0.1, 0.0]), dt=0.01)

t_plot = np.arange(0, 10, pendulum.dt)
x_plot = pendulum.state.copy()

for t in t_plot:
    u = np.array([0.0])  # No control input
    pendulum.update(u)
    x_plot = np.vstack((x_plot, pendulum.state))

animate_double_pendulum(
    t_plot,
    x_plot,
)


# plt.figure(figsize=(10, 6))
# plt.subplot(3, 1, 1)
# plt.plot(t_plot, x_plot[:-1, 0], label='Cart Position (z)')
# plt.ylabel('Position (m)')
# plt.legend()
# plt.subplot(3, 1, 2)
# plt.plot(t_plot, x_plot[:-1, 1], label='Pendulum 1 Angle (theta1)')
# plt.ylabel('Angle (rad)')
# plt.legend()
# plt.subplot(3, 1, 3)
# plt.plot(t_plot, x_plot[:-1, 2], label='Pendulum 2 Angle (theta2)')
# plt.xlabel('Time (s)')
# plt.ylabel('Angle (rad)')
# plt.legend()
# plt.tight_layout()
# plt.show()