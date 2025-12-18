import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

g = 9.81
l = 1

def pendulum(t, y):
    theta, omega = y
    return [omega, -g/l*np.sin(theta)]

theta0 = 0.5
omega0 = 10
y0 = [theta0, omega0]

sol = solve_ivp(
    pendulum,
    (0,10),
    y0,
    method="RK45",
    t_eval=np.linspace(0,10,10*120+1),
    rtol=1e-6,
    atol=1e-9,
)

x = l * np.sin(sol.y[0])
y = -l * np.cos(sol.y[0])

fig, ax = plt.subplots()
ax.set_xlim(-l, l)
ax.set_ylim(-l, l)
ax.set_aspect("equal")
line, = ax.plot([], [], "o-", lw=2)

def update(i):
    line.set_data([0,x[i]], [0, y[i]])
    return line

ani = FuncAnimation(fig, update, frames= len(x), interval = 100/120)
plt.show()

plt.plot(sol.y[0], sol.y[1])
plt.xlabel("theta")
plt.ylabel("omega")

plt.show()