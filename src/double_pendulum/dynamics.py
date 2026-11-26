import numpy as np
import eom_generated as EOM
import params as Params
from utils import rk4_step


class DoublePendulum:
    def __init__(self, state0=np.zeros(6), u_min=-10.0, u_max=10.0, dt=Params.dt):
        self.state = state0.copy()
        self.u_min = u_min
        self.u_max = u_max
        self.dt = dt

        self.g = Params.g
        self.m_cart = Params.m_cart
        self.m1 = Params.m1
        self.m2 = Params.m2
        self.l1 = Params.l1
        self.l2 = Params.l2

    def update(self, u):
        """
        Update the state of the double pendulum using the current control input u.

        Parameters
        ----------
        u : array_like
            The control input vector [F].
        """
        u = np.clip(u, self.u_min, self.u_max)
        self.state = rk4_step(self.f, self.state, u, self.dt)

    def f(self, x, u):
        """
        Compute the time derivative of the state vector x given the current
        state and control input u.

        Parameters
        ----------
        x : array_like
            The current state vector [z, theta1, theta2, zd, theta1d, theta2d].
        u : array_like
            The control input vector [F].

        Returns
        -------
        xdot : ndarray
            The time derivative of the state vector.
        """
        x = np.asarray(x).flatten()
        u = np.asarray(u).flatten()

        M = self.M(x)
        G = self.G(x)
        C = self.C(x)
        qd = x[3:]
        Dqd = Params.B_damping @ qd

        qdd = np.linalg.solve(M, u - C @ qd - G - Dqd)
        return np.concatenate((x[3:], qdd))

    def h(self, x):
        """
        Compute the output vector h(x) given the current state x.

        Parameters
        ----------
        x : array_like
            The current state vector [z, theta1, theta2, zd, theta1d, theta2d].

        Returns
        -------
        y : ndarray
            The output vector [z, theta1, theta2].
        """
        x = np.asarray(x).flatten()
        return x[:3]

    def M(self, x):
        return EOM.calculate_M(
            x,
            self.g,
            self.l1,
            self.l2,
            self.m1,
            self.m2,
            self.m_cart,
        )

    def G(self, x):
        return EOM.calculate_G(
            x,
            self.g,
            self.l1,
            self.l2,
            self.m1,
            self.m2,
            self.m_cart,
        )

    def C(self, x):
        return EOM.calculate_C(
            x,
            self.g,
            self.l1,
            self.l2,
            self.m1,
            self.m2,
            self.m_cart,
        )
