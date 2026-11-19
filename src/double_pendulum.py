import numpy as np
import sympy as sp
from sympy import symbols, Matrix, sin, cos, pprint
from sympy.physics.mechanics import dynamicsymbols
from sympy.printing.numpy import NumPyPrinter

import params as Params
from utils import save_functions

z = dynamicsymbols("z")
theta1 = dynamicsymbols("theta1")
theta2 = dynamicsymbols("theta2")
l1, l2, m_cart, m1, m2, g = symbols("l1 l2 m_cart m1 m2 g", positive=True)
F, t = symbols("F t")


zd = z.diff(t)  # pyright: ignore[reportAttributeAccessIssue]
theta1d = theta1.diff(t)  # pyright: ignore[reportAttributeAccessIssue]
theta2d = theta2.diff(t)  # pyright: ignore[reportAttributeAccessIssue]

q = Matrix([z, theta1, theta2])
qd = q.diff(t)
state = Matrix([q, q.diff(t)])
input = Matrix([F])

p_cart = Matrix([z, 0, 0])
p1 = p_cart + Matrix([l1 * sin(theta1), -l1 * cos(theta1), 0])
p2 = p1 + Matrix([l2 * sin(theta2), -l2 * cos(theta2), 0])

v_cart = p_cart.diff(t)
v1 = p1.diff(t)
v2 = p2.diff(t)

K = (
    sp.Rational(1,2) * m_cart * v_cart.T * v_cart
    + sp.Rational(1,2) * m1 * v1.T * v1
    + sp.Rational(1,2) * m2 * v2.T * v2
)
P = m1 * g * p1[1] + m2 * g * p2[1]

# L = K - P

tau = Matrix([F, 0, 0])
B = Matrix(np.zeros((3, 3)))

M = sp.hessian(K, qd)
M.simplify()
# pprint(M)

G = P.diff(q)
G.simplify()
# pprint(G)

C = sp.Matrix(
    3,
    3,
    lambda i, j: sp.simplify(
        sum([
            sp.Rational(1, 2)
            * (
                M[i, j].diff(q[k])
                + M[i, k].diff(q[j])
                - M[j, k].diff(q[i])
            )
            * qd[k]
            for k in range(3)
        ])
    ),
)
C.simplify()
# pprint(C)

save_functions(
    "src/eom_generated.py",
    {"M": M, "G": G, "C": C},
    q,
    qd,
)

param_subs = {
    l1: Params.l1,
    l2: Params.l2,
    m_cart: Params.m_cart,
    m1: Params.m1,
    m2: Params.m2,
    g: Params.g
}
# M = M.subs(param_subs)
# G = G.subs(param_subs)
# C = C.subs(param_subs)

# qdd_eom = M.LUsolve(tau - G - C * qd)
# pprint(qdd_eom[0])
# print(qdd_eom[0].count_ops())
