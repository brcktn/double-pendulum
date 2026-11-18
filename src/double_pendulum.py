import numpy as np
import sympy as sp
from sympy import symbols, Matrix, sin, cos, pprint
from sympy.physics.mechanics import dynamicsymbols
from sympy.printing.numpy import NumPyPrinter

import params as Params

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
    (1 / 2) * m_cart * v_cart.T * v_cart
    + (1 / 2) * m1 * v1.T * v1
    + (1 / 2) * m2 * v2.T * v2
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
        sp.Rational(1, 2) *
        sum([
            M[i, j].diff(q[k])
            + M[i, k].diff(q[j])
            - M[j, k].diff(q[i])
            for k in range(3)
        ]) # pyright: ignore[reportOperatorIssue]
    ),
)
C.simplify()
# pprint(C)

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

qdd_eom = M.inv() * (tau - C * qd - G)
qdd_eom.simplify()
# pprint(qdd_eom[0])
# print(qdd_eom[0].count_ops())

args = [*q, *qd, m_cart, m1, m2, l1, l2, F, g]
pr = NumPyPrinter()

# Common subexpression elimination keeps the generated code compact and fast
repl, reduced = sp.cse(list(qdd_eom), symbols=sp.numbered_symbols('t'))

lines = [
    "import numpy",
    "",
    f"def qddot({', '.join(map(str, args))}):",
]
lines += [f"    {s} = {pr.doprint(e)}" for s, e in repl]
lines.append("    return numpy.array([" + ", ".join(pr.doprint(e) for e in reduced) + "])")

with open("qdd_eom.py", "w") as fh:
    fh.write("\n".join(lines) + "\n")