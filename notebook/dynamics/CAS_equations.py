from sympy import symbols, cos, sin, simplify, solve, pprint, Eq

theta1, theta2, omega1, omega2, alpha1, alpha2, F21x, F21y, m1, m2, l1, l2, g, r1x, r1y, r2x, r2y = symbols("theta_1 theta_2 omega_1 omega_2 alpha_1 alpha_2 F_21x F_21y m_1 m_2 l_1 l_2 g r_1x r_1y r_2x r_2y")

eqs = [
    # Equations for r_1 and r_2 double derivatives,
    Eq(r1x, (alpha1 * l1 * cos(theta1) - omega1**2 * l1 * sin(theta1))),
    Eq(r1y, (alpha1 * l1 * sin(theta1) + omega1**2 * l1 * cos(theta1))),
    Eq(r2x, (alpha1 * l1 * sin(theta1) + omega1**2 * l1 * cos(theta1) + alpha2 * l2 * cos(theta2) - omega2**2 * l2 * sin(theta2))),
    Eq(r2y, (alpha1 * l1 * cos(theta1) - omega1**2 * l1 * sin(theta1) + alpha2 * l2 * sin(theta2) + omega2**2 * l2 * cos(theta2))),
    # Moment equations,
    Eq(0, cos(theta1) * F21x + sin(theta1) * F21y - m1 * cos(theta1) * r1x - m1 * cos(theta1) * r1x - m1 * sin(theta1) * r1y - sin(theta1) * m1 * g),
    Eq(0, -m2 * cos(theta2) * r2x - m2 * sin(theta2) * r2y - sin(theta2) * m2 * g),
    Eq(0, -cos(theta2) * (-F21x) - sin(theta2) * (-F21y)),
]

sol = solve(eqs, (alpha1, alpha2))

pprint(simplify(sol[alpha1]))
pprint(simplify(sol[alpha2]))

