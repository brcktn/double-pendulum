from sympy import symbols, cos, sin, simplify, solve, Eq, trigsimp, factor, fraction, together, pprint

# 1. Define Symbols
theta1, theta2, omega1, omega2, alpha1, alpha2 = symbols("theta_1 theta_2 omega_1 omega_2 alpha_1 alpha_2")
m1, m2, l1, l2, g = symbols("m1 m2 l1 l2 g")

# 2. Define Kinematics as EXPRESSIONS (not Equations)
# We store the math logic in the variable directly
r1x = alpha1 * l1 * cos(theta1) - omega1**2 * l1 * sin(theta1)
r1y = alpha1 * l1 * sin(theta1) + omega1**2 * l1 * cos(theta1)

# Note: Using (theta1 + theta2) for absolute angle of rod 2
r2x = l1 * alpha1 * cos(theta1) - l1 * omega1**2 * sin(theta1) + l2 * (alpha1 + alpha2) * cos(theta1 + theta2) - l2 * (omega1 + omega2)**2 * sin(theta1 + theta2)
r2y = l1 * alpha1 * sin(theta1) + l1 * omega1**2 * cos(theta1) + l2 * (alpha1 + alpha2) * sin(theta1 + theta2) + l2 * (omega1 + omega2)**2 * cos(theta1 + theta2)

# 3. Define Forces (Substitute Newton's 2nd Law directly)
# F21 = -F12 = -m2 * a2
F21x = -m2 * r2x
F21y = -m2 * r2y - m2 * g

# 4. Define Moment Equations
# Now we just plug the expressions above into the moment sum
# Moment about O1
eqn1 = Eq(0, cos(theta1) * F21x + sin(theta1) * F21y 
             - m1 * cos(theta1) * r1x - m1 * sin(theta1) * r1y 
             - sin(theta1) * m1 * g)

# Moment about O2 (Pivot of Mass 2)
# Note: Using (theta1 + theta2) for the projection directions
eqn2 = Eq(0, -m2 * cos(theta1 + theta2) * r2x 
             - m2 * sin(theta1 + theta2) * r2y 
             - sin(theta1 + theta2) * m2 * g)

# 5. Solve
# Now we only have 2 equations and 2 unknowns!
print("Solving...")
sol = solve([eqn1, eqn2], (alpha1, alpha2))

def clean_result(expr):
    # 1. Combine all partial fractions into one single fraction
    combined = together(expr)
    
    # 2. Separate numerator and denominator
    num, den = fraction(combined)
    
    # 3. Simplify the numerator aggressively
    # expand() helps breaking down compound angles like sin(A+B)
    # trigsimp() recombines them into simpler forms
    num = trigsimp(num.expand())
    
    # 4. Factor the denominator to make it readable
    den = factor(trigsimp(den))
    
    return num / den

print("--- Cleaned Alpha 1 ---")
# We simplify the result from the solver
a1_clean = clean_result(sol[alpha1])
print(a1_clean)

print("\n--- Cleaned Alpha 2 ---")
a2_clean = clean_result(sol[alpha2])
print(a2_clean)