"""Generated numerical dynamics functions."""
import numpy

def calculate_M(x, g, l1, l2, m1, m2, m_cart):
    x = numpy.asarray(x).flatten()
    [z, theta1, theta2, zd, theta1d, theta2d] = x
    M = numpy.array([
        [m1 + m2 + m_cart,
        l1*(m1 + m2)*numpy.cos(theta1),
        l2*m2*numpy.cos(theta2)],
        [l1*(m1 + m2)*numpy.cos(theta1),
        l1**2*(m1 + m2),
        l1*l2*m2*numpy.cos(theta1 - theta2)],
        [l2*m2*numpy.cos(theta2),
        l1*l2*m2*numpy.cos(theta1 - theta2),
        l2**2*m2]
    ], dtype=float)
    return M.squeeze()

def calculate_G(x, g, l1, l2, m1, m2, m_cart):
    x = numpy.asarray(x).flatten()
    [z, theta1, theta2, zd, theta1d, theta2d] = x
    G = numpy.array([
        [0],
        [g*l1*m1*numpy.sin(theta1) + g*l1*m2*numpy.sin(theta1)],
        [g*l2*m2*numpy.sin(theta2)]
    ], dtype=float)
    return G.squeeze()

def calculate_C(x, g, l1, l2, m1, m2, m_cart):
    x = numpy.asarray(x).flatten()
    [z, theta1, theta2, zd, theta1d, theta2d] = x
    C = numpy.array([
        [0,
        -l1*theta1d*(m1 + m2)*numpy.sin(theta1),
        -l2*m2*theta2d*numpy.sin(theta2)],
        [0,
        0,
        l1*l2*m2*theta2d*numpy.sin(theta1 - theta2)],
        [0,
        -l1*l2*m2*theta1d*numpy.sin(theta1 - theta2),
        0]
    ], dtype=float)
    return C.squeeze()
