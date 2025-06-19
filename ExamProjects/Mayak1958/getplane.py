import numpy as np
from scipy.optimize import minimize_scalar

def case1V(r,p,S):
    """Function to evaluate the volume of a partially filled inclined cylinder
    if the bottom is fully covered and the top is dry.

    Parameters
    ----------
    r : float
        radius of cylinder
    p : float
        angle of tilt (radians)
    S : float
        liquid height on side
    
    """
    G=S+r/np.tan(p)
    return np.pi*r**2*G

def case2V(r, p, S):
    """Function to evaluate the volume of a partially filled inclined cylinder
    if the bottom is partially covered and the top is dry.

    
    """
    G=S+r/np.tan(p)
    sqrt_expr = np.sqrt(r**2 - (G * np.tan(p))**2)

    first_term = (2 / (3 * np.tan(p))) * (r**2 - (G * np.tan(p))**2)**(3/2)

    second_term = 2 * G * (
        (np.pi * r**2) / 4 +
        (G * np.tan(p) / 2) * sqrt_expr +
        (r**2 / 2) * np.arcsin((G * np.tan(p)) / r)
    )
    return first_term + second_term

def caseV(V,r,p):
    Vedge=case1V(r,p,0)
    if V>Vedge:
        return case1V
    else:
        return case2V

def getS(V_target, r, p,Hc, search_width=200.0, S_guess=0.0):
    """function to get the S distance.
    Note, the distance might be negative for partially covered bottom.
    """
    integral_func=caseV(V_target,r,p)
    def objective(S):
        V = integral_func(r,p,S)
        if not np.isfinite(V):
            return 1e20  # Penalize NaN or inf
        return (V - V_target)**2  # Squared error to minimize

    if integral_func==case2V:
        bou=(-2*r/np.tan(p),0.0)
    else:
        return (V_target-np.pi*r**3/np.tan(p))/(np.pi*r**2)
        bou=(0, Hc-2*r/np.tan(p))
    result = minimize_scalar(
        objective,
        bounds=bou,
        method='bounded'
    )

    if result.success:
        return result.x
    else:
        raise RuntimeError("Minimization failed.")

def getPlane(r,p,S):
    """Function to evaluate the coefficients of the plane"""
    A=0
    B=1/np.tan(p)
    C=-1
    D=-S-r/np.tan(p)

    return A,B,C,D

def getPlaneSign(point, A, B, C, D):
    """function to get the sign of the plane.
    Give a test location which is on the side
    you want to use, and get the sign for openMC
    """
    x, y, z = point
    value = A * x + B * y + C * z - D
    return '+' if value > 0 else '-' if value < 0 else '0'