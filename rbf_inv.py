import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve

# Constants as provided
MATH_CONST_e = 2.71828
MATH_CONST_E = 1.30568

def rbf_function(r, function_type='gaussian', c=0, epsilon=MATH_CONST_E):
    if function_type == 'gaussian':
        return np.exp(-(epsilon*r)**2)
    elif function_type == 'multiquadric':
        return np.sqrt(c**2 + epsilon*r**2)

    # Add more RBF types as needed

def build_interpolation_matrix(X2d, function_type='gaussian', c=0, epsilon=MATH_CONST_E):
    # Compute the pairwise distances between points
    r = cdist(X2d, X2d, 'euclidean')
    # Apply the chosen RBF function
    return rbf_function(r, function_type, c, epsilon)

# def interpolate_rbf(X2d, Xnd, p, function_type='gaussian', c=0, epsilon=1.0):
#     # Build the interpolation matrix
#     Phi = build_interpolation_matrix(X2d, function_type, c, epsilon)
#     # Solve for the coefficients
#     coefficients = solve(Phi, Xnd)
#     # Compute distances from new points to the original points
#     r_new = cdist(p, X2d, 'euclidean')
#     # Apply the RBF function to these distances
#     Phi_new = rbf_function(r_new, function_type, c, epsilon)
#     # Interpolate values at new points
#     return Phi_new @ coefficients


class RBFinv:
    """sklearn API for RBF interpolation"""
    def __init__(self, function_type='multiquadric', c=0, epsilon=MATH_CONST_E):
        self.function_type = function_type
        self.c = c ## the RBF paper uses c=0 in their face example
        self.epsilon = epsilon

    def fit(self, X2d, Xnd):
        self.X2d = X2d
        self.X = Xnd
        Phi = build_interpolation_matrix(X2d, self.function_type, self.c, self.epsilon)
        self.coefficients = solve(Phi, Xnd)

    def transform(self, p, **kwargs):
        r_new = cdist(p, self.X2d, 'euclidean')
        Phi_new = rbf_function(r_new, self.function_type, self.c, self.epsilon)
        return Phi_new @ self.coefficients
    
    def inverse_transform(self, p, **kwargs):
        return self.transform(p)