import numpy as np

class Quadrature_Rules:
    def __init__(self):
        self.weights = np.zeros(3)+1./6 #assuming number of integration of points, nip =3
        self.quad_points = np.array([[0.5,0.5],[0.5,0],[0,0.5]])
        return None

