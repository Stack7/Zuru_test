import numpy as np

class Base_function:
    def __init__(self):

        pass
    
    def N1(self,csi,eta):
        return 1-csi-eta
    
    def N2(self,csi,eta):

        return csi
    def N3(self,csi,eta):
        return eta
        
    def grad_N1(self,csi,eta):
        return np.array([-1,-1])
    
    def grad_N2(self,csi,eta):
        return np.array([1,0])
    
    def grad_N3(self,csi,eta):
        return np.array([0,1])
    

    def get_xy(self, csi, eta, p1,p2,p3):
        x = p1[0] + csi*(p2[0]-p1[0]) + (p3[0]-p1[0])*eta
        y = p1[1] + csi*(p2[1]-p1[1]) + (p3[1]-p1[1])*eta
        return x, y
    
    def jacobian_matrix(self,csi, eta, p1,p2,p3):
        J = np.zeros((2,2))
        J[0][0] = p2[0]-p1[0]
        J[0][1] = p2[1]-p1[1]
        J[1][0] = p3[0]-p1[0]
        J[1][1] = p3[1]-p1[1]

        return J 
    def inverse_jacobian_matrix(self,csi, eta, p1,p2,p3):
        return np.linalg.inv(self.jacobian_matrix(csi, eta, p1,p2,p3))
    
    def determinant_jacobian_matrix(self,csi, eta, p1,p2,p3 ):
        return np.linalg.det(self.jacobian_matrix(csi, eta, p1,p2,p3))






