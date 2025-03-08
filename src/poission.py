import numpy as np
from scipy.sparse import dok_array,linalg


class Poisson:
    def __init__(self, mesh,base, quadrature,ffunction,bc):
       self.mesh = mesh 
       self.base = base
       self.quadrature = quadrature
       #Dirichlet_boundary_condition --> Dbc
       self.Dbc = bc
       self.ffunction = ffunction
       self.precomputed_base = [[self.base.N1(csi,eta),self.base.N2(csi,eta),self.base.N3(csi,eta)] for csi,eta in self.quadrature.quad_points]
       self.precomputed_grad = [[self.base.grad_N1(csi,eta),self.base.grad_N2(csi,eta),self.base.grad_N3(csi,eta)] for csi,eta in self.quadrature.quad_points]
       
       self.DOF =  len(self.mesh.points)
       self.A   = dok_array((self.DOF,self.DOF),dtype=np.float)
       self.b   = np.zeros(self.DOF, dtype=np.float)
       self.u   = np.zeros(self.DOF,dtype=np.float)

    def local_system_calculation(self,p1,p2,p3):
        local_A = np.zeros((3,3),dtype=np.float)
        local_b = np.zeros(3, dtype=np.float)
        for i in range(len(self.quadrature.quad_points)):
            csi =   self.quadrature.quad_points[i][0]
            eta =   self.quadrature.quad_points[i][1]
            w   =   self.quadrature.weights[i]
            jac = self.base.inverse_jacobian_matrix(csi,eta,p1,p2,p3)
            
            det = self.base.determinant_jacobian_matrix(csi,eta,p1,p2,p3)
            for j in range(3):
                for k in range(3):
                    local_A[j][k] += jac @ self.precomputed_grad[i][j] @ jac @ self.precomputed_grad[i][k]*det*w
                
                local_b[j] += self.precomputed_base[i][j] * self.ffunction(*self.base.get_xy(csi,eta,p1,p2,p3))*det*w
        
        return local_A, local_b
    
    def building_global_matrix(self):

        for p in self.mesh.triangulation.simplices:
           p1,p2,p3 = self.mesh.triangulation.points[p]
           local_a, local_b = self.local_system_calculation(p1,p2,p3)
           column_matrix = np.array([p,p,p])
           row_matrix = column_matrix.T
           self.A[row_matrix,column_matrix]+= local_a
           self.b[p]+=local_b
        return None

    def build_system(self):
        self.building_global_matrix()
        self.assign_bc()
        return None


    def assign_bc(self):
        for p in self.mesh.boundary:
            self.A[p,:] = 0
            self.A[p,p] = 1
            self.b[p] = self.Dbc
        return None
    
    def solve(self):
        self.u,self.info = linalg.cgs(self.A.tocsc(),self.b)

        return None
    
    def calculate_error(self,u_true):

        self.err = 0
        for p in self.mesh.triangulation.simplices:
            p1,p2,p3 = self.mesh.triangulation.points[p]
            for i in range(len(self.quadrature.quad_points)):
                csi =   self.quadrature.quad_points[i][0]
                eta =   self.quadrature.quad_points[i][1]
                w   =   self.quadrature.weights[i]          
                det = self.base.determinant_jacobian_matrix(csi,eta,p1,p2,p3)
                x,y =  self.base.get_xy(csi,eta,p1,p2,p3)
                u_num = self.precomputed_base[i] @ self.u[p].T
                self.err += w*(u_num - u_true(x,y))**2 *det

        return self.err
    
    def error_triangle_calculation(self, u_true):

        self.triangle_error = np.zeros(len( self.mesh.triangulation.simplices))

        for index,p in enumerate(self.mesh.triangulation.simplices):
            p1,p2,p3 = self.mesh.triangulation.points[p]
            for i in range(len(self.quadrature.quad_points)):
                csi =   self.quadrature.quad_points[i][0]
                eta =   self.quadrature.quad_points[i][1]
                w   =   self.quadrature.weights[i]          
                det = self.base.determinant_jacobian_matrix(csi,eta,p1,p2,p3)
                x,y =  self.base.get_xy(csi,eta,p1,p2,p3)
                u_num = self.precomputed_base[i] @ self.u[p].T
                self.triangle_error[index]+= w*(u_num - u_true(x,y))**2 *det

 
        return self.triangle_error
    
    def mesh_refinement(self,threshold):
        points_to_add = []
        triangles_to_refine = self.mesh.triangulation.simplices[self.triangle_error >= threshold]
        for p in triangles_to_refine:
            p1,p2,p3 = self.mesh.triangulation.points[p]
            points_to_add.append((p1+p2)*0.5)
            points_to_add.append((p1+p3)*0.5)
            points_to_add.append((p2+p3)*0.5)
        points_to_add = np.unique(np.array(points_to_add).reshape(-1,2),axis=0)
        self.mesh.add_point(points_to_add)
        self.DOF =  len(self.mesh.points)
        self.A   = dok_array((self.DOF,self.DOF),dtype=np.float)
        self.b   = np.zeros(self.DOF, dtype=np.float)
        self.u   = np.zeros(self.DOF,dtype=np.float)
        return None 
    
    




           
           
           



