import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from src.mesh_implementation import Mesh
from src.base_function import Base_function
from src.calcolate_quadrature import Quadrature_Rules
from src.poission import Poisson

#Geometrical  paramters
xstart = 0
ystart = 0
yend = 2
xend = 1
xnum_points = 4
ynum_points = 4
bc = 0
err = 0
# chosen right-hand side function
def f(x,y):
    return 5*np.pi**2 * np.sin(2*np.pi*x) *np.sin(np.pi*y)

#analitical solution
u = lambda x,y: np.sin(2*np.pi*x)*np.sin(np.pi*y)

#################################################################################################
# Defining the mesh
mesh = Mesh(xstart,xend,ystart,yend,xnum_points,ynum_points)

# Definition of tha bse function and quadrature points
base_f = Base_function()
quadrature = Quadrature_Rules()

# Definition Poisson equation on the defined mesh with given basis and quadrature points
#Assembling the discrete formulation and solving the sparse linear system
poisson_system = Poisson(mesh,base_f, quadrature,f,bc)
poisson_system.build_system()
poisson_system.solve()
# Plotting a comparison between analitical solution and true solution
u_true = np.sin(2*np.pi*poisson_system.mesh.triangulation.points[:,0])*np.sin(np.pi*poisson_system.mesh.triangulation.points[:,1])

# Calculating the error for each triangle 
#  add points if the error is bigger than a chosen threshold
threshold = 1e-5 
err = poisson_system.error_triangle_calculation(u)
while np.sum(err>threshold):
    print(np.sum(err))
    poisson_system.mesh_refinement(threshold)
    poisson_system.build_system()
    poisson_system.solve()
    poisson_system.mesh.view_Mesh()
    err = poisson_system.error_triangle_calculation(u)
