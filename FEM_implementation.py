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
xnum_points = 20
ynum_points = 16
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
mesh.view_Mesh()

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

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 12), subplot_kw={'projection': '3d'})

ax1.plot_trisurf(poisson_system.mesh.points[:,0],poisson_system.mesh.points[:,1],poisson_system.u)

ax2.plot_trisurf(poisson_system.mesh.points[:,0],poisson_system.mesh.points[:,1],u_true)
plt.show()