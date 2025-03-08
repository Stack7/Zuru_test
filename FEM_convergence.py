import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from src.mesh_implementation import Mesh
from src.base_function import Base_function
from src.calcolate_quadrature import Quadrature_Rules
from src.poission import Poisson

# Geometrical  paramters
xstart = 0
ystart = 0
yend = 2
xend = 1
bc = 0
err = []
# Average radius of excircle circle
h  = []
base_f = Base_function()
quadrature = Quadrature_Rules()

# Chosen right-hand side function
def f(x,y):
    return 5*np.pi**2 * np.sin(2*np.pi*x) *np.sin(np.pi*y)
# Analitical solution
u = lambda x,y: np.sin(2*np.pi*x)*np.sin(np.pi*y)

#################################################################################################

# Excircle_radius calculation

def excircle_radius_calculation(p1,p2,p3):
    triangle_points = np.array([[p1[0],p1[1],1],[p2[0],p2[1],1],[p3[0],p3[1],1]])
    area = 0.5*np.linalg.det(triangle_points)
    product_sides = np.linalg.norm(p2-p1)*np.linalg.norm(p3-p2) * np.linalg.norm(p3-p1)
    return product_sides / (4*area)

# Evaluating the error from different meshes

for n in range(5,50,5):
    mesh = Mesh(xstart,xend,ystart,yend,n,n)
    poisson_system = Poisson(mesh,base_f, quadrature,f,bc)
    poisson_system.build_system()
    poisson_system.solve()
    err.append(poisson_system.calculate_error(u))
    radius = np.zeros(len(mesh.triangulation.simplices))

    for j,p in enumerate(mesh.triangulation.simplices):
        p1,p2,p3 = mesh.triangulation.points[p]
        radius[j] =  excircle_radius_calculation(p1,p2,p3)
    h.append(np.mean(radius))

# Plotting a the error behaviour as a function of the excircle mean value
fig1= plt.figure(figsize=(5,5))
plt.plot(h,err, 'bo',markersize= 4)
plt.loglog()
plt.grid(which='both')
plt.ylabel('Error', fontsize=12)
plt.xlabel('Excirle radius', fontsize=12)
plt.title('Mesh error as function of excircle radius value', fontsize =14)
plt.show()

