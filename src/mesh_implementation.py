import numpy as np
from scipy.spatial import Delaunay


class Mesh:
    """This class designs a Delaunay mesh in a given domain, it allows it visuations through pyplot 
    and to add a point in the mesh """
    def __init__(self,xstart,xend,ystart,yend,xnum_points,ynum_points):

        xpoints =  np.linspace(xstart,xend,xnum_points,endpoint=True)
        ypoints =  np.linspace(ystart,yend,ynum_points,endpoint=True)
        xp,yp = np.meshgrid(xpoints,ypoints,indexing='ij')
        xp = xp.reshape(-1,1)
        yp = yp.reshape(-1,1)
        self.points = np.concatenate((xp,yp),axis=1)
        self.triangulation = Delaunay(self.points, incremental=True)
        self.boundary = np.unique(self.triangulation.convex_hull.flatten())

    def view_Mesh(self):
        
        from matplotlib import pyplot as plt
        plt.triplot(self.triangulation.points[:,0], self.triangulation.points[:,1], self.triangulation.simplices)

        plt.plot(self.triangulation.points[:,0], self.triangulation.points[:,1], 'o')

        plt.show()
        return None
    
    
    def add_point(self,new_point):
        self.triangulation.add_points(new_point)
        self.boundary = np.unique(self.triangulation.convex_hull.flatten())
        self.points = self.triangulation.points
        return None
