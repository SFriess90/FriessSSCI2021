from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
import numpy as np

def pointcloud(mesh_list):

	tmp_list = [v for poly in mesh_list.vectors for v in poly]
	point_list = list(map(list,set(map(tuple,tmp_list))))
	return point_list
	
def centered(point_list):

	mean = np.mean(point_list,axis=0)
	centered_point_list = [(p-mean) for p in point_list]
	return centered_point_list
    
def calculatebasis(centered_point_list):
    # Calculate basis vectors
    mins = np.min(centered_point_list, axis=0)
    maxs = np.max(centered_point_list, axis=0)

    delta = (np.array(maxs)-np.array(mins))

    x0 = -delta/2
    svec, tvec, uvec = np.array([delta[0],0,0]), np.array([0,delta[1],0]), np.array([0,0,delta[2]])
    return svec, tvec, uvec, x0
    
def visualize3D(centered_points, sz=5, save=False):

    # Visualize Mesh
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.set_axis_off()
    
    x_vals = np.array([p[0] for p in centered_points])
    y_vals = np.array([p[1] for p in centered_points])
    z_vals = np.array([p[2] for p in centered_points])

    ax.scatter(x_vals, y_vals, z_vals, s=sz)

    max_range = np.array([x_vals.max() - x_vals.min(), y_vals.max() - y_vals.min(), z_vals.max() - z_vals.min()]).max() / 2.0

    mid_x = (x_vals.max() + x_vals.min()) * 0.5
    mid_y = (y_vals.max() + y_vals.min()) * 0.5
    mid_z = (z_vals.max() + z_vals.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if(save==True):
        pyplot.savefig("out.pdf")

    pyplot.show()
    
def visualize2D(centered_points, c = [0,1],sz=0.5):

    x_vals = [p[0] for p in centered_points]
    y_vals = [p[1] for p in centered_points]
    z_vals = [p[2] for p in centered_points]
    set_vals = [x_vals, y_vals, z_vals]

    # Create data
    x = set_vals[c[0]]
    y = set_vals[c[1]]
    colors = (0,0,0)
    area = sz

    # Plot
    pyplot.scatter(x, y, s=area, alpha=0.5)
    ax = pyplot.gca()
    ax.set_aspect('equal')
    pyplot.title('Scatter plot')
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.show()
    
    
def visualizecontrolvolume2D(centered_points, control_points, c = [0,1], sz1=5, sz2=5):

    x_vals, y_vals, z_vals = [p[0] for p in centered_points], [p[1] for p in centered_points], [p[2] for p in centered_points]
    set_vals = [x_vals, y_vals, z_vals]
    
    cp_x, cp_y, cp_z = [p[0] for p in control_points],  [p[1] for p in control_points], [p[2] for p in control_points]
    cp_vals = [cp_x, cp_y, cp_z]


    # Create data
    x = set_vals[c[0]]
    y = set_vals[c[1]]
    
    cpx = cp_vals[c[0]]
    cpy = cp_vals[c[1]]
    
    colors = (0,0,0)



    # Plot
    pyplot.scatter(x, y, s=sz1, alpha=0.5)
    pyplot.scatter(cpx, cpy, s=sz2, alpha=1, c='r')
    ax = pyplot.gca()
    ax.set_aspect('equal')
    pyplot.title('Scatter plot')
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.show()
    
    
def visualizecontrolvolume(centered_points, control_points, sz1=5, sz2=5):

    # Visualize Mesh
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.set_axis_off()
    
    x_vals = np.array([p[0] for p in centered_points])
    y_vals = np.array([p[1] for p in centered_points])
    z_vals = np.array([p[2] for p in centered_points])

    ax.scatter(x_vals, y_vals, z_vals, s=sz1)
    ax.scatter([p[0] for p in control_points], [p[1] for p in control_points], [p[2] for p in control_points], s=sz2, alpha=1, c='r')

    max_range = np.array([x_vals.max() - x_vals.min(), y_vals.max() - y_vals.min(), z_vals.max() - z_vals.min()]).max() / 2.0

    mid_x = (x_vals.max() + x_vals.min()) * 0.5
    mid_y = (y_vals.max() + y_vals.min()) * 0.5
    mid_z = (z_vals.max() + z_vals.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    pyplot.show()


    
def controlpointmethod(b,xz,p=[4, 4, 4]):

    l, m, n = p[0], p[1], p[2]

    def pijk(i,j,k):
        out = xz + (i/(l-1))*b[0] + (j/(m-1))*b[1] + (k/(n-1))*b[2]
        return out
        
    return pijk

def createcplist(inputf, p=[4,4,4]):
    # Initialize all control points
    l, m, n = p[0], p[1], p[2]
    p = np.zeros(shape=(l,m,n,3))
    cp_list = []
    for i in range(0,l):
        for j in range(0,m):
            for k in range(0,n):
                cp = inputf(i,j,k)
                cp_list.append(cp)
                p[i][j][k] = inputf(i,j,k)
    
    return p, cp_list