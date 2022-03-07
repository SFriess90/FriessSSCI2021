import copy
import numpy as np
from scipy.special import comb
from scipy.spatial.distance import cdist

def calculateCoordinates(x,svec,tvec,uvec,x0):
    dx = x - x0
    s = np.dot(np.cross(tvec,uvec),dx)/np.dot(np.cross(tvec,uvec),svec)
    t = np.dot(np.cross(svec,uvec),dx)/np.dot(np.cross(svec,uvec),tvec)
    u = np.dot(np.cross(svec,tvec),dx)/np.dot(np.cross(svec,tvec),uvec)
    return s, t, u

def polynom(idx,planes,coord):
    return comb(planes,idx)*((1-coord)**(planes-idx))*(coord**(idx))

def BernsteinPolynomial(i, j, k, l,m,n, s, t, u):
    out = polynom(i, l, s)*polynom(j, m, t)*polynom(k, n, u)
    return out

def deformedPoint(coords,l,m,n,p):
    s,t,u = coords
    out = 0    
    for i in range(0,l+1):
        for j in range(0,m+1):
            for k in range(0,n+1):
                bp = BernsteinPolynomial(i,j,k,l,m,n,s,t,u)
                out+=bp*p[i][j][k]
    return out

def deformPolygon(polygon,svec,tvec,uvec,l,m,n,x0,p):
    out = []
    for v in polygon:
        coords = calculateCoordinates(v,svec,tvec,uvec,x0)
        out.append(deformedPoint(coords,l,m,n,p))
    return out
    
def deformPoint(point,svec,tvec,uvec,l,m,n,x0,p):
    out = []
    coords = calculateCoordinates(point,svec,tvec,uvec,x0)
    out.append(deformedPoint(coords,l,m,n,p))
    return out

def hausdorffDistance(list1, list2):
    length = len(list1)
    res = cdist(np.array(list1), np.array(list2), 'sqeuclidean')
    a_part = sum(np.min(res,axis=0))/length
    b_part = sum(np.min(res,axis=1))/length
    return max(a_part,b_part)

def calculateDeformedMesh(input_shape, input_base, deformation_function, input_x):
    svec,tvec,uvec, x0, p, cp_list = input_base
    plist = copy.deepcopy(cp_list)
    deformation_function(plist, input_x)
    return deformMesh(svec,tvec,uvec, x0, p, plist, input_shape), plist
    
def buildObjectivefunction(shapes, input_base, deformation_function):
     
    target_shape, start_shape = shapes
    svec,tvec,uvec, x0, p, cp_list = input_base
    
    def objectiveFunction(input_x):
        plist = copy.deepcopy(cp_list)
        deformation_function(plist, input_x)
        deformed_shape = deformMesh(svec,tvec,uvec, x0, p, plist, start_shape)
        
        return hausdorffDistance(target_shape, deformed_shape),
        
    return objectiveFunction
    
def deformMesh(svec,tvec,uvec, x0, p, plist, centered_point_list):
    l, m, n = p[0], p[1], p[2]
    deformed_mesh = [deformPoint(pnt,svec,tvec,uvec,l-1,m-1,n-1,x0,plist)[0] for pnt in centered_point_list]
    return deformed_mesh