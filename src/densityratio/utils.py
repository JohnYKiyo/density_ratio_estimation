import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
from jax import jit, vmap

from functools import partial

def gauss_kernel(r,centers,sigma):
    dists = pairwise_distances(euclid_distance)
    return np.exp(-0.5*dists(r,centers) / (sigma**2))
    #return np.exp(-0.5*pairwise_euclid_distances(r,centers) / (sigma**2))

def transform_data(x):
    if isinstance(x,np.ndarray):
        if len(x.shape)==1:
            return np.atleast_2d(x.astype(np.float64)).T
        else:
            return np.atleast_2d(x.astype(np.float64))
    elif isinstance(x,list):
        return transform_data(np.array(x))
    else:
        raise ValueError("Cannot convert to numpy.array")

def euclid_distance(x,y, square=True):
    '''
    \sum_m (X_m - Y_m)^2
    '''
    XX=np.dot(x,x)
    YY=np.dot(y,y)
    XY=np.dot(x,y)
    if not square:
        return np.sqrt(XX+YY-2*XY)
    return XX+YY-2*XY

def pairwise_distances(dist,**arg):
    '''
    d_ij = dist(X_i , Y_j)
    "i,j" are assumed to indicate the data index.
    '''
    return jit(vmap(vmap(partial(dist,**arg),in_axes=(None,0)),in_axes=(0,None)))

def pairwise_euclid_distances(x,y,square=True):
    XX = np.einsum('ik,ik->i',x,x)
    YY = np.einsum('ik,ik->i',y,y)
    XY = np.einsum('ik,jk->ij',x,y)
    if not square:
        return np.sqrt(XX[:,np.newaxis]+YY[np.newaxis,:] - 2*XY)
    return XX[:,np.newaxis]+YY[np.newaxis,:] - 2*XY
