# param x is numeric vector or matrix. Data from a numerator distribution p(x).   
# param y is numeric vector or matrix. Data from a denominator distribution q(x).   
# param sigma is positive numeric vector. Search range of Gussian kernel bandwidth.   
# param lambda is positive numeric vector. Search range of regularization parameter for uLSIF.   
# param kernel_num is positive integer. Number of kernels.
# 

from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import jit, vmap

import random as rand
from functools import partial
from .progressbar import progbar

class densratio: 
    def __init__(self):
        pass 
    def __init__(self, x, y, alpha=0., sigma=None, lamb=None, kernel_num=None, verbose=False):
        self.__clear_disp = not verbose
        self.__x = transform_data(x)
        self.__y = transform_data(y)

        if self.__x.shape[1] != self.__y.shape[1]:
            raise ValueError("x and y must be same dimentions.")
        
        if sigma is None:
            sigma = np.logspace(-4,9,14)
        
        if lamb is None:
            lamb = np.logspace(-4,9,14)
        
        
        
        self._RuLSIF(x = self.__x,
                     y = self.__y,
                     alpha = alpha,
                     s_sigma = np.atleast_1d(sigma),
                     s_lambda = np.atleast_1d(lamb),
                     kernel_num = kernel_num)
    
    def __call__(self,val):
        return self.calculate_density_ratio(val)
    
    def calculate_density_ratio(self, val):
        val = transform_data(val)
        phi_x = gauss_kernel(val, self.__centers, self.__sigma)
        density_ratio = np.dot(phi_x, self.__weights)
        return density_ratio
        
    @property
    def x(self):
        return self.__x
    @x.setter
    def x(self,v):
        self.__x = transform_data(v)

    @property
    def y(self):
        return self.__y
    @y.setter
    def y(self,v):
        self.__y = transform_data(v)
    
    @property
    def alpha(self):
        return self.__alpha
    
    @property
    def sigma(self):
        return self.__sigma
    
    @property
    def lambda_(self):
        return self.__lambda

    def help(self):
        help_text =     u'Estimate density ratio p(x)/q(y)\n'
        help_text +=    u'param x is from a numerator distribution p(x).\n'  
        help_text +=    u'param y is from a denominator distribution q(y).\n'
        print(help_text)
    
    #main
    def _RuLSIF(self,x,y,alpha,s_sigma,s_lambda,kernel_num):
        if kernel_num is None:
            kernel_num = 100
        
        x_num_row = x.shape[0]
        y_num_row = y.shape[0]
        kernel_num = np.min([kernel_num,x_num_row]).item() #kernel number is the minimum number of x's lines and the number of kernel.
        centers = x[rand.sample(range(x_num_row),kernel_num)] #randomly choose candidates of rbf kernel centroid.
        if len(s_sigma)==1 and len(s_lambda)==1:
            sigma = s_sigma[0]
            lambda_ = s_lambda[0]
        else:
            optimized_params = self._optimize_sigma_lambda(x,y,alpha,centers,s_sigma,s_lambda)
            sigma = optimized_params['sigma']
            lambda_ = optimized_params['lambda']
        
        phi_x = gauss_kernel(r = x, centers = centers, sigma = sigma)
        phi_y = gauss_kernel(r = y, centers = centers, sigma = sigma) 
        H = (1.-alpha)*(np.dot(phi_y.T, phi_y) / y_num_row) + alpha*(np.dot(phi_x.T, phi_x)/x_num_row) #Phi* Phi
        h = np.average(phi_x,axis=0).T
        weights = np.linalg.solve(H + lambda_*np.identity(kernel_num), h).ravel()
        #weights[weights < 0] = 0.
        weights = jax.ops.index_update(weights,weights<0,0) #G2[G2<0]=0
        
        self.__alpha = alpha
        self.__weights = weights
        self.__lambda = lambda_
        self.__sigma = sigma
        self.__centers = centers
        self.__kernel_num = kernel_num
        
    
    def _optimize_sigma_lambda(self,x,y,alpha,centers,s_sigma,s_lambda):
        x_num_row = x.shape[0]
        y_num_row = y.shape[0]
        n_minimum = min(x_num_row, y_num_row)
        kernel_num = centers.shape[0]
        score_new = np.inf
        sigma_new = 0 
        lamb_new = 0
    
        #if s_sigma.size == 1:
        #    sigma = np.atleast_1d(sigma)
        #if s_lambda.size == 1:
        #    lamb = np.atleast_1d(lamb)
        
        pbar = progbar(len(s_sigma),clear_display=self.__clear_disp)
        for i,sig in enumerate(s_sigma):
            phi_x = gauss_kernel(x, centers, sig)
            phi_y = gauss_kernel(y, centers, sig)
            H = (1.- alpha)*(np.dot(phi_y.T, phi_y) / y_num_row) + alpha*(np.dot(phi_x.T, phi_x) / x_num_row)
            h = phi_x.mean(axis=0,keepdims=True).T
            phi_x = phi_x[:n_minimum].T
            phi_y = phi_y[:n_minimum].T
        
            for lam in s_lambda:
                G = H + np.identity(kernel_num) * (lam*(y_num_row - 1)/ y_num_row)
                GinvX = np.linalg.solve(G,phi_y)
                XGinvX = phi_y * GinvX
            
                den = np.ones(n_minimum)*y_num_row - np.dot(np.ones(kernel_num),XGinvX)
                diag_G0 = np.diag((np.dot(h.T,GinvX)/den).ravel())
                G0 = np.linalg.solve(G, h*np.ones(n_minimum)) + np.dot(GinvX, diag_G0)
                
                diag_G1 = np.diag(np.dot(np.ones(kernel_num),phi_x*GinvX).ravel())
                G1 = np.linalg.solve(G, phi_x) + np.dot(GinvX, diag_G1)
            
                G2 = (y_num_row - 1) * (x_num_row * G0 - G1)/(y_num_row *(x_num_row - 1))
                #G2[G2<0] = 0.
                G2 = jax.ops.index_update(G2,G2<0,0) #G2[G2<0]=0
                
                r_x = (phi_x * G2).sum(axis=0).T
                r_y = (phi_y * G2).sum(axis=0).T
            
                score = ((np.dot(r_y.T,r_y).ravel()/2. - r_x.sum(axis=0))/n_minimum).item()
            
                if (score < score_new):
                    score_new = score
                    sigma_new = sig
                    lamb_new = lam
            pbar.update(i,info='sigma:{0},lambda:{1}, score={2}'.format(sigma_new,lamb_new,score_new))
        del pbar
        print('Found optimal sigma = {0}, lambda = {1}, score={2}'.format(sigma_new,lamb_new, score_new))
        return {'sigma':sigma_new, 'lambda':lamb_new}
    
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
