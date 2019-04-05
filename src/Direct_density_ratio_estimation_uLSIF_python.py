# coding: utf-8

# param x is numeric vector or matrix. Data from a numerator distribution p(x).   
# param y is numeric vector or matrix. Data from a denominator distribution q(x).   
# param sigma is positive numeric vector. Search range of Gussian kernel bandwidth.   
# param lambda is positive numeric vector. Search range of regularization parameter for uLSIF.   
# param kernel_num is positive integer. Number of kernels.
# 

import numpy as np
import random as rand        

class Density_ratio_estimation:
    
    def __init__(self):
        pass
    
    def __init__(self,x,y):
        self.__x = np.array(x)
        self.__y = np.array(y)
        self.__sigma = None
        self.__lambda = None
        self.__results = None
        self.__kernel_num = None
    
    def set_optional_sigma(self,val):
        self.__sigma = val
    
    def set_optional_lambda(self,val):
        self.__lambda = val
    
    def run(self):
        self.__results = self._core(x = self.__x, 
                                    y = self.__y, 
                                    p_sigma = self.__sigma, 
                                    p_lambda = self.__lambda,
                                    kernel_num = self.__kernel_num)
        
   
    def get_x(self):
        return self.__x
    
    def get_y(self):
        return self.__y
    
    def get_results(self):
        if self.__results is None:
            print(u'Error: results do not exist')
            return 1
        else:
            return self.__results
    
    def get_alpha(self):
        if self.__results is None:
            print(u'Error: results do not exist')
            return 1
        else:
            return self.__results['alpha']
    
    def get_lambda(self):
        if self.__results is None:
            print(u'Error: results do not exist')
            return 1
        else:
            return self.__results['lambda']
    
    def get_sigma(self):
        if self.__sigma is None:
            print(u'Error: sigma do not exist')
        else:
            return self.__results['sigma']
    
    def calculate_density_ratio(self,x):
        if self.__results is None:
            print(u'Error: results do not exist')
            return 1
        else:
            return self.__results['cal_density_ratio'](x)
        

    def help(self):
        help_text =     u'Estimate density ratio p(x)/q(y)\n'
        help_text +=    u'param x is from a numerator distribution p(x).\n'  
        help_text +=    u'param y is from a denominator distribution q(y).\n'
        print(help_text)
    
    #main
    def _core(self,x,y,p_sigma=None,p_lambda=None,kernel_num = None):
        if kernel_num is None:
            kernel_num = 100

        param = {'x':x, 'y':y, 'kernel_num':kernel_num}

        if p_sigma is not None:
            param['sigma'] = p_sigma
        if p_lambda is not None:
            param['lambda'] = p_lambda
        
        val = self._fn_uLSIF(**param)
        return val
    
    def _fn_uLSIF(self,x,y,
                  sigma = 10 ** np.linspace(-3,1,9),
                  lamb = 10 ** np.linspace(-3,1,9),
                  kernel_num = None):
        if kernel_num is None:
            kernel_num = 100
        
        x = np.array(x)
        y = np.array(y)
        sigma = np.array(sigma)
        lamb = np.array(lamb)
        
        print(u'start calculate density ratio using uLSIF.')
        
        if len(x.shape) == 1:
            x = x.reshape(len(x),1)
        if len(y.shape) == 1:
            y = y.reshape(len(y),1)
        if x.shape[1] != y.shape[1]:
            print(u'Error: x and y must be same dimensitons.')
            return 1 ##error return.
        
        x_num_row = x.shape[0]
        y_num_row = y.shape[0]
        kernel_num = np.min([kernel_num,x_num_row]) #kernel number is the minimum number of x's lines and the number of kernel.
        centers = x[rand.sample(range(x_num_row),kernel_num)] #randomly choose candidates of rbf kernel centroid.
        
        if(len(sigma) != 1 or len(lamb) != 1):
            print(u'search sigma and lambda')
            opt_params = self._fn_search_sigma_lambda(x,y,centers,sigma,lamb)
            sigma = opt_params['sigma']
            lamb = opt_params['lambda']
            print(u'optimal sigma = {sigma:.4f}, lamda = {lamb:.4f}'.format(sigma = sigma, lamb = lamb))
            
        print(u'alpha optimization')
        phi_x = self._fn_calculate_kernel_gauss(r = x, centers = centers, sigma = sigma)
        phi_y = self._fn_calculate_kernel_gauss(r = y, centers = centers, sigma = sigma)
        H = np.dot(phi_y.transpose(),phi_y) / y_num_row #Phi* Phi
        h = np.array([np.average(phi_x,axis = 0)]).transpose() 
        alpha = np.dot(np.linalg.inv(H + lamb*np.identity(kernel_num)) , h) 
        alpha[alpha < 0] = 0
        print(u'alpha optimization end')
        
        kernelinfo = {'kernel':"Gaussian RBF", 'kernel_num':kernel_num, 'sigma':sigma, 'centers':centers}
        
        def _ratio_uLSIF(x):
            if len(x.shape) == 1:
                x = x.reshape(len(x),1)
            phi_x = self._fn_calculate_kernel_gauss(r = x,
                                                    centers = centers,
                                                    sigma = sigma)
            ratio = np.dot(phi_x,alpha)
            return ratio
        
        val = {'alpha':alpha, 'lambda':lamb, 
               'kernel_info':kernelinfo, 
               'cal_density_ratio':_ratio_uLSIF}
        
        print(u'start calculate density ratio using uLSIF.')
        return val
    
    def _fn_search_sigma_lambda(self,x,y,centers,sigma,lamb):
        x_num_row = x.shape[0]
        y_num_row = y.shape[0]
        n_minimum = min(x_num_row, y_num_row)
        kernel_num = centers.shape[0]
        score_new = np.inf
        sigma_new = 0 
        lamb_new = 0
    
        if sigma.size == 1:
            sigma = [sigma]
        if lamb.size == 1:
            lamb = [lamb]
    
        for sig in sigma:
            phi_x = self._fn_calculate_kernel_gauss(x, centers, sig)
            phi_y = self._fn_calculate_kernel_gauss(y, centers, sig)
        
            H = np.dot(phi_y.transpose(),phi_y) / y_num_row
            h = np.array([np.average(phi_x,axis=0)]).transpose()
            phi_x = phi_x[0:n_minimum].transpose()
            phi_y = phi_y[0:n_minimum].transpose()
        
            for lam in lamb:
                B = H + np.identity(kernel_num) *(lam*(y_num_row - 1)/ y_num_row)
                B_inv = np.linalg.inv(B)
                BinvX = np.dot(B_inv,phi_y)
                XBinvX = phi_y * BinvX
            
                den = np.ones(n_minimum)*y_num_row - np.dot(np.ones(kernel_num),XBinvX)
                diag_B0 = np.diag((np.dot(h.transpose(),BinvX)/den).ravel())
                B0 = np.dot(B_inv, h * np.ones(n_minimum)) + np.dot(BinvX,diag_B0)
            
                diag_B1 = np.diag((np.dot(np.ones(kernel_num),(phi_x*BinvX))/den).ravel())
                B1 = np.dot(B_inv,phi_x) + np.dot(BinvX,diag_B1)
            
                B2 = (y_num_row - 1) * (x_num_row * B0 - B1)/(y_num_row *(x_num_row - 1))
                B2[B2 < 0] = 0
        
                r_x = np.array([ np.dot(np.ones(kernel_num),(phi_x * B2))]).transpose()
                r_y = np.array([ np.dot(np.ones(kernel_num),(phi_y * B2))]).transpose()
            
                score = (np.dot(r_y.transpose(),r_y)/2. - np.dot(np.ones(n_minimum),r_x))/n_minimum
            
                if (score < score_new):
                    print(u'sigma = {sigma:.4f}, lambda = {lamb:.4f}, score = {score:.4f}'.format(sigma = sig, lamb = lam, score = float(score)))
                    score_new = score
                    sigma_new = sig
                    lamb_new = lam
        val = {'sigma':sigma_new, 'lambda':lamb_new}
        return val
    
    def _fn_calculate_kernel_gauss(self,r,centers,sigma):
        def _fn_kernel(x,y,simga):
            dist = np.linalg.norm(x-y)
            return np.exp(-0.5*dist / (sigma**2))
        
        val = []
        for cent in centers:
            vec = np.array([_fn_kernel(x,cent,sigma) for x in r])
            val.append(vec.ravel())
    
        return np.array(val).transpose()

