"""densityratio

"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
from jax import jit, vmap

import random as rand
from functools import partial
from tqdm import tqdm

from .utils import *

class Densratio(object):
    """Densratio
    The densratio class estimates the density ratio r(x) = p(x) / q(x) from two-samples x1 and x2 generated from two unknown distributions p(x), q(x), respectively, where x1 and x2 are d-dimensional real numbers.

    Args:
        x (array-like of float):
            Numerator samples array. x is generated from p(x).
        y (array-like of float):
            Denumerator samples array. y is generated from q(x).
        alpha (float or array-like, optional):
            The alpha is a parameter that can adjust the mixing ratio r(x) = p(x)/(alpha*p(x)+(1-alpha)q(x))
            , and is set in the range of 0-1.
            Defaults to 0.
        sigma (float or array-like, optional):
            Bandwidth of kernel. If a value is set for sigma, that value is used for kernel bandwidth
            , and if a numerical array is set for sigma, Densratio selects the optimum value by using CV.
            Defaults to array of 10e-4 to 10e+9 divided into 14 on the log scale.
        lamb (float or array-like, optional):
            Regularization parameter. If a value is set for lamb, that value is used for hyperparameter
            , and if a numerical array is set for lamb, Densratio selects the optimum value by using CV.
            Defaults to array of 10e-4 to 10e+9 divided into 14 on the log scale.
        kernel_num (int, optional): The number of kernels in the linear model. Defaults to 100.

        Raises:
            ValueError: [description]
    """

    def __init__(self, x, y, alpha=0., sigma=None, lamb=None, kernel_num=100):

        self.__x = transform_data(x)
        self.__y = transform_data(y)

        if self.__x.shape[1] != self.__y.shape[1]:
            raise ValueError("x and y must be same dimentions.")

        if sigma is None:
            sigma = np.logspace(-4, 9, 14)

        if lamb is None:
            lamb = np.logspace(-4, 9, 14)

        self.__x_num_row = self.__x.shape[0]
        self.__y_num_row = self.__y.shape[0]
        self.__kernel_num = min([kernel_num, self.__x_num_row])  # kernel number is the minimum number of x's lines and the number of kernel.
        self.__centers = np.array(rand.sample(list(self.__x), k=self.__kernel_num))  # randomly choose candidates of rbf kernel centroid.
        self.__n_minimum = min(self.__x_num_row, self.__y_num_row)
        self.__kernel = jit(partial(gauss_kernel, centers=self.__centers))

        self._RuLSIF(
            x=self.__x,
            y=self.__y,
            alpha=alpha,
            s_sigma=np.atleast_1d(sigma),
            s_lambda=np.atleast_1d(lamb),
        )

    def __call__(self, val):
        """__call__ method
        call calculate_density_ratio.
        Args:
            val (`float` or `array_like of float`):

        Returns:
            array_like of float. Density ratio at input val. r(val)
        """
        return self.calculate_density_ratio(val)

    def calculate_density_ratio(self, val):
        """calculate_density_ratio method

        Args:
            val (`float` or `array-like of float`) : [description]

        Returns:
            array-like of float : Density ratio at input val. r(val)
        """

        val = transform_data(val)
        phi_x = self.__kernel(val, sigma=self.__sigma)
        density_ratio = np.dot(phi_x, self.__weights)
        return density_ratio

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def alpha(self):
        return self.__alpha

    @property
    def sigma(self):
        return self.__sigma

    @property
    def lambda_(self):
        return self.__lambda

    @property
    def kernel_centers(self):
        return self.__centers

    @property
    def N_kernels(self):
        return self.__kernel_num

    @property
    def KLDiv(self):
        return np.log(np.dot(self.__phi_x, self.__weights)).mean()

    # main
    def _RuLSIF(self, x, y, alpha, s_sigma, s_lambda):
        if len(s_sigma) == 1 and len(s_lambda) == 1:
            sigma = s_sigma[0]
            lambda_ = s_lambda[0]
        else:
            optimized_params = self._optimize_sigma_lambda(x, y, alpha, s_sigma, s_lambda)
            sigma = optimized_params['sigma']
            lambda_ = optimized_params['lambda']

        phi_x = self.__kernel(r=x, sigma=sigma)
        phi_y = self.__kernel(r=y, sigma=sigma)
        H = (1. - alpha) * (np.dot(phi_y.T, phi_y) / self.__y_num_row) + alpha * (np.dot(phi_x.T, phi_x) / self.__x_num_row)  # Phi* Phi
        h = np.average(phi_x, axis=0).T
        weights = np.linalg.solve(H + lambda_ * np.identity(self.__kernel_num), h).ravel()
        #  weights[weights < 0] = 0.
        weights = jax.ops.index_update(weights, weights < 0, 0)  # G2[G2<0]=0

        self.__alpha = alpha
        self.__weights = weights
        self.__lambda = lambda_
        self.__sigma = sigma
        self.__phi_x = phi_x
        self.__phi_y = phi_y

    def _optimize_sigma_lambda(self, x, y, alpha, s_sigma, s_lambda):
        score_new = np.inf
        sigma_new = 0
        lamb_new = 0

        with tqdm(total=len(s_sigma)) as pbar:
            for i, sig in enumerate(s_sigma):
                phi_x = self.__kernel(x, sigma=sig)
                phi_y = self.__kernel(y, sigma=sig)
                H = (1. - alpha) * (np.dot(phi_y.T, phi_y) / self.__y_num_row) + alpha * (np.dot(phi_x.T, phi_x) / self.__x_num_row)
                h = phi_x.mean(axis=0, keepdims=True).T
                phi_x = phi_x[:self.__n_minimum].T
                phi_y = phi_y[:self.__n_minimum].T

                for lam in s_lambda:
                    B = H + np.identity(self.__kernel_num) * (lam * (self.__y_num_row - 1) / self.__y_num_row)

                    BinvX = np.linalg.solve(B, phi_y)
                    XBinvX = phi_y * BinvX
                    D0 = np.ones(self.__n_minimum) * self.__y_num_row - np.dot(np.ones(self.__kernel_num), XBinvX)
                    diag_D0 = np.diag((np.dot(h.T, BinvX) / D0).ravel())
                    B0 = np.linalg.solve(B, h * np.ones(self.__n_minimum)) + np.dot(BinvX, diag_D0)

                    diag_D1 = np.diag(np.dot(np.ones(self.__kernel_num), phi_x * BinvX).ravel())
                    B1 = np.linalg.solve(B, phi_x) + np.dot(BinvX, diag_D1)

                    B2 = (self.__y_num_row - 1) * (self.__x_num_row * B0 - B1) / (self.__y_num_row * (self.__x_num_row - 1))
                    B2 = jax.ops.index_update(B2, B2 < 0, 0)  # G2[G2<0]=0

                    r_x = (phi_x * B2).sum(axis=0).T
                    r_y = (phi_y * B2).sum(axis=0).T

                    score = ((np.dot(r_y.T, r_y).ravel() / 2. - r_x.sum(axis=0)) / self.__n_minimum).item()  # LOOCV

                    if (score < score_new):
                        score_new = score
                        sigma_new = sig
                        lamb_new = lam

                pbar.set_description()
                pbar.set_postfix_str(f"sigma:{sigma_new},lambda:{lamb_new}, score:{score_new:.4f}", refresh=True)
                pbar.update(1)
            pbar.clear()
            print(f'Found optimal sigma = {sigma_new}, lambda = {lamb_new}, score={score_new}')
        return {'sigma': sigma_new, 'lambda': lamb_new}
