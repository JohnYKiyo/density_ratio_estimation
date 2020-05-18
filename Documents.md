# CLASS densityratio.densratio(*x, y, alpha=0., sigma=None, lamb=None, kernel_num=None, verbose=False*)
Densityratio class for estimation relative-densratio and densratio culculated from two multi-dimentional samples.   

## Parameters
- **x(*array_like*)**   Numerator samples array.   
- **y(*array_like*)**   Denumerator samples array.   
the density ratio `r(x) = p(x) / q(x)` from two-samples `x` and `y` generated from two unknown distributions `p(x), q(x)`, respectively.
- **alpha(*float range of [0,1], default=0*)**   
The *alpha* is a parameter that can adjust the mixing ratio `r(x) = p(x)/(alpha*p(x)+(1-alpha)q(x))`, and is set in the range of \[0,1.\].   
- **sigma(*float or array_like, default=None*)**   Bandwidth of kernel.
- **lamb(*float or array_like, default=None*)**   Regularization parameter.
If a value is set for *sigma, lamb*, that value is used, and if a numerical array is set, the *densityratio()* selects the optimum value by CV.   
- **kernel_num(*int, default=None*)**   
The *kernel_num* is the number of kernels in the linear model.
- **verbose(*bool, default=False*)**   
Whether to override the progress bar display or show a newline.

## Methods
- *\_\_call\_\_(self,val)*   
    return calculate_density_ratio(self, val) 

- *calculate_density_ratio(self, val)*
    This method returns an estimated densityratio value of an input val.