# A Python Package for Density Ratio Estimation by uLSIF.

## 1\. Overview


## 2\. Installation

You can install the package from
[GitHub](https://github.com/JohnYKiyo/density_ratio_estimation)

``` :sh
$ pip install git+https://github.com/JohnYKiyo/density_ratio_estimation.git
```

### Dependencies   
densityratio requires:

- Python (>= 3.6)   
- Jax (>=0.1.57)   
- Jaxlib (>=0.1.37)   
- Ipython (>=7.12.0)


## 3\. Quick start

### 3.1

## 4\. References

\[1\] M. Suigyama et al., **Direct Importance Estimation with Model Selection and Its Application to Covariate Shift Adaptation,** Proc. 20th Int. Conf. Neural Inf. Process. Syst., 2007.

\[2\] T. Kanamori, S. Hido, and M. Sugiyama, **A least-squares approach to direct importance estimation,** J. Mach. Learn. Res., vol. 10, pp. 1391–1445, 2009.

\[3\] T. Kanamori, T. Suzuki, and M. Sugiyama, **Condition number analysis of kernel-based density ratio estimation**.,[arXiv](https://arxiv.org/abs/0912.2800)., (2009).

\[4\] S. Hido, Y. Tsuboi, H. Kashima, M. Sugiyama, and T. Kanamori, **Statistical outlier detection using direct density ratio estimation,** Knowl. Inf. Syst., vol. 26, no. 2, pp. 309–336, 2011.

\[5\] M. Sugiyama, T. Suzuki, and T. Kanamori, **Density Ratio Estimation in Machine Learning.** Cambridge University Press 2012.

\[6\] S. Liu, M. Yamada, N. Collier, and M. Sugiyama, **Change-point detection in time-series data by relative density-ratio estimation,** Neural Networks, 2013.

\[7\] M. Yamada, T. Suzuki, T. Kanamori, H. Hachiya, and M. Sugiyama, **Relative density-ratio estimation for robust distribution comparison,** Neural Computation. 2013.


## 6\. Related Work
- uLSIF for MATLAB R C++ <http://www.ms.k.u-tokyo.ac.jp/software.html>
- RuLSIF for MATLAB <https://riken-yamada.github.io/RuLSIF.html>

    ## Covariate shift
    - M. Sugiyama, M. Krauledat, and K. R. Müller, **Covariate shift adaptation by importance weighted cross validation,** J. Mach. Learn. Res., 2007.

    - H. Shimodaira, **Improving predictive inference under covariate shift by weighting the log-likelihood function,** J. Stat. Plan. Inference, 2000.

    ### Outlier detection
    - S. Hido, Y. Tsuboi, H. Kashima, M. Sugiyama, and T. Kanamori, **Statistical outlier detection using direct density ratio estimation,** Knowl. Inf. Syst., vol. 26, no. 2, pp. 309–336, 2011.

    - H. Nam and M. Sugiyama, **Direct density ratio estimation with convolutional neural networks with application in outlier detection,** IEICE Trans. Inf. Syst., 2015.

    - M. C. du Plessis, H. Shiino, and M. Sugiyama, **Online direct density-ratio estimation applied to inlier-based outlier detection,** Neural Computation. 2015.

    ### Change point detection
    - Y. Kawahara and M. Sugiyama, **Sequential change-point detection based on direct density-ratio estimation,** Stat. Anal. Data Min., 2012.

    - S. Liu, M. Yamada, N. Collier, and M. Sugiyama, **Change-point detection in time-series data by relative density-ratio estimation,** Neural Networks, 2013.

    - M. Yamada, A. Kimura, F. Naya, and H. Sawada, **Change-point detection with feature selection in high-dimensional time-series data,** in IJCAI International Joint Conference on Artificial Intelligence, 2013.