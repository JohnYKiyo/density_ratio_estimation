# Algorithm of direct density ratio estimation.

The advantage of directly estimating the ratio of the probability density functions from the two samples is that the estimation error can be reduced by calculating the ratio of the probability density functions of the distributions to be compared and taking the ratio.   
Direct estimation of the density ratio uses the unconstrained least squares importance fitting algorithm (uLSIF) \[1-3\].

First, the density ratio $r({\bf x}) = p({\bf x})/q({\bf x})$ is defined as the sum of kernel functions as 
\begin{eqnarray}
    {\bf \hat{r}}_{\bf w} = {\displaystyle \sum_{i=1}^{N_{kernel}}}{{\rm w}_i\psi_i({\bf x})} = {\bf w}^{tr}{\bf \psi({\bf x})}
\end{eqnarray} 
, where ${\bf \psi({\bf x})}$ is non-negative kernel function which is employed gaussian kernel, 
$p({\bf x})$ and $q({\bf x})$ are probability density function, x is d-dimensional real number, 
${\bf w}$ are linear model parameters, $N_{kernel}$ is the number of kernels.   

The uLSIF algorithm considers the minimum value of the following function to determine alpha:
\begin{eqnarray}
    E({\bf w}) &=& \frac{1}{2}\int d{\bf x}\, \left\{\hat{r}_{\bf w}({\bf x}) - r({\bf x}) \right\}^2 q(\bf x) \\
              &=& \frac{1}{2}\int d{\bf x}\, {\bf w}^{tr}{\bf \psi({\bf x})} {\bf \psi^{tr}({\bf x})}{\bf w}q({\bf x}) -  \int d{\bf x}\,{\bf w}^{tr}{\bf \psi({\bf x})} p({\bf x}) + Const. \label{eq:uLSIF}
\end{eqnarray}   

Although the generalized KL divergence is well known as the objective function, the use of KL-div is a convex optimization problem, and the demerit is that the calculation is slow because the optimal solution is obtained by the gradient method.
On the other hand, uLSIF has better computational cost than the gradient method because it can be solved analytically and is theoretically shown that A has excellent properties in terms of convergence and numerical stability of the algorithm \[3\].

The following equation is obtained by replacing the expected value of the objective function with the sample mean and adding regularization.   
\begin{eqnarray}   
    \displaystyle \min_{\bf w} \left[ \frac{1}{2}{\bf w}^{tr}\hat{H} {\bf w} - {\bf w}^{tr}\hat{h} + \frac{\lambda}{2}|{\bf w}|^2 \right]
\end{eqnarray}   

\begin{eqnarray}   
    \hat{H} = \frac{1}{N_{q}} \sum_{i=1}^{N_{q}}{\bf \psi({\bf x}^{q}_{i})}{\bf \psi^{tr}({\bf x}^{q}_{i})} \,\, , \,\, \hat{h} = \frac{1}{N_p} \sum_{i=1}^{N_p} \psi({\bf x}^{p}_{i})
\end{eqnarray}   
,where two data samples ${\bf x}_i^p$ and ${\bf x}_i^q$ are obtained from the pdf $p(\bf x)$ and $q(\bf x)$, respectively, $N_p$ and $N_q$ are the number of two samples.

The analytical {\bf w} is obtained as
\begin{eqnarray}
    {\bf w} = (\hat{H} + \lambda I)^{-1}\hat{h}.
\end{eqnarray}
If {\bf w} is negative during the calculation, it is set to 0.

The bandwidth of RBF $\sigma$ is set by leave one-out cross validation (LOOCV).   
LOOCV can also be analytically obtained.   
In the current setup, two sets of samples, $\{ x_i^p\}^{N_p}_{i=1}$ and  $\{ x_j^q\}^{N_q}_{j=1}$, generally have different sample sizes. For $i=1,\dots,n$, where $n:= \min(N_p,N_q)$, suppose that the $i-$th $p(x)$ sample $x_i^p$ and $i-$th $q(x)$ sample $x_i^q$ are held out at the same time in the LOOCV procedure.

Let $\hat{r}_i(x)$ be a density-ratio estimate obtained without the $i-$th $p(x)$ and $q(x)$ samples $x_i^p$. Then the LOOCV score is expressed as 
\begin{eqnarray}
    LOOCV = \frac{1}{n_t}\sum_{i=1}^{n_t}\left[ \frac{1}{2}(\hat{r}_i({\bf x}^q_i))^2 - \hat{r}_i({\bf x}^p_i) \right].
\end{eqnarray}

This package provides a direct density estimation module in which uLSIF \[1-3\] is implemented.
In addition, the relative density ratio (RuLSIF) $p(x)/(\alpha * p(x) + (1 - \alpha) * q(x))$ ,where $\alpha$ is in the range \[0,1\], is also implimented \[4\].
By substituting $q(x)$ into $\alpha * p(x) + (1 - \alpha) * q(x)$, we get RuLSIF algorithm from the above development of formula.

### References
\[1\] M. Sugiyama, T. Suzuki, and T. Kanamori, **Density Ratio Estimation in Machine Learning.** Cambridge University Press 2012.

\[2\] T. Kanamori, S. Hido, and M. Sugiyama, **A least-squares approach to direct importance estimation,** J. Mach. Learn. Res., vol. 10, pp. 1391–1445, 2009.

\[3\] T. Kanamori, T. Suzuki, and M. Sugiyama, **Condition number analysis of kernel-based density ratio estimation**.,[arXiv](https://arxiv.org/abs/0912.2800)., (2009).

\[4\] M. Yamada, T. Suzuki, T. Kanamori, H. Hachiya, and M. Sugiyama, **Relative density-ratio estimation for robust distribution comparison,** Neural Computation. 2013.
