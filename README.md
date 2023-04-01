
# Multivariate Stochastic Volatility Models

Matlab implementation of the Multivariate Stochastic Volatility model based on the two-step sparse estimation procedure (two-step sparse MSV) developed in the paper:

*High-dimensional Sparse Multivariate Stochastic Volatility Models* by Benjamin Poignard and Manabu Asai, **Journal of Time Series Analysis** 

Link: https://onlinelibrary.wiley.com/doi/abs/10.1111/jtsa.12647

# Reproducible results

Two simulated experiments can be reproduced: 

* in-sample comparison between the true variance-covariance matrix and the variance-covariance matrix estimated from the models (two-step sparse MSV, DCC, CCC). The true variance-covariance matrix is deduced from MARCH/BEKK type dynamics: use **simulation_experiments.m**
* out-of-sample analysis based on simulated returns: the returns are deduced from various data generating processes; the variance-covariance models (two-step sparse MSV with different penalization and number of lags; scalar DCC model) are estimated in the in-sample period; the out-of-sample variance-covariance forecasts are based on the in-sample estimated parameters. Out-of-sample optimal portfolio allocations are deduced from the GMVP problem; those portfolio allocations are used in the Diebold-Mariano/Model Confidence Set tests: use **simulated_portfolio_analysis.m**

The full Matlab code for the DCC model (and alternative MGARCH models) can be found at https://www.kevinsheppard.com/code/matlab/mfe-toolbox/
