# Notes on [Lasso.jl](https://juliastats.org/Lasso.jl/stable/)

For CBIOMES Julia meeting discussion lead by [@jtsiddons](https://www.github.com/jtsiddons) in October.

---

## Plan

We look at using `Lasso.jl` and `GLMNet.jl` to perform penalised regression in Julia. I have created a quick dataset from the Narragansett Bay [NABATS.org](nabats.org), from which we will use environmental and meteorlogical variables to estimate surface salinity in the bay. I will start with a simple linear regression using `GLM.jl`.

### Introduction to Lasso Regression

- What is Lasso Regression? 1st order penalty on the regression coefficients.
- How do coefficients change with $\lambda$ (plot)

### Ridge Regression

2nd order penalty to the coefficients.

### Elastic Net

- How does changing $\lambda$ and/or $\alpha$ affect the result

### $\lambda$ selection

Use cross-validation using K-fold from `MLBase`.
