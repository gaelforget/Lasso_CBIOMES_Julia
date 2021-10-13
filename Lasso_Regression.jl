### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ a7e00cab-9287-4912-a34d-e19bb570552f
using Pkg; Pkg.activate(".") # Use local Project.toml/Environment

# ╔═╡ 8b6e08f0-520a-44fc-bf49-1503dd2016ed
 begin
	using DataFrames, CSV # Data
	using GLM # Standard Regression
	using Lasso, GLMNet # Penalised regression
	using MLBase # Required for Lasso.jl cross-validation
	using CairoMakie # Plotting
	using Statistics, LinearAlgebra # Analysis tools
	using PlutoUI, BenchmarkTools # Some utilities
end

# ╔═╡ 4717f0ba-2b75-46de-bd3e-78ea0141a7dd
# Quick function for adding line between fit and data
function find_line(x, y)
	x̄ = mean(x); ȳ = mean(y);
	σ_x = std(x); σ_y = std(y);
	r = cor(x, y);
	
	a = r * σ_y/σ_x;
	b = ȳ - a*x̄;
	return a, b
end

# ╔═╡ 07f3e0f8-2c25-11ec-1ad3-63c909b28ea3
md"""
# Penalised Regression in Julia

I have created a small dataset from the historical Narragansett from [Nabats.org](https://www.nabats.org/). I have taken a subset of the data and will try to predict salinity from a selection of other covariates.

## 0. The data

**Note of advice!** Using `CSV.read(file, DataFrame)` sets columns to `SentinelArrays.ChainedVector{Float64, Vector{Float64}}`, for the models we need each column to just be `Vector{Float64}`. To solve this use `CSV.File(file) |> DataFrame`.
"""

# ╔═╡ 580d6b3a-95c4-4296-8c1e-25e93a993477
begin
	file = "./Narragansett_Environmental.csv";
	df = CSV.File(file) |> DataFrame;
	describe(df)
end

# ╔═╡ 8b85abb5-167f-4e1c-9730-517a81d8a560
hist(df.salinity_surface_psu,bins=25)

# ╔═╡ f1a36f60-616b-4667-baad-16715d4622a6
md"""
If our model is $y\sim 1 + \hat{X}$, where $y$ is an $n$-dimensional response vector to predictors $\hat{X}$ which is an $n\times p$ matrix containing $p$ predictors, we are trying to find the arguments/coeficients $\beta = \left[\beta_1, ... \beta_p \right]$ and $\beta_0$ that minimises:

$$\frac{1}{n}\|y - \beta_0 - \hat{X}\beta\|_2^2$$
"""

# ╔═╡ da9f2f14-ddf6-4306-a75f-be7937c2302e
# string for formula!
begin
	modelstr = uppercasefirst(replace(names(df)[1],"_"=>" ")) *  " ~ Intercept";
	for n in names(df)[2:end]
		global modelstr *= " + " * uppercasefirst(replace(n,"_"=>" "))
	end
end		

# ╔═╡ 0ef97167-55f1-462f-9817-fe6a076136c7
md"""
This selection of data isn't necessarily appropriate for **linear** regression but it's a quick example and should hopefilly demonstrate penalised regression effectively.

## 1. Linear Regression

We'll start with a quick linear regression for context. We will use the following linear model:

---

$modelstr

---

We use the `GLM` package. It can either take a `@formula` or `Matrix`, `Vector` pair to define the model. We can constuct a formula programatically using `term`, this requires a symbol for each term.

"""

# ╔═╡ 6a751311-0743-464b-84bf-980e2c712e2d
begin
	terms = term.(names(df))
	form = terms[1] ~ sum(terms[2:end]) # no intercept (required for Lasso.jl)
	form_1 = terms[1] ~ term(1) + sum(terms[2:end]) # Include intercept
end

# ╔═╡ 93f53604-984e-46c7-a518-49048a50088d
lr = lm(
	form_1, 
	df
)

# ╔═╡ 8134dba1-b085-42e6-b27d-f144b9ec626d
md"""
### Predicted Salinity
"""

# ╔═╡ 12e0575f-b3c4-4d93-84cc-8a514356f417
lr_pred = GLM.predict(lr)

# ╔═╡ ca8c3d50-1ce8-444e-b073-022532b1be96
norm(df.salinity_surface_psu - lr_pred)

# ╔═╡ 5c394294-2833-4441-b18b-07fee7768a1c
cor(df.salinity_surface_psu, lr_pred)

# ╔═╡ 278ac3fd-8d0c-4fb0-9826-bf2af4be9a84
begin
	lr_min, lr_max = extrema(vcat(lr_pred, df.salinity_surface_psu))
	lr_pts = lr_min:0.1:lr_max
	lr_a, lr_b = find_line(df.salinity_surface_psu, lr_pred)
end

# ╔═╡ e9387f27-e640-48e1-a646-48ad0b2c6dbd
begin
	F_lr = Figure()
	Ax_lr = Axis(F_lr[1,1], ylabel="Fit", xlabel="Data", aspect=1)
	
	scatter!(Ax_lr, df.salinity_surface_psu, lr_pred)
	
	lines!(Ax_lr, lr_pts, lr_pts, color=:red, linestyle=:dash)
	lines!(Ax_lr, lr_pts, lr_b .+ lr_a.*lr_pts, color=:black)
	
	F_lr
end

# ╔═╡ f51b1c16-9df7-4e57-9ae1-978efee7a9c7
md"""
## 2. Lasso Regression

The aim of Lasso regression is to reduce the dimensionality of our regression. In that it only returns coeficients for predictors that are significant for the response. 

It does this by applying a penalty constraint to the coeficients, we solve:

$$\underset{\beta_0,\beta}{\text{argmin}}\left(\frac{1}{n}\|y - \beta_0 - \hat{X}\beta\|_2^2\right),$$

subject to

$$\sum_{i=1}^p \beta_i = \|\beta\|_1 \leq t.$$

This can be rewritten in *Lagrangian* form as

$$\underset{\beta_0, \beta}{\text{argmin}}\left(\frac{1}{n}\|y - \beta_0 - \hat{X}\beta\|_2^2 + \lambda\|\beta\|_1\right).$$

We can use either `Lasso.jl` or `GLMNet.jl` to solve this for a given λ or *path* of λs. Both packages use coordinate descent to find the coeficients. `GLMNet` is a wrapper for a fortran library (similar packages for R and python) whereas `Lasso.jl` is a fully Julia solution.

Both packages automatically add an **unpenalised** intercept term.

First we'll see results from for different values of λ. `Lasso.jl` can take a predefined formula wheras `GLMNet` needs a predictor `Matrix` and response `Vector`.
"""

# ╔═╡ ed4b03e1-72bc-4d0c-82c6-ce710c471d6c
@bind λ PlutoUI.Slider(0.0:0.01:0.5)

# ╔═╡ 7df3a5e5-888b-410a-bff2-51ce194d20fd
"λ = $λ"

# ╔═╡ 719eb86f-8f15-43aa-a88b-ff77ee6315e5
las1 = Lasso.coef(fit(LassoModel, form, df, λ=[λ]))[2:end]		

# ╔═╡ 3c0265b0-8b7e-4db8-9a50-25e0014c418f
glmnet1 = glmnet(Matrix(df[:,2:6]), df.salinity_surface_psu, lambda=float.([λ])).betas[:]

# ╔═╡ 3bd52b3d-22ae-4111-8d96-8343998b03db
md"""
As λ increases some of the coeficients become 0.

For `Lasso.jl` if we just want the best result we can use `LassoModel` as above. If we want to see the results for all λ in the path we need to use `LassoPath`

Let's see this in a plot
"""

# ╔═╡ c4f383af-6122-4668-a016-7078f415b18b
las_test = fit(LassoPath, form, df, λ=collect(0.01:0.01:0.3))

# ╔═╡ b641931a-7d6a-4872-946a-565914d0ae4d
las_test.model.λ

# ╔═╡ 52e099f3-3582-43e0-8079-07d49ed084a6
begin
	F_path = Figure()
	Ax_path = Axis(F_path[1,1], xlabel = "λ")
	
	for i = 2:6
		lines!(Ax_path, las_test.model.λ, Lasso.coef(las_test)[i,:])
	end
	
	F_path
end

# ╔═╡ 1aed4dad-ed9f-4b67-bf66-2891fcfcc8eb
md"""
The best λ is automatically chosen using `MinAIC`. We can change the selection criteria. One of the main uses of `GLMNet` is with cross-validation. This is fully built into `GLMNet` as it`s all done in fortran. For `Lasso.jl` we need to load `MLBase`.

### Cross-Validation

We can let `Lasso.jl` and `GLMNet` choose their own λ-path and select using cross-validation. Here the data is repeatedly split into training and testing data. I'll use `KFold(k)`. This splits the shuffled data into k equally sized subgroups, k-1 subgroups are used to train the model while the remanining group is used to test the model. This is repeated k times. If k=n (the number of samples) this is called *Leave-one-out* cross-validation.

We will split the data into 10 subgroups.
"""

# ╔═╡ f8da630b-4d1e-4891-bf12-adbfa811b456
begin
	n = nrow(df)
	k = 10
end;

# ╔═╡ 98b1f658-1960-4d2f-9510-d101500766b4
lasso = fit(LassoModel, form, df; select=MinCVmse(Kfold(n,k)))

# ╔═╡ cea0a9f8-c3aa-48cc-b981-d3bbde4fef33
glmnet_cv = glmnetcv(Matrix(df[:,2:6]), df.salinity_surface_psu)

# ╔═╡ 3397d6fb-a20e-4f22-ad5f-34332cc7b1f0
glmnet_cv.path

# ╔═╡ db61d409-7b7a-4192-a454-9c74ff65009f
md"""
### Timing the two packages
"""

# ╔═╡ 14ed5aa9-3aee-4dd2-9022-8f3e0f3c37ea
# with_terminal() do
# 	@btime glmnetcv(Matrix(df[:,2:6]), df.salinity_surface_psu)
# 	@btime Lasso.selectmodel(fit(LassoPath, form, df).model,MinCVmse(Kfold(n,k)))
# end

# ╔═╡ 8486af06-3ebd-4bef-ad58-d62dd0d7e160
lasso_pred = Lasso.predict(lasso)

# ╔═╡ 4ca2f3ce-b47a-4267-9c5b-0e0fc0d8ad23
glmnet_cv_pred = GLMNet.predict(glmnet_cv, Matrix(df[:,2:6]))

# ╔═╡ 9abbc3ed-62aa-4810-aaeb-f6c4f09f2db9
[
	cor(lasso_pred, df.salinity_surface_psu) norm(lasso_pred- df.salinity_surface_psu) ;
	cor(glmnet_cv_pred, df.salinity_surface_psu) norm(glmnet_cv_pred- df.salinity_surface_psu) 
]

# ╔═╡ 3fb7d4c1-0a5b-425d-839c-2da1c17f1805
md"""
Little difference between the models. `GLMNet` runs faster, I tried to ensure we are using the same selection criteria but since I was not able to specify that in `GLMNet` I am not 100% sure.
"""

# ╔═╡ 06345b5d-a59a-4170-a486-ba770bcb948e
begin
	minpt, maxpt = extrema(vcat(vcat(lasso_pred, df.salinity_surface_psu),glmnet_cv_pred))
	pts = minpt:0.1:maxpt
	lasso_a, lasso_b = find_line(df.salinity_surface_psu, lasso_pred)
	glmnet_cv_a, glmnet_cv_b = find_line(df.salinity_surface_psu, glmnet_cv_pred)
end;

# ╔═╡ b0fa7cad-d72b-4704-8fb0-cbe2d380f6d9
begin
	F_cv = Figure()
	Ax_lasso = Axis(F_cv[1,1], aspect=1, ylabel = "Data", xlabel="Fit", title="Lasso.jl")
	
	scatter!(Ax_lasso, df.salinity_surface_psu, lasso_pred)
	
	lines!(Ax_lasso, pts, pts, color=:red, linestyle=:dash)
	lines!(Ax_lasso, pts, lasso_b .+ lasso_a.*pts, color=:black)
	
	
	Ax_glmnet = Axis(F_cv[1,2], aspect=1, ylabel = "Data", xlabel="Fit", title="GLMNet.jl")
	
	scatter!(Ax_glmnet, df.salinity_surface_psu, glmnet_cv_pred)
	
	lines!(Ax_glmnet, pts, pts, color=:red, linestyle=:dash)
	lines!(Ax_glmnet, pts, glmnet_cv_b .+ glmnet_cv_a.*pts, color=:black)
	
	F_cv
end

# ╔═╡ 3f98b661-557a-45bb-9585-b1b47c118b61
md"""
## Ridge Regression


"""

# ╔═╡ Cell order:
# ╠═a7e00cab-9287-4912-a34d-e19bb570552f
# ╠═8b6e08f0-520a-44fc-bf49-1503dd2016ed
# ╟─4717f0ba-2b75-46de-bd3e-78ea0141a7dd
# ╠═07f3e0f8-2c25-11ec-1ad3-63c909b28ea3
# ╠═580d6b3a-95c4-4296-8c1e-25e93a993477
# ╠═8b85abb5-167f-4e1c-9730-517a81d8a560
# ╟─0ef97167-55f1-462f-9817-fe6a076136c7
# ╠═f1a36f60-616b-4667-baad-16715d4622a6
# ╟─da9f2f14-ddf6-4306-a75f-be7937c2302e
# ╠═6a751311-0743-464b-84bf-980e2c712e2d
# ╠═93f53604-984e-46c7-a518-49048a50088d
# ╟─8134dba1-b085-42e6-b27d-f144b9ec626d
# ╠═12e0575f-b3c4-4d93-84cc-8a514356f417
# ╠═ca8c3d50-1ce8-444e-b073-022532b1be96
# ╠═5c394294-2833-4441-b18b-07fee7768a1c
# ╠═278ac3fd-8d0c-4fb0-9826-bf2af4be9a84
# ╠═e9387f27-e640-48e1-a646-48ad0b2c6dbd
# ╠═f51b1c16-9df7-4e57-9ae1-978efee7a9c7
# ╠═ed4b03e1-72bc-4d0c-82c6-ce710c471d6c
# ╠═7df3a5e5-888b-410a-bff2-51ce194d20fd
# ╠═719eb86f-8f15-43aa-a88b-ff77ee6315e5
# ╠═3c0265b0-8b7e-4db8-9a50-25e0014c418f
# ╠═3bd52b3d-22ae-4111-8d96-8343998b03db
# ╠═c4f383af-6122-4668-a016-7078f415b18b
# ╠═b641931a-7d6a-4872-946a-565914d0ae4d
# ╠═52e099f3-3582-43e0-8079-07d49ed084a6
# ╠═1aed4dad-ed9f-4b67-bf66-2891fcfcc8eb
# ╠═f8da630b-4d1e-4891-bf12-adbfa811b456
# ╠═98b1f658-1960-4d2f-9510-d101500766b4
# ╠═3397d6fb-a20e-4f22-ad5f-34332cc7b1f0
# ╠═cea0a9f8-c3aa-48cc-b981-d3bbde4fef33
# ╠═db61d409-7b7a-4192-a454-9c74ff65009f
# ╠═14ed5aa9-3aee-4dd2-9022-8f3e0f3c37ea
# ╠═8486af06-3ebd-4bef-ad58-d62dd0d7e160
# ╠═4ca2f3ce-b47a-4267-9c5b-0e0fc0d8ad23
# ╠═9abbc3ed-62aa-4810-aaeb-f6c4f09f2db9
# ╠═3fb7d4c1-0a5b-425d-839c-2da1c17f1805
# ╠═06345b5d-a59a-4170-a486-ba770bcb948e
# ╠═b0fa7cad-d72b-4704-8fb0-cbe2d380f6d9
# ╠═3f98b661-557a-45bb-9585-b1b47c118b61
