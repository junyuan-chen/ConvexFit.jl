# ConvexFit.jl

*Fit vectors with convex combinations of data columns*

[![CI-stable][CI-stable-img]][CI-stable-url]
[![codecov][codecov-img]][codecov-url]
[![PkgEval][pkgeval-img]][pkgeval-url]

[CI-stable-img]: https://github.com/junyuan-chen/ConvexFit.jl/workflows/CI-stable/badge.svg
[CI-stable-url]: https://github.com/junyuan-chen/ConvexFit.jl/actions?query=workflow%3ACI-stable

[codecov-img]: https://codecov.io/gh/junyuan-chen/ConvexFit.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/junyuan-chen/ConvexFit.jl

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/C/ConvexFit.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/C/ConvexFit.html

[ConvexFit.jl](https://github.com/junyuan-chen/ConvexFit.jl)
is a lightweight Julia package for fitting vectors with convex combinations of data columns.
Notably, the coefficients are always restricted to be nonnegative and sum to one.
This restriction arises naturally in circumstances
where predictions involving extrapolation are undesirable,
for instance, when constructing weights for synthetic controls.

## The Optimization Problem

The coefficients for the convex combinations are obtained by
solving a constrained optimization problem of the following form:

<p align="center">
min<sub>x</sub> ||Ax - b||<sup>2</sup> + λ||x||<sup>2</sup> <br>
st. &nbsp x<sub>i</sub> &#8805 0 for all i and Σ<sub>i</sub>x<sub>i</sub> = 1
</p>

where `A` is a matrix containing the data columns;
`x` is a vector of coefficients on the unit simplex;
`b` is a vector to be fitted by `Ax`;
and `λ` is a nonnegative regularization parameter.
Only the Euclidean norm is supported at this moment.

The optimization problem is solved iteratively with a conditional gradient method,
the Frank-Wolfe algorithm,
that directly searches solution candidates on the unit simplex.
In practice, some extent of regularization is often desired
and that can be controlled by the magnitude of `λ`.
The choice of `λ` depends on the context of the specific problem
and is left to be zero by default.
If appropriate, one may consider selecting `λ` based on the leave-one-out cross validation,
which is implemented in this package.

## Quick Start

Most of the functionality can be accessed by calling `convexfit`.
To fit a vector `b` with a convex combination of columns in `A`
and regularize the coefficients `x` with some `λ`:

```julia
using ConvexFit
r = convexfit(A, b, λ)
```

The results are stored in `SolverResult` and can be retrieved from the corresponding fields.
For example, `r.sol` gives the optimal `x`; while `r.fit` gives the fitted values.

To select an optimal `λ` based on the leave-one-out cross validation,
one may either provide a `grid` in place of `λ` to `convexfit` for exhaustive search
or specify a solver that searches the optimal `λ` over an interval.
An example for the latter case that uses a solver from
[`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) is as follows:

```julia
using ConvexFit, Optim
# Specify a solver in a function that takes an objective function as argument
# The returned object must be a tuple of the solver result and the minimizer
function fmin(f::Function)
    r = optimize(f, 0.0, 100.0, Brent(), abs_tol=1e-4, store_trace=true)
    return r, minimizer(r)
end
# Fit b under the optimal λ selected from [0, 100] based on leave-one-out cross validation
r = convexfit(A, b, optim(fmin))
```

More details can be found in the
[help](https://docs.julialang.org/en/v1/stdlib/REPL/#Help-mode) mode of Julia REPL.

## Reference

**Jaggi, Martin.** 2013. "Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization."
*Proceedings of the 30th International Conference on Machine Learning* 28 (1): 427-435.
