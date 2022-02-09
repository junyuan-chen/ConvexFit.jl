module ConvexFit

using LinearAlgebra: mul!, dot
using Printf
using Random: AbstractRNG, randperm!

import Base: ==, show, push!, getindex

export convexfit,

       ModelSelection,
       GridSearch,
       grid,
       OptimSearch,
       optim,
       ModelSelectionResult,
       GridSearchResult,
       kfoldcv,
       loocv

if VERSION < v"1.3"
    import LinearAlgebra: mul!
    using LinearAlgebra: lmul!
    function mul!(C::Vector, A::AbstractMatrix, B::Vector, α::Bool, β::Real)
        lmul!(β, C)
        C .= C + A*B
    end
    using Random: GLOBAL_RNG
    default_rng() = GLOBAL_RNG
else
    using Random: default_rng
end

include("utils.jl")
include("result.jl")
include("fit.jl")
include("select.jl")

end # module
