module ConvexFit

using LinearAlgebra: mul!, dot
using Printf

import Base: ==, show, push!, getindex

export convexfit,

       ModelSelection,
       GridSearch,
       grid,
       OptimSearch,
       optim,
       ModelSelectionResult,
       GridSearchResult,
       loocv

if VERSION < v"1.3"
    import LinearAlgebra: mul!
    using LinearAlgebra: lmul!
    function mul!(C::Vector, A::AbstractMatrix, B::Vector, α::Bool, β::Real)
        lmul!(β, C)
        C .= C + A*B
    end
end

include("utils.jl")
include("result.jl")
include("fit.jl")
include("select.jl")

end # module
