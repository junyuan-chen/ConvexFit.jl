module ConvexFit

using LinearAlgebra: mul!, dot
using Printf

import Base: show, push!, getindex

export convexfit

if VERSION < v"1.3"
    import LinearAlgebra: mul!
    using LinearAlgebra: lmul!
    function mul!(C::Vector, A::AbstractMatrix, B::Vector, α::Bool, β::Real)
        lmul!(β, C)
        C .= C + A*B
    end
end

include("result.jl")
include("fit.jl")

end # module
