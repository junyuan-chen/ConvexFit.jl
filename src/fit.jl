struct Cache{TF<:AbstractFloat}
    Ax::Vector{TF}
    resid::Vector{TF}
    dresid::Vector{TF}
    g::Vector{TF}
    dx::Vector{TF}
end

function Cache(A::AbstractMatrix{TF}) where TF<:AbstractFloat
    M, N = size(A)
    Ax = Vector{TF}(undef, M)
    resid = Vector{TF}(undef, M)
    dresid = Vector{TF}(undef, M)
    g = Vector{TF}(undef, N)
    dx = Vector{TF}(undef, N)
    return Cache(Ax, resid, dresid, g, dx)
end

function updaterss!(A, b, x, ca)
    mul!(ca.Ax, A, x)
    @inbounds for i in eachindex(ca.resid)
        ca.resid[i] = ca.Ax[i] - b[i]
    end
    return sum(abs2, ca.resid)
end

function direction!(A, x, λ, ca)
    copyto!(ca.g, x)
    mul!(ca.g, A', ca.resid, true, λ)
    gmin, ix = findmin(ca.g)
    mul!(ca.dx, -one(eltype(A)), x)
    ca.dx[ix] += one(eltype(x))
    return ix
end

function stepsize!(ix, A, λ, ca)
    @inbounds for i in eachindex(ca.dresid)
        ca.dresid[i] = A[i,ix] - ca.Ax[i]
    end
    ss = -dot(ca.g, ca.dx) / (sum(abs2,ca.dresid) + λ*sum(abs2,ca.dx))
    TF = eltype(A)
    return min(one(TF), max(zero(TF), ss))
end

function updatex!(ss, x, ca)
    @inbounds for i in eachindex(x)
        x[i] += ss*ca.dx[i]
    end
end

function convexfit(A::Matrix{TF}, b::Vector{TF}, λ::Real=zero(TF);
        x0::AbstractVector{<:Real}=fill(convert(TF,1/size(A,2)), size(A,2)),
        ftol::Real=convert(TF, 1e-6),
        xtol::Real=convert(TF, 1e-6),
        maxiter::Integer=1000,
        store_trace::Bool=false,
        show_trace::Bool=false,
        show_thread::Bool=false,
        cache::Cache=Cache(A)) where TF<:AbstractFloat

    sum(x0) == 1 || throw(ArgumentError("elements of x0 do not sum up to 1"))
    λ = convert(TF, λ)
    λ < 0 && throw(ArgumentError("regularization parameter cannot be negative"))

    f_converged = false
    x_converged = false
    rss = -one(TF)
    xnorm = -one(TF)
    flast = -one(TF)
    f = -one(TF)
    x = Vector{TF}(undef, length(x0))
    copyto!(x, x0)
    dfnorm = convert(TF, NaN)
    dxnorm = convert(TF, NaN)
    iter = 0
    if store_trace || show_trace
        tr = SolverTrace(TF)
    else
        tr = nothing
    end

    for _ in 1:maxiter
        flast = f
        rss = updaterss!(A, b, x, cache)
        xnorm = sum(abs2, x)
        f = rss + λ * xnorm
        dfnorm = abs(f-flast)
        if dfnorm < ftol
            f_converged = true
            break
        elseif x_converged
            break
        elseif iszero(f)
            f_converged = true
            x_converged = true
            dfnorm = zero(TF)
            dxnorm = zero(TF)
            break
        end
        ix = direction!(A, x, λ, cache)
        if iszero(cache.dx[ix])
            f_converged = true
            x_converged = true
            dxnorm = zero(TF)
            break
        end
        ss = stepsize!(ix, A, λ, cache)
        dxnorm = ss*maximum(abs, cache.dx)
        if dxnorm < xtol
            x_converged = true
            # Do not immediately break to allow checking f_converged with the new x
        end
        updatex!(ss, x, cache)
        # Ensure that iter always represents the number of steps from x0r
        iter += 1
        # xnorm should not be updated until the next iteration
        update!(tr, iter, f, rss, xnorm, ix, ss, store_trace, show_trace, show_thread)
    end
    return SolverResult{TF}(x0, x, copy(cache.Ax), iter, f, rss, λ, xnorm,
        dfnorm, ftol, f_converged, dxnorm, xtol, x_converged, tr)
end

function convexfit(A::AbstractMatrix, b::AbstractVecOrMat, λ=0.0; kwargs...)
    TF = promote_type(eltype(A), eltype(b))
    TF <: Integer && (TF = Float64)
    A = convert(Matrix{TF}, A)
    b = convert(Array{TF}, b)
    x0 = get(kwargs, :x0, nothing)
    if x0 !== nothing
        x0 = convert(Vector{TF}, x0)
        return convexfit(A, b, λ; x0=x0, kwargs...)
    else
        return convexfit(A, b, λ; kwargs...)
    end
end

function convexfit(A::Matrix{TF}, B::Matrix{TF}, λ=0.0;
        multithreads::Bool=false, show_thread::Bool=multithreads, cache=Cache(A),
        kwargs...) where TF<:AbstractFloat
    N = size(B, 2)
    rs = Vector{SolverResult}(undef, N)
    if multithreads
        Threads.@threads for i in axes(B, 2)
            # Create new Cache for each call no matter whether cache is specified
            rs[i] = convexfit(A, view(B,:,i), λ;
                cache=Cache(A), show_thread=show_thread, kwargs...)
        end
    else
        for i in axes(B, 2)
            get(kwargs, :show_trace, false) && println("target $i of $N:")
            rs[i] = convexfit(A, view(B,:,i), λ;
                cache=cache, show_thread=show_thread, kwargs...)
        end
    end
    return rs
end

function convexfit(A::AbstractMatrix, λ=0.0; kwargs...)
    TF = eltype(A)
    TF <: Integer && (TF = Float64)
    A = convert(Matrix{TF}, A)
    x0 = get(kwargs, :x0, nothing)
    if x0 !== nothing
        x0 = convert(Vector{TF}, x0)
        return convexfit(A, λ; x0=x0, kwargs...)
    else
        return convexfit(A, λ; kwargs...)
    end
end

function convexfit(A::Matrix{TF}, λ=0.0;
        loo::Bool=true, multithreads::Bool=false, show_thread::Bool=multithreads,
        kwargs...) where TF<:AbstractFloat
    if loo
        N = size(A, 2)
        N > 1 || throw(ArgumentError("matrix A must contain at least two columns if loo=true"))
        rs = Vector{SolverResult}(undef, N)
        if multithreads
            Threads.@threads for i in axes(A, 2)
                Aloo = view(A,:,axes(A,2).!=i)
                # Create new Cache for each call no matter whether cache is specified
                rs[i] = convexfit(Aloo, view(A,:,i), λ;
                    cache=Cache(Aloo), show_thread=show_thread, kwargs...)
            end
        else
            ca = get(kwargs, :cache, nothing)
            ca === nothing && (ca = Cache(view(A,:,2:N)))
            for i in axes(A, 2)
                get(kwargs, :show_trace, false) && println("target $i of $N:")
                rs[i] = convexfit(view(A,:,axes(A,2).!=i), view(A,:,i), λ;
                    cache=ca, show_thread=show_thread, kwargs...)
            end
        end
        return rs
    else
        ca = get(kwargs, :cache, nothing)
        ca === nothing && (ca = Cache(A))
        return convexfit(A, A, λ; cache=ca, kwargs...)
    end
end
