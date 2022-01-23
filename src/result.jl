struct SolverState{TF<:AbstractFloat}
    iter::Int
    f::TF
    rss::TF
    xnorm::TF
    idx::Int
    step::TF
end

struct SolverTrace{TF<:AbstractFloat}
    states::Vector{SolverState{TF}}
end

SolverTrace(TF::Type) = SolverTrace(Vector{SolverState{TF}}(undef, 0))

function show(io::IO, st::SolverState)
    @printf io " iter=%-8d  f=%-8.3g  rss=%-8.3g  xnorm=%-8.3g  idx=%-8d  step=%-8.3g\n" st.iter st.f st.rss st.xnorm st.idx st.step
end

push!(tr::SolverTrace, st::SolverState) = push!(tr.states, st)

getindex(tr::SolverTrace, i::Integer) = getindex(tr.states, i)

function show(io::IO, tr::SolverTrace)
    for state in tr.states
        print(io, state)
    end
end

function update!(tr::SolverTrace, iter, f, rss, xnorm, idx, step,
        store_trace::Bool, show_trace::Bool)
    st = SolverState(iter, f, rss, xnorm, idx, step)
    store_trace && push!(tr, st)
    show_trace && print(st)
    return tr
end

update!(::Nothing, args...) = nothing

struct SolverResult{TF<:AbstractFloat}
    x0::Vector{TF}
    sol::Vector{TF}
    fit::Vector{TF}
    iter::Int
    f::TF
    rss::TF
    λ::TF
    xnorm::TF
    dfnorm::TF
    ftol::TF
    f_converged::Bool
    dxnorm::TF
    xtol::TF
    x_converged::Bool
    trace::Union{SolverTrace{TF},Nothing}
end

converged(r::SolverResult) = r.f_converged || r.x_converged

function show(io::IO, ::MIME"text/plain", r::SolverResult)
    print(io, "convexfit ", converged(r) ? "converged " : "did not converge ")
    print(io, "after ", r.iter, " iteration")
    r.iter > 1 && print(io, "s")
    @printf io ":\n |f(x) - f(x')| = %.1e < %.1e: %s\n" r.dfnorm r.ftol r.f_converged
    @printf io " |x - x'| = %.1e < %.1e: %s\n" r.dxnorm r.xtol r.x_converged
    @printf io " f=%-8.3g  rss=%-8.3g  λ=%-8.3g  xnorm=%-8.3g" r.f r.rss r.λ r.xnorm
end
