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

@fieldequal SolverTrace

SolverTrace(TF::Type) = SolverTrace(Vector{SolverState{TF}}(undef, 0))

function show(io::IO, st::SolverState, show_thread::Bool=false)
    if show_thread
        @printf io " thread %2d:  iter=%-8d  f=%-8.3g  rss=%-8.3g  xnorm=%-8.3g  idx=%-8d  step=%-8.3g\n" Threads.threadid() st.iter st.f st.rss st.xnorm st.idx st.step
    else
        @printf io " iter=%-8d  f=%-8.3g  rss=%-8.3g  xnorm=%-8.3g  idx=%-8d  step=%-8.3g\n" st.iter st.f st.rss st.xnorm st.idx st.step
    end
end

push!(tr::SolverTrace, st::SolverState) = push!(tr.states, st)

getindex(tr::SolverTrace, i::Integer) = getindex(tr.states, i)

function show(io::IO, tr::SolverTrace)
    for state in tr.states
        print(io, state)
    end
end

function update!(tr::SolverTrace, iter, f, rss, xnorm, idx, step,
        store_trace::Bool, show_trace::Bool, show_thread::Bool)
    st = SolverState(iter, f, rss, xnorm, idx, step)
    store_trace && push!(tr, st)
    show_trace && show(stdout, st, show_thread)
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

@fieldequal SolverResult

converged(r::SolverResult) = r.f_converged || r.x_converged

function show(io::IO, r::SolverResult)
    print(io, typeof(r).name.name, "(converged=", converged(r), ", iter=$(r.iter)")
    print(IOContext(io, :compact=>true), ", rss=$(r.rss), λ=$(r.λ), xnorm=$(r.xnorm))")
end

function show(io::IO, ::MIME"text/plain", r::SolverResult)
    print(io, "convexfit ", converged(r) ? "converged " : "did not converge ")
    print(io, "after ", r.iter, " iteration")
    r.iter > 1 && print(io, "s")
    @printf io ":\n |f(x) - f(x')| = %.1e < %.1e: %s\n" r.dfnorm r.ftol r.f_converged
    @printf io " |x - x'| = %.1e < %.1e: %s\n" r.dxnorm r.xtol r.x_converged
    @printf io " f=%-8.3g  rss=%-8.3g  λ=%-8.3g  xnorm=%-8.3g" r.f r.rss r.λ r.xnorm
end
