using Test
using ConvexFit

using ConvexFit: SolverState, SolverTrace, update!, Cache, CVCache, LOOCVCache
using Optim: Brent, optimize

# Threads.@threads may cause the CI to freeze on Windows in older versions
const testthreads = VERSION >= v"1.2" && !Sys.iswindows() || VERSION >= v"1.7"

function fmin(f::Function)
    r = optimize(f, 0.0, 100.0, Brent(), abs_tol=1e-4, store_trace=true)
    return r, r.minimizer
end

const tests = [
    "fit",
    "select"
]

printstyled("Running tests:\n", color=:blue, bold=true)

@time for test in tests
    include("$test.jl")
    println("\033[1m\033[32mPASSED\033[0m: $(test)")
end
