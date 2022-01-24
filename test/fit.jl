@testset "trace" begin
    stpara = (12345.6, 10.123, 10.0, 5, 1.2)
    st = SolverState(1, stpara...)
    @test sprint(show, st) == """
         iter=1         f=1.23e+04  rss=10.1      xnorm=10        idx=5         step=1.2     
        """
    @test sprint(show, st, true) == """
         thread  1:  iter=1         f=1.23e+04  rss=10.1      xnorm=10        idx=5         step=1.2     
        """
    tr = SolverTrace(Float64)
    update!(tr, 1, stpara..., true, true, true)
    @test length(tr.states) == 1
    @test tr.states[1] == st
    update!(tr, 2, stpara..., true, false, true)
    @test length(tr.states) == 2
    @test tr[1] == st
    @test sprint(show, tr) == """
         iter=1         f=1.23e+04  rss=10.1      xnorm=10        idx=5         step=1.2     
         iter=2         f=1.23e+04  rss=10.1      xnorm=10        idx=5         step=1.2     
        """
    update!(tr, 3, stpara..., false, false, false)
    @test length(tr.states) == 2
    @test update!(nothing, 4, stpara..., false, false, false) === nothing
end

@testset "fit" begin
    A = [1.0 0.0; 0.0 1.0]
    b = [1.0, 1.0]
    x0 = [1.0, 0.0]
    ca = Cache(A)
    r = convexfit(A, b, x0=x0, show_trace=true, store_trace=true, cache=ca)
    @test r.x0 === x0
    @test r.sol == [0.5, 0.5]
    @test r.fit == [0.5, 0.5]
    @test r.iter == 2
    @test r.f == 0.5
    @test r.rss == 0.5
    @test r.λ == 0
    @test r.xnorm == 0.5
    @test r.dfnorm == 0
    @test r.dxnorm == 0
    @test r.f_converged && r.x_converged
    @test sprint(show, r) == "SolverResult(converged=true, iter=2, rss=0.5, λ=0.0, xnorm=0.5)"
    @test sprint(show, MIME("text/plain"), r) == """
        convexfit converged after 2 iterations:
         |f(x) - f(x')| = 0.0e+00 < 1.0e-06: true
         |x - x'| = 0.0e+00 < 1.0e-06: true
         f=0.5       rss=0.5       λ=0         xnorm=0.5     """
    @test sprint(show, r.trace) == """
         iter=1         f=1         rss=1         xnorm=1         idx=2         step=0.5     
         iter=2         f=0.5       rss=0.5       xnorm=0.5       idx=1         step=0       
        """

    r = convexfit(A, b, x0=[1,0], show_trace=true)
    @test r.x0 == [1.0, 0.0]
    r = convexfit(A, b, x0=view(A,:,1))
    @test r.x0 == [1.0, 0.0]

    A = [1 0 1; 0 1 1]
    b = [1, 1]
    ca = Cache(convert(Matrix{Float64}, A))
    r = convexfit(A, b, show_trace=true, cache=ca)
    @test all(r.x0.==1/3)
    @test r.sol[3] ≈ 1
    @test r.fit == b
    @test r.iter == 1
    @test r.f == r.rss ≈ 0
    @test r.dfnorm ≈ 0
    @test r.dxnorm ≈ 0
    @test r.f_converged && r.x_converged

    r = convexfit(A, b, 0.5, xtol=0, ftol=eps(Float64), show_trace=true)
    @test r.sol ≈ [0.2, 0.2, 0.6]
    @test r.f ≈ 0.3

    r = convexfit(A, A, 1)
    @test r[1].sol[1] ≈ r[2].sol[2]
    @test r[1].fit[1] ≈ r[2].fit[2]
    @test r[3].sol ≈ [0.25, 0.25, 0.5]
    @test r[3].f ≈ 0.5

    r1 = convexfit(A, A, 1, multithreads=false)
    @test r1 == r

    r = convexfit(A, 0.5, show_trace=true)
    @test length(r) == 3
    @test r[1].sol ≈ [0.25, 0.75]
    @test r[2].sol ≈ [0.25, 0.75]
    @test r[3].sol ≈ [0.5, 0.5]

    r1 = convexfit(A, 0.5, show_trace=true, multithreads=false)
    @test r1 == r

    r = convexfit(A, loo=false, 0.5)
    @test length(r) == 3
    @test r[3].sol ≈ [0.2, 0.2, 0.6]

    r1 = convexfit(A, loo=false, 0.5, multithreads=false)
    @test r1 == r

    @test_throws ArgumentError convexfit(A, b, x0=[2.0,0.0])
    @test_throws ArgumentError convexfit(A, b, -1.0)
    @test_throws ArgumentError convexfit(reshape([1.0,0.0],2,1))
end
