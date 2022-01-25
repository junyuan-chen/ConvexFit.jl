@testset "GridSearch" begin
    @test grid(1) == GridSearch([1.0])
    @test grid(1:3) == GridSearch([1.0, 2.0, 3.0])
    @test grid([2.0, 0.0]) == GridSearch([2.0, 0.0])
    @test grid().v == exp.(LinRange(-6, 6, 50))
    @test_throws ArgumentError grid([2.0, -1.0])

    s = grid()
    @test sprint(show, s) == "GridSearch"
    @test sprint(show, MIME("text/plain"), s)[1:64] == """
        GridSearch{Float64} across 50 candidate values:
          [0.00247875, 0"""

    sr = GridSearchResult(2, rand(2))
    @test sprint(show, sr) == "GridSearchResult"
    @test sprint(show, MIME("text/plain"), sr) == """
        GridSearchResult across 2 candidate values: 2"""

    A = [1.0 1.0; 0.0 1.0]
    b = [1.0, 1.0]
    x0 = [1.0, 0.0]
    r = convexfit(A, b, grid(), x0=x0)
    @test r[1].f_converged
    @test r[1].λ == exp(-6)
    @test r[2].iopt == 1
    @test all(r[2].loocv.==0.125)
    @test length(r[2].loocv) == 50

    ca = Cache(A)
    fill!(ca.Ax, 0.0)
    ca1 = deepcopy(ca)
    looca = Cache(view(A,1:1,:))
    fill!(looca.Ax, 0.0)
    looca1 = deepcopy(looca)
    r = convexfit(A, A, grid(), cache=ca, looargs=(cache=looca,), x0=x0)
    @test ca.Ax != ca1.Ax
    @test looca.Ax != looca1.Ax
    @test r[1][1].x0 === x0

    if VERSION >= v"1.2"
        ca = Cache(A)
        fill!(ca.Ax, 0.0)
        ca1 = deepcopy(ca)
        looca = Cache(view(A,1:1,:))
        fill!(looca.Ax, 0.0)
        looca1 = deepcopy(looca)
        r = convexfit(A, A, grid(), cache=ca, looargs=(cache=looca,), x0=x0, multithreads=true)
        @test ca.Ax == ca1.Ax
        # looca is not handled by the wrapper
        @test looca.Ax != looca1.Ax
        @test r[1][1].x0 === x0
    end

    r = convexfit(A, grid())
    @test r[1][1].sol == [1.0]

    @test_throws ArgumentError convexfit([1.0 2.0], [1.0], grid())
end

@testset "OptimSearch" begin
    s = optim(fmin)
    @test sprint(show, s) == "OptimSearch"
    @test sprint(show, MIME("text/plain"), s) == """
        OptimSearch:
          fmin (generic function with 1 method)"""

    A = [1.0 1.0; 0.0 1.0]
    b = [1.0, 1.0]
    x0 = [1.0, 0.0]
    r = convexfit(A, b, optim(fmin), x0=x0)
    @test r[1].f_converged
    @test minimum(r[2]) == 0.125

    ca = Cache(A)
    fill!(ca.Ax, 0.0)
    ca1 = deepcopy(ca)
    looca = Cache(view(A,1:1,:))
    fill!(looca.Ax, 0.0)
    looca1 = deepcopy(looca)
    r = convexfit(A, A, optim(fmin), cache=ca, looargs=(cache=looca,), x0=x0)
    @test ca.Ax != ca1.Ax
    @test looca.Ax != looca1.Ax
    @test r[1][1].x0 === x0

    r = convexfit(A, optim(fmin))
    @test r[1][1].sol == [1.0]

    @test_throws ArgumentError convexfit([1.0 2.0], [1.0], optim(fmin))
end
