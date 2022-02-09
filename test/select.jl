@testset "CVCache" begin
    ca1 = CVCache(Float32, 4, 3, 2)
    @test ca1.inds == 1:4
    @test eltype(ca1.Ain) == Float32
    @test size(ca1.Ain) == (2, 3)
    ca2 = LOOCVCache(Float32, 4, 3)
    @test eltype(ca2.bin) == Float32
    @test size(ca2.Ain) == (3, 3)

    @test_throws ArgumentError CVCache(Float64, 4, 2, 4)
    @test_throws ArgumentError CVCache(Float64, 4, 2, 0)
end

@testset "kfoldcv loocv" begin
    A = [1.0 1.0; 0.0 1.0]
    A = vcat(A, A)
    b = ones(4)
    cv = kfoldcv(A, b, 0, 2)
    @test cv == 0.125 || cv == 0
    ca = CVCache(Float64, 4, 2, 2)
    ca.inds .= [1,3,2,4]
    @test kfoldcv(A, b, 0, cvcache=ca, rng=nothing) == 0.125
    @test kfoldcv(A, b, 1.0, cvcache=ca, rng=nothing) == 0.125
    ca.inds .= 1:4
    @test kfoldcv(A, b, 0.5, cvcache=ca, rng=nothing) == 0.03125

    @test_throws ArgumentError kfoldcv(A, b, 0, 5)

    @test loocv(A, b, 0.5) == 0.03125
    ca = LOOCVCache(Float64, 4, 2)
    Ain1 = copy(ca.Ain)
    loocv(A, b, 0, cvcache=ca)
    @test Ain1 != ca.Ain
end

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
    @test all(r[2].cvs.==0.125)
    @test length(r[2].cvs) == 50

    ca = Cache(A)
    fill!(ca.Ax, 0.0)
    ca1 = deepcopy(ca)
    looca = Cache(view(A,1:1,:))
    fill!(looca.Ax, 0.0)
    looca1 = deepcopy(looca)
    r = convexfit(A, A, grid(), cache=ca, cvargs=(cache=looca,), x0=x0)
    @test ca.Ax != ca1.Ax
    @test looca.Ax == looca1.Ax
    @test r[1][1].x0 === x0

    if testthreads
        ca = Cache(A)
        fill!(ca.Ax, 0.0)
        ca1 = deepcopy(ca)
        looca = Cache(view(A,1:1,:))
        fill!(looca.Ax, 0.0)
        looca1 = deepcopy(looca)
        r = convexfit(A, A, grid(), cache=ca, cvargs=(cache=looca,), x0=x0, multithreads=true)
        @test ca.Ax == ca1.Ax
        @test looca.Ax == looca1.Ax
        @test r[1][1].x0 === x0
    end

    r = convexfit(A, grid())
    @test r[1][1].sol == [1.0]

    A = [1.0 1.0; 0.0 1.0]
    A = vcat(A, A)
    b = ones(4)
    r = convexfit(A, b, grid(), k=2)
    @test all(r[2].cvs.<=0.125)

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
    r = convexfit(A, A, optim(fmin), cache=ca, cvargs=(cache=looca,), x0=x0)
    @test ca.Ax != ca1.Ax
    @test looca.Ax == looca1.Ax
    @test r[1][1].x0 === x0

    r = convexfit(A, optim(fmin), k=2)
    @test r[1][1].sol == [1.0]

    @test_throws ArgumentError convexfit([1.0 2.0], [1.0], optim(fmin))
end
