using PeriodicSchurDecompositions

using Test
using LinearAlgebra
using Random
# needed to test exotic eltypes
using GenericSchur

# extra randomness is good for development but risks nasty surprises in CI
iseed = parse(Int,get(ENV,"PS_DEV","-1"))
developing = iseed >= 0
if iseed > 0
    @info "Using seed $iseed for PS test"
    Random.seed!(iseed)
elseif iseed == 0
    @info "Throwing dice for PS test"
else
    Random.seed!(1234)
end

# Can't test exotic types unless someone implements
# lmul!(HessenbergQ{T},Vector{T}), equiv. to LAPACK.ormhr!
# or Matrix(HessenbergQ{T}), equiv to LAPACK.orghr!
for T in [Float64,Complex{Float64}]
  @testset "Periodic Hessenberg $T" begin
    for p in [1,2,5]
        n = 5
        tol = 20
        qtol = 10
        A = [rand(T,n,n) for j in 1:p]
        Awrk = [copy(A[j]) for j in 1:p]
        H1,pH = phessenberg!(Awrk)
        if p==1
            Ax = [H1.Q * H1.H * H1.Q']
        else
            Ax = [H1.Q * H1.H * pH[1].Q']
        end
        Hj = [pH[j].R for j in 1:p-1]
        Qj = [pH[j].Q for j in 1:p-1]
        for j in 2:p-1
            push!(Ax,Qj[j-1]*Hj[j-1]*Qj[j]')
        end
        if p > 1
            push!(Ax,Qj[p-1]*Hj[p-1]*H1.Q')
        end
        # H = [Matrix(pH[j].H) for j in 1:p]
        # Q = [Matrix(pH[j].Q) for j in 1:p]
        # @test istriu(H[j],j==1 ? -1 : 0)
        @test istriu(H1.H,-1)
        @test norm(H1.Q * H1.Q' - I) < qtol * eps(real(T)) * n
        for j in 1:p-1
            @test istriu(Hj[j])
            @test norm(Qj[j]*Qj[j]' - I) < qtol * eps(real(T)) * n
        end
        for j in 1:p
            @test norm(A[j] - Ax[j]) < tol * eps(real(T)) * n
        end
    end
  end
end

# this is for a mixture of real and conjugate pairs, not general
function compare_eigvals(λ,λx,λtol)
    n = length(λ)
    idx = sortperm(abs.(λ))
    idxx = sortperm(abs.(λx))
    λscale = abs(λ[idx[end]])
    i = 1
    while i <= n
        λ1 = λ[idx[i]]
        λ1x = λx[idxx[i]]
        if isreal(λ1)
            @test isreal(λ1x)
            @test abs(λ1 - λ1x) < λtol * λscale
            i = i+1
        else
            λ2 = λ[idx[i+1]]
            λ2x = λx[idxx[i+1]]
            if imag(λ1) * imag(λ1x) < 0
                λ1x, λ2x = λ2x, λ1x
            end
            @test abs(λ1 - λ1x) < λtol * λscale
            @test abs(λ2 - λ2x) < λtol * λscale
            i = i+2
        end
    end
end

function pschur_test(A::AbstractVector{TM}) where {TM <: AbstractMatrix{T}} where T
    p = length(A)
    n = size(A[1],1)
    qtol = 10
    tol = 32
    λtol = 1000 # these are non-normal problems, so curb your expectations
    Awrk = [copy(A[j]) for j in 1:p]
    pS = pschur!(Awrk, wantZ=true)
    Ts = [Matrix(pS.T1)]
    for j in 1:p-1
        push!(Ts, Matrix(pS.T[j]))
    end
    Zs = pS.Z
    l1 = (p==1) ? 1 : 2
    Ax = [Zs[j]*Ts[j]*Zs[mod(j,p)+1]' for j in 1:p]
    for j in 1:p
        # TODO: use hard triu tests when we have logic to clear detritus
        #@test istriu(Ts[j], j==1 ? -1 : 0)
        @test norm(tril(Ts[j],j==1 ? -2 : -1)) < 100 * eps(real(T)) * n
        @test norm(Zs[j]*Zs[j]' - I) < qtol * eps(real(T)) * n
        if developing
            r = norm(A[j] - Ax[j]) / (eps(real(T)) * n)
            if r > tol
                @warn "residual[$j]: $r"
            elseif r > 20
                println("residual[$j]: $r")
            end
        else
            @test norm(A[j] - Ax[j]) < tol * eps(real(T)) * n
        end
    end
    if p > 1
        Aprod = foldl(*,A)
    else
        Aprod = copy(A[1])
    end
    λ = eigvals(Aprod)

    λx = pS.values
    compare_eigvals(λ,λx,λtol*eps(real(T)))
end

# "easy" case: trivial p.Hessenberg reduction
for T in [Float64]
  @testset "Periodic Schur Hess+UT $T" begin
   for p in [1,2,3,5]
        n = 5
        A = [triu(rand(T,n,n)) for j in 1:p]
        A[1] = triu(rand(T,n,n),-1) # Hessenberg
        pschur_test(A)
    end
  end
end

for T in [Float64, BigFloat]
  @testset "Periodic Schur full $T" begin
    for p in [1,2,3,5]
        n = 5
        A = [rand(T,n,n) for j in 1:p]
        pschur_test(A)
    end
  end
end

# make sure quick versions run
@testset "Periodic Schur fast paths (real)" begin
    n = 5
    tol = 20
    qtol = 10
    λtol = 1000
    T = Float64
    for p in [1,5]
        A = [rand(n,n) for j in 1:p]
        Awrk = [copy(A[j]) for j in 1:p]
        pS2 = pschur!(Awrk, wantZ=true)
        λ = pS2.values

        Awrk = [copy(A[j]) for j in 1:p]
        pS0 = pschur!(Awrk, wantT=false, wantZ=false)
        λx = pS0.values
        # currently we leave the mangled matrices in the result, so no check on Ts
        @test (length(pS0.Z) == 0) || (size(pS0.Z[1],1) == 0)
        compare_eigvals(λ,λx,λtol*eps(real(T)))

        Awrk = [copy(A[j]) for j in 1:p]
        pS1 = pschur!(Awrk, wantT=true, wantZ=false)
        λx = pS1.values
        @test (length(pS1.Z) == 0) || (size(pS1.Z[1],1) == 0)
        @test norm(pS1.T1 - pS2.T1) < tol * eps(real(T)) * n
        for j in 2:p-1
            @test norm(pS1.T[j] - pS2.T[j]) < tol * eps(real(T)) * n
        end
        compare_eigvals(λ,λx,λtol*eps(real(T)))
    end
end

include("generalized.jl") # temporarily up front
