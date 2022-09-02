using PeriodicSchurDecompositions

using Test
using LinearAlgebra
using Random
# needed to test exotic eltypes
using GenericSchur

include("testfuncs.jl")

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

# "easy" case: trivial p.Hessenberg reduction
for T in [Float64]
  @testset "Periodic Schur Hess+UT $T" begin
   for p in [1,2,3,5]
       n = 5
       A = [triu(rand(T,n,n)) for j in 1:p]
       A[1] = triu(rand(T,n,n),-1) # Hessenberg
       pschur_test(A)
       if p > 1
           A[1],A[p] = A[p],A[1]
       end
       pschur_test(A, left=true)
    end
  end
end

for T in [Float64, ComplexF64]
  @testset "Periodic Schur exp. split Hess+UT $T" begin
   for p in [5,20]
       A,λ = expsplit(p,T)
       Awrk = deepcopy(A)
       ps  = pschur!(Awrk, :R, wantZ=true)
       pschur_check(A, ps, checkλ=false, tol=128)
       λs = ps.values
       for λj in λ
           dmin, idmin = findmin(abs.(λs .- λj))
           developing && println("ES $p $T λ=$λj min err: $dmin")
           @test (dmin .< 0.001 * abs(λj)) || (max(abs(λj),abs(λs[idmin])) < eps(real(T))^2)
       end
       A[1], A[p] = A[p], A[1]
       Awrk = deepcopy(A)
       ps  = pschur!(Awrk, :L, wantZ=true)
       pschur_check(A, ps, checkλ=false, tol=128)
    end
  end
end

for T in [Float64, BigFloat]
  @testset "Periodic Schur full $T" begin
    for p in [1,2,3,5]
        n = 5
        A = [rand(T,n,n) for j in 1:p]
        pschur_test(A)
        if T == Float64
            pschur_test(A, left=true)
        end
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
        compare_reigvals(λ,λx,λtol*eps(real(T)))

        Awrk = [copy(A[j]) for j in 1:p]
        pS1 = pschur!(Awrk, wantT=true, wantZ=false)
        λx = pS1.values
        @test (length(pS1.Z) == 0) || (size(pS1.Z[1],1) == 0)
        @test norm(pS1.T1 - pS2.T1) < tol * eps(real(T)) * n
        for j in 2:p-1
            @test norm(pS1.T[j] - pS2.T[j]) < tol * eps(real(T)) * n
        end
        compare_reigvals(λ,λx,λtol*eps(real(T)))
    end
end

include("generalized.jl")

include("ordschur.jl")

include("krylov.jl")
