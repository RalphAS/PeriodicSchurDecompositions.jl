using PeriodicSchurDecompositions: _phessenberg!
for T in [ComplexF64,Float64,Complex{BigFloat},BigFloat]
    @testset "Generalized Periodic  Hessenberg $T" begin
      for p in [2,5]
        @testset "Generalized Periodic Hessenberg $T p=$p" begin
            n = 5
            S = [true]
            for l in 2:p
                push!(S,!S[l-1])
            end
            tol = 20
            qtol = 10
            A = [rand(T,n,n) for j in 1:p]
            Awrk = [copy(A[j]) for j in 1:p]
            Hs,Qs = _phessenberg!(Awrk,S)
            if p==1
                Ax = [Qs[1] * Hs[1] * Qs[1]']
            else
                Ax = [Qs[1] * Hs[1] * Qs[2]']
            end
            for j in 2:p
                if S[j]
                    push!(Ax,Qs[j]*Hs[j]*Qs[mod(j,p)+1]')
                else
                    push!(Ax,Qs[mod(j,p)+1]*Hs[j]*Qs[j]')
                end
            end
            @test istriu(Hs[1],-1)
            @test norm(Qs[1] * Qs[1]' - I) < qtol * eps(real(T)) * n
            for j in 2:p
                @test istriu(Hs[j])
                @test norm(Qs[j]*Qs[j]' - I) < qtol * eps(real(T)) * n
            end
            for j in 1:p
                @test norm(A[j] - Ax[j]) < tol * eps(real(T)) * n
            end
        end
      end
    end
end

for T in [Float64]
    @testset "Generalized Periodic Schur left full $T" begin
        for p in [4]
            @testset "Generalized Periodic Schur left full $T p=$p" begin
                n = 5
                S = fill(true,p)
                for j in p-1:-2:1
                    S[j] = false
                end
                A = [rand(T,n,n) for j in 1:p]
                gpschur_test(A,S;left=true)
            end
        end
    end
    @testset "Generalized Periodic Schur full $T" begin
        for p in [4]
            @testset "Generalized Periodic Schur full $T p=$p" begin
                n = 5
                S = [true,false,true,false]
                A = [rand(T,n,n) for j in 1:p]
                gpschur_test(A,S)
            end
        end
    end
end

for T in [Float64,BigFloat,Complex{Float64},Complex{BigFloat}]
    @testset "Generalized Periodic Schur Hess+UT $T (small)" begin
        for p in [2,3,5]
            @testset "Generalized Periodic Schur Hess+UT $T p=$p" begin
                n = 5
                S = [true,false,trues(p-2)...]
                A = [triu(rand(T,n,n)) for j in 1:p]
                A[1] = triu(rand(T,n,n),-1) # Hessenberg
                gpschur_test(A,S)
            end
        end
        p = 5
        @testset "Generalized Periodic Schur Hess+UT $T p=$p w/ early +hole" begin
            n = 5
            if SINGLE_MINUS_SIG[]
                S = [true,true,false,true,true]
            else
                S = [true,true,false,true,false]
            end
            A = [triu(rand(T,n,n)) for j in 1:p]
            A[1] = triu(rand(T,n,n),-1) # Hessenberg
            A[2][3,3] = 0
            gpschur_test(A,S)
        end
        @testset "Generalized Periodic Schur Hess+UT $T p=$p w/ late +hole" begin
            n = 5
            if SINGLE_MINUS_SIG[]
                S = [true,true,false,true,true]
            else
                S = [true,true,false,true,false]
            end
            A = [triu(rand(T,n,n)) for j in 1:p]
            A[1] = triu(rand(T,n,n),-1) # Hessenberg
            A[4][3,3] = 0
            gpschur_test(A,S)
        end
        @testset "Generalized Periodic Schur Hess+UT $T p=$p w/ late upper -hole" begin
            n = 5
            if SINGLE_MINUS_SIG[]
                S = [true,true,true,false,true]
            else
                S = [true,false,true,false,true]
            end
            A = [triu(rand(T,n,n)) for j in 1:p]
            A[1] = triu(rand(T,n,n),-1) # Hessenberg
            A[4][2,2] = 0
            gpschur_test(A,S)
        end
        @testset "Generalized Periodic Schur Hess+UT $T p=$p w/ late lower -hole" begin
            n = 5
            if SINGLE_MINUS_SIG[]
                S = [true,true,true,false,true]
            else
                S = [true,false,true,false,true]
            end
            A = [triu(rand(T,n,n)) for j in 1:p]
            A[1] = triu(rand(T,n,n),-1) # Hessenberg
            A[4][4,4] = 0
            gpschur_test(A,S)
        end
        @testset "Generalized Periodic Schur Hess+UT $T p=$p w/ upper -hole" begin
            n = 5
            if SINGLE_MINUS_SIG[]
                S = [true,false,true,true,true]
            else
                S = [true,false,true,false,true]
            end
            A = [triu(rand(T,n,n)) for j in 1:p]
            A[1] = triu(rand(T,n,n),-1) # Hessenberg
            A[2][2,2] = 0
            gpschur_test(A,S)
        end
        @testset "Generalized Periodic Schur Hess+UT $T p=$p w/ lower -hole" begin
            n = 5
            if SINGLE_MINUS_SIG[]
                S = [true,false,true,true,true]
            else
                S = [true,false,true,false,true]
            end
            A = [triu(rand(T,n,n)) for j in 1:p]
            A[1] = triu(rand(T,n,n),-1) # Hessenberg
            A[2][4,4] = 0
            gpschur_test(A,S)
        end
    end
end
for T in [Complex{Float64}]
    @testset "Periodic Schur Hess+UT, $T moderate N" begin
        p=4
        n=32
        S = trues(p)
        A = [triu(rand(T,n,n)) for j in 1:p]
        A[1] = triu(rand(T,n,n),-1) # Hessenberg
        A[2][3,3] = 0
        gpschur_test(A,S)
    end
    @testset "Generalized Periodic Schur Hess+UT, $T moderate N" begin
        p=4
        n=32
        S = [true,false,true,false]
        A = [triu(rand(T,n,n)) for j in 1:p]
        A[1] = triu(rand(T,n,n),-1) # Hessenberg
        A[2][3,3] = 0 # make it singular just because we can
        gpschur_test(A,S)
    end
end

for T in [Complex{Float64}]
    @testset "Periodic Schur left full $T" begin
        for p in [5]
            @testset "Periodic Schur left full $T p=$p" begin
                n = 5
                S = trues(p)
                A = [rand(T,n,n) for j in 1:p]
                gpschur_test(A,S;left=true)
            end
        end
    end
    @testset "Generalized Periodic Schur left full $T" begin
        for p in [4]
            @testset "Generalized Periodic Schur left full $T p=$p" begin
                n = 5
                S = fill(true,p)
                for j in p-1:-2:1
                    S[j] = false
                end
                A = [rand(T,n,n) for j in 1:p]
                gpschur_test(A,S;left=true)
            end
        end
    end
end

for T in [Complex{Float64}, Complex{BigFloat}]
    @testset "Periodic Schur full $T" begin
        for p in [5]
            @testset "Periodic Schur full $T p=$p" begin
                n = 5
                S = trues(p)
                A = [rand(T,n,n) for j in 1:p]
                gpschur_test(A,S)
            end
        end
    end
    @testset "Generalized Periodic Schur full $T" begin
        for p in [4]
            @testset "Generalized Periodic Schur full $T p=$p" begin
                n = 5
                S = [true,false,true,false]
                A = [rand(T,n,n) for j in 1:p]
                gpschur_test(A,S)
            end
        end
    end
end

for T in [Complex{Float64}]
    @testset "Periodic Schur Hess+UT $T" begin
        for p in [1,2,3,5]
            @testset "Periodic Schur Hess+UT $T p=$p" begin
                n = 5
                S = trues(p)
                A = [triu(rand(T,n,n)) for j in 1:p]
                A[1] = triu(rand(T,n,n),-1) # Hessenberg
                gpschur_test(A,S)
            end
        end
        for p in [2,3,5]
            @testset "Periodic Schur Hess+UT $T p=$p w/ hole" begin
                n = 5
                S = trues(p)
                A = [triu(rand(T,n,n)) for j in 1:p]
                A[1] = triu(rand(T,n,n),-1) # Hessenberg
                A[2][3,3] = 0
                gpschur_test(A,S)
            end
        end
    end
end

using PeriodicSchurDecompositions: checkpsd
@testset "checkpsd" begin
    T = ComplexF64
    p=4
    n=5
    S = [true,false,true,false]
    A = [triu(rand(T,n,n)) for j in 1:p]
    A[1] = triu(rand(T,n,n),-1) # Hessenberg
    Awrk = [copy(A[j]) for j in 2:p]
    pS = pschur!(copy(A[1]), Awrk, S, wantZ=true)
    goodresult, errs = checkpsd(pS,A)
    @test goodresult
    @test all(errs .< 100)
    pS.T1[3,3] += one(T)
    badresult, errs = checkpsd(pS,A,quiet=true)
    @test !badresult
    @test !all(errs .< 100)
end

# make sure quick versions run
@testset "Periodic Schur fast paths (complex)" begin
    n = 5
    tol = 20
    qtol = 10
    λtol = 1000
    T = ComplexF64
    for p in [1,5]
        A = [rand(T,n,n) for j in 1:p]
        # reference result
        Awrk = [copy(A[j]) for j in 1:p]
        pS2 = pschur!(Awrk, wantZ=true)
        @test isa(pS2, PeriodicSchur)
        λ = pS2.values

        Awrk = [copy(A[j]) for j in 1:p]
        pS0 = pschur!(Awrk, wantT=false, wantZ=false)
        @test isa(pS0, PeriodicSchur)
        λx = pS0.values
        # currently we leave the mangled matrices in the result, so no check on Ts
        @test (length(pS0.Z) == 0) || (size(pS0.Z[1],1) == 0)
        # Warning: we happen to know that computational path is not changed,
        # so we can be lazy here. This week, anyway.
        @test λ ≈ λx

        Awrk = [copy(A[j]) for j in 1:p]
        pS1 = pschur!(Awrk, wantT=true, wantZ=false)
        @test isa(pS1, PeriodicSchur)
        λx = pS1.values
        @test (length(pS1.Z) == 0) || (size(pS1.Z[1],1) == 0)
        @test norm(pS1.T1 - pS2.T1) < tol * eps(real(T)) * n
        for j in 2:p-1
            @test norm(pS1.T[j] - pS2.T[j]) < tol * eps(real(T)) * n
        end
        @test λ ≈ λx
    end
end

# TODO:
# error returns
# !S[1]
# dimension errors
