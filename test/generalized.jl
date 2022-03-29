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

# mods needed if we add real versions of GPSD
function gpschur_test(A::AbstractVector{TM},S; left=false
                      ) where {TM <: AbstractMatrix{T}} where T <: Complex
    p = length(S)
    n = size(A[1],1)
    qtol = 10
    tol = 100
    λtol = 1000
    simple = all(S)
    # check for Hessenberg+triangular case
    # We want to call the specialized Hess+UT method so that roundoff does not
    # prevent checking the edge-case branches.
    tri0 = istriu(A[1],-1)
    for l in 2:p-1
        tri0 &= istriu(A[l])
    end
    if !tri0
        Awrk = [copy(A[j]) for j in 1:p]
        pS = pschur!(Awrk, S, left ? :L : :R, wantZ=true)
    else
        left && @error "test for Hess+UT circumvents orientation logic"
        if p > 1
            Awrk = [copy(A[j]) for j in 2:p]
        else
            Awrk = Vector{typeof(A[1])}(undef,0)
        end
        pS = pschur!(copy(A[left ? p : 1]), Awrk, S, rev = left, wantZ=true)
    end
    js = pS.schurindex
    jt = 0
    Ts = []
    for j in 1:p
        if j == js
            push!(Ts, Matrix(pS.T1))
        else
            jt += 1
            push!(Ts, Matrix(pS.T[jt]))
        end
    end
    Zs = pS.Z
    l1 = (p==1) ? 1 : 2
    if S[1] ⊻ left
        Ax = [Zs[1]*Ts[1]*Zs[l1]']
    else
        Ax = [Zs[l1]*Ts[1]*Zs[1]']
    end
    for l in 2:p
        if S[l] ⊻ left
            push!(Ax,Zs[l]*Ts[l]*Zs[mod(l,p)+1]')
        else
            push!(Ax,Zs[mod(l,p)+1]*Ts[l]*Zs[l]')
        end
    end
    for l in 1:p
        if developing
            if norm(tril(Ts[l]),-1) > 100 * eps(real(T)) * n
                @warn "triangularity failure for l=$l"
                display(Ts[l]); println()
            end
        else
            # if/when we have logic to ensure this now
            @test istriu(Ts[l],-1)
            # @test norm(tril(Ts[l],-1) < 100 * eps(real(T)) * n
        end
        @test norm(Zs[l]*Zs[l]' - I) < qtol * eps(real(T)) * n
        if developing
            r = norm(A[l] - Ax[l]) / (eps(real(T)) * n)
            if r > tol
                @warn "residual[$l]: $r"
            elseif r > 20
                println("residual[$l]: $r")
            end
        else
            @test norm(A[l] - Ax[l]) < tol * eps(real(T)) * n
        end
    end
    # Check consistency of eigenvalues w/ Schur matrices
    # TODO: hard tests in case of over/underflow
    λ = pS.values
    λs = diag(pS.T1)
    if !pS.S[pS.schurindex]
        λs = one(T) ./ λs
    end
    il = 0
    for l in 1:p
        l == pS.schurindex && continue
        il += 1
        if pS.S[l]
            λs .*= diag(pS.T[il])
        else
            λs .*= (one(T) ./ diag(pS.T[il]))
        end
    end
    for j in 1:n
        if isfinite(λs[j])
            @test λ[j] ≈ λs[j]
        else
            @test !isfinite(λ[j])
        end
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
for T in [Complex{Float64},Complex{BigFloat}]
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
            S = [true,true,false,true,false]
            A = [triu(rand(T,n,n)) for j in 1:p]
            A[1] = triu(rand(T,n,n),-1) # Hessenberg
            A[2][3,3] = 0
            gpschur_test(A,S)
        end
        @testset "Generalized Periodic Schur Hess+UT $T p=$p w/ late +hole" begin
            n = 5
            S = [true,true,false,true,false]
            A = [triu(rand(T,n,n)) for j in 1:p]
            A[1] = triu(rand(T,n,n),-1) # Hessenberg
            A[4][3,3] = 0
            gpschur_test(A,S)
        end
        @testset "Generalized Periodic Schur Hess+UT $T p=$p w/ late upper -hole" begin
            n = 5
            S = [true,false,true,false,true]
            A = [triu(rand(T,n,n)) for j in 1:p]
            A[1] = triu(rand(T,n,n),-1) # Hessenberg
            A[4][2,2] = 0
            gpschur_test(A,S)
        end
        @testset "Generalized Periodic Schur Hess+UT $T p=$p w/ late lower -hole" begin
            n = 5
            S = [true,false,true,false,true]
            A = [triu(rand(T,n,n)) for j in 1:p]
            A[1] = triu(rand(T,n,n),-1) # Hessenberg
            A[4][4,4] = 0
            gpschur_test(A,S)
        end
        @testset "Generalized Periodic Schur Hess+UT $T p=$p w/ upper -hole" begin
            n = 5
            S = [true,false,true,false,true]
            A = [triu(rand(T,n,n)) for j in 1:p]
            A[1] = triu(rand(T,n,n),-1) # Hessenberg
            A[2][2,2] = 0
            gpschur_test(A,S)
        end
        @testset "Generalized Periodic Schur Hess+UT $T p=$p w/ lower -hole" begin
            n = 5
            S = [true,false,true,false,true]
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
