# extra randomness is good for development but risks nasty surprises in CI
iseed = parse(Int,get(ENV,"PS_DEV","-1"))
if iseed > 0
    @info "Using seed $iseed for PS test"
    Random.seed!(iseed)
elseif iseed == 0
    let j = round(Int,mod(time(),10000))
        @info "Throwing dice for PS test: seed = $j"
        Random.seed!(j)
    end
else
    Random.seed!(1234)
end

# "developing" means some tests turn into warnings
# so we get more evidence in each run.
developing = iseed >= 0

# This allows us to compare eigvals against ordinary QZ
# (visually and only in developing mode);
# the alternative is more typical of real usage.
const SINGLE_MINUS_SIG = Ref(false)
if developing
    SINGLE_MINUS_SIG[] = parse(Int, get(ENV, "PS_QZ1", "0")) > 0
end

# this is designed for a mixture of real and conjugate pairs, not general
function compare_reigvals(λ,λx,λtol)
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

# Most cases are non-normal problems, so curb your expectations for λtol.
# Reference eigvals may be provided in λ, otherwise compute from product.
function pschur_check(A::AbstractVector{TM}, pS;
                      qtol = 10, tol = 32, λtol = 1000,
                      Aisreal = true, checkλ = true, λ = nothing
                     ) where {TM <: AbstractMatrix{T}} where T
    p = length(A)
    left = pS.orientation == 'L'
    n = size(A[1],1)
    js = pS.schurindex
    Ts = []
    jt = 0
    for j in 1:p
        # convert to ordinary Matrix so missing methods are not our problem.
        if j == js
            push!(Ts, Matrix(pS.T1))
        else
            jt += 1
            push!(Ts, Matrix(pS.T[jt]))
        end
    end
    Zs = pS.Z
    if left
        Ax = [Zs[mod(j,p)+1]*Ts[j]*Zs[j]' for j in 1:p]
    else
        Ax = [Zs[j]*Ts[j]*Zs[mod(j,p)+1]' for j in 1:p]
    end
    # check that we have logic to clear detritus
    for j in 1:p
        if developing
            @test norm(tril(Ts[j],j==js ? -2 : -1)) < 100 * eps(real(T)) * n
            if !istriu(Ts[j], ((j==js) && (T <: Real)) ? -1 : 0)
                @warn "subdiagonal junk found in matrix $j of $p of $(typeof(pS))"
            end
        else
            @test istriu(Ts[j], j==js ? -1 : 0)
        end
        if (j==js) && (T <: Real)
            for i in 1:n-1
                λi = pS.values[i]
                if developing
                    if isreal(λi) && (Ts[j][i+1,i] != 0)
                        @warn "subdiagonal junk found in Schur matrix ($j of $p) of $(typeof(pS))"
                        println("λ = $λi, subdiag is ",Ts[j][i+1,i])
                    end
                else
                    if isreal(λi)
                        @test Ts[j][i+1,i] == 0
                    end
                end
            end
        end

        # orthonormality of Schur vectors
        @test norm(Zs[j]*Zs[j]' - I) < qtol * eps(real(T)) * n

        # accuracy of decomposition
        if developing
            r = norm(A[j] - Ax[j]) / (eps(real(T)) * n)
            if r > tol
                @warn "residual[$j]: $r"
            elseif r > 20
                println("residual[$j]: $r")
            end
        else
            @test norm(A[j] - Ax[j]) < tol * eps(real(T)) * opnorm(A[j],1)
        end
    end
    if checkλ
        if λ === nothing
            if p > 1
                if left
                    Aprod = foldl((x,y) -> y*x, A)
                else
                    Aprod = foldl(*,A)
                end
            else
                Aprod = copy(A[1])
            end
            λ = eigvals(Aprod)
        end

        λx = pS.values
        if Aisreal
            #println("λ: $λ")
            #println("λx: $λx")
            compare_reigvals(λ,λx,λtol*eps(real(T)))
        elseif developing
            @warn "need proper test for general complex eigvals"
        end
    end
end

function pschur_test(A::AbstractVector{TM}; left=false
                     ) where {TM <: AbstractMatrix{T}} where T
    Awrk = deepcopy(A)
    pS = pschur!(Awrk, left ? :L : :R, wantZ=true)
    pschur_check(A, pS)
end

# mods needed if we add real versions of GPSD
function gpschur_check(A::AbstractVector{TM}, S, pS;
                       qtol = 10, tol = 100, λtol = 1000
                       ) where {TM <: AbstractMatrix{T}} where T <: Complex
    p = length(S)
    left = pS.orientation == 'L'
    n = size(A[1],1)
    simple = all(S)
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
            # we have logic to ensure this now
            @test istriu(Ts[l],-1)
        end
        # orthonormality
        @test norm(Zs[l]*Zs[l]' - I) < qtol * eps(real(T)) * n
        # accuracy of decomposition
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

# maintain a separate version for real eltype until it works
function gpschur_check(A::AbstractVector{TM}, S, pS;
                       qtol = 10, tol = 100, λtol = 1000
                       ) where {TM <: AbstractMatrix{T}} where T <: Real
    p = length(S)
    left = pS.orientation == 'L'
    n = size(A[1],1)
    simple = all(S)
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
        if (l==js)
            for i in 1:n-1
                λi = pS.values[i]
                if developing
                    if isreal(λi) && (Ts[l][i+1,i] != 0)
                        @warn "subdiagonal junk found in Schur matrix ($l of $p) of $(typeof(pS))"
                        println("λ = $λi, subdiag is ",Ts[l][i+1,i])
                    end
                else
                    if isreal(λi)
                        @test Ts[l][i+1,i] == 0
                    end
                end
            end
        end

        id = l == js ? -2 : -1
        if developing
            if norm(tril(Ts[l]), id) > 100 * eps(real(T)) * n
                @warn "triangularity failure for l=$l"
                display(Ts[l]); println()
            end
        else
            # we have logic to ensure this now
            @test istriu(Ts[l], id)
        end
        # orthonormality
        @test norm(Zs[l]*Zs[l]' - I) < qtol * eps(real(T)) * n
        # accuracy of decomposition
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
    # TODO: fix for complex eigvals
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
    if developing && T <: LinearAlgebra.BlasFloat
        if simple
            if left
                Aπ = foldl((x, y) -> y * x, A)
            else
                Aπ = prod(A)
            end
            λπ = eigvals(Aπ)
        elseif count(S) == p-1
            lb = findfirst(x -> !x, S)
            B = A[lb]
            Aπ = I
            for il in 1:p-1
                l = mod(lb + il - 1, p) + 1
                if left
                    Aπ = A[l] * Aπ
                else
                    Aπ = Aπ * A[l]
                end
            end
            λπ = eigvals(Aπ, B)
        elseif all(isfinite.(λ))
            Aπ = S[1] ? A[1] : inv(A[1])
            if left
                for l in 2:p
                    Aπ = (S[l] ? A[l] : inv(A[l])) * Aπ
                end
            else
                for l in 2:p
                    Aπ = Aπ * (S[l] ? A[l] : inv(A[l]))
                end
            end
            λπ = eigvals(Aπ)
        else
            λπ = nothing
        end
        if λπ !== nothing
            println("    ps.vals            diagprods             prodvals")
            display(hcat(λ, λs, λπ))
        else
            println("ps.vals diagprods")
            display(hcat(λ, λs))
        end
        println()
    end
    for j in 1:n
        isreal(λ[j]) || continue
        if isfinite(λs[j])
            @test λ[j] ≈ λs[j]
        else
            @test !isfinite(λ[j])
        end
    end
end

function gpschur_test(A::AbstractVector{TM}, S; left=false
                      ) where {TM <: AbstractMatrix{T}} where T
    p = length(S)

    # check for Hessenberg+triangular case
    # We want to call the specialized Hess+UT method so that roundoff does not
    # prevent checking the edge-case branches.
    tri0 = istriu(A[1],-1)
    for l in 2:p
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

    gpschur_check(A, S, pS)
end

# generate matrices with exponentially split eigvals, from Kressner 2001.
function expsplit(p,T)
    fac = 0.1
    A1 = T.([9 4 1 4 3 4; 6 8 2 4 0 2; 0 7 4 4 6 6; 0 0 8 4 6 7; 0 0 0 8 9 3; 0 0 0 0 5 0])
    Aj = Matrix(Diagonal(T.([fac,fac^2,fac^3,1,1,1])))
    A = vcat([A1],[copy(Aj) for _ in 1:p-1])
    # these are (obviously) approximate, and asymptotic values, so test accordingly
    λ = [15.6284,-1.31418-3.51424im,-1.31418+3.51424im,90*fac^p,(1600/3)*fac^(2*p),
         -(71750/11)*fac^(3*p)]
    return A, λ
end
