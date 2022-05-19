# test for periodic Krylov-Schur

using ArnoldiMethod: ArnoldiMethod

# verify that PS is a partial periodic Schur decomposition of order k for As
function check(PS::PartialPeriodicSchur{T}, As; otol=100) where {T}
    columns(A) = [view(A,:,j) for j in 1:size(A,2)]
    p = length(As)
    n = size(As[1],1)
    Us = PS.Z
    Bs = PS.T
    push!(Bs, PS.T1)
    k = size(Us[1],2)
    b = norm(view(Bs[p],1:k,1:k))
    res = Matrix(As[p]*Us[p]) - Us[1] * view(Bs[p],1:k,1:k)
    cnrms = norm.(columns(res))
    thresh = max.(abs.(PS.values[1:k]),b * eps(real(T)))
    @test all(cnrms .< thresh)
    for l in 1:p
        U = Us[l]
        resnorm = norm(U' * U - I)
        @test resnorm < otol * n * b
    end
    nothing
end

coven = Dict( :LR => ArnoldiMethod.LR, :SR => ArnoldiMethod.SR,
              :LI => ArnoldiMethod.LI, :SI => ArnoldiMethod.SI,
              :LM => ArnoldiMethod.LM)
byes = Dict(:LR => (real,true), :SR => (real,false),
            :LI => (imag,true), :SI => (imag,false),
            :LM => (abs,true))

function mkmats1(T, n=30, p=3, xpnd=1.25)
    As = [triu(randn(T,n,n)) for _ in 1:p]
    λs = [prod([As[l][j,j] for l in 1:p]) for j in 1:n]
    idxs = sortperm(λs,by=abs)
    if xpnd != 1
        # spread out the eigvals
        for j in 1:n
            jj = idxs[j]
            fac = float(xpnd)^(j-1)
            for l in 1:p
                As[l][jj,jj] *= fac
            end
        end
    end
    for l in 1:p
        q,r = qr(randn(T,n,n))
        l1 = mod(l,p)+1
        As[l] = As[l] * q
        As[l1] = q' * As[l1]
    end
    return As
end

function pkstest1(As, psfull, which;
                  nev=4, tol=1e-10, k0=6, logf=nothing, λtol=1e-5, niter=60)
    n = size(As[1],1)
    p = length(As)
    T = eltype(As[1])
    wh = coven[which]()
    b = byes[which]
    ps, hist = partial_pschur(As,nev,wh; restarts=niter,mindim=k0,maxdim=2*k0,tol=tol)
    nconv = size(ps.Z[1],2)
    @test nconv >= (nev >> 1)
    check(ps,As)
    if psfull !== nothing
        vfull = sort(psfull.values, by=b[1], rev=b[2])
        # we cannot guarantee that the best values were found, but we can
        # require that the ones found are valid
        for λ in ps.values
            @test any(isapprox.(vfull[1:2*nev], λ, rtol=λtol))
        end
    end
    return ps
end

for T in [ComplexF64]
    As = mkmats1(T)
    ps0 = pschur(As,:L)
    @testset "krylov, $T" begin
        for w in [:LM, :SR, :LR, :LI, :SI]
            @testset "$w" begin
                pkstest1(As, ps0, w)
            end
        end
    end
end

for T in [Float64]
    As = mkmats1(T)
    ps0 = pschur(As,:L)
    @testset "krylov, $T" begin
        for w in [:LM, :SR, :LR]
            @testset "$w" begin
                pkstest1(As, ps0, w)
            end
        end
    end
end

