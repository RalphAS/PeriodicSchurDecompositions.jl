for T in [Float64, Complex{Float64}]
    @testset "ordschur $T: distinct real" begin
        p = 5
        n = 7
        # try to avoid serious non-normality
        A = [0.01 * triu!(rand(T,n,n)) for _ in 1:p]
        for j in 1:n
            μ = 2.0^(2*j/p)
            for l in 1:p
                A[l][j,j] = μ
            end
        end
        for l in 1:p
            q,_ = qr(randn(T,n,n))
            Atmp = q*A[l]
            copyto!(A[l],Atmp)
            l1 = mod(l,p)+1
            Atmp = A[l1]*q'
            copyto!(A[l1],Atmp)
        end
        Awrk = deepcopy(A)
        ps0 = pschur!(Awrk, :L)
        λ0s = ps0.values
        nsel = 2
        @testset "smallest" begin
            idx = sortperm(λ0s, by=abs)
            select = falses(n)
            select[idx[1:nsel]] .= true
            ps1 = deepcopy(ps0)
            ps1 = ordschur!(ps1, select)
            pschur_check(A, ps1; checkλ = false)
            λ1sel = ps1.values[1:nsel]
            for j in 1:nsel
                λ0 = λ0s[idx[j]]
                # println("checking $λ0")
                @test any(λ1sel .≈ λ0)
            end
        end
        @testset "largest" begin
            idx = sortperm(λ0s, by=abs, rev=true)
            select = falses(n)
            select[idx[1:nsel]] .= true
            ps1 = deepcopy(ps0)
            ps1 = ordschur!(ps1, select)
            pschur_check(A, ps1; checkλ = false)
            λ1sel = ps1.values[1:nsel]
            for j in 1:nsel
                λ0 = λ0s[idx[j]]
                # println("checking $λ0")
                @test any(λ1sel .≈ λ0)
            end
        end
    end
end

"""
construct a periodic real Schur decomposition (left orientation, quasi-tri at left end)
with conjugate eigvals in locations specified by jcs
"""
function mkrps(n,p,jcs,T=Float64; tri=false, check=false, nnfac=1e-2)
    T1 = triu!(nnfac * rand(T,n,n))
    Ts = [nnfac * triu!(rand(T,n,n)) for _ in 1:p-1]
    jj = 0
    λs = Vector{complex(T)}(undef,n)
    local μ
    for j in 1:n
        if j-1 ∈ jcs
            T1[j,j-1] = μ
            T1[j-1,j] = -μ
            λs[j] = 2.0^(2*jj)*(1-im)
            λs[j-1] = 2.0^(2*jj)*(1+im)
            for l in 1:p-1
                # for reference, note that eigvals are very sensitive to these entries
                Ts[l][j-1,j] = 0
            end
        else
            jj += 1
            μ = 2.0^(2*jj/p)
            λs[j] = 2.0^(2*jj)
        end
        for l in 1:p-1
            T1[j,j] = μ
            Ts[l][j,j] = μ
        end
    end
    if tri
        Zs = [Matrix{T}(I,n,n) for _ in 1:p]
    else
        q,_ = qr(randn(T,n,n))
        Zs = [Matrix(q)]
        q,_ = qr(randn(T,n,n))
        for l in 1:p-1
            push!(Zs,Matrix(q))
        end
    end
    As = [Zs[l+1]*Ts[l]*Zs[l]' for l in 1:p-1]
    push!(As, Zs[1]*T1*Zs[p]')
    ps = PeriodicSchur(T1,Ts,Zs,λs,'L',p)
    check && pschur_check(As, ps)
    return ps,As
end

for T in [Float64]
    @testset "ordschur $T: conjugate pair(s)" begin
        p = 5
        n = 7
        jcs = [3,6]
        ps0, A = mkrps(n,p,jcs,T; tri=false)
        λ0s = ps0.values # [a,b,z,z',c,w,w']
        # println("initial vals:"); display(λ0s); println()
        for (selset, str) in (([1,2,5],"ℂ,ℝ"),
                              ([1,3,4],"ℝ,ℂ"),
                              ([1,2,6,7],"ℂ,ℂ")
                              )
            @testset "$str" begin
                select = falses(n)
                nsel = length(selset)
                select[selset] .= true
                ps1 = deepcopy(ps0)
                ps1 = ordschur!(ps1, select)
                pschur_check(A, ps1; checkλ = false)
                # println("T1:"); display(ps1.T1); println()
                # NOTE: the following logic needs elaboration to test cases
                # where user is too sloppy to select conjugates
                λ1sel = ps1.values[1:nsel]
                # println("$str expected,got:"); display(hcat(λ0s[selset],λ1sel)); println()
                for j in 1:nsel
                   λ0 = λ0s[selset[j]]
                    # println("checking $λ0")
                   @test any(λ1sel .≈ λ0)
                end
            end
        end
    end
end

for T in [Complex{Float64}] # [Float64, Complex{Float64}]
    @testset "gen. ordschur $T: distinct real" begin
        p = 5
        n = 7
        S = trues(p)
        for l in 1:2:p-1
            S[l] = false
        end
        # try to avoid serious non-normality
        A = [0.01 * triu!(rand(T,n,n)) for _ in 1:p]
        for j in 1:n
            μ = 2.0^(2*j/p)
            for l in 1:p
                A[l][j,j] = S[l] ? μ : (1 / μ)
            end
        end
        for l in 1:p
            q,_ = qr(randn(T,n,n))
            Atmp = S[l] ? (q*A[l]) : (A[l]*q')
            copyto!(A[l],Atmp)
            l1 = mod(l,p)+1
            Atmp = S[l1] ? (A[l1]*q') : (q*A[l1])
            copyto!(A[l1],Atmp)
        end
        Awrk = deepcopy(A)
        ps0 = pschur!(Awrk, S, :L)
        λ0s = ps0.values
        nsel = 2
        @testset "smallest" begin
            idx = sortperm(λ0s, by=abs)
            select = falses(n)
            select[idx[1:nsel]] .= true
            ps1 = deepcopy(ps0)
            ps1 = ordschur!(ps1, select)
            gpschur_check(A, S, ps1)
            λ1sel = ps1.values[1:nsel]
            for j in 1:nsel
                λ0 = λ0s[idx[j]]
                # println("checking $λ0")
                @test any(λ1sel .≈ λ0)
            end
        end
        @testset "largest" begin
            idx = sortperm(λ0s, by=abs, rev=true)
            select = falses(n)
            select[idx[1:nsel]] .= true
            ps1 = deepcopy(ps0)
            ps1 = ordschur!(ps1, select)
            gpschur_check(A, S, ps1)
            λ1sel = ps1.values[1:nsel]
            for j in 1:nsel
                λ0 = λ0s[idx[j]]
                # println("checking $λ0")
                @test any(λ1sel .≈ λ0)
            end
        end
    end
end

