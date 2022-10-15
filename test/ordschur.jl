for T in [Float64, Complex{Float64}]
  for p in [5,1]
    @testset "ordschur $T: distinct real, p=$p" begin
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
            pschur_check(A, ps1; checkλ = false, tol = (p == 1) ? 20000 : 32)
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
            pschur_check(A, ps1; checkλ = false, tol = (p == 1) ? 20000 : 32)
            λ1sel = ps1.values[1:nsel]
            for j in 1:nsel
                λ0 = λ0s[idx[j]]
                # println("checking $λ0")
                @test any(λ1sel .≈ λ0)
            end
        end
    end
  end
end

"""
construct a periodic real Schur decomposition (left orientation, quasi-tri at left end)
with conjugate eigvals in locations specified by jcs
(if `alt`, generalize w/alternating signatures; eigval order will probably be lost)
"""
function mkrps(n,p,jcs,T=Float64; tri=false, check=false, nnfac=1e-2, alt=false)
    T1 = triu!(nnfac * rand(T,n,n))
    if p > 1
        Ts = [nnfac * triu!(rand(T,n,n)) for _ in 1:p-1]
    else
        Ts = Vector{Matrix{T}}(undef, 0)
    end
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
        T1[j,j] = μ
        for l in 1:p-1
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
    if p > 1
        As = [Zs[l+1]*Ts[l]*Zs[l]' for l in 1:p-1]
        push!(As, Zs[1]*T1*Zs[p]')
    else
        As = [Zs[1]*T1*Zs[p]']
    end
    if alt
        S = trues(p)
        for l in 1:2:p-1
            S[l] = false
            Ts[l] = inv(Ts[l])
            As[l] = Zs[l]*Ts[l]*Zs[l+1]'
        end
        ps = GeneralizedPeriodicSchur(S,p,T1,Ts,Zs,λs,ones(T,n),zeros(Int,n),'L')
        #ps = pschur(As,S,:L)
        check && gpschur_check(As, S, ps)
        # gpschur_check(As, S, ps)
    else
        ps = PeriodicSchur(T1,Ts,Zs,λs,'L',p)
        check && pschur_check(As, ps)
    end
    check && println("test ps is ready")
    return ps,As
end

for T in [Float64]
  for p in [5,1]
    @testset "ordschur $T: conjugate pair(s), p=$p" begin
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
                pschur_check(A, ps1; checkλ = false, tol = (p == 1) ? 20000 : 32)
                # println("T1:"); display(ps1.T1); println()
                if developing
                    println("λ0 λ1 sel:")
                    display(hcat(ps0.values,ps1.values,select)); println()
                end
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
end

for T in [Float64, Complex{Float64}]
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

for T in [Float64]
    @testset "gen. ordschur $T: conjugate pair(s)" begin
        p = 5
        n = 7
        jcs = [3,6]
        ps0, A = mkrps(n,p,jcs,T; tri=false, alt=true, check=true)
        # @show (ps0.orientation, ps0.schurindex, ps0.S)
        # when we're sure we get [a,b,z,z',c,w,w'] we can copy from std case
        λ0s = ps0.values
        if developing
            println("initial vals:"); display(λ0s); println()
        end
        t = isreal.(λ0s)
        ir1 = findfirst(t)
        ir2 = findlast(t)
        t .= (!).(t)
        ic1 = findfirst(t)
        ic2 = findlast(t) - 1
        prs = [([ic2,ic2+1],"ℂ,ℂ")]
        if ir1 > ic1
            push!(prs, ([ir1],"ℂ,ℝ"))
        else
            push!(prs, ([ic1,ic1+1],"ℝ,ℂ"))
        end
        if developing
            @show prs
        end
        for (selset, str) in prs
            @testset "$str" begin
                select = falses(n)
                nsel = length(selset)
                select[selset] .= true
                ps1 = deepcopy(ps0)
                ps1 = ordschur!(ps1, select)
                gpschur_check(A, ps0.S, ps1)
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
