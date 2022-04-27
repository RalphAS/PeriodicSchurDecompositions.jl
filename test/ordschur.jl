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
