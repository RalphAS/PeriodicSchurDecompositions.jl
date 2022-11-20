function ev_check(As,Vs::AbstractVector{TM},λs, tol=sqrt(eps(real(T)))
                  ) where {TM <: AbstractMatrix{T}} where {T}
    p = length(As)
    nev = size(Vs[1],2)
    for ki in 1:nev
        μ = λs[ki] ^ (1/p)
        for l in 1:p
            ref = abs(μ) * norm(Vs[mod(l,p)+1][:,ki])
            err = norm(As[l] * Vs[l][:,ki] - μ * Vs[mod(l,p)+1][:,ki])
            @test err < tol * ref
        end
    end
end

for T in [Float64, Complex{Float64}]
  for p in [5,1]
    @testset "eigvecs $T: distinct real, p=$p" begin
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
            Vs = eigvecs(ps0, select)
            ev_check(A, Vs, λ0s[select])
        end
        @testset "largest" begin
            idx = sortperm(λ0s, by=abs, rev=true)
            select = falses(n)
            select[idx[1:nsel]] .= true
            Vs = eigvecs(ps0, select)
            ev_check(A, Vs, λ0s[select])
        end
    end
  end
end
