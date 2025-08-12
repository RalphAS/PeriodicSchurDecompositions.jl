for T in [Float64, Complex{Float64}]
  for p in [2,5]
    @testset "eigvecs $T: distinct complex, p=$p" begin
      for left in (false, true)
        dirstr = left ? "left" : "right"
        n = 8
        # try to avoid serious non-normality
        A = [0.01 * triu!(rand(T,n,n)) for _ in 1:p]
        for j in 1:2:n
            for l in 1:p-1
                A[l][j,j] = 1
                A[l][j+1,j+1] = 1
            end
            if T <: Complex
                λ = 2.0^(2*j) * (1 + im)
                A[p][j,j] = λ
                A[p][j+1,j+1] = conj(λ)
            else
                μ = 2.0^(2*j)
                A[p][j,j] = 1
                A[p][j+1,j+1] = 1
                A[p][j,j+1] = μ
                A[p][j+1,j] = -μ
            end
        end
        for l in 1:p
            q,_ = qr(randn(T,n,n))
            if left
                Atmp = q*A[l]
            else
                Atmp = A[l]*q'
            end
            copyto!(A[l],Atmp)
            l1 = mod(l,p)+1
            if left
                Atmp = A[l1]*q'
            else
                Atmp = q*A[l1]
            end
            copyto!(A[l1],Atmp)
        end
          Awrk = deepcopy(A)
          ps0 = pschur!(Awrk, left ? :L : :R)
            λ0s = ps0.values
            @show λ0s
            nsel = 2
            @testset "smallest $dirstr" begin
                idx = sortperm(λ0s, by=abs)
                select = falses(n)
                select[idx[1:nsel]] .= true
                Vs = eigvecs(ps0, select)
                ev_check(A, Vs, λ0s[select], left=left)
                V1 = eigvecs(ps0, select, shifted=false)
                @test V1[1] ≈ Vs[left ? 1 : p]
            end
            @testset "largest $dirstr" begin
                idx = sortperm(λ0s, by=abs, rev=true)
                select = falses(n)
                select[idx[1:nsel]] .= true
                Vs = eigvecs(ps0, select)
                ev_check(A, Vs, λ0s[select], left=left)
            end
        end
    end
  end
end

for T in [Float64, Complex{Float64}]
  for p in [1,5]
    @testset "eigvecs $T: distinct real, p=$p" begin
      for left in (false, true)
        dirstr = left ? "left" : "right"
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
            if left
                Atmp = q*A[l]
            else
                Atmp = A[l]*q'
            end
            copyto!(A[l],Atmp)
            l1 = mod(l,p)+1
            if left
                Atmp = A[l1]*q'
            else
                Atmp = q*A[l1]
            end
            copyto!(A[l1],Atmp)
        end
          Awrk = deepcopy(A)
          ps0 = pschur!(Awrk, left ? :L : :R)
            λ0s = ps0.values
            @show λ0s
            nsel = 2
            @testset "smallest $dirstr" begin
                idx = sortperm(λ0s, by=abs)
                select = falses(n)
                select[idx[1:nsel]] .= true
                Vs = eigvecs(ps0, select)
                ev_check(A, Vs, λ0s[select], left=left)
                V1 = eigvecs(ps0, select, shifted=false)
                @test V1[1] ≈ Vs[left ? 1 : p]
            end
            @testset "largest $dirstr" begin
                idx = sortperm(λ0s, by=abs, rev=true)
                select = falses(n)
                select[idx[1:nsel]] .= true
                Vs = eigvecs(ps0, select)
                ev_check(A, Vs, λ0s[select], left=left)
            end
        end
    end
  end
end
