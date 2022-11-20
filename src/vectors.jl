"""
    eigvecs(ps::PeriodicSchur, select::Vector{Bool}; shifted::Bool) => V::Vector{Matrix}

computes selected right eigenvectors of a product of matrices

For the `n×n` matrices `Aⱼ, j ∈ 1:p`, whose periodic schur decomposition
is in `ps`, this function computes vectors `v` such that
`Aₚ*...*A₂*A₁*v = λₖ v`  (or `A₁*A₂*...*Aₚ*v = λₖ v`) for left (right)
oriented `ps`. The returned vectors correspond to the eigenvalues in
`ps.values` corresponding to `true` entries in `select`.

If keyword `shifted` is true (the default), eigenvectors for circularly shifted
permutations of the `A` matrices are also returned.
The vectors are returned as columns of the elements of the vector `V`
of matrices. (*Note:* A vector of length one is returned if `shifted` is false.)
Vectors are normalized so that `Aⱼvⱼ = μ vⱼ₊₁` where `μ^p = λₖ` for
the left orientation.

If the element type of `ps` is real, `select` may be updated to include
conjugate pairs. `ps` itself is not modified.

The algorithm may fail if some selected eigenvalues are associated
with an invariant subspace that cannot be untangled.
"""
function LinearAlgebra.eigvecs(ps0::PeriodicSchur{T}, select::AbstractVector{Bool};
                 shifted=true, verbosity=1) where {T}
    RT = real(T)
    CT = complex(T)
    ps = deepcopy(ps0)
    if (length(ps.Z) == 0) || (size(ps.Z[1], 1) == 0)
        throw(ArgumentError("eigvecs requires Schur vectors in the PSD"))
    end
    n,m = size(ps.Z[1])
    if length(select) != m
        throw(ArgumentError("length of `select` must correspond to rank of Schur (sub-)space"))
    end
    p = ps.period
    left = ps.orientation == 'L'
    if !all(select)
        if T <: Real
            inpair = false
            for j in 1:m
                if inpair
                    if select[j - 1]
                        if verbosity > 0 && !select[j]
                            @info "adding $j to select for conjugacy"
                        end
                        select[j] = true
                    elseif select[j]
                        if verbosity > 0 && !select[j - 1]
                            @info "adding $(j-1) to select for conjugacy"
                        end
                        select[j - 1] = true
                    end
                    inpair = false
                    continue
                end
                inpair = !isreal(ps.values[j])
            end
        end
        ordschur!(ps, select)
    end
    nvec = count(select)
    sel = falses(m)
    sel[1:nvec] .= true
    nmat = shifted ? p : 1
    Vs = [Matrix{CT}(undef, n, nvec) for _ in 1:nmat]
    iλ = 1
    while iλ <= nvec
        if T <: Real && !isreal(ps.values[1])
            # I have a hammer so this must be a nail.
            # set up and solve the 2x2 cyclic problem
            μ = (ps.values[1] + 0im) ^ (1/RT(p))
            Zd = [-μ * Matrix{CT}(I, 2, 2) for _ in 1:p]
            Zl = [Matrix{CT}(undef, 2, 2) for _ in 1:p]
            il = 0
            for l in 1:p
                lx = left ? l : (p + 1 - l)
                if l == ps.schurindex
                    Tl = ps.T1
                else
                    il += 1
                    Tl = ps.T[il]
                end
                Zl[lx] .= Tl[1:2,1:2]
            end
            nsolve = 2
            rowx = 1
            colx = 1:nsolve
            y = zeros(CT, nsolve * p)
            # replace a row
            y[rowx] = 1
            Zd[1][rowx, :] .= 0
            Zl[p][rowx, :] .= 0
            Zd[1][rowx, colx] .= 1
            R, Zu, Zr, _ = _babd_qr!(Zd, Zl, y)
            x = _babd_solve!(R, Zu, Zr, y)
            t = 1 / norm(view(x, 1:nsolve))
            for l in 1:nmat
                if left
                    i0 = (l - 1) * nsolve
                else
                    i0 = l == 1 ? 0 : (p + 1 - l) * nsolve
                end
                vl = Vs[l]
                mul!(view(vl,:,iλ),
                     view(ps.Z[l],:,1:nsolve),
                     view(x, i0+1:i0+nsolve),
                     t, false)
                vl[:,iλ+1] .= conj.(vl[:,iλ])
            end
            nλ = 2
        else
            # A₁x₁ = T₁[1,1]*Z₂[:,1] = μ x₂, etc.
            il = 0
            fac = one(T)
            μ = (ps.values[1] + 0im) ^ (1/RT(p))
            for l in 1:nmat
                if l == ps.schurindex
                    Tl = ps.T1
                else
                    il += 1
                    Tl = ps.T[il]
                end
                Vs[l][:,iλ] .= fac .* ps.Z[l][:,1]
                fac *= (Tl[1,1] / μ)
            end
            nλ = 1
        end

        sel[1:nλ] .= false
        # should we try to recover if this fails?
        ordschur!(ps, sel)
        iλ += nλ
        circshift!(sel, -nλ)
    end
    return Vs
end
