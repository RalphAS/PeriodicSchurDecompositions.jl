using LinearAlgebra: Givens, givensAlgorithm

"""
GeneralizedPeriodicSchur

Matrix factorization type of the generalized periodic Schur factorization of a series
`A₁, A₂, ... Aₚ` of matrices. This is the return type of [`pschur!(_)`](@ref)
with a sign vector.

The `orientation` property may be `'L'`(left), corresponding to the product
 `Aₚ^sₚ * Aₚ₋₁^sₚ₋₁ * ... * A₂^s₂ * A₁^s₁`
or `'R'`(right), for the product `A₁^s₁ * A₂^s₂ * ... * Aₚ^sₚ`,
where the signs `sⱼ` are '±1`.

The decomposition for the "right" orientation is
`Zⱼ' * Aⱼ * Zᵢ = Tⱼ` where `i=mod(j,p)+1` if `sⱼ=1`, and
`Zᵢ' * Aⱼ * Zⱼ = Tⱼ` if `sⱼ=-1`.

The decomposition for the "left" orientation is
`Zᵢ' * Aⱼ * Zⱼ = Tⱼ`  where `i=mod(j,p)+1` if `sⱼ=1`, and
`Zⱼ' * Aⱼ * Zᵢ = Tⱼ` if `sⱼ=-1`.

For real element types, `Tₖ` is a quasi-triangular "real Schur" matrix,
where `k` is the value of the `schurindex` field. Otherwise
the `Tⱼ` are upper triangular. The `Zⱼ` are unitary (orthogonal for reals).

Given `F::GeneralizedPeriodicSchur`, the (quasi) triangular Schur factor `Tₖ` can be obtained via
`F.T1`.  `F.T` is a vector of the remaining triangular `Tⱼ`.
`F.Z` is a vector of the `Zⱼ`.
`F.values` is a vector of the eigenvalues of the product of the `Aⱼ`.
(The eigenvalues are stored internally in scaled form to avoid over/underflow.)
"""
struct GeneralizedPeriodicSchur{Ty,
                                St1<:AbstractMatrix, St<:AbstractMatrix, Sz<:AbstractMatrix,
                                Ss} <: AbstractPeriodicSchur{Ty}
    S::Ss
    schurindex::Int
    T1::St1
    T::Vector{St}
    Z::Vector{Sz}
    α::Vector
    β::Vector
    αscale::Vector{Int}
    orientation::Char
    function GeneralizedPeriodicSchur{Ty,St1,St,Sz,Ss}(S::AbstractVector{Bool},
                                                       schurindex::Int,
                                                       T1::AbstractMatrix{Ty},
                                                       T::Vector{<:AbstractMatrix{Ty}},
                                                       Z::Vector{<:AbstractMatrix{Ty}},
                                                       α::Vector{Ty},
                                                       β::Vector{Ty},
                                                       αscale::Vector{Int},
                                                       orientation::Char
                                                       ) where {Ty,St1,St,Sz,Ss}
        # maybe enforce sanity?
        new(S, schurindex, T1, T, Z, α, β, αscale, orientation)
    end
end
function GeneralizedPeriodicSchur(S::Ss, schurindex::Int,
                                  T1::St1,
                                  T::Vector{<:AbstractMatrix{Ty}},
                                  Z::Vector{<:AbstractMatrix{Ty}},
                                  α::Vector,
                                  β::Vector,
                                  αscale::Vector,
                                  orientation::Char='R'
                                  ) where {St1<:AbstractMatrix{Ty},
                                           Ss<:AbstractVector{Bool}
                                           } where {Ty}
    GeneralizedPeriodicSchur{Ty, St1, eltype(T), eltype(Z), Ss}(S, schurindex, T1, T, Z,
                                                                α, β, αscale, orientation)
end
function Base.getproperty(P::GeneralizedPeriodicSchur{T},s::Symbol) where {T}
    if s == :values
        return P.α ./ P.β .* T(2*one(real(T))) .^ P.αscale
    elseif s == :period
        return length(P.S)
    else
        return getfield(P,s)
    end
end

function pschur(A::AbstractVector{MT}, S::AbstractVector{Bool}, lr::Symbol=:R; kwargs...
                ) where {MT<:AbstractMatrix{T}} where {T}
    Atmp = [copy(Aj) for Aj in A]
    pschur!(Atmp, S, lr; kwargs...)
end

"""
    pschur!(A::Vector{<:StridedMatrix}, S::Vector{Bool}, lr::Symbol) -> F::GeneralizedPeriodicSchur

Computes a generalized periodic Schur decomposition of a series of general square matrices
with left (`lr=:L`) or right (`lr=:R`) orientation.

Optional arguments `wantT` and `wantZ`, defaulting to `true`, are booleans which may
be used to save time and memory by suppressing computation of the `T` and `Z`
matrices. See [`GeneralizedPeriodicSchur`](@ref) for the resulting structure.

Currently `Sⱼ` must be `true` for the leftmost term.
"""
function pschur!(A::AbstractVector{TA}, S::AbstractVector{Bool},
                 lr::Symbol=:R;
                 wantZ::Bool=true, wantT::Bool=true,
                 ) where {TA<:AbstractMatrix{T}} where {T<:Complex}
    orient = char_lr(lr)
    p = length(A)
    if orient == 'L'
        Aarg = similar(A)
        for j in 1:p
            Aarg[j] = A[p+1-j]
        end
        Sarg = reverse(S)
    else
        Aarg = A
        Sarg = S
    end
    if all(S)
        H1,pH = phessenberg!(Aarg)
        if wantZ
            Q = [_materializeQ(H1)]
            for j in 1:p-1
                push!(Q, Matrix(pH[j].Q))
            end
        else
            Q = nothing
        end
        Hs = [pH[j].R for j in 1:p-1]
        H1.H .= triu(H1.H,-1)
        F = pschur!(H1.H, Hs, Sarg, wantT=wantT, wantZ=wantZ, Q=Q, rev=(orient == 'L'))
    else
        # check this here so error message is less confusing
        Sarg[1] || throw(ArgumentError("The leftmost entry in S must be true"))
        Hs,Qs = _phessenberg!(Aarg,Sarg)
        H1 = popfirst!(Hs)
        Q = wantZ ? Qs : nothing
        F = pschur!(H1, Hs, Sarg, wantT=wantT, wantZ=wantZ, Q=Q, rev=(orient == 'L'))
    end
    return F
end

# We make this a macro so that @warn may usefully report the context.
macro _checkforms()
    return esc(quote
        if norm(tril(H1,-2)) > 1e-5
            @warn("H1 not Hessenberg")
        end
        for l in 1:p-1
            if norm(tril(Hs[l],-1)) > 1e-5
                @warn("H[$(l-1)] is not triangular")
            end
        end
    end)
end

# Mainly translated from SLICOT MB03BZ, by D.Kressner and V.Sima
# MB03BZ is Copyright (c) 2002-2020 NICONET e.V.
function pschur!(H1H::Th1, Hs::AbstractVector{Th},
                 S::AbstractVector{Bool}=trues(length(Hs)+1);
                 wantZ::Bool=true, wantT::Bool=true,
                 Q::Union{Nothing,Vector{Th}}=nothing, maxitfac=30,
                 rev::Bool=false
                 ) where {Th1<:Union{UpperHessenberg{T}, StridedMatrix{T}},
                          Th<:StridedMatrix{T}} where {T<:Complex}
    p = length(Hs)+1
    n = checksquare(H1H)
    for l in 1:p-1
        n1 = checksquare(Hs[l])
        n1 == n || throw(DimensionMismatch("H matrices must have the same size"))
    end
    S[1] || throw(ArgumentError("Signature entry S[1] must be true"))

    TR = real(T)
    α = zeros(T,n)
    β = zeros(T,n)
    αscale = zeros(Int,n)

    safmin = floatmin(real(T))
    unfl = floatmin(real(T))
    ovfl = 1 / unfl
    ulp = eps(TR)
    smlnum = unfl * (n / ulp)

    H1 = _gethess!(H1H)
    Hp = p==1 ? H1 : Hs[p-1]

    # decide whether we need an extra iteration w/ controlled zero shift
    ziter = (p >= log2(floatmin(TR)) / log2(ulp)) ? -1 : 0

    # Below, eigvals in ilast+1:n have been found.
    # Column ops modify rows ifirstm:whatever
    # Row ops modify columns whatever:ilastm
    # If only eigvals are needed, ifirstm is the row of the last split row above ilast.

    if wantZ
        if Q === nothing
            Z = [Matrix{T}(I,n,n) for l in 1:p]
        else
            for l in 1:p
                if size(Q[l],2) != n
                    throw(DimensionMismatch("second dimension of Q's must agree w/ H"))
                end
            end
            Z = Q
        end
    else
        Z = Vector{Matrix{T}}(undef,0)
    end

    Gtmp = Vector{Givens{T}}(undef,n)
    v4ev = zeros(T,p-1)

    ilast = n
    ifirst = -1
    ifirstm = 1
    ilastm = n
    iiter = 1
    maxit = maxitfac * n

    # diagnostic routines
    function showmat(str, j)
        print(str," H[$j] "); show(stdout, "text/plain", j==1 ? H1 : Hs[j-1]); println()
        nothing
    end
    function showallmats(str)
        for l in 1:p
            showmat(str, l)
        end
        nothing
    end
    function showprod(str)
        if verbosity[] > 1 && p > 1
            Htmp = copy(H1)
            for l in 2:p
                Htmp = Htmp * Hs[l-1]
            end
            print(str," ℍ "); show(stdout, "text/plain", Htmp); println()
            println("  ev: ",eigvals(Htmp)')
        end
        nothing
    end

    # computational routines

    function check_deflate_hess(H1,ilo,ilast)
        jlo = ilo
        for j in ilast:-1:ilo+1
            tol = abs(H1[j-1,j-1]) + abs(H1[j,j])
            if tol == 0
                tol = opnorm(view(H1,ilo:j,ilo:j),1)
            end
            tol = max(ulp*tol, smlnum)
            if abs(H1[j,j-1]) <= tol
                H1[j,j-1] = zero(T)
                jlo = j
                if j == ilast
                    return true, jlo
                end
                break
            end
        end
        return false,jlo
    end

    function check_deflate_tr(Hl,jlo,ilast)
        for j in ilast:-1:jlo
            if j == ilast
                tol = abs(Hl[j-1,j])
            elseif j==jlo
                tol = abs(Hl[j,j+1])
            else
                tol = abs(Hl[j-1,j]) + abs(Hl[j,j+1])
            end
            if tol == 0
                tol = opnorm(UpperTriangular(view(Hl,jlo:j,jlo:j)),1)
            end
            tol = max(ulp * tol, smlnum)
            if abs(Hl[j,j]) <= tol
                Hl[j,j] = zero(T)
                return true,j
            end
        end
        return false,0
    end

    done = false
    for jiter in 1:maxit
        split1block = false  # flag for 390
        ldeflate = -1
        jdeflate = -1
        deflate_pos = false # flag for 170
        deflate_neg = false # flag for 320
        doqziter = true
        czshift = false # for developer to check logic
        jlo = 1

        # Check for deflation
        for itmp in 1:1
            # not really a loop, but a block to break out of
            # SOMEDAY: rewrite as function(s)

            if ilast == 1
                # special case
                split1block = true
                break
            end

            # Test 1: deflation in the Hessenberg matrix.
            split1block, jlo = check_deflate_hess(H1, 1, ilast)
            split1block && break

            # Test 2: deflation in triangular matrices w/ index 1
            deflate_pos = false
            for l in 2:p
                if S[l]
                    Hl = Hs[l-1]
                    deflate_pos,jx = check_deflate_tr(Hl, jlo, ilast)
                    if deflate_pos
                        ldeflate = l
                        jdeflate = jx
                        break
                    end
                end
            end

            # Test 3: deflation in triangular matrices w/ index -1
            deflate_neg = false
            for l in 2:p
                if !S[l]
                    Hl = Hs[l-1]
                    deflate_neg,jx = check_deflate_tr(Hl, jlo, ilast)
                    if deflate_neg
                        ldeflate = l
                        jdeflate = jx
                        break
                    end
                end
            end

            # Test 4: controlled zero shift
            if ziter >= 7 || ziter < 0
                verbosity[] > 0 && println("controlled zero shift")
                # triangularize Hessenberg
                for j in jlo:ilast-1
                    c,s,r = givensAlgorithm(H1[j,j], H1[j+1,j])
                    H1[j,j] = r
                    H1[j+1,j] = zero(T)
                    G = Givens(j,j+1, complex(c), s)
                    lmul!(G, view(H1,:,j+1:ilastm))
                    Gtmp[j] = G
                end
                if wantZ
                    for j in jlo:ilast-1
                        rmul!(Z[1], Gtmp[j]')
                    end
                end
                # propagate transformations back to H1
                for l in p:-1:2
                    Hl = Hs[l-1]
                    if S[l]
                        for j in jlo:ilast-1
                            G = Gtmp[j]
                            if G.s != 0
                                rmul!(view(Hl,ifirstm:j+1,:), Gtmp[j]')
                                # check for deflation
                                tol = abs(Hl[j,j]) + abs(Hl[j+1,j+1])
                                if tol == 0
                                    tol = opnorm(view(Hl,jlo:j+1,jlo:j+1),1)
                                end
                                tol = max(ulp*tol, smlnum)
                                if abs(Hl[j+1,j]) <= tol
                                    c,s = one(T),zero(T)
                                    Hl[j+1,j] = zero(T)
                                    G = Givens(j,j+1,complex(c),s)
                                    Gtmp[j] = G
                                else
                                    c,s,r = givensAlgorithm(Hl[j,j], Hl[j+1,j])
                                    Hl[j,j] = r
                                    Hl[j+1,j] = zero(T)
                                    G = Givens(j,j+1,complex(c),s)
                                    lmul!(G, view(Hl, :,j+1:ilastm))
                                    Gtmp[j] = G
                                end
                            end
                        end # j loop
                    else # not S[l]
                        for j in jlo:ilast-1
                            G = Gtmp[j]
                            if G.s != 0
                                lmul!(G, view(Hl, :, j:ilastm))
                                # check for deflation
                                tol = abs(Hl[j,j]) + abs(Hl[j+1,j+1])
                                if tol == 0
                                    tol = opnorm(view(Hl,jlo:j+1,jlo:j+1),1)
                                end
                                tol = max(ulp*tol, smlnum)
                                if abs(Hl[j+1,j]) <= tol
                                    c,s = one(T),zero(T)
                                    Hl[j+1,j] = zero(T)
                                    G = Givens(j,j+1,complex(c),-s)
                                    Gtmp[j] = G
                                else
                                    c,s,r = givensAlgorithm(Hl[j+1,j+1], Hl[j+1,j])
                                    Hl[j+1,j+1] = r
                                    Hl[j+1,j] = zero(T)
                                    G = Givens(j+1,j,complex(c),s') # backwards!
                                    rmul!(view(Hl, ifirstm:j, :), G')
                                    Gtmp[j] = Givens(j,j+1,complex(c),-s)
                                end
                            end
                        end
                    end
                    if wantZ
                        for j in jlo:ilast-1
                            rmul!(Z[l], Gtmp[j]')
                        end
                    end
                end # loop over l propagating transformations back to H1

                # Apply transformations to right side of Hessenberg factor
                ziter = 0
                for j in jlo:ilast-1
                    G = Gtmp[j]
                    rmul!(view(H1, ifirstm:j+1, :), G')
                    if G.s == 0
                        ziter = 1
                    end
                end
                czshift = true
                doqziter = false
                break
            end
        end

        # Handle deflations

        if deflate_pos # Case II (170)
            verbosity[] > 0 && println("deflating S+ l=$ldeflate j=$jdeflate $ifirst:$ilast")
            # Do an unshifted pQZ step

            verbosity[] > 2 && showallmats("before deflation jlo=$jlo")

            # left of Hessenberg
            for j in jlo:jdeflate-1
                c,s,r = givensAlgorithm(H1[j,j], H1[j+1,j])
                H1[j,j] = r
                H1[j+1,j] = 0
                G = Givens(j,j+1,complex(c),s)
                lmul!(G, view(H1, :, j+1:ilastm))
                Gtmp[j] = G
            end
            if wantZ
                for j in jlo:jdeflate-1
                    rmul!(Z[1], Gtmp[j]')
                end
            end
            # propagate through triangular matrices
            for l in p:-1:2
                # due to zero on diagonal of H[ldeflate], decrement count
                ntra = (l < ldeflate) ? (jdeflate - 2) : (jdeflate - 1)
                Hl = Hs[l-1]
                if S[l]
                    for j in jlo:ntra
                        G = Gtmp[j]
                        rmul!(view(Hl, ifirstm:j+1, :), Gtmp[j]')
                        c,s,r = givensAlgorithm(Hl[j,j], Hl[j+1,j])
                        Hl[j,j] = r
                        Hl[j+1,j] = 0
                        G = Givens(j,j+1,complex(c),s)
                        lmul!(G, view(Hl, :, j+1:ilastm))
                        Gtmp[j] = G
                    end
                else
                    for j in jlo:ntra
                        lmul!(Gtmp[j], view(Hl, :, j:ilastm))
                        c,s,r = givensAlgorithm(Hl[j+1,j+1],Hl[j+1,j])
                        Hl[j+1,j+1] = r
                        Hl[j+1,j] = 0
                        G = Givens(j+1,j,complex(c),s')
                        rmul!(view(Hl, ifirstm:j, :), G')
                        Gtmp[j] = Givens(j,j+1, complex(c), -s)
                    end
                end
                if wantZ
                    for j in jlo:ntra
                        rmul!(Z[l], Gtmp[j]')
                    end
                end
            end
            # right of Hessenberg
            for j in jlo:jdeflate-2
                rmul!(view(H1, ifirstm:j+1, :), Gtmp[j]')
            end

            # Do another unshifted periodic QZ step
            # right of Hessenberg
            for j in ilast:-1:jdeflate+1
                c,s,r = givensAlgorithm(H1[j,j], H1[j,j-1])
                H1[j,j] = r
                H1[j,j-1] = 0
                G = Givens(j,j-1,complex(c),s')
                rmul!(view(H1,ifirstm:j-1,:), G')
                Gtmp[j] = Givens(j-1,j,complex(c),-s)
            end
            if wantZ
                for j in ilast:-1:jdeflate+1
                    rmul!(Z[2], Gtmp[j]')
                end
            end
            # propagate through triangular matrices
            for l in 2:p
                ntra = l > ldeflate ? (jdeflate + 2) : (jdeflate + 1)
                Hl = Hs[l-1]
                if !S[l]
                    for j in ilast:-1:ntra
                        rmul!(view(Hl, ifirstm:j, :), Gtmp[j]')
                        c,s,r = givensAlgorithm(Hl[j-1,j-1], Hl[j,j-1])
                        Hl[j-1,j-1] = r
                        Hl[j,j-1] = 0
                        G = Givens(j-1,j,complex(c),s)
                        lmul!(G, view(Hl, :, j:ilastm))
                        Gtmp[j] = Givens(j-1,j,complex(c),s)
                    end
                else
                    for j in ilast:-1:ntra
                        G = Gtmp[j]
                        lmul!(Gtmp[j], view(Hl, :, j-1:ilastm))
                        c,s,r = givensAlgorithm(Hl[j,j], Hl[j,j-1])
                        Hl[j,j] = r
                        Hl[j,j-1] = 0
                        G = Givens(j,j-1,complex(c),s')
                        rmul!(view(Hl, ifirstm:j-1, :), G')
                        Gtmp[j] = Givens(j-1,j,complex(c),-s)
                    end
                end
                if wantZ
                    ln = mod(l,p) + 1
                    for j in ilast:-1:ntra
                        rmul!(Z[ln], Gtmp[j]')
                    end
                end
            end
            # left of Hessenberg
            for j in ilast:-1:jdeflate+2
                G = Gtmp[j]
                lmul!(Gtmp[j], view(H1, :, j-1:ilastm))
            end
            verbosity[] > 2 && showallmats("after deflation")
            doqziter = false

        elseif deflate_neg # Case III (320)
            verbosity[] > 0 && println("deflating S- l=$ldeflate j=$jdeflate")
            if jdeflate > (ilast - jlo + 1) / 2 # bottom half
                # chase the zero down
                for j1 in jdeflate:ilast-1
                    j = j1
                    Hl = Hs[ldeflate-1]
                    c,s,r = givensAlgorithm(Hl[j,j+1], Hl[j+1,j+1])
                    Hl[j,j+1] = r
                    Hl[j+1,j+1] = 0
                    G = Givens(j,j+1,complex(c),s)
                    lmul!(G, view(Hl, :, j+2:ilastm))
                    ln = mod(ldeflate,p) + 1
                    if wantZ
                        rmul!(Z[ln],G')
                    end
                    for l in 1:p-1
                        if ln == 1
                            lmul!(G, view(H1,:,j-1:ilastm))
                            c,s,r = givensAlgorithm(H1[j+1,j], H1[j+1,j-1])
                            H1[j+1,j] = r
                            H1[j+1,j-1] = 0
                            G = Givens(j,j-1,complex(c),s')
                            rmul!(view(H1, ifirstm:j, :), G')
                            G = Givens(j-1,j,complex(c),-s)
                            j -= 1
                        elseif S[ln]
                            Hln = Hs[ln-1]
                            lmul!(G, view(Hln, :, j:ilastm))
                            c,s,r = givensAlgorithm(Hln[j+1,j+1], Hln[j+1,j])
                            Hln[j+1,j+1] = r
                            Hln[j+1,j] = 0
                            G = Givens(j+1,j,complex(c),s')
                            rmul!(view(Hln, ifirstm:j, :), G')
                            G = Givens(j,j+1,complex(c),-s)
                        else
                            Hln = Hs[ln-1]
                            rmul!(view(Hln, ifirstm:j+1, :), G')
                            c,s,r = givensAlgorithm(Hln[j,j], Hln[j+1,j])
                            Hln[j,j] = r
                            Hln[j+1,j] = 0
                            G = Givens(j,j+1,complex(c),s)
                            lmul!(G, view(Hln, :, j+1:ilastm))
                        end
                        ln = mod(ln,p) + 1
                        if wantZ
                            rmul!(Z[ln], G')
                        end
                    end # l loop
                    Hl = Hs[ldeflate-1]
                    rmul!(view(Hl, ifirstm:j, :), G')
                end # j1 loop (340)
                # deflate last element in Hessenberg
                j = ilast
                c,s,r = givensAlgorithm(H1[j,j], H1[j,j-1])
                H1[j,j] = r
                H1[j,j-1] = 0
                G = Givens(j,j-1,complex(c),s')
                rmul!(view(H1, ifirstm:j-1, :), G')
                G = Givens(j-1,j,complex(c),-s)
                if wantZ
                    rmul!(Z[2], G')
                end
                for l in 2:ldeflate-1
                    Hl = Hs[l-1]
                    if !S[l]
                        rmul!(view(Hl, ifirstm:j, :), G')
                        c,s,r = givensAlgorithm(Hl[j-1,j-1], Hl[j,j-1])
                        Hl[j-1,j-1] = r
                        Hl[j,j-1] = 0
                        G = Givens(j-1,j,complex(c),s)
                        lmul!(G, view(Hl, :, j:ilastm))
                    else
                        lmul!(G, view(Hl, :, j-1:ilastm))
                        c,s,r = givensAlgorithm(Hl[j,j], Hl[j,j-1])
                        Hl[j,j] = r
                        Hl[j,j-1] = 0
                        G = Givens(j,j-1,complex(c),s')
                        rmul!(view(Hl, ifirstm:j-1, :), G')
                        G = Givens(j-1,j,complex(c),-s)
                    end
                    if wantZ
                        ln = mod(l,p)+1
                        rmul!(Z[ln], G')
                    end
                end # l loop (350)
                Hl = Hs[ldeflate-1]
                rmul!(view(Hl, ifirstm:j, :), G')
            else # jdeflate in top half
                # chase the zero up to the first position
                for j1 in jdeflate:-1:jlo+1
                    j = j1
                    Hl = Hs[ldeflate-1]
                    c,s,r = givensAlgorithm(Hl[j-1,j],Hl[j-1,j-1])
                    Hl[j-1,j] = r
                    Hl[j-1,j-1] = 0
                    G = Givens(j,j-1,complex(c),conj(s))
                    rmul!(view(Hl,ifirstm:j-2,:), G')
                    G = Givens(j-1,j,complex(c),-s)
                    if wantZ
                        rmul!(Z[ldeflate], G')
                    end
                    ln = ldeflate-1
                    for l in 1:p-1
                        Hln = ln == 1 ? H1 : Hs[ln-1]
                        if ln == 1
                            rmul!(view(Hln, ifirstm:j+1, :), G')
                            c,s,r = givensAlgorithm(Hln[j,j-1], Hln[j+1,j-1])
                            Hln[j,j-1] = r
                            Hln[j+1,j-1] = 0
                            G = Givens(j,j+1,complex(c),s)
                            lmul!(G, view(Hln, :, j:ilastm))
                            j += 1
                        elseif !S[ln]
                            lmul!(G, view(Hln, :, j-1:ilastm))
                            c,s,r = givensAlgorithm(Hln[j,j], Hln[j,j-1])
                            Hln[j,j] = r
                            Hln[j,j-1] = 0
                            G = Givens(j,j-1,complex(c),s')
                            rmul!(view(Hln, ifirstm:j-1, :), G')
                            G = Givens(j-1,j,complex(c),-s)
                        else
                            rmul!(view(Hln, ifirstm:j, :), G')
                            c,s,r = givensAlgorithm(Hln[j-1,j-1], Hln[j,j-1])
                            Hln[j-1,j-1] = r
                            Hln[j,j-1] = 0
                            G = Givens(j-1,j,complex(c),s)
                            lmul!(G, view(Hln, :, j:ilastm))
                        end
                        if wantZ
                            rmul!(Z[ln], G')
                        end
                        ln = (ln == 1) ? p : (ln-1)
                    end # l loop (360)
                    Hl = Hs[ldeflate-1]
                    lmul!(G, view(Hl, :, j:ilastm))
                end # j1 loop
                # Deflate the first element in Hessenberg
                j = jlo
                c,s,r = givensAlgorithm(H1[j,j], H1[j+1,j])
                H1[j,j] = r
                H1[j+1,j] = 0
                G = Givens(j,j+1,complex(c),s)
                lmul!(G, view(H1,:,j+1:ilastm))
                if wantZ
                    rmul!(Z[1], G')
                end
                for l in p:-1:ldeflate+1
                    Hl = Hs[l-1]
                    if S[l]
                        rmul!(view(Hl,ifirstm:j+1,:), G')
                        c,s,r = givensAlgorithm(Hl[j,j], Hl[j+1,j])
                        Hl[j,j] = r
                        Hl[j+1,j] = 0
                        G = Givens(j,j+1,complex(c),s)
                        lmul!(G, view(Hl,:, j+1:ilastm))
                    else
                        lmul!(G, view(Hl,:,j:ilastm))
                        c,s,r = givensAlgorithm(Hl[j+1,j+1],Hl[j+1,j])
                        Hl[j+1,j+1] = r
                        Hl[j+1,j] = 0
                        G = Givens(j+1,j,complex(c),conj(s))
                        rmul!(view(Hl,ifirstm:j,:), G')
                        G = Givens(j,j+1,complex(c),-s)
                    end
                    if wantZ
                        rmul!(Z[l], G')
                    end
                end # trailing l loop (380)
                Hl = Hs[ldeflate-1]
                lmul!(G, view(Hl, :, j+1:ilastm))
            end # jdeflate top/bottom branches
            doqziter = false
        elseif split1block # (390)
            verbosity[] > 0 && println("splitting 1x1 index $ilast")
            for l in 1:p-1
                v4ev[l] = Hs[l][ilast,ilast]
            end
            α[ilast], β[ilast], αscale[ilast] = _safeprod(S,H1[ilast,ilast],v4ev)
            ilast -= 1
            if ilast < 1
                done = true
                break
            end
            iiter = 0
            if ziter != -1
                ziter = 0
            end
            if !wantT
                ilastm = ilast
                if ifirstm > ilast
                    ifirstm = 1
                end
            end
            doqziter = false
        else
            if (verbosity[] > 0) && (ifirst != jlo)
                println("deflating from Hessenberg")
            end
            ifirst = jlo
        end

        if doqziter # (400)
            verbosity[] > 0 && println("starting QZ loop $ifirst:$ilast")
            # showZ("start")
            iiter += 1
            ziter += 1
            if !wantT
                ifirstm = ifirst
            end
            if mod(iiter, 10) == 0
                # exceptional shift
                # (from here, results must differ from Fortran version)
                verbosity[] > 0 && println("using exceptional shift")
                fgtmp = rand(T,2)
                c,s,_ = givensAlgorithm(fgtmp[1], fgtmp[2])
            else
                # normal single shift
                c,s,_ = givensAlgorithm(one(T), one(T))
                for l in p:-1:2
                    Hl = Hs[l-1]
                    if S[l]
                        c,s,_ = givensAlgorithm(Hl[ifirst,ifirst]*c,
                                                Hl[ilast,ilast]*conj(s))
                    else
                        c,s,_ = givensAlgorithm(Hl[ilast,ilast]*c,
                                                -Hl[ifirst,ifirst]*conj(s))
                        s = -s
                    end
                end
                c,s,_ = givensAlgorithm(H1[ifirst,ifirst]*c -H1[ilast,ilast]*conj(s),
                                        H1[ifirst+1,ifirst]*c)
                verbosity[] > 2 && println("initial c,s: $c, $s")
            end
            # do the sweeps
            for j1 in ifirst-1:ilast-2
                j = j1+1
                # Create a bulge or chase it
                if j1 >= ifirst
                    c,s,r = givensAlgorithm(H1[j,j-1], H1[j+1,j-1])
                    H1[j,j-1] = r
                    H1[j+1,j-1] = 0
                end
                G = Givens(j,j+1,complex(c),s)
                verbosity[] > 2 && println("initial G: $G")
                lmul!(G, view(H1, :, j:ilastm))
                if wantZ
                    rmul!(Z[1], G')
                end
                # propagate
                for l in p:-1:2
                    Hl = Hs[l-1]
                    if S[l]
                        rmul!(view(Hl, ifirstm:j+1, :), G')
                        c,s,r = givensAlgorithm(Hl[j,j], Hl[j+1,j])
                        Hl[j,j] = r
                        Hl[j+1,j] = 0
                        G = Givens(j,j+1,complex(c),s)
                        lmul!(G, view(Hl, :, j+1:ilastm))
                    else
                        lmul!(G, view(Hl, :, j:ilastm))
                        c,s,r = givensAlgorithm(Hl[j+1,j+1], Hl[j+1,j])
                        Hl[j+1,j+1] = r
                        Hl[j+1,j] = 0
                        G = Givens(j+1,j,complex(c),s')
                        rmul!(view(Hl, ifirstm:j, :), G')
                        s = -s
                        G = Givens(j,j+1,complex(c),s)
                    end
                    if wantZ
                        rmul!(Z[l], G')
                    end
                end
                itmp = min(j+2, ilastm)
                rmul!(view(H1, ifirstm:itmp, :), G')
                if verbosity[] > 2
                    str = (j1 >= ifirst) ? "chasing j=$j" : "creating bulge j=$j"
                    showallmats(str)
                end
            end # end of qz loop
            verbosity[] == 2 && showallmats("after qz loop")
        end # if doqziter

    end # iteration loop
    if !done
        throw(ErrorException("convergence failed at level $ilast"))
    end

    if wantT
        # scale factors 2:p
        scalefacs = zeros(T,n)
        for l in p:-1:2
            Hl = Hs[l-1]
            if S[l]
                for j in 1:n
                    abst = abs(Hl[j,j])
                    if abst > safmin
                        z = conj(Hl[j,j] / abst)
                        Hl[j,j] = abst
                        if j < n
                            Hl[j,j+1:n] .*= z
                        end
                    else
                        z = one(T)
                    end
                    scalefacs[j] = z
                end
            else
                for j in 1:n
                    abst = abs(Hl[j,j])
                    if abst > safmin
                        z = conj(Hl[j,j] / abst)
                        Hl[j,j] = abst
                        Hl[1:j-1,j] .*= z
                    else
                        z = one(T)
                    end
                    scalefacs[j] = conj(z)
                end
            end
            if wantZ
                for j in 1:n
                    Z[l][:,j] .*= conj(scalefacs[j])
                end
            end
            Hlm1 = l==2 ? H1 : Hs[l-2]
            if S[l-1]
                for j in 1:n
                    Hlm1[1:j,j] .*= conj(scalefacs[j])
                end
            else
                for j in 1:n
                    Hlm1[j,j:n] .*= scalefacs[j]
                end
            end
        end

    end

    if rev
        if wantZ
            Zr = similar(Z)
            Zr[1] = Z[1]
            for l in 2:p
                Zr[l] = Z[p+2-l]
            end
        else
            Zr = Z
        end
        Hr = similar(Hs)
        for l in 1:p-1
            Hr[l] = Hs[p-l]
        end
        return GeneralizedPeriodicSchur(S,p,H1,Hr,Zr,α,β,αscale,'L')
    else
        return GeneralizedPeriodicSchur(S,1,H1,Hs,Z,α,β,αscale)
    end
end

"""
_safeprod(s,x0,x) -> alpha,beta,scale::Int

represent `x0^s[1] * cumprod(x.^s[2:end])` as `α / β * 2^scale` where
`α = 0` or `abs(α) ∈ [1,2)`, `β ∈ {0,1}`, avoiding overflow/underflow.
"""
function _safeprod(s::AbstractVector{Bool},x0::T,v::Vector{T}) where T
    # WARNING: assumes base 2, like IEEE
    RT = real(T)
    base = RT(2)
    p = length(v) + 1
    α = one(T)
    β = 1
    scale = 0
    for i in 1:p
        xi = i == 1 ? x0 : v[i-1]
        if s[i]
            α *= xi
        else
            if xi == 0
                β = 0
            else
                α /= xi
            end
        end
        if abs(α) == 0
            α = zero(T)
            scale = 0
            if β == 0
                return α,β,scale
            end
        else
            while abs(α) < one(real(T))
                α *= base
                scale -= 1
            end
            while abs(α) >= base
                α /= base
                scale += 1
            end
        end
    end
    return α,β,scale
end

# Generalized Periodic Hessenberg reduction.
# Needs RQ, so just BLAS types for now.

using LinearAlgebra: BlasFloat
using LinearAlgebra.LAPACK: geqrf!, gerqf!, ormqr!, ormrq!
using MatrixFactorizations: rq!

# NOTE: this differs from the standard case, returning full Q matrices since
# there are extra transformations and we want full Q here anyway.
# Based on Kressner's 2001 paper
function _phessenberg!(A::AbstractVector{TA}, S::AbstractVector{Bool};
                       wantQ = true
                       ) where {TA<:AbstractMatrix{T}} where {T<:BlasFloat}
    S[1] || throw(ArgumentError("The first entry in S must be true"))
    p = length(A)
    n = checksquare(A[1])
    require_one_based_indexing(A[1])
    for l in 2:p
        require_one_based_indexing(A[l])
        if checksquare(A[l]) != n
            throw(DimensionMismatch())
        end
    end
    if wantQ
        Qs = [Matrix{T}(I,n,n) for l in 1:p]
    else
        Qs = nothing
    end

    # Stage 1. Triangular decompositions
    τ = zeros(T,n)
    tC = (T <: Complex) ? 'C' : 'T'
    for l in p:-1:2
        if S[l]
            qrA, τl = geqrf!(A[l],τ)
            if S[l-1]
                ormqr!('R','N', qrA, τl, A[l-1])
            else
                ormqr!('L',tC, qrA, τl, A[l-1])
            end
            wantQ && ormqr!('R','N', qrA, τl, Qs[l])
        else
            rqA,τl = gerqf!(A[l],τ)
            if S[l-1]
                ormrq!('R', tC, rqA, τl, A[l-1])
            else
                ormrq!('L', 'N', rqA, τl, A[l-1])
            end
            wantQ && ormrq!('R',tC, rqA, τl, Qs[l])
        end
        triu!(A[l])
    end
    # Stage 2. Hessenberg reduction of A₁
    # Following Bojanczyk et al., we use Givens here to facilitate re-triangularization
    # in the !S[l] case.
    Gtmp = Vector{Givens{T}}(undef,n)
    A1 = A[1]
    for j in 1:n-2
        for i in n:-1:j+2
            c,s,r = givensAlgorithm(A1[i-1,j], A1[i,j])
            A1[i-1,j] = r
            A1[i,j] = 0
            G = Givens(i-1,i,T(c),s)
            lmul!(G, view(A1, :, j+1:n))
            wantQ && rmul!(Qs[1], G')
            Gtmp[i] = G
        end
        # and propagation "as usual"
        for l in p:-1:2
            Al = A[l]
            if S[l]
                for i in n:-1:j+2
                    G = Gtmp[i]
                    rmul!(view(Al, 1:i, :), G')
                    c,s,r = givensAlgorithm(Al[i-1,i-1], Al[i,i-1])
                    Al[i-1,i-1] = r
                    Al[i,i-1] = 0
                    G = Givens(i-1,i,T(c),s)
                    lmul!(G, view(Al, :, i:n))
                    Gtmp[i] = G
                end
            else
                for i in n:-1:j+2
                    G = Gtmp[i]
                    lmul!(G, view(Al, :, i-1:n))
                    c,s,r = givensAlgorithm(Al[i,i], Al[i,i-1])
                    Al[i,i] = r
                    Al[i,i-1] = 0
                    G = Givens(i,i-1,T(c),s')
                    rmul!(view(Al, 1:i-1, :), G')
                    Gtmp[i] = Givens(i-1,i,T(c),-s)
                end
            end
            if wantQ
                for i in n:-1:j+2
                    rmul!(Qs[l], Gtmp[i]')
                end
            end
        end
        for i in n:-1:j+2
            rmul!(A1, Gtmp[i]') # view(A1, 1:n, :) is all of A1
        end
    end
    # claim the above already cleared out the detritus, so no triu! calls.
    return A,Qs
end

# generic version
function _phessenberg!(A::AbstractVector{TA}, S::AbstractVector{Bool};
                       wantQ = true
                       ) where {TA<:AbstractMatrix{T}} where {T}
    S[1] || throw(ArgumentError("The first entry in S must be true"))
    p = length(A)
    n = checksquare(A[1])
    require_one_based_indexing(A[1])
    for l in 2:p
        require_one_based_indexing(A[l])
        if checksquare(A[l]) != n
            throw(DimensionMismatch())
        end
    end
    if wantQ
        Qs = [Matrix{T}(I,n,n) for l in 1:p]
    else
        Qs = nothing
    end

    # Stage 1. Triangular decompositions
    for l in p:-1:2
        if S[l]
            F = qr!(A[l])
            if S[l-1]
                rmul!(A[l-1], F.Q)
            else
                lmul!(F.Q', A[l-1])
            end
            wantQ && rmul!(Qs[l], F.Q)
        else
            F = rq!(A[l])
            if S[l-1]
                rmul!(A[l-1], F.Q')
            else
                lmul!(F.Q, A[l-1])
            end
            wantQ && rmul!(Qs[l], F.Q')
        end
        # We are done with the Q part of QR/RQ(A[l]), so clear out the lower triangle
        # to prepare for Stage 2.
        triu!(A[l])
    end
    # Stage 2. Hessenberg reduction of A₁
    # Following Bojanczyk et al., we use Givens here to facilitate re-triangularization
    # in the !S[l] case.
    Gtmp = Vector{Givens{T}}(undef,n)
    A1 = A[1]
    for j in 1:n-2
        for i in n:-1:j+2
            c,s,r = givensAlgorithm(A1[i-1,j], A1[i,j])
            A1[i-1,j] = r
            A1[i,j] = 0
            G = Givens(i-1,i,T(c),s)
            lmul!(G, view(A1, :, j+1:n))
            wantQ && rmul!(Qs[1], G')
            Gtmp[i] = G
        end
        # and propagation "as usual"
        for l in p:-1:2
            Al = A[l]
            if S[l]
                for i in n:-1:j+2
                    G = Gtmp[i]
                    rmul!(view(Al, 1:i, :), G')
                    c,s,r = givensAlgorithm(Al[i-1,i-1], Al[i,i-1])
                    Al[i-1,i-1] = r
                    Al[i,i-1] = 0
                    G = Givens(i-1,i,T(c),s)
                    lmul!(G, view(Al, :, i:n))
                    Gtmp[i] = G
                end
            else
                for i in n:-1:j+2
                    G = Gtmp[i]
                    lmul!(G, view(Al, :, i-1:n))
                    c,s,r = givensAlgorithm(Al[i,i], Al[i,i-1])
                    Al[i,i] = r
                    Al[i,i-1] = 0
                    G = Givens(i,i-1,T(c),s')
                    rmul!(view(Al, 1:i-1, :), G')
                    Gtmp[i] = Givens(i-1,i,T(c),-s)
                end
            end
            if wantQ
                for i in n:-1:j+2
                    rmul!(Qs[l], Gtmp[i]')
                end
            end
        end
        for i in n:-1:j+2
            rmul!(A1, Gtmp[i]') # view(A1, 1:n, :) is all of A1
        end
    end
    # claim the above already cleared out the detritus, so no triu! calls.
    return A,Qs
end

"""
    gpschur(As, Bs) -> F::GeneralizedPeriodicSchur

Computes a generalized periodic Schur decomposition corresponding to the formal product
        `Bₚ⁻¹Aₚ...B₁⁻¹A₁`
of paired series of matrices in left operator order `[A₁,...,Aₚ]`,`[B₁,...,Bₚ]`.

Terms in the decomposition are actually shifted by one; this does not change the
eigenvalues but requires attention when dealing with invariant subspaces.
"""
function gpschur(As::AbstractVector{MT},Bs::AbstractVector{MT}; kwargs...
                 ) where {MT<:AbstractMatrix{T}} where {T}
    Cs, Ss = _mkpsargs(As, Bs)
    pschur!(Cs, Ss; kwargs...)
end

# construct argument `Cs,S` to pschur!
function _mkpsargs(As::Vector{MT},Bs) where {MT<:AbstractMatrix{T}} where {T}
    ph = length(As)
    ib = (ph==1) ? 1 : (ph-1)
    Cs = [copy(As[ph]) .+ 0im, copy(Bs[ib]) .+ 0im]
    Ss = [true,false]
    for j in ph-1:-1:1
        push!(Cs,copy(As[j]) .+ 0im)
        jx = (j == 1) ? ph : (j-1)
        push!(Cs,copy(Bs[jx]) .+ 0im)
        push!(Ss,true)
        push!(Ss,false)
    end
    return Cs,Ss
end


"""
    checkpsd(P::AbstractPeriodicSchur{T}, As::Vector{Matrix{T}})

Verify integrity of a (generalized) periodic Schur decomposition.
Returns a status code (Bool) and a vector of
normalized factorization errors (which should be O(1)).
"""
function checkpsd(P::AbstractPeriodicSchur{T}, Hs::AbstractVector;
                  quiet=false, thresh=100, strict=false) where T
    # Hs could be structured matrices of different varieties, caveat emptor.
    p = length(Hs)
    n = size(P.T1,1)
    S = isa(P,GeneralizedPeriodicSchur) ? P.S : trues(p)
    if P.period != p
        throw(DimensionMismatch("length of Hs vector must match period of P"))
    end
    for l in 1:p
        n1 = checksquare(Hs[l])
        if n1 != n
            throw(DimensionMismatch("size of Hs matrices must match P"))
        end
    end
    ttol = 10
    qtol = 10
    err = zeros(p)
    result = true
    Ts = []
    js = P.schurindex
    jt = 0
    for j in 1:p
        if j == P.schurindex
            push!(Ts, Matrix(P.T1))
        else
            jt += 1
            push!(Ts, Matrix(P.T[jt]))
        end
    end
    for l in 1:p
        l1 = mod(l,p)+1
        Tl = Ts[l]
        cmp = strict ? 0 : ttol * eps(real(T))*n
        if norm(tril(Tl,-1)) > cmp
            if !quiet
                @warn "triangularity fails for l=$l"
            end
            result = false
        end
        if norm(P.Z[l]*P.Z[l]'-I) > qtol * eps(real(T))*n
            if !quiet
                @warn "orthogonality fails for l=$l"
            end
            result = false
        end
        Hl = Hs[l]
        # Note: MB03BZ description has conjugation on the wrong side
        if S[l] ⊻ (P.orientation == 'L')
            Hx = P.Z[l] * Tl * P.Z[l1]'
        else
            Hx = P.Z[l1] * Tl * P.Z[l]'
        end
        err[l] = norm(Hx - Hl) / eps(real(T)) / n
        if err[l] > thresh
            if !quiet
                @warn "large factorization error for l=$l"
            end
            result = false
        end
    end
    return result, err
end
