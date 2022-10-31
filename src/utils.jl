"""
    _circshift(P::AbstractPeriodicSchur, n) -> P_new

construct circular shift of `P`, aliasing content
"""
function _circshift(P::GeneralizedPeriodicSchur, n)
    p = P.period
    n = mod(n, p)
    n == 0 && return P
    ks = P.schurindex
    ksnew = mod(ks + n - 1, p) + 1
    Znew = circshift(P.Z, n)
    Snew = circshift(P.S, n)
    Tnew = similar(P.T)
    # println("ks old $ks new $ksnew")
    tidx = circshift(1:(p - 1), mod(ksnew - ks, p - 1))
    # println("1:p-1 -> $tidx")
    for l in 1:(p - 1)
        Tnew[l] = P.T[tidx[l]]
    end
    return GeneralizedPeriodicSchur(Snew, ksnew, P.T1, Tnew, Znew, P.α, P.β, P.αscale,
                                    P.orientation)
end

function _circshift(P::PeriodicSchur, n)
    p = length(P.T) + 1
    n = mod(n, p)
    n == 0 && return P
    ks = P.schurindex
    ksnew = mod(ks + n - 1, p) + 1
    Znew = circshift(P.Z, n)
    Tnew = similar(P.T)
    # println("ks old $ks new $ksnew")
    tidx = circshift(1:(p - 1), mod(ksnew - ks, p - 1))
    # println("1:p-1 -> $tidx")
    for l in 1:(p - 1)
        Tnew[l] = P.T[tidx[l]]
    end
    return PeriodicSchur(P.T1, Tnew, Znew, P.values, P.orientation, ksnew)
end

"""
    _rev_alias(P::AbstractPeriodicSchur) -> P_new

Create a new decomposition structure with reversed orientation,
aliasing the matrix content.
Warning: does not alias eigenvalue vectors.
"""
function _rev_alias(P::GeneralizedPeriodicSchur)
    #P.schurindex == 1 || throw(ArgumentError("only implemented for schurindex=1"))
    p = P.period
    lorient = P.orientation == 'L' ? 'R' : 'L'
    ksl = p + 1 - P.schurindex
    Zl = similar(P.Z)
    if length(P.Z) > 0
        # what I think it should be to shift at same time
        #Zl[1] = Z[2]
        #Zl[2] = Z[1]
        #for l in 3:p
        #    Zl[l] = Z[p+3-l]
        #end
        Zl[1] = P.Z[1]
        for l in 2:p
            Zl[l] = P.Z[p + 2 - l]
        end
    end
    Tl = reverse(P.T)
    Sl = reverse(P.S)
    return GeneralizedPeriodicSchur(Sl, ksl, P.T1, Tl, Zl, P.α, P.β, P.αscale, lorient)
end

function _rev_alias(P::PeriodicSchur)
    p = P.period
    ksl = p + 1 - P.schurindex
    lorient = P.orientation == 'L' ? 'R' : 'L'
    Zl = similar(P.Z)
    if length(P.Z) > 0
        Zl[1] = P.Z[1]
        for l in 2:p
            Zl[l] = P.Z[p + 2 - l]
        end
    end
    Tl = reverse(P.T)
    return PeriodicSchur(P.T1, Tl, Zl, P.values, lorient, ksl)
end

@inline _isnotinv(P::PeriodicSchur, l) = true
@inline _isnotinv(P::GeneralizedPeriodicSchur, l) = P.S[l]

function _safeprod(P, as::AbstractVector{T}) where {T}
    α = one(T)
    β = one(real(T))
    k = length(as)
    scal = 0
    for l in 1:k
        if _isnotinv(P, l)
            α *= as[l]
        else
            if as[l] == 0
                β = zero(real(T))
            else
                α /= as[l]
            end
        end
        if abs(α) == 0
            scal = 0
            α = zero(T)
        else
            while abs(α) < 1
                α *= 2
                scal -= 1
            end
            while abs(α) >= 2
                α /= 2
                scal += 1
            end
        end
    end
    return α, β, scal
end

# avoid crypic LAPACK exception
function _checkqr(R)
    @inbounds for jj in diagind(R)
        if R[jj] == 0
            n = size(R, 1)
            j = (jj - 1) ÷ n + 1
            throw(SingularException(j))
        end
    end
end
