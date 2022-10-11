include("sylvester.jl")
include("sylswap.jl")
include("rpschur2x2.jl")

"""
    ordschur!(P::AbstractPeriodicSchur{T}, select::AbstractVector{Bool})

reorder a periodic Schur decomposition so that the eigenvalues corresponding
to `true` entries in `select` and their associated subspace are moved to the top.
"""
function LinearAlgebra.ordschur!(P::AbstractPeriodicSchur{T}, select::AbstractVector{Bool};
                                 wantZ = true, Z = nothing) where {T <: Complex}
    # restrict to complex eltype for 1x1 swaps
    p = P.period
    cshift = 0
    rev = false
    specialQ = wantZ && Z !== nothing

    # swap routine requires left orientation and schur index 1
    if P.orientation == 'R'
        Parg = P
        P = _rev_alias(P)
        rev = true
    end

    if P.schurindex == 1
        Px = P
    elseif P.schurindex == p
        cshift = 1
        Px = _circshift(P, 1)
    else
        throw(ArgumentError("only implemented for schurindex in (1,p)"))
    end
    if wantZ
        if specialQ
            if rev
                throw(NotImplemented("no logic for reversing supplementary Z"))
            end
            Q = circshift(Z, cshift)
        else
            Q = Px.Z
        end
    else
        Q = nothing
    end

    A1 = Px.T1
    As = Px.T

    p = length(As) + 1
    n = size(A1, 1)
    m = sum(select)
    js = 0 # destination
    for j in 1:n
        if select[j]
            js += 1
            # move j to js by swapping neighbors upwards
            if j != js
                for i in (j - 1):-1:js
                    ok = _swapschur1!(Px, i, wantZ, Q)
                    ok || throw(IllConditionedException(j))
                end
            end
        end
    end
    # TODO:
    # _rebalance!(P)
    if rev
        P = Parg
    end
    _updateλ!(P)
    return P
end

function _updateλ!(P::GeneralizedPeriodicSchur{T}) where {T <: Complex}
    p = length(P.T) + 1
    v4ev = zeros(T, p)
    A1 = P.T1
    As = P.T
    n = size(A1, 1)
    for j in 1:n
        il = 1
        for l in 1:p
            if l == P.schurindex
                v4ev[l] = A1[j, j]
            else
                v4ev[l] = As[il][j, j]
                il += 1
            end
        end
        a, b, sc = _safeprod(P, v4ev)
        P.α[j] = a
        P.β[j] = b
        P.αscale[j] = sc
    end
end
function _updateλ!(P::PeriodicSchur{T}) where {T <: Complex}
    p = length(P.T) + 1
    v4ev = zeros(T, p)
    A1 = P.T1
    As = P.T
    n = size(A1, 1)
    for j in 1:n
        v4ev[1] = A1[j, j]
        if j < n && abs(A1[j + 1, j]) > 100 * eps(real(T))
            @error "unexpected subdiag in complex Schur form"
        end
        for l in 1:(p - 1)
            v4ev[l + 1] = As[l][j, j]
            if j < n && abs(As[l][j + 1, j]) > 100 * eps(real(T))
                @error "unexpected subdiag in complex Schur form"
            end
        end
        a, b, sc = _safeprod(P, v4ev)
        #P.α[j] = a
        #P.β[j] = b
        #P.αscale[j] = sc
        P.values[j] = a / b * T(2 * one(real(T)))^sc
    end
end

function _updateλ!(P::PeriodicSchur{T}; strict = true) where {T <: Real}
    p = length(P.T) + 1
    sdtol = strict ? 0 : 100 * eps(T)
    v4ev = zeros(complex(T), p)
    A1 = P.T1
    As = P.T
    n = size(A1, 1)
    xS = trues(p)
    xAord = 1:p # we will rearrange here

    j = 1
    pairflag = false
    while j <= n
        il = 0
        for l in 1:p
            if l == P.schurindex
                Al = A1
            else
                il += 1
                Al = As[il]
            end
            v4ev[l] = Al[j, j]
            if j < n && abs(Al[j + 1, j]) > sdtol
                if P.schurindex == l
                    pairflag = true
                else
                    @error "unexpected subdiag in triang factor $l at $j: $(Al[j+1,j])"
                end
            end
        end
        if pairflag
            xA1 = view(A1, j:(j + 1), j:(j + 1))
            si = P.schurindex
            if P.orientation == 'L'
                if si in (1, p)
                    # simply reverse
                    xAs = [view(As[p - l], j:(j + 1), j:(j + 1)) for l in 1:(p - 1)]
                else
                    xAs = [view(As[si - 1], j:(j + 1), j:(j + 1))]
                    for l in (si - 2):-1:1
                        push!(xAs, view(As[l], j:(j + 1), j:(j + 1)))
                    end
                    for l in (p - 1):-1:si
                        push!(xAs, view(As[l], j:(j + 1), j:(j + 1)))
                    end
                end
            else
                if si in (1, p)
                    xAs = [view(As[l], j:(j + 1), j:(j + 1)) for l in 1:(p - 1)]
                else
                    xAs = [view(As[si], j:(j + 1), j:(j + 1))]
                    for l in (si + 1):(p - 1)
                        push!(xAs, view(As[l], j:(j + 1), j:(j + 1)))
                    end
                    for l in 1:(si - 1)
                        push!(xAs, view(As[l], j:(j + 1), j:(j + 1)))
                    end
                end
            end
            α, β, scal, cvg, good = _rpeigvals2x2(xA1, xAs, xS, xAord, 1)
            if !cvg
                @warn "recomputation of eigvals did not converge; accuracy is suspect"
            elseif !good
                @warn "recomputation of eigvals deviates from reality; accuracy is suspect"
            end
            P.values[j] = α[1] * T(2 * one(real(T)))^scal[1]
            P.values[j + 1] = α[2] * T(2 * one(real(T)))^scal[2]
            j += 2
            pairflag = false
        else
            a, b, sc = _safeprod(P, v4ev)
            #P.α[j] = a
            #P.β[j] = b
            #P.αscale[j] = sc
            P.values[j] = a / b * T(2 * one(real(T)))^sc
            j += 1
        end
    end
end

# swap 1×1 blocks at `i1:i1+1`
function _swapschur1!(P::PeriodicSchur, i1, wantZ, Q)
    if P.orientation != 'L' || P.schurindex != 1
        throw(ArgumentError("only implemented for left orientation w/ schurindex=1"))
    end
    _swapadj1x1g!(P.T1, P.T, wantZ ? Q : nothing, i1)
end
function _swapschur1!(P::GeneralizedPeriodicSchur, i1, wantZ, Q)
    if P.orientation != 'L' || P.schurindex != 1
        throw(ArgumentError("only implemented for left orientation w/ schurindex=1"))
    end
    _swapadj1x1g!(P.T1, P.T, wantZ ? Q : nothing, P.S, i1)
end

include("rordschur.jl")
