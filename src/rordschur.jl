# top-level reordering routines for real periodic schur
const _rord_verby = Ref(0)

function LinearAlgebra.ordschur!(P::AbstractPeriodicSchur{T}, select::AbstractVector{Bool};
                    wantZ=true, Z = nothing) where {T <: Real}
    p = P.period
    cshift = 0
    rev = false
    specialQ = wantZ && Z !== nothing
    vb = _rord_verby[] > 0

    # swap routine requires left orientation and schur index 1
    if P.orientation == 'R'
        P = _rev_alias(P)
        rev = true
    end

    if P.schurindex == 1
        Px = P
    elseif P.schurindex == p
        cshift = 1
        Px = _circshift(P,1)
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

    p = length(As)+1
    n = size(A1,1)

    # find subspace dimension
    m0 = sum(select)
    m = 0
    pair = false
    for l in 1:n
        if pair
            pair = false
            continue
        end
        if l < n
            if A1[l+1,l] == 0
                if select[l]
                    m += 1
                end
            else
                pair = true
                if select[l] || select[l+1]
                    m += 2
                end
            end
        else
            if select[n]
                m += 1
            end
        end
    end
    if m > m0
        vb && println("ordschur added $(m-m0) to select to handle conjugate pairs")
        #    maybe throw something if caller didn't indicate nonchalance?
    end

    jdest = 0
    pair = false
    for j in 1:n
        if pair
            pair = false
            continue
        end
        swap = select[j]
        if j < n
            if A1[j+1,j] != 0
                pair = true
                swap = swap || select[j+1]
            end
        end
        if swap
            jdest += 1
            jsrc = j
            # move j to js by swapping neighbors upwards
            if j != jdest
                vb && println("moveblock $jsrc -> $jdest"); j0i=jsrc; j1i=jdest;
                jsrc, jdest, ok = _moveblock!(Px, jsrc, jdest, wantZ, Q)
                ok || error("move $jsrc -> $jdest failed, sorry.")
                if jsrc != j0i || jdest != j1i
                    vb && println("actual $jsrc $jdest")
                end
            end
            if pair
                jdest += 1
            end
        end
    end
    # TODO:
    # _rebalance!(P)
    _updateλ!(P)
    return P
end

# compare to MB03KA
# jsrc, jdest -> IFST, ILST
# nbsrc, nbdest -> NBF, NBL
# except splitsrc is NBF == 3

# try to move singleton or pair at jsrc to jdest where jdest < jsrc
# this is painful because of possible real <--> complex transitions
function _moveblock!(P, jsrc, jdest, wantZ, Q)
    vb = _rord_verby[] > 0
    A1 = P.T1
    As = P.T
    p = length(As) + 1
    n = size(A1,1)
    # make sure jsrc and jdest point to singleton or first of a pair
    # and determine their block sizes
    if jsrc > 1 && A1[jsrc,jsrc-1] != 0
        vb && println("bump jsrc A1 sd $(jsrc-1):", A1[jsrc,jsrc-1])
        jsrc -= 1
    end
    nbsrc = 1
    if jsrc < n && A1[jsrc+1,jsrc] != 0
        vb && println("nbsrc=2 A1 sd $jsrc:", A1[jsrc+1,jsrc])
        nbsrc = 2
    end
    if jdest > 1 && A1[jdest,jdest-1] != 0
        vb && println("bump jdest A1 sd $(jdest-1):", A1[jdest,jdest-1])
        jdest -= 1
    end
    nbdest = 1
    if jdest < n && A1[jdest+1,jdest] != 0
        vb && println("nbdest=2 A1 sd $jdest:", A1[jdest+1,jdest])
        nbdest = 2
    end
    if jsrc == jdest
        return jsrc, jdest, true
    end
    (jdest < jsrc) || throw(ArgumentError("only jdest < jsrc is implemented"))

    ok = true

    vb && println("effective jsrc,jdest: $jsrc,$jdest")

    here = jsrc
    splitsrc = false # flag for two 1×1 blocks to move together
    while here > jdest
        if !splitsrc
            nbnext = 1
            if (here >= 3) && (A1[here-1, here-2] != 0)
                nbnext = 2
            end
            vb && println("swap $nbnext, $nbsrc at $(here-nbnext)")
            ok = _swapschur!(P,here-nbnext, nbnext, nbsrc, Q)
            if !ok
                jdest = here
                return jsrc, jdest, ok
            end
            here -= nbnext
            # check for split of 2x2
            if (nbsrc == 2) && (A1[here+1, here] == 0)
                splitsrc = true
            end
        else
            # source block has split
            nbnext = 1
            if (here >= 3) && (A1[here-1, here-2] != 0)
                nbnext = 2
            end
            vb && println("split; swap $nbnext, 1 at $(here-nbnext)")
            ok = _swapschur!(P,here-nbnext, nbnext, 1, Q)
            if !ok
                jdest = here
                return jsrc, jdest, ok
            end
            if nbnext == 1
                # swap two 1×1
                vb && println("-1 swap $nbnext, 1 at $here")
                ok = _swapschur!(P,here, nbnext, 1, Q)
                if !ok
                    jdest = here
                    return jsrc, jdest, ok
                end
            else
                # check for 2×2 split
                if A1[here, here-1] == 0
                    nbnext = 1
                end
                if nbnext == 2
                    # no split
                    vb && println("-2 swap 2,1 at $(here-1)")
                    ok = _swapschur!(P,here-1, 2, 1, Q)
                    if !ok
                        jdest = here
                        return jsrc, jdest, ok
                    end
                    here -= 2
                else
                    # split
                    vb && println("-3 swap 1,1 at $(here)")
                    ok = _swapschur!(P,here, 1, 1, Q)
                    if !ok
                        jdest = here
                        return jsrc, jdest, ok
                    end
                    vb && println("-4 swap 1,1 at $(here-1)")
                    ok = _swapschur!(P,here-1, 1, 1, Q)
                    if !ok
                        jdest = here
                        return jsrc, jdest, ok
                    end
                    here -= 2
                end
            end
        end
    end
    jdest = here

    return jsrc, jdest, ok
end

function _swapschur!(P, i1, nb1, nb2, Q)
    if (nb1 == 1) && (nb2 == 1)
        ok = _swapschur1!(P,i1,Q !== nothing, Q)
    else
        ok = _swapadjqr!(P.T1, P.T, Q, i1, nb1, nb2)
    end
    return ok
end
