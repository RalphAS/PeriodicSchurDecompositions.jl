# Real generalized periodic Schur decomposition

function pschur!(A::AbstractVector{TA}, S::AbstractVector{Bool},
                 lr::Symbol = :R;
                 wantZ::Bool = true,
                 wantT::Bool = true,
                 aggressive::Bool = false,
                 ) where {TA <: AbstractMatrix{T}} where {T <: Real}
    orient = char_lr(lr)
    p = length(A)
    if orient == 'L'
        Aarg = similar(A)
        for j in 1:p
            Aarg[j] = A[p + 1 - j]
        end
        Sarg = reverse(S)
    else
        Aarg = A
        Sarg = S
    end
    if all(S)
        H1, pH = phessenberg!(Aarg)
        if wantZ
            Q = [_materializeQ(H1)]
            for j in 1:(p - 1)
                push!(Q, Matrix(pH[j].Q))
            end
        else
            Q = nothing
        end
        Hs = [pH[j].R for j in 1:(p - 1)]
        H1.H .= triu(H1.H, -1)
        F = pschur!(H1.H, Hs, Sarg, wantT = wantT, wantZ = wantZ, Q = Q,
                    rev = (orient == 'L'))
    else
        # check this here so error message is less confusing
        Sarg[1] || throw(ArgumentError("The leftmost entry in S must be true"))
        Hs, Qs = _phessenberg!(Aarg, Sarg)
        H1 = popfirst!(Hs)
        Q = wantZ ? Qs : nothing
        F = pschur!(H1, Hs, Sarg, wantT = wantT, wantZ = wantZ, Q = Q,
                    rev = (orient == 'L'))
    end
    return F
end

# Mainly translated from SLICOT MB03BD, by D.Kressner and V.Sima
# MB03BD is Copyright (c) 2002-2020 NICONET e.V.
function pschur!(H1H::Th1, Hs::AbstractVector{Th},
                 S::AbstractVector{Bool};
                 wantZ::Bool = true, wantT::Bool = true,
                 Q::Union{Nothing, Vector{Tq}} = nothing, maxitfac = 120,
                 rev::Bool = false,
                 aggressive::Bool = false
                 ) where {
                     Th1 <: Union{UpperHessenberg{T}, StridedMatrix{T}},
                     Th <: StridedMatrix{T},
                     Tq <: StridedMatrix{T}
                 } where {T <: Real}
    p = length(Hs) + 1
    n = checksquare(H1H)
    recip = !S[1]

    # how many iters w/ each kind of shift to take in deflated subproblem
    # before switching
    nimplicit = 10
    nexplicit = 1

    for l in 1:(p - 1)
        n1 = checksquare(Hs[l])
        n1 == n || throw(DimensionMismatch("H matrices must have the same size"))
    end
    S[1] || throw(ArgumentError("Signature entry S[1] must be true"))


    α = zeros(complex(T), n)
    β = zeros(T, n)
    αscale = zeros(Int, n)

    safmin = floatmin(T)
    unfl = floatmin(T)
    ovfl = 1 / unfl
    ulp = eps(T)
    smlnum = unfl * (n / ulp)

    H1 = _gethess!(H1H)
    Hp = p == 1 ? H1 : Hs[p - 1]

    if wantZ
        if Q === nothing
            Z = [Matrix{T}(I, n, n) for l in 1:p]
        else
            for l in 1:p
                if size(Q[l], 2) != n
                    throw(DimensionMismatch("second dimension of Q's must agree w/ H"))
                end
            end
            Z = Q
        end
    else
        Z = Vector{Matrix{T}}(undef, 0)
    end

    @_dbg_rgpschur fcheck = _FacChecker(H1, Hs, Z, wantZ, S)

    # decide whether we need an extra iteration w/ controlled zero shift
    ziter = (p >= log2(floatmin(T)) / log2(ulp)) ? -1 : 0

    # Frobenius norms this time
    hnorms = zeros(T, p) # dwork[pnorm+1:pnorm+p]
    for j in 2:p
        triu!(Hs[j - 1], -1)
        hnorms[j] = norm(Hs[j - 1])
    end

    Gtmp = Vector{Givens{T}}(undef, n)
    v4ev = zeros(T, p - 1)

    # some of the 2x2 block logic wants independent mutables
    H2s = [zeros(T, 2, 2) for _ in 1:p]
    # ... and a deviant ordering
    S2 = circshift(S,-1)

    # diagnostic routines
    function showmat(str, j)
        print(str, " H[$j] ")
        show(stdout, "text/plain", j == 1 ? H1 : Hs[j - 1])
        println()
        nothing
    end
    function showallmats(str)
        for l in 1:p
            showmat(str, l)
        end
        nothing
    end
    function showprod(str)
        if verbosity[] > 2 && p > 1
            Htmp = copy(H1)
            for l in 2:p
                Htmp = Htmp * Hs[l - 1]
            end
            print(str, " ℍ ")
            show(stdout, "text/plain", Htmp)
            println()
            println("  ev: ", eigvals(Htmp)')
        end
        nothing
    end

    # Below, eigvals in ilast+1:n have been found.
    # Column ops modify rows ifirstm:whatever
    # Row ops modify columns whatever:ilastm
    # If only eigvals are needed, ifirstm is the row of the last split row above ilast.

    ilast = n
    ifirst = 1
    ifirstm = 1
    ilastm = n
    iiter = 1
    maxit = maxitfac * n
    iwarn = 0
    iimplicit = 0
    iexplicit = 0

    verbosity[] > 0 && println("Real generalized periodic QR p=$p n=$n")

    done = false
    for jiter in 1:maxit
        verbosity[] > 0 && println("iter $jiter $ifirst:$ilast")
        done && break
        split1block = false  # flag for 390
        ldeflate = -1
        jdeflate = -1
        deflate_pos = false # flag for 170
        deflate_neg = false # flag for 320
        doqziter = true
        czshift = false # for developer to check logic
        jlo = 1
        ilo = 1
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
            split1block, jlo =  _check_deflate_hess(H1, 1, ilast, ulp, smlnum,
                        aggressive ? max(safmin, hnorms[1] * ulp) : nothing)
            # split1block && println("preparing to split 1")
            split1block && break

            # Test 2: deflation in triangular matrices w/ index 1
            deflate_pos = false
            for l in 2:p
                if S[l]
                    Hl = Hs[l - 1]
                    deflate_pos, jx = _check_deflate_tr(Hl, jlo, ilast, ulp, smlnum,
                          aggressive ? max(safmin, hnorms[l] * ulp) : nothing)
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
                    Hl = Hs[l - 1]
                    deflate_neg, jx = _check_deflate_tr(Hl, jlo, ilast, ulp, smlnum,
                          aggressive ? max(safmin, hnorms[l] * ulp) : nothing)
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
                for j in jlo:(ilast - 1)
                    c, s, r = givensAlgorithm(H1[j, j], H1[j + 1, j])
                    H1[j, j] = r
                    H1[j + 1, j] = zero(T)
                    G = Givens(j, j + 1, c, s)
                    lmul!(G, view(H1, :, (j + 1):ilastm))
                    Gtmp[j] = G
                end
                if wantZ
                    for j in jlo:(ilast - 1)
                        rmul!(Z[1], Gtmp[j]')
                    end
                end
                if aggressive
                    throw(NotImplemented("mods for aggressive zero shift"))
                end
                # propagate transformations back to H1
                for l in p:-1:2
                    Hl = Hs[l - 1]
                    if S[l]
                        for j in jlo:(ilast - 1)
                            G = Gtmp[j]
                            if G.s != 0
                                rmul!(view(Hl, ifirstm:(j + 1), :), Gtmp[j]')
                                # check for deflation
                                tol = abs(Hl[j, j]) + abs(Hl[j + 1, j + 1])
                                if tol == 0
                                    tol = opnorm(view(Hl, jlo:(j + 1), jlo:(j + 1)), 1)
                                end
                                tol = max(ulp * tol, smlnum)
                                if abs(Hl[j + 1, j]) <= tol
                                    c, s = one(T), zero(T)
                                    Hl[j + 1, j] = zero(T)
                                    G = Givens(j, j + 1, c, s)
                                    Gtmp[j] = G
                                else
                                    c, s, r = givensAlgorithm(Hl[j, j], Hl[j + 1, j])
                                    Hl[j, j] = r
                                    Hl[j + 1, j] = zero(T)
                                    G = Givens(j, j + 1, c, s)
                                    lmul!(G, view(Hl, :, (j + 1):ilastm))
                                    Gtmp[j] = G
                                end
                            end
                        end # j loop
                    else # not S[l]
                        for j in jlo:(ilast - 1)
                            G = Gtmp[j]
                            if G.s != 0
                                lmul!(G, view(Hl, :, j:ilastm))
                                # check for deflation
                                tol = abs(Hl[j, j]) + abs(Hl[j + 1, j + 1])
                                if tol == 0
                                    tol = opnorm(view(Hl, jlo:(j + 1), jlo:(j + 1)), 1)
                                end
                                tol = max(ulp * tol, smlnum)
                                if abs(Hl[j + 1, j]) <= tol
                                    c, s = one(T), zero(T)
                                    Hl[j + 1, j] = zero(T)
                                    G = Givens(j, j + 1, c, -s)
                                    Gtmp[j] = G
                                else
                                    c, s, r = givensAlgorithm(Hl[j + 1, j + 1],
                                                              Hl[j + 1, j])
                                    Hl[j + 1, j + 1] = r
                                    Hl[j + 1, j] = zero(T)
                                    G = Givens(j + 1, j, c, s') # backwards!
                                    rmul!(view(Hl, ifirstm:j, :), G')
                                    Gtmp[j] = Givens(j, j + 1, c, -s)
                                end
                            end
                        end
                    end
                    if wantZ
                        for j in jlo:(ilast - 1)
                            rmul!(Z[l], Gtmp[j]')
                        end
                    end
                end # loop over l propagating transformations back to H1

                # Apply transformations to right side of Hessenberg factor
                ziter = 0
                for j in jlo:(ilast - 1)
                    G = Gtmp[j]
                    rmul!(view(H1, ifirstm:(j + 1), :), G')
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
            verbosity[] > 0 &&
                println("deflating S+ l=$ldeflate j=$jdeflate $ifirst:$ilast")
            # Do an unshifted pQZ step

            verbosity[] > 2 && showallmats("before deflation jlo=$jlo")

            # left of Hessenberg
            for j in jlo:(jdeflate - 1)
                c, s, r = givensAlgorithm(H1[j, j], H1[j + 1, j])
                H1[j, j] = r
                H1[j + 1, j] = 0
                G = Givens(j, j + 1, c, s)
                lmul!(G, view(H1, :, (j + 1):ilastm))
                Gtmp[j] = G
            end
            if wantZ
                for j in jlo:(jdeflate - 1)
                    rmul!(Z[1], Gtmp[j]')
                end
            end
            # propagate through triangular matrices
            for l in p:-1:2
                # due to zero on diagonal of H[ldeflate], decrement count
                ntra = (l < ldeflate) ? (jdeflate - 2) : (jdeflate - 1)
                Hl = Hs[l - 1]
                if S[l]
                    for j in jlo:ntra
                        G = Gtmp[j]
                        rmul!(view(Hl, ifirstm:(j + 1), :), Gtmp[j]')
                        c, s, r = givensAlgorithm(Hl[j, j], Hl[j + 1, j])
                        Hl[j, j] = r
                        Hl[j + 1, j] = 0
                        G = Givens(j, j + 1, c, s)
                        lmul!(G, view(Hl, :, (j + 1):ilastm))
                        Gtmp[j] = G
                    end
                else
                    for j in jlo:ntra
                        lmul!(Gtmp[j], view(Hl, :, j:ilastm))
                        c, s, r = givensAlgorithm(Hl[j + 1, j + 1], Hl[j + 1, j])
                        Hl[j + 1, j + 1] = r
                        Hl[j + 1, j] = 0
                        G = Givens(j + 1, j, c, s')
                        rmul!(view(Hl, ifirstm:j, :), G')
                        Gtmp[j] = Givens(j, j + 1, c, -s)
                    end
                end
                if wantZ
                    for j in jlo:ntra
                        rmul!(Z[l], Gtmp[j]')
                    end
                end
            end
            # right of Hessenberg
            for j in jlo:(jdeflate - 2)
                rmul!(view(H1, ifirstm:(j + 1), :), Gtmp[j]')
            end

            # Do another unshifted periodic QZ step
            # right of Hessenberg
            for j in ilast:-1:(jdeflate + 1)
                c, s, r = givensAlgorithm(H1[j, j], H1[j, j - 1])
                H1[j, j] = r
                H1[j, j - 1] = 0
                G = Givens(j, j - 1, c, s')
                rmul!(view(H1, ifirstm:(j - 1), :), G')
                Gtmp[j] = Givens(j - 1, j, c, -s)
            end
            if wantZ
                for j in ilast:-1:(jdeflate + 1)
                    rmul!(Z[2], Gtmp[j]')
                end
            end
            # propagate through triangular matrices
            for l in 2:p
                ntra = l > ldeflate ? (jdeflate + 2) : (jdeflate + 1)
                Hl = Hs[l - 1]
                if !S[l]
                    for j in ilast:-1:ntra
                        rmul!(view(Hl, ifirstm:j, :), Gtmp[j]')
                        c, s, r = givensAlgorithm(Hl[j - 1, j - 1], Hl[j, j - 1])
                        Hl[j - 1, j - 1] = r
                        Hl[j, j - 1] = 0
                        G = Givens(j - 1, j, c, s)
                        lmul!(G, view(Hl, :, j:ilastm))
                        Gtmp[j] = Givens(j - 1, j, c, s)
                    end
                else
                    for j in ilast:-1:ntra
                        G = Gtmp[j]
                        lmul!(Gtmp[j], view(Hl, :, (j - 1):ilastm))
                        c, s, r = givensAlgorithm(Hl[j, j], Hl[j, j - 1])
                        Hl[j, j] = r
                        Hl[j, j - 1] = 0
                        G = Givens(j, j - 1, c, s')
                        rmul!(view(Hl, ifirstm:(j - 1), :), G')
                        Gtmp[j] = Givens(j - 1, j, c, -s)
                    end
                end
                if wantZ
                    ln = mod(l, p) + 1
                    for j in ilast:-1:ntra
                        rmul!(Z[ln], Gtmp[j]')
                    end
                end
            end
            # left of Hessenberg
            for j in ilast:-1:(jdeflate + 2)
                G = Gtmp[j]
                lmul!(Gtmp[j], view(H1, :, (j - 1):ilastm))
            end
            verbosity[] > 2 && showallmats("after deflation")
            doqziter = false

        elseif deflate_neg # Case III (320)
            verbosity[] > 0 && println("deflating S- l=$ldeflate j=$jdeflate")
            if jdeflate > (ilast - jlo + 1) / 2 # bottom half
                # chase the zero down
                for j1 in jdeflate:(ilast - 1)
                    j = j1
                    Hl = Hs[ldeflate - 1]
                    c, s, r = givensAlgorithm(Hl[j, j + 1], Hl[j + 1, j + 1])
                    Hl[j, j + 1] = r
                    Hl[j + 1, j + 1] = 0
                    G = Givens(j, j + 1, c, s)
                    lmul!(G, view(Hl, :, (j + 2):ilastm))
                    ln = mod(ldeflate, p) + 1
                    if wantZ
                        rmul!(Z[ln], G')
                    end
                    for l in 1:(p - 1)
                        if ln == 1
                            lmul!(G, view(H1, :, (j - 1):ilastm))
                            c, s, r = givensAlgorithm(H1[j + 1, j], H1[j + 1, j - 1])
                            H1[j + 1, j] = r
                            H1[j + 1, j - 1] = 0
                            G = Givens(j, j - 1, c, s')
                            rmul!(view(H1, ifirstm:j, :), G')
                            G = Givens(j - 1, j, c, -s)
                            j -= 1
                        elseif S[ln]
                            Hln = Hs[ln - 1]
                            lmul!(G, view(Hln, :, j:ilastm))
                            c, s, r = givensAlgorithm(Hln[j + 1, j + 1], Hln[j + 1, j])
                            Hln[j + 1, j + 1] = r
                            Hln[j + 1, j] = 0
                            G = Givens(j + 1, j, c, s')
                            rmul!(view(Hln, ifirstm:j, :), G')
                            G = Givens(j, j + 1, c, -s)
                        else
                            Hln = Hs[ln - 1]
                            rmul!(view(Hln, ifirstm:(j + 1), :), G')
                            c, s, r = givensAlgorithm(Hln[j, j], Hln[j + 1, j])
                            Hln[j, j] = r
                            Hln[j + 1, j] = 0
                            G = Givens(j, j + 1, c, s)
                            lmul!(G, view(Hln, :, (j + 1):ilastm))
                        end
                        ln = mod(ln, p) + 1
                        if wantZ
                            rmul!(Z[ln], G')
                        end
                    end # l loop
                    Hl = Hs[ldeflate - 1]
                    rmul!(view(Hl, ifirstm:j, :), G')
                end # j1 loop (340)
                # deflate last element in Hessenberg
                j = ilast
                c, s, r = givensAlgorithm(H1[j, j], H1[j, j - 1])
                H1[j, j] = r
                H1[j, j - 1] = 0
                G = Givens(j, j - 1, c, s')
                rmul!(view(H1, ifirstm:(j - 1), :), G')
                G = Givens(j - 1, j, c, -s)
                if wantZ
                    rmul!(Z[2], G')
                end
                for l in 2:(ldeflate - 1)
                    Hl = Hs[l - 1]
                    if !S[l]
                        rmul!(view(Hl, ifirstm:j, :), G')
                        c, s, r = givensAlgorithm(Hl[j - 1, j - 1], Hl[j, j - 1])
                        Hl[j - 1, j - 1] = r
                        Hl[j, j - 1] = 0
                        G = Givens(j - 1, j, c, s)
                        lmul!(G, view(Hl, :, j:ilastm))
                    else
                        lmul!(G, view(Hl, :, (j - 1):ilastm))
                        c, s, r = givensAlgorithm(Hl[j, j], Hl[j, j - 1])
                        Hl[j, j] = r
                        Hl[j, j - 1] = 0
                        G = Givens(j, j - 1, c, s')
                        rmul!(view(Hl, ifirstm:(j - 1), :), G')
                        G = Givens(j - 1, j, c, -s)
                    end
                    if wantZ
                        ln = mod(l, p) + 1
                        rmul!(Z[ln], G')
                    end
                end # l loop (350)
                Hl = Hs[ldeflate - 1]
                rmul!(view(Hl, ifirstm:j, :), G')
            else # jdeflate in top half
                # chase the zero up to the first position
                for j1 in jdeflate:-1:(jlo + 1)
                    j = j1
                    Hl = Hs[ldeflate - 1]
                    c, s, r = givensAlgorithm(Hl[j - 1, j], Hl[j - 1, j - 1])
                    Hl[j - 1, j] = r
                    Hl[j - 1, j - 1] = 0
                    G = Givens(j, j - 1, c, conj(s))
                    rmul!(view(Hl, ifirstm:(j - 2), :), G')
                    G = Givens(j - 1, j, c, -s)
                    if wantZ
                        rmul!(Z[ldeflate], G')
                    end
                    ln = ldeflate - 1
                    for l in 1:(p - 1)
                        Hln = ln == 1 ? H1 : Hs[ln - 1]
                        if ln == 1
                            rmul!(view(Hln, ifirstm:(j + 1), :), G')
                            c, s, r = givensAlgorithm(Hln[j, j - 1], Hln[j + 1, j - 1])
                            Hln[j, j - 1] = r
                            Hln[j + 1, j - 1] = 0
                            G = Givens(j, j + 1, c, s)
                            lmul!(G, view(Hln, :, j:ilastm))
                            j += 1
                        elseif !S[ln]
                            lmul!(G, view(Hln, :, (j - 1):ilastm))
                            c, s, r = givensAlgorithm(Hln[j, j], Hln[j, j - 1])
                            Hln[j, j] = r
                            Hln[j, j - 1] = 0
                            G = Givens(j, j - 1, c, s')
                            rmul!(view(Hln, ifirstm:(j - 1), :), G')
                            G = Givens(j - 1, j, c, -s)
                        else
                            rmul!(view(Hln, ifirstm:j, :), G')
                            c, s, r = givensAlgorithm(Hln[j - 1, j - 1], Hln[j, j - 1])
                            Hln[j - 1, j - 1] = r
                            Hln[j, j - 1] = 0
                            G = Givens(j - 1, j, c, s)
                            lmul!(G, view(Hln, :, j:ilastm))
                        end
                        if wantZ
                            rmul!(Z[ln], G')
                        end
                        ln = (ln == 1) ? p : (ln - 1)
                    end # l loop (360)
                    Hl = Hs[ldeflate - 1]
                    lmul!(G, view(Hl, :, j:ilastm))
                end # j1 loop
                # Deflate the first element in Hessenberg
                j = jlo
                c, s, r = givensAlgorithm(H1[j, j], H1[j + 1, j])
                H1[j, j] = r
                H1[j + 1, j] = 0
                G = Givens(j, j + 1, c, s)
                lmul!(G, view(H1, :, (j + 1):ilastm))
                if wantZ
                    rmul!(Z[1], G')
                end
                for l in p:-1:(ldeflate + 1)
                    Hl = Hs[l - 1]
                    if S[l]
                        rmul!(view(Hl, ifirstm:(j + 1), :), G')
                        c, s, r = givensAlgorithm(Hl[j, j], Hl[j + 1, j])
                        Hl[j, j] = r
                        Hl[j + 1, j] = 0
                        G = Givens(j, j + 1, c, s)
                        lmul!(G, view(Hl, :, (j + 1):ilastm))
                    else
                        lmul!(G, view(Hl, :, j:ilastm))
                        c, s, r = givensAlgorithm(Hl[j + 1, j + 1], Hl[j + 1, j])
                        Hl[j + 1, j + 1] = r
                        Hl[j + 1, j] = 0
                        G = Givens(j + 1, j, c, conj(s))
                        rmul!(view(Hl, ifirstm:j, :), G')
                        G = Givens(j, j + 1, c, -s)
                    end
                    if wantZ
                        rmul!(Z[l], G')
                    end
                end # trailing l loop (380)
                Hl = Hs[ldeflate - 1]
                lmul!(G, view(Hl, :, (j + 1):ilastm))
            end # jdeflate top/bottom branches
            doqziter = false
        elseif split1block # (390)
            verbosity[] > 0 && println("splitting 1x1 index $ilast")
            for l in 1:(p - 1)
                v4ev[l] = Hs[l][ilast, ilast]
            end
            α[ilast], β[ilast], αscale[ilast] = _safeprod(S, H1[ilast, ilast], v4ev)
            if verbosity[] > 0 && isreal(α[ilast]) && ilast < n
                println("claim zero subdiag[$ilast]: ", H1[ilast+1,ilast])
            end
            # TODO: check for loss of accuracy
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

#     end # fake out emacs indenter
# end # fake
# function foo() # fake
#     for jiter in 1:maxit # fake

        if doqziter # (420)
            iiter += 1
            ziter += 1
            if !wantT
                ifirstm = ifirst
            end
            if ifirst+1 == ilast
                # ll.1294-1492
                verbosity[] > 0 && println("attacking 2x2 block $ifirst:$ilast")
                titer = 0
                done2x2 = false
                j = ilast - 1
                # CHECKME: deviate from SLICOT: reuse H2s
                # PUZZLE: mb03bd might inadvertently shuffle the order here
                for l in 1:p
                    Hl = l == p ? H1 : Hs[l]
                    H2s[l] .= view(Hl, j:j+1, j:j+1)
                end
                while !done2x2 && titer < 2
                    titer += 1
                    # attempt to deflate 2x2
                    # PUZZLE: mb03bd tries this "twice", but I don't see
                    # what changes in the second iteration without
                    # moving this copy out of the loop
                    # for l in 1:p
                    #     Hl = l == p ? H1 : Hs[l - 1]
                    #     H2s[l] .= view(Hl, j:j+1, j:j+1)
                    # end
                    # real single-shifted pqz,
                    # Hessenberg is in last place
                    _rp2x2ssr!(H2s, S2)
                    # PUZZLE: mb03bd uses H2s[schurindex],
                    #    but mb03bf always uses last one for the quasi-tri
                    H2p = H2s[p]
                    verbosity[] > 0 && println("try $titer h21=$(H2p[2, 1])")
                    if abs(H2p[2, 1]) < ulp * max(abs(H2p[1, 1]),
                                                  abs(H2p[1, 2]),
                                                  abs(H2p[2, 2]))
                        done2x2 = true
                        # construct perfect shift polynomial
                        c1, s1 = one(T), one(T)
                        for l in p:-1:2
                            r = H2s[l - 1][2,2] # was H2s[l]
                            Hl = Hs[l - 1]
                            if S[l]
                                c1, s1, r = givensAlgorithm(c1 * Hl[j,j], s1 * r)
                            else
                                c1, s1, r = givensAlgorithm(c1 * r, s1 * Hl[j,j])
                            end
                        end
                        r = H2s[p][2,2] # was H2s[1]
                        Hl = H1
                        c1, s1, r = givensAlgorithm(c1 * Hl[j,j] - r * s1,
                                                  c1 * Hl[j + 1, j])
                        G2 = Givens(j, j+1, c1, s1)

                        # taken from section 510 in MB03BD:

                        # needs the usual series of rotations
                        # first applying G to H1, then H[l] l=k:-1:2
                        # with updates to G, finally H1

                        lmul!(G2, view(H1, :, j:ilastm))
                        if wantZ
                            rmul!(Z[1], G2')
                        end
                        for l in p:-1:2
                            Hl = Hs[l-1]
                            if S[l]
                                rmul!(view(Hl, ifirstm:j+1, :), G2')
                                c1, s1, r1 = givensAlgorithm(Hl[j, j], Hl[j + 1, j])
                                Hl[j, j] = r1
                                Hl[j + 1, j] = zero(T)
                                G2 = Givens(j, j + 1, c1, s1)
                                lmul!(G2, view(Hl, :, j+1:ilastm))
                            else
                                lmul!(G2, view(Hl, :, j:ilastm))
                                c1, s1, r1 = givensAlgorithm(Hl[j + 1, j + 1], -Hl[j + 1, j])
                                Hl[j + 1, j + 1] = r1
                                Hl[j + 1, j] = zero(T)
                                G2 = Givens(j, j + 1, c1, s1)
                                rmul!(view(Hl, ifirstm:j, :), G2')
                            end
                            if wantZ
                                rmul!(Z[l], G2')
                            end
                        end
                        rmul!(view(H1, ifirstm:ilastm, :), G2')

                        verbosity[] > 0 && println("deflated 2x2 block: reals")
                        @_dbg_rgpschur fcheck("after deflation", H1, Hs, Z, S)
                    end
                end # perfect shift trial loop
                if !done2x2
                    # apparently complex block
                    # todo: optional SVD (ll. 1349ff.)
                    # complex single-shifted pqz,
                    for l in 1:p
                        Hl = l == 1 ? H1 : Hs[l - 1]
                        H2s[l] .= view(Hl, j:j+1, j:j+1)
                    end
                    # α[j], β[j], converged, good = _rp2x2ssc!(H2s, S)
                    α2, β2, scal2, converged, good =
                        _rpeigvals2x2(H2s[1], view(H2s, 2:p), S, collect(1:p), 1; recip)
                    λs = α2 .* (T(2) .^ scal2)
                    verbosity[] > 0 && println("using 2x2 eigvals $λs")
                    α[j:j+1] .= α2
                    β[j:j+1] .= β2
                    αscale[j:j+1] .= scal2
                    if !converged
                        iwarn = max(iwarn, j)
                    elseif (!good) && (iwarn == 0)
                        iwarn = n
                    end
                    # ll 1419ff.
                    # FIXME: do this
                    # todo("check for reals, singularity, or loss of accuracy")

                    # prepare for next block; reset counters
                    ilast = ifirst - 1
                    if ilast < 1
                        done = true
                    end
                    iiter = 0
                    iimplicit = 0
                    iexplicit = 0
                    if ziter != -1
                        ziter = 0
                    end
                    if !wantT
                        ilastm = ilast
                        if ifirstm > ilast
                            ifirstm = ilo
                        end
                    end
                end
            else # not a 2x2
                verbosity[] > 0 && println("starting QZ loop $ifirst:$ilast")
                if ilast - ifirst + 1 < 3
                    @warn "QZ logic error ilast = $ilast ifirst = $ifirst"
                end
                if iimplicit < nimplicit
                    iimplicit += 1
                    # normal QZ
                    c1, s1, c2, s2 = _qzrots(H1, Hs, S, ifirst, ilast-ifirst+1)
                    if verbosity[] > 1
                        println("implicit shift")
                        @show (c1, s1, c2, s2)
                    end
                elseif iexplicit < nexplicit
                    # ll 1516-1666
                    # compute trailing eigvals to find shifts
                    i = ilast - 1
                    recip && error("missing logic for recip")
                    for l in 1:p
                        Hl = l == 1 ? H1 : Hs[l - 1]
                        H2s[l] .= view(Hl, i:i+1, i:i+1)
                    end
                    # α[j], β[j], converged, good = _rp2x2ssc!(H2s, S)
                    α2, β2, scal2, converged, good =
                        _rpeigvals2x2(H2s[1], view(H2s, 2:p), S, collect(1:p), 1; recip)
                    if !converged || !good
                        verbosity[] > 0 && println("rp2x2 unconverged")
                        # try an exceptional transformation
                        t2 = T(2)^scal2[1]
                        if imag(α2[1]) != 0
                            t = (abs(real(α2[1])) + abs(imag(α2[1]))) * t2
                        else
                            t = max(abs(real(α2[2])) * T(2)^scal2[2],
                                    abs(real(α2[1])) * t2)
                        end
                        if t < sqrt(ulp) * hnorm[1]
                            α2 .= hnorms[1]
                            scal2 .= one(T)
                            converged = true
                            verbosity[] > 0 && println("replace w/ exceptional value")
                        end
                    elseif verbosity[] > 0
                        println("using 2x2 eigvals $α2 in explicit shift")
                    end
                    α[i:i+1] .= α2
                    β[i:i+1] .= β2
                    αscale[i:i+1] .= scal2
                    if !converged
                        # normal periodic QZ step
                        c1, s1, c2, s2 = _qzrots(H1, Hs, S, ifirst)
                        iexplicit = 0
                        iimplicit = 0
                    else
                        # explicit shifts
                        iexplicit += 1
                        λ1 = α[i] * T(2) ^ αscale[i]
                        if imag(λ1) != 0
                            shft = :c
                            λ2 = conj(λ1)
                        else
                            # try 2 identical real shifts
                            # if no convergence, try single shift using
                            # closest to last elt of current product
                            λ2 = α[ilast] * T(2) ^ αscale[ilast]
                            for l in 1:p-1
                                v4ev[l] = Hs[l][ilast, ilast]
                            end
                            α0, β0, α0scale = _safeprod(S, H1[ilast, ilast], v4ev)
                            t = α0 * T(2) ^ α0scale
                            a1 = abs(t - real(λ1))
                            a2 = abs(t - real(λ2))
                            if iexplicit <= max(1, nexplicit / 2)
                                shft = :d
                                if a1 < a2
                                    λ2 = λ1
                                else
                                    λ1 = λ1
                                end
                            else
                                shft = :s
                                if a1 < a2
                                    λ2 = λ1
                                end
                            end
                        end
                        verbosity[] > 0 && println("explicit shift key $shft")
                        c1, s1, c2, s2 = _shift2rot(shft, H1, Hs, S, ifirst,
                                                    ilast - ifirst + 1, λ1, λ2)
                        if verbosity[] > 1
                            @show (c1, s1, c2, s2)
                        end
                    end
                    if iimplicit + iexplicit >= nimplicit + nexplicit
                        iimplicit = 0
                        iexplicit = 0
                    end
                end
                # just to make things entertaining, this is quite different
                # from the complex case...
                if p > 1
                    # initial transformation is handled separately
                    i1 = ifirst + 1
                    i2 = ilast - 2
                    j = ifirst
                    G1 = Givens(j + 1, j + 2, c2, s2)
                    G2 = Givens(j, j + 1, c1, s1)
                    rmul!(view(H1, ifirstm:ilast, :), G1')
                    rmul!(view(H1, ifirstm:ilast, :), G2')
                    if wantZ
                        rmul!(Z[2], G1')
                        rmul!(Z[2], G2')
                    end
                    for l in 2:p
                        Hl = Hs[l-1]
                        if S[l]
                            lmul!(G1, view(Hl, :, j:ilastm))
                            c2, s2, r = givensAlgorithm(Hl[j + 2, j + 2],
                                                        - Hl[j + 2, j + 1])
                            Hl[j + 2, j + 2] = r
                            Hl[j + 2, j + 1] = zero(T)
                            G1 = Givens(j + 1, j + 2, c2, s2)
                            rmul!(view(Hl, ifirstm:j+1, :), G1')

                            lmul!(G2, view(Hl, :, j:ilastm))
                            c1, s1, r = givensAlgorithm(Hl[j + 1, j + 1],
                                                        - Hl[j + 1, j])
                            Hl[j + 1, j + 1] = r
                            Hl[j + 1, j] = zero(T)
                            G2 = Givens(j, j + 1, c1, s1)
                            rmul!(view(Hl, ifirstm:j, :), G2')
                        else # not S[l]
                            rmul!(view(Hl, ifirstm:j+2, :), G1')
                            c2, s2, r = givensAlgorithm(Hl[j + 1, j + 1], Hl[j + 2, j + 1])
                            Hl[j + 1, j + 1] = r
                            Hl[j + 2, j + 1] = zero(T)
                            G1 = Givens(j + 1, j + 2, c2, s2)
                            lmul!(G1, view(Hl, :, j+2:ilastm))
                            rmul!(view(Hl, ifirstm:j+1, :),G2')
                            c1, s1, r = givensAlgorithm(Hl[j, j], Hl[j + 1, j])
                            Hl[j, j] = r
                            Hl[j + 1, j] = zero(T)
                            G2 = Givens(j, j + 1, c1, s1)
                            lmul!(G2, view(Hl, :, j+1:ilastm))
                        end
                        if wantZ
                            ln = mod(l, p) + 1
                            rmul!(Z[ln], G1')
                            rmul!(Z[ln], G2')
                        end
                    end # for l
                    lmul!(G1, view(H1, :, ifirst:ilastm))
                    lmul!(G2, view(H1, :, ifirst:ilastm))
                    @_dbg_rgpschur fcheck("after QZ initial", H1, Hs, Z, S)
                else
                    i1 = ifirst - 1
                    i2 = ilast - 3
                    jt = ifirst
                    G1 = Givens(jt + 1, jt + 2, c2, s2)
                    G2 = Givens(jt, jt + 1, c1, s1)
                end # if p > 1

                @_dbg_rgpschur println("ifirst, i1, i2: $ifirst, $i1, $i2")
                for j1 in i1:i2
                    # create or chase a bulge
                    if j1 < ifirst
                        j = j1 + 1
                        lmul!(G1, view(H1, :, j:ilastm))
                        lmul!(G2, view(H1, :, j:ilastm))
                    else
                        j = (p == 1) ? (j + 1) : j1
                        c2, s2, r2 = givensAlgorithm(H1[j + 1, j - 1],
                                                     H1[j + 2, j - 1])
                        c1, s1, r1 = givensAlgorithm(H1[j, j - 1], r2)
                        H1[j, j - 1] = r1
                        H1[j + 1, j - 1] = zero(T)
                        H1[j + 2, j - 1] = zero(T)
                        G1 = Givens(j + 1, j + 2, c2, s2)
                        G2 = Givens(j, j + 1, c1, s1)
                        lmul!(G1, view(H1, :, j:ilastm))
                        lmul!(G2, view(H1, :, j:ilastm))
                    end
                    if wantZ
                        rmul!(Z[1], G1')
                        rmul!(Z[1], G2')
                    end

                    for l in p:-1:2
                        Hl = Hs[l-1]
                        if S[l]
                            rmul!(view(Hl, ifirstm:j+2, :), G1')
                            c2, s2, r2 = givensAlgorithm(Hl[j + 1, j + 1], Hl[j + 2, j + 1])
                            Hl[j + 1, j + 1] = r2
                            Hl[j + 2, j + 1] = zero(T)
                            G1 = Givens(j + 1, j + 2, c2, s2)
                            lmul!(G1, view(Hl, :, j + 2:ilastm))
                            rmul!(view(Hl, ifirstm:j+1, :), G2')
                            c1, s1, r1 = givensAlgorithm(Hl[j, j], Hl[j + 1, j])
                            Hl[j, j] = r1
                            Hl[j + 1, j] = zero(T)
                            G2 = Givens(j, j + 1, c1, s1)
                            lmul!(G2, view(Hl, :, j + 1:ilastm))
                        else # !S[l]
                            lmul!(G1, view(Hl, :, j:ilastm))
                            c2, s2, r2 = givensAlgorithm(Hl[j + 2, j + 2], -Hl[j + 2, j + 1])
                            Hl[j + 2, j + 2] = r2
                            Hl[j + 2, j + 1] = zero(T)
                            G1 = Givens(j + 1, j + 2, c2, s2)
                            rmul!(view(Hl, ifirstm:j+1, :), G1')
                            lmul!(G2, view(Hl, :, j:ilastm))
                            c1, s1, r1 = givensAlgorithm(Hl[j + 1, j + 1], -Hl[j + 1, j])
                            Hl[j + 1, j + 1] = r1
                            Hl[j + 1, j] = zero(T)
                            G2 = Givens(j, j + 1, c1, s1)
                            rmul!(view(Hl, ifirstm:j, :), G2')
                        end
                        if wantZ
                            rmul!(Z[l], G1')
                            rmul!(Z[l], G2')
                        end
                    end
                    lm = min(j + 3, ilastm)
                    rmul!(view(H1, ifirstm:lm, :), G1')
                    rmul!(view(H1, ifirstm:lm, :), G2')
                end # for j1
                j = ilast - 1
                c1, s1, r1 = givensAlgorithm(H1[j, j - 1], H1[j + 1, j - 1])
                H1[j, j - 1] = r1
                H1[j + 1, j - 1] = zero(T)

                # (510)
                # this is the same block we use in the deflation stage
                G2 = Givens(j, j + 1, c1, s1)
                lmul!(G2, view(H1, :, j:ilastm))
                if wantZ
                    rmul!(Z[1], G2')
                end
                for l in p:-1:2
                    Hl = Hs[l-1]
                    if S[l]
                        rmul!(view(Hl, ifirstm:j+1, :), G2')
                        c1, s1, r1 = givensAlgorithm(Hl[j, j], Hl[j + 1, j])
                        Hl[j, j] = r1
                        Hl[j + 1, j] = zero(T)
                        G2 = Givens(j, j + 1, c1, s1)
                        lmul!(G2, view(Hl, :, j+1:ilastm))
                    else
                        lmul!(G2, view(Hl, :, j:ilastm))
                        c1, s1, r1 = givensAlgorithm(Hl[j + 1, j + 1], -Hl[j + 1, j])
                        Hl[j + 1, j + 1] = r1
                        Hl[j + 1, j] = zero(T)
                        G2 = Givens(j, j + 1, c1, s1)
                        rmul!(view(Hl, ifirstm:j, :), G2')
                    end
                    if wantZ
                        rmul!(Z[l], G2')
                    end
                end
                rmul!(view(H1, ifirstm:ilastm, :), G2')
                if verbosity[] > 2
                    str = "after QZ loop"
                    showallmats(str)
                end
                @_dbg_rgpschur fcheck("after QZ loop", H1, Hs, Z, S; check_A1 = true)
            end # qz loop
        end # if doqziter
    end # iteration loop
    if !done
        throw(ErrorException("convergence failed at level $ilast"))
    end

    # postprocessing
    if rev
        if wantZ
            Zr = similar(Z)
            Zr[1] = Z[1]
            for l in 2:p
                Zr[l] = Z[p + 2 - l]
            end
        else
            Zr = Z
        end
        Hr = similar(Hs)
        for l in 1:(p - 1)
            Hr[l] = Hs[p - l]
        end
        #        return GeneralizedPeriodicSchur(reverse(S),p,H1,Hr,Zr,α,β,αscale,'L')
        # somehow this seems to circumvent a type-intersection bug in the compiler:
        f() = GeneralizedPeriodicSchur(reverse(S), p, H1, Hr, Zr, α, β, αscale, 'L')
        return f()
    else
        return GeneralizedPeriodicSchur(S, 1, H1, Hs, Z, α, β, αscale)
    end
end


function _check_deflate_hess(H1::StridedMatrix{T}, ilo, ilast, ulp, smlnum, tol) where {T}
    aggressive = tol !== nothing
    jlo = ilo
    xmin = Inf
    for j in ilast:-1:(ilo + 1)
        if !aggressive
            tol = abs(H1[j - 1, j - 1]) + abs(H1[j, j])
            if tol == 0
                tol = opnorm(view(H1, ilo:j, ilo:j), 1)
            end
            tol = max(ulp * tol, smlnum)
        end
        xmin = min(xmin, abs(H1[j, j - 1]))
        if abs(H1[j, j - 1]) <= tol
            H1[j, j - 1] = zero(T)
            jlo = j
            if j == ilast
                return true, jlo
            else
                verbosity[] > 1 && println("deflating at $jlo")
                return false, jlo
            end
        end
    end
    verbosity[] > 1 && println("min subdiag in H1: $xmin jlo=$jlo")
    return false, jlo
end

function _check_deflate_tr(Hl::StridedMatrix{T}, jlo, ilast, ulp, smlnum, tol) where {T}
    aggressive = tol !== nothing
    for j in ilast:-1:jlo
        if !aggressive
            if j == ilast
                tol = abs(Hl[j - 1, j])
            elseif j == jlo
                tol = abs(Hl[j, j + 1])
            else
                tol = abs(Hl[j - 1, j]) + abs(Hl[j, j + 1])
            end
            if tol == 0
                tol = opnorm(UpperTriangular(view(Hl, jlo:j, jlo:j)), 1)
            end
            tol = max(ulp * tol, smlnum)
        end
        if abs(Hl[j, j]) <= tol
            Hl[j, j] = zero(T)
            return true, j
        end
    end
    return false, 0
end

# compute 2 rotations to start a PQZ sweep
# translation of MB03AF('Double'), except Hessenberg is in H1
function _qzrots(H1::AbstractMatrix{T}, Hs, S, i1, n) where {T}
    p = length(Hs) + 1
    c1, s1, r = givensAlgorithm(H1[i1, i1], H1[i1+1, i1])
    c2, s2, r = givensAlgorithm(r, one(T))
    i2 = i1 + n - 1
    for l in p:-1:2
        Hl = Hs[l-1]
        if S[l]
            α = c2 * (c1 * Hl[i1,i1] + s1 * Hl[i1,i1+1])
            β = s1 * c2 * Hl[i1+1, i1+1]
            γ = s2 * Hl[i2, i2]
            c1, s1, r = givensAlgorithm(α, β)
            c2, s2, val1 = givensAlgorithm(r, γ)
        else
            α = c1 * s2 * Hl[i1,i1]
            γ = s1 * Hl[i1,i1]
            β = s2 * (c1 * Hl[i1,i1+1] + s1 * Hl[i1+1, i1+1])
            δ = c1 * Hl[i1+1, i1+1] - s1 * Hl[i1, i1+1]
            c1, s1, r = givensAlgorithm(δ, γ)
            α = c1 * α + s1 * β
            β = c2 * Hl[i2, i2]
            c2, s2, r = givensAlgorithm(β, α)
        end
    end
    i = p
    α = s2 * H1[i2, i2] - c1 * c2
    β = -s1 * c2

    m = n - 1
    nh = size(H1,1)
    Hlv = view(H1, i1:nh, i1:nh)
    γ = -s2 * Hlv[n,m]
    c2, s2, r = givensAlgorithm( α, γ)
    c1, s1, r = givensAlgorithm( r,  β)
    cx = c1 * c2
    sx = c1 * s2
    β  = s1 * Hlv[n,m]
    α = cx * Hlv[n,m] + sx * Hlv[n,n]
    γ = s1 * Hlv[m,m]
    δ = cx * Hlv[m,m] + sx * Hlv[m,n]
    val1  = s1 * Hlv[3,2]
    val2  = cx * Hlv[2,1] + s1 * Hlv[2,2]
    val3  = cx * Hlv[1,1] + s1 * Hlv[1,2]
    c1, s1, r = givensAlgorithm( α, β)
    c2, s2, r = givensAlgorithm( γ, r)
    c3, s3, r = givensAlgorithm( δ, r)
    c4, s4, r = givensAlgorithm( val1,  r)
    c5, s5, r = givensAlgorithm( val2,  r)
    c6, s6, r = givensAlgorithm( val3,  r)

    for i in p:-1:2

        Hlv = view(Hs[i - 1], i1:nh, i1:nh)
        if S[i]
            ss    = s3 * s4
            sss   = s2 * ss
            ssss  = s1 * sss
            val1  = c4 * Hlv[1,3]
            val2  = c4 * Hlv[2,3]
            val3  = c4 * Hlv[3,3]
            α = s4 * c3 * Hlv[m,m] + sss  * c1 * Hlv[m,n]
            β  = ss * c2 * Hlv[m,m] + ssss * Hlv[m,n]
            γ = sss  * c1 * Hlv[n,n]
            δ = ssss * Hlv[n,n]

            ss    = s5 * s6
            cs    = c5 * s6
            val1  = ss * val1 + cs * Hlv[1,2] + c6 * Hlv[1,1]
            val2  = ss * val2 + cs * Hlv[2,2]
            val3  = ss * val3
            α = ss * α
            β  = ss * β
            γ = ss * γ
            δ = ss * δ

            c1, s1, r = givensAlgorithm( γ, δ)
            c2, s2, r = givensAlgorithm( β,  r)
            c3, s3, r = givensAlgorithm( α, r)
            c4, s4, r = givensAlgorithm( val3,  r)
            c5, s5, r = givensAlgorithm( val2,  r)
            c6, s6, r = givensAlgorithm( val1,  r)

        else

            δ =  c1 * Hlv[n,n]
            ϵ =  s1 * Hlv[n,n]

            α =  c2 * Hlv[m,m]
            β  =  s2 * δ
            γ = -s2 * Hlv[m,m]
            ζ  =  c2 * Hlv[m,n] + s2 * ϵ
            η   = -s2 * Hlv[m,n] + c2 * ϵ

            #  Update the entry (2n+1,2n+1) for G1'.

            δ = c1 * c2 * δ + s1*η

            #  Compute the new, right rotation G2.

            c2R, s2R, r = givensAlgorithm( δ, -γ)

            #  Apply G3 to the 2-by-4 submatrix in
            #  (n+1:n+2,[n+1:n+2 2n+1:2n+1]).

            δ =  c3 * Hlv[m,m]
            ϵ =  s3 * α
            η   =  c3 * Hlv[m,n] + s3 * β
            θ =  s3 * ζ
            γ = -s3 * Hlv[m,m]
            β  = -s3 * Hlv[m,n] + c3 * β

            #  Update the entry (n+2,n+2) for G1' and G2R'.

            α = c2R * c3 * α + s2R * ( c1 * β +
                                       s1 * c3 * ζ )

            #  Compute the new G3.

            c3R, s3R, r = givensAlgorithm( α, -γ)

            #  Apply G4 to the 2-by-5 submatrix in
            #  ([3 n+1],[3 n+1:n+2 2n+1:2n+1]).

            val1  =  c4 * Hlv[3,3]
            val2  =  s4 * δ
            val3  =  s4 * ϵ
            val4  =  s4 * η
            val5  =  s4 * θ
            β  = -s4 * Hlv[3,3]
            δ =  c4 * δ
            ϵ =  c4 * ϵ
            ζ  =  c4 * η
            η   =  c4 * θ

            #  Update the entry (n+1,n+1) for G1', G2R', and G3R'.

            α = c3R * δ + s3R * ( c2R * ϵ + s2R *
                                  ( c1 * ζ  + s1 * η ) )

            #  Compute the new G4.

            c4R, s4R, r = givensAlgorithm( α, -β)

            #  Apply G5 to the 2-by-6 submatrix in
            #  (2:3,[2:3 n+1:n+2 2n+1:2n+2]).

            β  =  c5 * Hlv[2,2]
            δ =  c5 * Hlv[2,3] + s5 * val1
            ϵ =  s5 * val2
            ζ  =  s5 * val3
            η   =  s5 * val4
            θ =  s5 * val5
            γ = -s5 * Hlv[2,2]
            val1  =  c5 * val1 - s5 * Hlv[2,3]
            val2  =  c5 * val2
            val3  =  c5 * val3
            val4  =  c5 * val4
            val5  =  c5 * val5

            #  Update the entry (3,3) for G1', G2R', G3R', and G4R'.

            α = c4R * val1 + s4R * ( c3R * val2 + s3R *
                                     ( c2R * val3 + s2R *
                                       ( c1 * val4 + s1 * val5 ) ) )

            #  Compute the new G5.

            c5R, s5R, r = givensAlgorithm( α, -γ)

            #  Apply G6 to the 2-by-7 submatrix in
            #  (1:2,[1:3 n+1:n+2 2n+1:2n+2]).

            γ = -s6 * Hlv[1,1]
            β  =  c6 * β  - s6 * Hlv[1,2]
            δ =  c6 * δ - s6 * Hlv[1,3]
            ϵ =  c6 * ϵ
            ζ  =  c6 * ζ
            η   =  c6 * η
            θ =  c6 * θ

            #  Update the entry (2,2) for G1', G2R', G3R', G4R', and
            #  G5R'.

            α = c5R * β + s5R * ( c4R * δ + s4R *
                                  ( c3R * ϵ + s3R *
                                    ( c2R * ζ  + s2R *
                                      ( c1  * η   + s1  * θ )
                                      ) ) )

            #  Compute the new G5.

            c6R, s6R, r = givensAlgorithm( α, -γ)

            c2 = c2R
            s2 = s2R
            c3 = c3R
            s3 = s3R
            c4 = c4R
            s4 = s4R
            c5 = c5R
            s5 = s5R
            c6 = c6R
            s6 = s6R

        end

    end

    # Last step: let the rotations collapse into the first factor.

    val1  =  s5 * s6
    val2  =  s4 * val1
    val3  =  s3 * val2
    α =  c3 * val2 - c6
    β  =  c2 * val3 - c5 * s6
    γ = -c4 * val1
    c2, s2, r = givensAlgorithm( β,  γ)
    c1, s1, r = givensAlgorithm( α, r)
    return c1, s1, c2, s2
end

# compute a rotation to start a PQZ sweep
# Hessenberg is in last place
# based on MB03AF('Single',N=2)
function _qzrot2x2(H2s, S)
    T = eltype(H2s[1])
    p = length(H2s)
    Hl = H2s[p]
    i1 = 1 # for ease of comparison w/ original
    i2 = 2
    c1, s1, r = givensAlgorithm(Hl[i1, i1], Hl[i1+1, i1])
    c2, s2, r = givensAlgorithm(r, one(T))
    for l in p-1:-1:1
        Hl = H2s[l]
        if S[l]
            α = c2 * (c1 * Hl[i1,i1] + s1 * Hl[i1,i1+1])
            β = s1 * c2 * Hl[i1+1, i1+1]
            γ = s2 * Hl[i2, i2]
            c1, s1, r = givensAlgorithm(α, β)
            c2, s2, val1 = givensAlgorithm(r, γ)
        else
            α = c1 * s2 * Hl[i1,i1]
            γ = s1 * Hl[i1,i1]
            β = s2 * (c1 * Hl[i1,i1+1] + s1 * Hl[i1+1, i1+1])
            δ = c1 * Hl[i1+1, i1+1] - s1 * Hl[i1, i1+1]
            c1, s1, r = givensAlgorithm(δ, γ)
            α = c1 * α + s1 * β
            β = c2 * Hl[i2, i2]
            c2, s2, r = givensAlgorithm(β, α)
        end
    end
    Hl = H2s[p]
    α = s2 * Hl[i2, i2] - c1 * c2
    β = -s1 * c2
    c1, s1, _ = givensAlgorithm(α, β)
    return c1, s1
end

# compute terms in 2 Givens rotations s.t.
# G₁G₂ projects the Wilkinson double shift polynomial of the matrix product
# into direction e₁, using precomputed eigvals
# translation of MB03AB
function _shift2rot(key, H1::AbstractMatrix{T}, Hs, S, i1, nord, λ1, λ2) where T
    # nord is not used in MB03AB, but should (optionally) be used to check
    # dimensions
    p = length(Hs) + 1
    single = key == :s
    n = size(H1,1)
    if single
        @assert nord >= 2
    else
        @assert nord >= 3
    end
    @assert n >= nord
    i2 = i1 + nord - 1

    is_real = key != :c
    c1, s1, r = givensAlgorithm(H1[i1 + 1,i1], one(T))
    c2, s2, _ = givensAlgorithm(H1[i1, i1], r)
    for i in p:-1:2
        Ai = view(Hs[i-1], i1:i2, i1:i2)
        if S[i]
            α = Ai[1, 1] * c2 + Ai[1, 2] * c1 * s2
            β = Ai[2, 2] * c1
            γ = s1
            c1, s1, r = givensAlgorithm(β, γ)
            c2, s2, _ = givensAlgorithm(α, r * s2)
        else
            α = s2 * Ai[1, 1]
            β = c1 * c2 * Ai[2, 2] - s2 * Ai[1, 2]
            γ = s1 * Ai[2, 2]
            cx, sx = c1, s1
            c1, s1, _ = givensAlgorithm(cx, γ)
            r = c1 * β + sx * c2 * s1
            c2, s2, _ = givensAlgorithm(r, α)
        end
    end
    if is_real
        c2, s2, _ = givensAlgorithm(c2 - real(λ2) * s1 * s2, c1 * s2)
        if single
            c1, s1 = c2, s2
            c2, s2 = one(T), zero(T)
            return c, s, c2, s2
        else
            cx, sx = c2, s2
        end
    else
        t = s1 * s2
        α = c2 - real(λ1) * t
        β = c1 * s2
        γ = real(λ2) * t
        c1, s1, r = givensAlgorithm(β, γ)
        c2, s2, _ = givensAlgorithm(α, r)
        cx, sx = c1, s1
        cy, sy = c2, s2
    end
    Ai = view(H1, i1:i1+2, i1:i1+1)
    Ai = H1
    α = Ai[1, 2] * s2 + Ai[1, 1] * s2
    β = Ai[2, 2] * s2 + Ai[2, 1] * c2
    γ = Ai[3, 2] * s2
    c1, s1, r = givensAlgorithm(γ, one(T))
    c3, s3, _ = givensAlgorithm(β, r)
    c2, s2, _ = givensAlgorithm(α, c3 * β + s3 * r)
    for i in p:-1:2
        Ai = view(Hs[i-1], i1:i2, i1:i2)
        if S[i]
            t = c1 * s3
            α = (Ai[1, 3] * t + Ai[1, 2] * c3) * s2 + Ai[1, 1] * c2
            β = (Ai[2, 3] * t + Ai[2, 2] * c3) * s2
            γ = Ai[3, 3] * c1
            δ = s1
            c1, s1, r = givensAlgorithm(γ, δ)
            c3, s3, r3 = givensAlgorithm(β, r)
            c2, s2, _ = givensAlgorithm(α, r3)
        else
            c23 = c2 * c3
            t = c2 * s3
            α = c1 * c3 * Ai[3, 3] - s3 * Ai[2, 3]
            β = s1 * c3
            γ = c2 * t * Ai[3, 3] + c23 * Ai[3, 3] - s2 * Ai[1, 3]
            δ = s1 * t
            t = c1
            c1, s1, _ = givensAlgorithm(t, s1 * Ai[3, 3])
            t = α * c1 + β * s1
            c3, s3, r = givensAlgorithm(t, s3 * Ai[2, 2])
            t = (c23 * Ai[2, 2] - s2 * Ai[1, 2]) * c3 + (γ * c1 + δ * s1) * s3
            c2, s2, _ = givensAlgorithm(t, s2 * Ai[1, 1])
        end
    end
    # last step: let rotations collapse into first factor
    if is_real
        t = real(λ1) * s1 * s3
        α = c2 - cx * t * s2
        β = (c3 - sx * t) * s2
        γ = c1 * s2 * s3
    else
        pp = s1 * s3
        α = c2 + (imag(λ1) * sx * sy - real(λ1) * cy) * pp * s2
        β = c3 - real(λ1) * cx * sy * pp
        γ = c1 * s3
        pp = s2
    end
    c2, s2, r = givensAlgorithm(β, γ)
    if !is_real
        r *= pp
    end
    c1, s1, _ = givensAlgorithm(α, t)
    return c1, s1, c2, s2
end
