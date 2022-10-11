# swap functions using Sylvester equations

# TODO:
# @inbounds
# StaticArrays

if !isfile(joinpath(@__DIR__, "debugging.jl"))
    macro _dbg_sylswap(expr)
        nothing
    end
end

# this is for left ordering: we are following Granat's papers
function _swapadjqr!(T1::AbstractMatrix{T}, Ts, Zs, i1, p1, p2) where {T}
    tol = T(100)
    vb = _ss_verby[]
    i2 = i1 + p1
    i2new = i1 + p2
    i3 = i2 + p2 - 1
    pp = p1 * p2
    tnrm = norm(view(T1, i1:i3, i1:i3))
    T11 = [T1[i1:(i2 - 1), i1:(i2 - 1)]]
    T12 = [T1[i1:(i2 - 1), i2:i3]]
    T22 = [T1[i2:i3, i2:i3]]
    k = length(Ts) + 1
    ok = true
    for l in 2:k
        Tl = Ts[l - 1]
        tnrm = hypot(tnrm, norm(view(Tl, i1:i3, i1:i3)))
        push!(T11, Tl[i1:(i2 - 1), i1:(i2 - 1)])
        push!(T12, Tl[i1:(i2 - 1), i2:i3])
        push!(T22, Tl[i2:i3, i2:i3])
    end
    Xv, scale = _psylsolve(T11, T22, T12)
    thresh = max(floatmin(real(T)), tol * eps(real(T)) * tnrm)

    @_dbg_sylswap if vb > 1
        X1 = reshape(Xv[1:pp], p1, p2)
        X2 = reshape(Xv[(pp + 1):(2 * pp)], p1, p2)
        Xk = reshape(Xv[((k - 1) * pp + 1):(k * pp)], p1, p2)
        resid = norm(T11[1] * X1 - X2 * T22[1] + T12[1])
        println("psyl residual 1: ", resid)
        resid = norm(T11[k] * Xk - X1 * T22[k] + T12[k])
        println("psyl residual k: ", resid)
    end
    # use a working copy to facilitate stability tests
    Txx = [[T11[l] T12[l]; zeros(T, p2, p1) T22[l]] for l in 1:k]
    # separate l=1 to set up data structs
    l = 1
    i0 = 0
    X = reshape(Xv[(i0 + 1):(i0 + pp)], p1, p2)
    Xi = Matrix(vcat(X, I(p2)))
    m = p1 + p2
    q, r = qr(Xi)
    Tl = Txx[l]
    Tp = Txx[k]
    rmul!(view(Tl, :, 1:m), q)
    lmul!(q', view(Tp, 1:m, :))
    Qs = [q]

    # compute and save p2 * k reflectors
    for l in 2:k
        i0 = (l - 1) * pp
        X = reshape(Xv[(i0 + 1):(i0 + pp)], p1, p2)
        Xi = Matrix(vcat(X, I(p2)))
        q, r = qr(Xi)
        push!(Qs, q)
        Tl = Txx[l]
        Tp = Txx[l == 1 ? k : l - 1]
        rmul!(view(Tl, :, 1:m), q)
        lmul!(q', view(Tp, 1:m, :))
        # TODO: weak stability test, i.e. is Qk' [Xk I]  actually UT?
    end
    if max(p1, p2) > 2
        throw(ArgumentError("only implemented for block sizes 1 or 2"))
    end
    # check for fill-in for any pj > 1
    fillin1 = false
    fillin2 = false
    if p2 > 1
        for l in 1:k
            fillin1 |= abs(Txx[l][2, 1]) > thresh
        end
        vb > 1 && fillin1 && @info "fill-in detected in new top block"
    end
    if p1 > 1
        ii = p2 + 1
        for l in 1:k
            fillin2 |= abs(Txx[l][ii + 1, ii]) > thresh
        end
        vb > 1 && fillin2 && @info "fill-in detected in new lower block"
    end
    # If fill-in occurred, run 2x2 p.Hessenberg reduction(s), saving reflectors.
    # The current implementation repeats some computations to avoid the need
    # for yet another function with confusing reflector indexing.
    fillin = fillin1 || fillin2
    if fillin
        Ws = [Matrix{T}(I, m, m) for _ in 1:k]
        Trows = similar(Txx[1], 2, m) # workspace
        Tcols = similar(Txx[1], m, 2)
        if fillin1
            _filled2hess!(Txx, Ws, 1, Trows, Tcols)
        end
        if fillin2
            _filled2hess!(Txx, Ws, p2 + 1, Trows, Tcols)
        end
    end
    # strong stability test, i.e. is Wl1 Ql1 Txxl Ql' Wl' ≈ Tl?
    for l in 1:k
        l1 = mod(l, k) + 1
        Tl = l == 1 ? T1 : Ts[l - 1]
        if fillin
            Ttmp = Ws[l1] * Txx[l] * Ws[l]'
            Ttmp = Qs[l1] * Ttmp * Qs[l]'
        else
            Ttmp = Qs[l1] * Txx[l] * Qs[l]'
        end
        if norm(Ttmp - view(Tl, i1:i3, i1:i3)) > thresh
            @warn "failing strong stability test"
            ok = false
        end
    end

    for l in 1:k
        i0 = (l - 1) * pp
        Tl = l == 1 ? T1 : Ts[l - 1]
        Tp = (l == 2 || k == 1) ? T1 : (l == 1 ? Ts[k - 1] : Ts[l - 2])
        Zl = Zs[l]
        q = Qs[l]
        rmul!(view(Tl, :, i1:i3), q)
        rmul!(view(Zl, :, i1:i3), q)
        lmul!(q', view(Tp, i1:i3, :))
        if fillin
            Ttmp = Tl[:, i1:i3]
            Ztmp = Zl[:, i1:i3]
            Tptmp = Tp[i1:i3, :]
            mul!(view(Tl, :, i1:i3), Ttmp, Ws[l])
            mul!(view(Zl, :, i1:i3), Ztmp, Ws[l])
            mul!(view(Tp, i1:i3, :), Ws[l]', Tptmp)
        end
    end
    # sweep up the dust
    T1[i2new:i3, i1:(i2new - 1)] .= 0
    for l in 1:(k - 1)
        Tl = Ts[l]
        triu!(view(Tl, i1:i3, i1:i3))
    end
    return ok
end

function _filled2hess!(Txx,Ws,j0,Trows,Tcols,S=nothing)
    k = length(Txx)
    j1 = j0 + 1
    Th = [Txx[l][j0:j1, j0:j1] for l in 1:k]
    notinverted(l) = (S === nothing) ? true : S[l]
    H11, H1qs = _phess2x2!(Th,1,S)
    for l in 1:k
        lp = mod(l, k) + 1
        q = H1qs[l]
        qp = H1qs[lp]
        Tl = Txx[l]
        if notinverted(l)
            # rmul!(view(Tl, :, j0:j1), q)
            v = view(Tl, :, j0:j1)
            Tcols .= v
            mul!(v, Tcols, q)
            v = view(Tl, j0:j1, :)
            Trows .= v
            mul!(v, qp', Trows)
        else
            # lmul!(q', view(Tl, j0:j1, :))
            v = view(Tl, j0:j1, :)
            Trows .= v
            mul!(v, q', Trows)
            v = view(Tl, :, j0:j1)
            Tcols .= v
            mul!(v, Tcols, qp)
        end
        v = view(Ws[l], :, j0:j1)
        Tcols .= v
        mul!(v, Tcols, q)
    end
end

# left ordering, circ-shifted so T1 is rightmost
# version using Givens
function _swapadj1x1g!(T1::AbstractMatrix{T}, Ts, Zs, i1;
                       strong = true, threshfac = 20) where {T}
    verbosity = _ss_verby[]
    ok = true
    i2 = i1 + 1
    i3 = i2 + 1
    T11 = [T1[i1, i1]]
    T12 = [T1[i1, i2]]
    T22 = [T1[i2, i2]]
    k = length(Ts) + 1
    for l in 2:k
        Tl = Ts[l - 1]
        push!(T11, Tl[i1, i1])
        push!(T12, Tl[i1, i2])
        push!(T22, Tl[i2, i2])
    end
    # CHECKME: maybe use scaled SSQ as in LAPACK instead of norm()
    thresh = max(threshfac * hypot(norm(T11), norm(T12), norm(T22)) * eps(real(T)),
                 floatmin(real(T)))
    Xv, scale = _psylsolve1(T11, T22, T12)
    # use a working copy to facilitate stability tests
    Txx = [[T11[l] T12[l]; zero(T) T22[l]] for l in 1:k]
    c, s, r = givensAlgorithm(Xv[1], one(T))
    G = Givens(1, 2, c, s)
    Gs = [G]
    rmul!(Txx[1], G')
    lmul!(G, Txx[k])
    for l in 2:k
        c, s, r = givensAlgorithm(Xv[l], one(T))
        G = Givens(1, 2, c, s)
        rmul!(Txx[l], G')
        lp = l == 1 ? k : l - 1
        lmul!(G, Txx[lp])
        push!(Gs, G)
    end
    ws = sum((l) -> abs(Txx[l][2, 1]), 1:k)
    if ws > thresh
        if verbosity > 0
            @warn "failing weak test for swap at $i1:$i2 ws=$ws vs $thresh"
        end
        ok = false
    end
    if strong
        # TODO: just apply Gs directly
        Ws = [Matrix{T}(I, 2, 2) for _ in 1:k]
        for l in 1:k
            rmul!(Ws[l], Gs[l]')
        end
        for l in 1:k
            l1 = l == k ? 1 : l + 1
            Txx[l] = Ws[l1] * Txx[l] * Ws[l]'
        end
        # println("remade T1:"); display(Txx[1]); println()
        # println("orig T1:"); display(T1[i1:i2,i1:i2]); println()
        ss = norm(Txx[1] - view(T1, i1:i2, i1:i2))
        for l in 2:k
            # println("remade T[$l]:"); display(Txx[l]); println()
            # println("orig T[$l]:"); display(Ts[l-1][i1:i2,i1:i2]); println()
            ss = hypot(ss, norm(Txx[l] - view(Ts[l - 1], i1:i2, i1:i2)))
        end
        if ss > thresh
            if verbosity > 0
                @warn "failing strong test for swap at $i1:$i2 $ss > $thresh"
            end
            ok = false
        else
            if verbosity > 1
                @info "strong test for swap at $i1:$i2 $ss <= $thresh"
            end
        end
    end
    for l in 1:k
        G = Gs[l]
        Tl = l == 1 ? T1 : Ts[l - 1]
        Tp = (l == 2 || k == 1) ? T1 : (l == 1 ? Ts[k - 1] : Ts[l - 2])
        rmul!(view(Tl, :, i1:(i1 + 1)), G')
        lmul!(G, view(Tp, i1:(i1 + 1), :))
        if Zs !== nothing
            Zl = Zs[l]
            rmul!(view(Zl, :, i1:(i1 + 1)), G')
        end
    end
    # some upstream tests require zero subdiags
    T1[i1 + 1, i1] = 0
    for l in 1:(k - 1)
        Ts[l][i1 + 1, i1] = 0
    end
    return ok
end

# generalized version
function _swapadj1x1g!(T1::AbstractMatrix{T}, Ts, Zs, S::AbstractVector{Bool}, i1;
                       strong = true, threshfac = 20) where {T}
    verbosity = _ss_verby[]
    verbosity > 1 && println("swapping 1,1 at $i1 via gSyl/Givens")
    ok = true
    i2 = i1 + 1
    i3 = i2 + 1
    T11 = [T1[i1, i1]]
    T12 = [T1[i1, i2]]
    T22 = [T1[i2, i2]]
    k = length(Ts) + 1
    for l in 2:k
        Tl = Ts[l - 1]
        push!(T11, Tl[i1, i1])
        push!(T12, Tl[i1, i2])
        push!(T22, Tl[i2, i2])
    end
    # CHECKME: maybe use scaled SSQ as in LAPACK instead of norm()
    thresh = max(threshfac * hypot(norm(T11), norm(T12), norm(T22)) * eps(real(T)),
                 floatmin(real(T)))
    Xv, scale = _pgsylsolve1(T11, T22, T12, S)
    # use a working copy to facilitate stability tests
    Txx = [[T11[l] T12[l]; zero(T) T22[l]] for l in 1:k]
    c, s, r = givensAlgorithm(Xv[1], one(T))
    G = Givens(1, 2, c, s)
    Gs = [G]
    rmul!(Txx[1], G')
    if S[k]
        lmul!(G, Txx[k])
    else
        rmul!(Txx[k], G')
    end
    for l in 2:k
        if S[l]
            c, s, r = givensAlgorithm(Xv[l], one(T))
            G = Givens(1, 2, c, s)
            rmul!(Txx[l], G')
        else
            c, s, r = givensAlgorithm(-Xv[l], one(T))
            G = Givens(2, 1, c, s')
            lmul!(G, Txx[l])
        end
        lp = l == 1 ? k : l - 1
        if S[lp]
            lmul!(G, Txx[lp])
        else
            rmul!(Txx[lp], G')
        end
        push!(Gs, G)
    end
    if verbosity > 2
        for l in 2:k
            println("trial T[$l]:"); display(Txx[l]); println()
        end
    end
    ws = sum((l) -> abs(Txx[l][2, 1]), 1:k)
    if ws > thresh
        if verbosity > 0
            @warn "failing weak test for swap at $i1:$i2 ws=$ws vs $thresh"
        end
        ok = false
    elseif verbosity > 1
        @info "weak test for swap at $i1:$i2 $ws <= $thresh"
    end
    if strong
        # TODO: just apply Gs directly
        Ws = [Matrix{T}(I, 2, 2) for _ in 1:k]
        for l in 1:k
            rmul!(Ws[l], Gs[l]')
        end
        for l in 1:k
            l1 = l == k ? 1 : l + 1
            if S[l]
                Txx[l] = Ws[l1] * Txx[l] * Ws[l]'
            else
                Txx[l] = Ws[l] * Txx[l] * Ws[l1]'
            end
        end
        if verbosity > 2
            println("remade T1:"); display(Txx[1]); println()
            println("orig T1:"); display(T1[i1:i2,i1:i2]); println()
        end
        ss = norm(Txx[1] - view(T1, i1:i2, i1:i2))
        for l in 2:k
            if verbosity > 2
                println("remade T[$l]:"); display(Txx[l]); println()
                println("orig T[$l]:"); display(Ts[l-1][i1:i2,i1:i2]); println()
            end
            ss = hypot(ss, norm(Txx[l] - view(Ts[l - 1], i1:i2, i1:i2)))
        end
        if ss > thresh
            if verbosity > 0
                @warn "failing strong test for swap at $i1:$i2 $ss > $thresh"
            end
            ok = false
        else
            if verbosity > 1
                @info "strong test for swap at $i1:$i2 $ss <= $thresh"
            end
        end
    end
    for l in 1:k
        G = Gs[l]
        Tl = l == 1 ? T1 : Ts[l - 1]
        Tp = (l == 2 || k == 1) ? T1 : (l == 1 ? Ts[k - 1] : Ts[l - 2])
        if S[l]
            rmul!(view(Tl, :, i1:(i1 + 1)), G')
        else
            lmul!(G, view(Tl, i1:(i1 + 1), :))
        end
        if S[l == 1 ? k : (l-1)]
            lmul!(G, view(Tp, i1:(i1 + 1), :))
        else
            rmul!(view(Tp, :, i1:(i1 + 1)), G')
        end
        if Zs !== nothing
            Zl = Zs[l]
            rmul!(view(Zl, :, i1:(i1 + 1)), G')
        end
    end
    # some upstream tests require zero subdiags
    T1[i1 + 1, i1] = 0
    for l in 1:(k - 1)
        Ts[l][i1 + 1, i1] = 0
    end
    return ok
end
