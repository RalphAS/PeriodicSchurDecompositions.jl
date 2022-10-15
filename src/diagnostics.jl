# diagnostic stuff for PeriodicSchurDecompositions

# `verbosity` applies to the main periodic QZ codes.
# values: 0 - silent, 1 - steps, 2 - convergence info, 3 - various matrices
const verbosity = Ref(0)

# Diagnosing the Krylov-Schur code is a special adventure, so we handle it separately.
const _kry_verby = Ref(0)

# likewise the swapping codes
const _ss_verby = Ref(0)
const _rord_verby = Ref(0)

# Styling is sometimes awkward so make it optional.
const _dgn_styled = Ref(true)

function setverbosity(j, key = nothing)
    if key === nothing
        verbosity[] = j
    elseif startswith(String(key), "kry")
        _kry_verby[] = j
    elseif startswith(String(key), "rord")
        _rord_verby[] = j
    elseif startswith(String(key), "syl")
        _ss_verby[] = j
    elseif startswith(String(key), "sty")
        _dgn_styled[] = j != 0
    else
        @warn "incomprehensible key $key"
    end
end

_printsty(c, xs...) =
    if _dgn_styled[]
        printstyled(xs...; color = c)
    else
        print(xs...)
    end

# This is a debugging facility for use inside the pschur!(Hessenberg) codes.
# When debugging is enabled, verify the transformation chains as we go, to hunt
# for translation errors etc.
# Note that this implementation assumes nontrivial initial `Z`, since
# primarily used in the codes operating on periodic Hessenberg series.
struct _FacChecker{TM}
    A1init::TM
    Aπinit::TM
    p::Int
    valid::Bool
    left::Bool
    ischur::Int
end

function _FacChecker(H1, Hs, Z, wantZ, S=nothing; left=false, ischur=1)
    sx(l) = S === nothing ? true : S[l]
    if !wantZ
        _printsty(:cyan, "H1, ΠH checks not available w/o Z")
        return _FacChecker(1,1,1,false,false,1)
    end
    if !sx(1)
        _printsty(:cyan, "H1, ΠH checks not available for !S[1]")
        return _FacChecker(1,1,1,false,false,1)
    end
    p = length(Hs) + 1
    A1 = (ischur == 1) ? H1 : Hs[1]
    if left
        A1init = Z[p > 1 ? 2 : 1] * A1 * Z[1]'
        Aπinit = A1 * Z[1]';
    else
        A1init = Z[1] * H1 * Z[p > 1 ? 2 : 1]'
        Aπinit = Z[1] * H1;
    end

    il = (ischur == 1) ? 0 : 1
    for l in 2:p
        if l == ischur
            Hl = H1
        else
            il += 1
            Hl = Hs[il]
        end
        if left
            if sx(l)
                Aπinit = Hl * Aπinit
            else
                Aπinit = inv(Hl) * Aπinit
            end
        else
            if sx(l)
                Aπinit = Aπinit * Hl
            else
                Aπinit = Aπinit * inv(Hl)
            end
        end
    end
    if left
        Aπinit = Z[1] * Aπinit
    else
        Aπinit = Aπinit * Z[1]'
    end
    if _dbg_detail[] > 0
        print("A1 (initial):")
        show(stdout, "text/plain",  A1init)
        println()
        print("ΠA (initial):")
        show(stdout, "text/plain",  Aπinit)
        println()
    end
    return _FacChecker(A1init, Aπinit, p, true, left, ischur)
end

function (fc::_FacChecker)(str, H1, Hs, Z, S = nothing;
                           check_Aπ = true, check_A1 = false, pd=nothing)
    fc.valid || return nothing
    sx(l) = S === nothing ? true : S[l]
    if check_A1
        if fc.left
            A1tmp = Z[fc.p > 1 ? 2 : 1] * H1 * Z[1]'
        else
            A1tmp = Z[1] * H1 * Z[fc.p > 1 ? 2 : 1]'
        end
        t = norm(A1tmp - fc.A1init)
        println(str, " H1 factor error: ", t, " rel. ", t/norm(fc.A1init))
    end
    if check_Aπ
        Hl = (fc.ischur == 1) ? H1 : Hs[1]
        Aπtmp = Hl
        il = (fc.ischur == 1) ? 0 : 1
        for l in 2:fc.p
            if l == fc.ischur
                Hl = H1
            else
                il += 1
                Hl = Hs[il]
            end
            if fc.left
                if sx(l)
                    Aπtmp = Hl * Aπtmp
                else
                    Aπtmp = inv(Hl) * Aπtmp
                end
            else
                if sx(l)
                    Aπtmp = Aπtmp * Hl
                else
                    Aπtmp = Aπtmp * inv(Hl)
                end
            end
        end
        if pd !== nothing
            println("  block at [$(pd-1),$(pd-1)]: ", Aπtmp[pd-1:pd,pd-1:pd])
        end
        Aπtmp = Z[1] * Aπtmp * Z[1]'
        t = norm(Aπtmp - fc.Aπinit)
        println(str, " ΠH error: ", t, " rel. ", t / norm( fc.Aπinit))
    end
    nothing
end




"""
    checkpsd(P::AbstractPeriodicSchur{T}, As::Vector{Matrix{T}})

Verify integrity of a (generalized) periodic Schur decomposition.
Returns a status code (Bool) and a vector of
normalized factorization errors (which should be O(1)).

WARNING: this function assumes dominant entries in `As` are O(1).
"""
function checkpsd(P::AbstractPeriodicSchur{T}, Hs::AbstractVector;
                  quiet = false, thresh = 100, strict = true) where {T}
    # Hs could be structured matrices of different varieties, caveat emptor.
    # thresh should probably scale w/ period; leave it to user for now.
    p = length(Hs)
    n = size(P.T1, 1)
    S = isa(P, GeneralizedPeriodicSchur) ? P.S : trues(p)
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
        l1 = mod(l, p) + 1
        Tl = Ts[l]
        cmp = strict ? 0 : ttol * eps(real(T)) * n
        if T <: Real && l == P.schurindex
            # check first subdiag for real eigvals
            for j in 1:n-1
                if isreal(P.values[j]) && Tl[j + 1, j] != 0
                    @warn "unexpected subdiagonal for j=$j"
                end
            end
            tval = tril(Tl, -2)
        else
            tval = tril(Tl, -1)
        end
        if norm(tval) > cmp
            if !quiet
                @warn "triangularity fails for l=$l"
            end
            result = false
        end
        if norm(P.Z[l] * P.Z[l]' - I) > qtol * eps(real(T)) * n
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
                @warn "large factorization error ($(err[l]) nϵ) for l=$l"
            end
            result = false
        end
    end
    return result, err
end
