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
# Note that this implementation assumes rightward orientation, as used in
# the codes operating on periodic Hessenberg series.
struct _FacChecker{TM}
    A1init::TM
    Aπinit::TM
    p::Int
    valid::Bool
end

function _FacChecker(H1, Hs, Z, wantZ, S=nothing)
    sx(l) = S === nothing ? true : S[l]
    if !wantZ
        _printsty(:cyan, "H1, ΠH checks not available w/o Z")
        return _FacChecker(1,1,1,false)
    end
    p = length(Hs) + 1
    A1init = Z[1] * H1 * Z[p > 1 ? 2 : 1]'
    Aπinit = Z[1] * H1;
    for l in 2:p
        if sx(l)
            Aπinit = Aπinit * Hs[l-1]
        else
            Aπinit = Aπinit * inv(Hs[l-1])
        end
    end
    Aπinit = Aπinit * Z[1]'
    if _dbg_detail[] > 0
        print("A1 (initial):")
        show(stdout, "text/plain",  A1init)
        println()
        print("ΠA (initial):")
        show(stdout, "text/plain",  Aπinit)
        println()
    end
    return _FacChecker(A1init, Aπinit, p, true)
end

function (fc::_FacChecker)(str, H1, Hs, Z, S = nothing;
                           check_Aπ = true, check_A1 = false)
    fc.valid || return nothing
    sx(l) = S === nothing ? true : S[l]
    if check_A1
        A1tmp = Z[1] * H1 * Z[fc.p > 1 ? 2 : 1]'
        println(str, " H1 factor error: ", norm(A1tmp - fc.A1init))
    end
    if check_Aπ
        Aπtmp = Z[1] * H1;
        for l in 2:fc.p
            if sx(l)
                Aπtmp = Aπtmp * Hs[l-1]
            else
                Aπtmp = Aπtmp * inv(Hs[l-1])
            end
        end
        Aπtmp = Aπtmp * Z[1]'
        println(str, " ΠH error: ", norm(Aπtmp - fc.Aπinit))
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
