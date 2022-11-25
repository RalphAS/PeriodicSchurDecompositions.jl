# periodic Krylov-Schur scheme
# based on D. Kressner, Numer. Math. 2006

export PartialPeriodicSchur, partial_pschur

using ArnoldiMethod
using ArnoldiMethod:
                     vtype,
                     IsConverged,
                     History,
                     get_order,
                     RitzValues,
                     include_conjugate_pair,
                     OrderPerm,
                     partition!,
                     Target
using Random

include("rhessx.jl")

struct PKSFailure <: Exception
    msg::String
end

"""
Structure holding a periodic Krylov decomposition (D.Kressner, Numer. Math. 2006).

Intended for internal use.
"""
struct PKrylov{T, TV <: StridedMatrix{T}, TB <: StridedMatrix{T}}
    Vs::Vector{TV}
    Bs::Vector{TB}
    kcur::Base.RefValue{Int}
    vrand!::Any
end
function PKrylov{T}(p::Int, n::Int, k::Int, randfunc!) where {T}
    k <= n * p || throw(ArgumentError("Krylov dimension may not exceed matrix order."))
    V = vcat([Matrix{T}(undef, n, k + 1)], [Matrix{T}(undef, n, k) for _ in 1:(p - 1)])
    H = [zeros(T, k, k) for _ in 1:(p - 1)]
    push!(H, zeros(T, k + 1, k))
    return PKrylov{T, eltype(V), eltype(H)}(V, H, Ref(0), randfunc!)
end
function describe(PK::PKrylov{T}) where {T}
    println("PKrylov{$T} n=$(PK.n) p=$(PK.p) k=$(PK.kcur[]) kmax=$(PK.kmax)")
end

function Base.getproperty(PK::PKrylov, s::Symbol)
    if s ∈ (:Vs, :Bs)
        return getfield(PK, s)
    end
    if s == :p
        Vs = getfield(PK, :Vs)
        return length(Vs)
    elseif s == :n
        Vs = getfield(PK, :Vs)
        return size(Vs[1], 1)
    elseif s == :kmax
        Bs = getfield(PK, :Bs)
        return size(Bs[1], 1)
    elseif s == :k
        return getfield(PK, :kcur)[]
    else
        return getfield(PK, s)
    end
end

if isfile(joinpath(@__DIR__, "krylov_debugging.jl"))
    include("krylov_debugging.jl")
else
    macro _dbg_arnoldi(expr)
        nothing
    end
    macro _dbg_pk(expr)
        nothing
    end
end

"""
PartialPeriodicSchur

A partial periodic Schur decomposition of a series of matrices, with `k` Schur vectors
of length `n` in the `Zⱼ` where typically `k ≪ n`.

The decomposition for the "left" orientation is
`A₁ * Z₁ = Z₂ * T₁; A₂ * Z₂ = Z₃ * T₂; ...; Aₚ * Zₚ = Z₁ * Tₚ.`

Properties are similar to [`PeriodicSchur`](@ref).
"""
struct PartialPeriodicSchur{
                            Ty,
                            St1 <: AbstractMatrix,
                            St <: AbstractMatrix,
                            Sz <: AbstractMatrix,
                            Tλ <: Complex
                            }  <: AbstractPeriodicSchur{Ty}
    "Quasi upper triangular matrix"
    T1::St1
    "Upper triangular matrices"
    T::Vector{St}
    "Matrices of orthonormal Schur vectors"
    Z::Vector{Sz}
    "Complex-valued vector of eigenvalues"
    values::Vector{Tλ}
    orientation::Char
    schurindex::Int
    function PartialPeriodicSchur{Ty, St1, St, Sz, Tλ}(T1::AbstractMatrix{Ty},
                                                       T::Vector{<:AbstractMatrix{Ty}},
                                                       Z::Vector{<:AbstractMatrix{Ty}},
                                                       values::Vector{Tλ},
                                                       orientation::Char = 'L',
                                                       schurindex::Int = 1) where {Ty, St1,
                                                                                   St, Sz,
                                                                                   Tλ}
        new(T1, T, Z, values, orientation, schurindex)
    end
end
function PartialPeriodicSchur(T1::St1,
                              T::Vector{<:AbstractMatrix{Ty}},
                              Z::Vector{<:AbstractMatrix{Ty}},
                              values::Vector{Tλ},
                              orientation::Char = 'L',
                              schurindex::Int = length(Z)) where {St1 <: AbstractMatrix{Ty}
                                                                  } where {Ty, Tλ}
    PartialPeriodicSchur{Ty, St1, eltype(T), eltype(Z), Tλ}(T1,
                                                            T,
                                                            Z,
                                                            values,
                                                            orientation,
                                                            schurindex)
end
function Base.getproperty(P::PartialPeriodicSchur, s::Symbol)
    if s == :period
        return length(P.T) + 1
    else
        return getfield(P, s)
    end
end
Base.propertynames(P::PartialPeriodicSchur) = (:period, fieldnames(typeof(P))...)

# parameter for refined orthogonalization
const _η_orth = Ref(1 / sqrt(2))

function _reinitialize!(PK::PKrylov, l::Integer, j::Integer)
    _kry_verby[] > 0 && println(" reinitializing: l=$l j=$j")
    # describe(PK)
    n = PK.n
    Us = PK.Vs
    v = view(Us[l], :, j + 1)
    PK.vrand!(v)
    rnorm = norm(v)
    if j == 0
        return true
    end
    η = _η_orth[]
    Uprev = view(Us[l], :, 1:j)
    #mul!(hj, Uprev', v)
    hj = Uprev' * v
    mul!(v, Uprev, hj, -1, true)
    wnorm = norm(v)
    if wnorm < η * rnorm
        rnorm = wnorm
        mul!(hj, Uprev', v)
        mul!(v, Uprev, hj, -1, true)
        wnorm = norm(v)
    end
    if wnorm <= η * rnorm
        return false
    else
        rmul!(v, 1 / wnorm)
        return true
    end
end
@noinline throw_reinit() = throw(PKSFailure("Arnoldi reinitialization failed"))

function _deflate!(H1::AbstractMatrix{T}, Hs, Z, ldeflate, jdeflate) where {T}
    n = size(H1, 2)
    p = length(Hs) + 1
    Gtmp = Vector{Givens{T}}(undef, n)

    # left of Hessenberg
    for j in 1:(jdeflate - 1)
        c, s, r = givensAlgorithm(H1[j, j], H1[j + 1, j])
        H1[j, j] = r
        H1[j + 1, j] = 0
        cc = T <: Complex ? complex(c) : c
        G = Givens(j, j + 1, cc, s)
        lmul!(G, view(H1, :, (j + 1):n))
        Gtmp[j] = G
    end
    for j in 1:(jdeflate - 1)
        rmul!(Z[1], Gtmp[j]')
    end
    # propagate through triangular matrices
    for l in 1:(p - 1)
        ntra = jdeflate - 1
        Hl = Hs[l]
        for j in 1:ntra
            G = Gtmp[j]
            rmul!(view(Hl, 1:(j + 1), :), Gtmp[j]')
            c, s, r = givensAlgorithm(Hl[j, j], Hl[j + 1, j])
            Hl[j, j] = r
            Hl[j + 1, j] = 0
            cc = T <: Complex ? complex(c) : c
            G = Givens(j, j + 1, cc, s)
            lmul!(G, view(Hl, :, (j + 1):n))
            Gtmp[j] = G
        end
        for j in 1:ntra
            rmul!(Z[l + 1], Gtmp[j]')
        end
    end
    # right of Hessenberg
    for j in 1:(jdeflate - 2)
        rmul!(view(H1, 1:(j + 1), :), Gtmp[j]')
    end
    nothing
end

function periodic_arnoldi!(As::Vector{TA},
                           PK,
                           krange,
                           uj1;
                           tol1 = 100 * eps(real(vtype(As[1])))) where {TA}
    T = vtype(As[1])
    p = length(As)
    n = size(As[1], 1)
    Hs = PK.Bs
    Us = PK.Vs

    v = Vector{T}(undef, n)
    η = 1 / sqrt(2)
    k1 = krange[1]
    k2 = krange[end]
    _kry_verby[] > 0 && println("Arnoldi working on $k1:$k2")
    h = zeros(T, k2)
    Us[1][:, k1] .= uj1
    j = k1
    singularities = 0
    η = _η_orth[]
    while j <= k2
        @_dbg_arnoldi print("Arnoldi iter $j")
        ldeflate = 0
        jdeflate = 0
        hj = view(h, 1:(j - 1))
        null1 = false
        for l in 1:(p - 1)
            ujl = view(Us[l], :, j)
            if any(isnan.(ujl))
                @error "NaN in ujl"
            end
            mul!(v, As[l], ujl)
            inspan = false
            rnorm = norm(v)
            reorth = false
            if j > 1
                Uprev = view(Us[l + 1], :, 1:(j - 1))
                mul!(hj, Uprev', v)
                mul!(v, Uprev, hj, -1, 1)

                wnorm = norm(v)
                if wnorm < η * rnorm
                    reorth = true
                    rnorm = wnorm
                    correction = Uprev' * v
                    mul!(v, Uprev, correction, -1, 1)
                    hj .+= correction
                    wnorm = norm(v)
                end
                if wnorm <= η * rnorm
                    hjj = zero(T)
                    inspan = true
                    if ldeflate == 0
                        ldeflate = l
                        jdeflate = j
                    end
                else
                    hjj = wnorm
                end
            else # j==1
                if rnorm < tol1
                    # starting with a nullvector? Surely you jest.
                    hjj = zero(T)
                    null1 = true
                    break
                else
                    hjj = rnorm
                end
            end
            @_dbg_arnoldi print(" l=$l hⱼⱼ=$hjj" * (reorth ? "(R)" : ""))
            if inspan
                @_dbg_arnoldi println()
                _reinitialize!(PK, l + 1, j - 1) || throw_reinit()
            else
                ujlp1 = view(Us[l + 1], :, j)
                mul!(ujlp1, 1 / hjj, v)
            end
            Hs[l][1:(j - 1), j] .= hj
            Hs[l][j, j] = hjj
        end # l loop
        if null1
            # just start over
            _kry_verby[] > 0 && println("initialized with nullvector; starting over.")
            _reinitialize!(PK, 1, 0)
            continue
        end

        mul!(v, As[p], view(Us[p], :, j))
        rnorm = norm(v)
        Uprev = view(Us[1], :, 1:j)
        hj = view(h, 1:j)
        mul!(hj, Uprev', v)
        mul!(v, Uprev, hj, -1, 1)
        wnorm = norm(v)
        reorth = false
        if wnorm < η * rnorm
            reorth = true
            rnorm = wnorm
            correction = Uprev' * v
            mul!(v, Uprev, correction, -1, 1)
            hj .+= correction
            wnorm = norm(v)
        end
        if wnorm <= η * rnorm
            hjp1j = zero(T)
            if ldeflate == 0
                ldeflate = p
                jdeflate = j
            end
            inspan = true
        else
            hjp1j = wnorm
            mul!(view(Us[1], :, j + 1), (1 / hjp1j), v)
        end
        @_dbg_arnoldi print(" l=$p hⱼ₊₁ⱼ=$hjp1j" * (reorth ? "(R)" : ""))
        if ldeflate == p
            @_dbg_arnoldi println()
            _kry_verby[] > 0 && println("trivial deflation j=$jdeflate")
            _reinitialize!(PK, 1, j) || throw_reinit()
            ldeflate = 0
        else
            ujlp1 = view(Us[1], :, j + 1)
            mul!(ujlp1, 1 / hjp1j, v)
        end
        Hs[p][1:j, j] .= hj
        Hs[p][j + 1, j] = hjp1j
        if ldeflate > 0
            @_dbg_arnoldi println()
            _kry_verby[] > 0 && println("deflating for singularity j=$jdeflate l=$ldeflate")
            Z = [Matrix{T}(I, j, j) for _ in 1:p]
            _deflate!(Hs[p], [Hs[ll] for ll in 1:(p - 1)], Z, ldeflate, jdeflate)
            for l in 1:p
                # no change to Vs[p][:,kmax+1] (cf. Kressner)
                vtmp = Us[l][:, 1:j]
                mul!(view(Us[l], :, 1:j), vtmp, Z[l])
            end
            hn = norm(view(Hs[p], 1:jdeflate, 1:jdeflate))
            if abs(Hs[p][jdeflate + 1, jdeflate]) < 100 * eps(real(T)) * hn
                _kry_verby[] > 0 && println("accepting null")
            else
                singularities += 1
                if singularities > 5
                    # TODO: reconsider this if it ever happens
                    @info "arnoldi: too many singularities"
                    return false
                end
                if jdeflate < k2
                    _reinitialize!(PK, 1, jdeflate + 1) || throw_reinit()
                end
                #continue
            end
        end
        @_dbg_arnoldi println()
        PK.kcur[] = j
        j += 1
    end
    return true
end

"""
    partial_pschur(As, nev, which; kwargs...) → PartialPeriodicSchur, ArnoldiMethod.History

Find a `nev`-order partial periodic Schur decomposition  of the product `Aₚ*...*A₂*A₁`
with eigenvalues near a specified region of the spectral boundary.

The elements `Aⱼ` can be matrices or any linear maps that implement `mul!(y, Aⱼ, x)`,
`eltype` and `size`.

The method will run iteratively until the Schur vectors are approximated to
the prescribed tolerance or until `restarts` restarts have passed.

## Arguments

| Name | Type | Default | Description |
|------:|:-----|:----|:------|
| `nev` | `Int` | `min(6, size(A, 1))` |Number of eigenvalues |
| `which` | `Target` | `LM()` | One of `LM()`, `LR()`, `SR()`, `LI()`, `SI()`, see below. |

The most important keyword arguments:

| Keyword | Type | Default | Description |
|------:|:-----|:----|:------|
| `tol` | `Real` | `√eps` | Tolerance for convergence: ‖AV - VT‖₂ < tol * ‖λ‖ |
| `maxdim` | `Int` | `max(20,2*nev)` | order of working Krylov subspace |
| `restarts` | `Int` | 100 | limit on restart iterations |

The target `which` can be any appropriate subtype of
[`ArnoldiMethod.Target`](https://julialinearalgebra.github.io/ArnoldiMethod.jl/stable/usage/01_getting_started.html).
"""
function partial_pschur(As::Vector{TA},
                        nev::Integer,
                        which::Target = LM();
                        mindim::Integer = min(max(10, nev), size(As[1], 1)),
                        maxdim::Integer = min(max(20, 2nev), size(As[1], 1)),
                        u1 = nothing,
                        tol = sqrt(eps(real(vtype(As[1])))),
                        tol1 = 100 * eps(real(vtype(As[1]))),
                        vrand! = Random.rand!,
                        restarts = 100,
                        purgebuffer = 2) where {TA}
    T = vtype(As[1])
    p = length(As)
    n = size(As[1])[1]
    for l in 1:p
        if checksquare(As[l]) != n
            throw(ArgumentError("all As must have the same (square) size"))
        end
    end
    if nev < 1
        throw(ArgumentError("nev cannot be less than 1"))
    end
    nev ≤ mindim ≤ maxdim ≤ p * n ||
        throw(ArgumentError("nev ≤ mindim ≤ maxdim does not hold, got $nev ≤ $mindim ≤ $maxdim"))
    PK = PKrylov{T}(p, n, maxdim, vrand!)
    _partial_pschur!(As,
                     PK,
                     vtype(As[1]),
                     mindim,
                     maxdim,
                     nev,
                     tol,
                     tol1,
                     restarts,
                     which,
                     u1,
                     purgebuffer)
end

function _showritz(ritz, str, n = length(ritz.rs))
    _printsty(:cyan, "Ritzvals$(str):")
    println()
    show(stdout, "text/plain", ritz.λs[1:n]')
    println()
    println("Ritz resids:")
    show(stdout, "text/plain", ritz.rs[1:n]')
    println()
    nothing
end

function _partial_pschur!(As,
                          PK,
                          ::Type{T},
                          kmin,
                          kmax,
                          nev,
                          tol,
                          tol1,
                          restarts,
                          which,
                          u1,
                          purgebuffer;
                          suspicion = 0) where {T}
    _kry_verby[] > 0 &&
        println("Partial pschur{$T} $which kmin,kmax,nev = $kmin,$kmax,$nev")
    p = length(As)
    n = size(As[1], 1)
    Hs = PK.Bs
    Vs = PK.Vs
    # workspace for change of basis
    Vtmp = Matrix{T}(undef, n, kmax)
    Htmp = similar(PK.Bs[1])
    # FIXME: use views of these when ready
    # Qs = [Matrix{T}(undef, kmax, kmax) for _ in 1:p]
    x = zeros(complex(T), kmax)

    # Approximate residual norms for all Ritz values, and Ritz values
    ritz = RitzValues{T}(kmax)
    isconverged = IsConverged(ritz, tol)
    ordering = get_order(which)
    groups = zeros(Int, kmax)

    active = 1
    k = kmin
    effective_nev = nev

    if u1 === nothing
        v = Vector{T}(undef, n)
        PK.vrand!(v)
    else
        if length(u1) != n
            throw(ArgumentError("u1 must have length matching first matrix/operator"))
        end
        v = copy(u1)
    end
    rmul!(v, 1 / norm(v))

    pa_ok = periodic_arnoldi!(As, PK, 1:kmin, v)
    @_dbg_pk _check(PK, As, "after initial Arnoldi", nw = true)

    nprods = p * kmin
    nlock = 0

    for iter in 1:restarts
        if iter > 1
            _restore_hessenberg!(PK, active, k, Vtmp, Htmp)
            @_dbg_pk _check(PK, As, "after Hessenberg", detail = checkdet[])
        end

        _kry_verby[] > 0 && _printsty(:green, "Extend/restart $iter k=$k active=$active\n")

        v .= PK.Vs[1][:, k + 1]
        pa_ok = periodic_arnoldi!(As, PK, (k + 1):kmax, v)
        @_dbg_pk _check(PK, As, "after Arnoldi", detail = checkdet[], nw = true)

        nprods += p * length((k + 1):kmax)

        # for l in 1:p
        #     copyto!(Qs[l], I)
        # end
        # just allocate and copy back, at least until the logic appears correct
        nk = length(active:kmax)
        Qtw = [Matrix{T}(I, nk, nk) for _ in 1:p]
        H1 = UpperHessenberg(Hs[p][active:kmax, active:kmax])
        # it is simplest (but confusing) to call the 'R' (non-reversed) version of pschur!
        if p == 1
            Hx = similar(Hs,0)
        else
            Hx = [triu(Hs[p - l][active:kmax, active:kmax]) for l in 1:(p - 1)]
        end
        if T <: Complex
            gps = pschur!(H1, Hx, Q = Qtw)
            PS = PeriodicSchur(gps.T1,
                               gps.T,
                               gps.Z,
                               gps.values,
                               gps.orientation,
                               gps.schurindex)
        else
            PS = pschur!(H1, Hx, Q = Qtw)
        end

        # Kressner Eq. 22 involves only the one matrix H1
        Hpnorm = norm(H1)
        isconverged.H_frob_norm[] = Hpnorm

        ritz.λs[active:kmax] .= PS.values
        Hpfoot = Hs[p][(kmax + 1):(kmax + 1), active:kmax] # rowvector

        # old logic (sorting 1:kmax) was wrong if we acquire an unconverged eigval
        # preferable to the previously locked ones. (This is true of AM too.)

        copyto!(ritz.ord, Base.OneTo(kmax))
        sort!(ritz.ord, active, kmax, QuickSort, OrderPerm(ritz.λs, ordering))
        _kry_verby[] > 0 && println("ritz.ord for locking test: ", ritz.ord)
        effective_nev = include_conjugate_pair(T, ritz, nev)
        # in principle we could get only lockable resids now, but that makes
        # for extra confusing logic later.
        # _compute_ritz_resids!( ritz, PS, Hpfoot, active:kmax, active, isconverged)
        _compute_ritz_resids!(ritz, PS, Hpfoot, active:kmax, active, i -> true)

        if _kry_verby[] > 0
            # _showritz(ritz," (after Arnoldi: ignore unlocked resids)")
            _showritz(ritz, " (after Arnoldi)")
        end

        # partition!(pred, seq, rg) reorders seq[rg] into preimages of pred -> true,false
        # returning index of first false

        # find how many preferred ev may have converged
        first_not_conv_idx = partition!(isconverged, ritz.ord, active:effective_nev)
        _kry_verby[] > 0 && println("ritz.ord after lock ordering: ", ritz.ord)
        nlockprev = nlock
        nlock = first_not_conv_idx === nothing ? effective_nev : first_not_conv_idx - 1
        if (_kry_verby[] > 0) && (nlock > 0)
            # println("first_not_conv_idx = $first_not_conv_idx effective_nev = $effective_nev")
            adv = (nlock == nlockprev) ? "" : "tentatively "
            println("$(adv)locking $nlock indices: ", ritz.ord[1:nlock])
        end

        kgood = kmax
        if nlock >= active
            # Select Ritz values to lock
            j0 = active - 1
            select = falses(kmax - j0)
            @inbounds for i in 1:nlock
                j = ritz.ord[i]
                if j in active:kmax
                    select[j - j0] = true
                end
            end
            # move 1:nlock to top
            selected = findall(select)
            nsel = length(selected)
            @assert nsel > 0
            ordschur!(PS, select)
            if _kry_verby[] > 0
                println("newly locked PS indices: ", selected)
                # println("reordered PS values: ",PS.values)
                # println("PS prod: ");
                # Tprod = PS.T1*prod(PS.T)
                # show(stdout, "text/plain", Tprod); println()
            end

            _update_ritz!(ritz, PS, select, active, kmax, kmax, ordering)

            kgood = kmax
            # CHECKME: may need paranoid check of locked Schur pairs here

            if _kry_verby[] > 0
                _showritz(ritz, " (after locking ordschur)")
                println("ritz.ord:")
                show(stdout, "text/plain", ritz.ord')
                println()
            end
        end
        # if we want to (re-) compute resids, we need to apply ordschur Q to Hpfoot
        # _compute_ritz_resids!( ritz, PS, Hpfoot, active:kmax, active, x -> true)
        if _kry_verby[] > 0
            ncx = sum(isconverged, 1:kmax)
            println("Apparently converged: $ncx, ‖Hₚ‖ = $(isconverged.H_frob_norm[])")
        end

        if nlock < nev
            # reorder so converged unwanted Schur pairs will be purged on truncation
            # pace AM, there is no reason to expect them to converge in order of preference,
            # so we normally allow a "buffer" of partially converged hopefuls.
            istart = nlock + 1 + purgebuffer
            nopurge(i) = !isconverged(i)
            conv_idx = partition!(nopurge, ritz.ord, istart:kgood)
        else
            conv_idx = nothing
        end

        # set length of truncated subspace
        k = include_conjugate_pair(T, ritz, min(nlock + kmin, (kmin + kmax) ÷ 2))
        if _kry_verby[] > 0 && conv_idx !== nothing
            ncx = kmax - conv_idx + 1
            println("unwanted converged: $ncx;  purging $(ritz.ord[max(k+1,conv_idx):kmax])")
        end

        # Select Ritz values to retain
        select = falses(kmax - active + 1)
        @inbounds for i in 1:k
            j = ritz.ord[i]
            if j in active:kmax
                select[j - active + 1] = true
            end
        end
        # move wanted ones to top
        _kry_verby[] > 0 && println("wanted PS idx: ", findall(select))
        PS0 = deepcopy(PS)
        ord_ok = true
        try
            ordschur!(PS, select)
        catch JE
            if JE isa IllConditionedException
                @warn "reordering failed, start praying."
            else
                rethrow(JE)
            end
            ord_ok = false
            PS = PS0
            Qtw = PS.Z
        end
        # if _kry_verby[] > 0
        #     println("leading active Ritzvals:")
        #     show(stdout, "text/plain", PS.values[1:min(5,k)]'); println()
        # end
        if ord_ok
            _update_ritz!(ritz, PS, select, active, kmax, kmax, ordering)
        end
        _kry_verby[] > 0 && _showritz(ritz, " (after trunc ordschur)")

        # stuff PS back into PK
        Qs = similar(Qtw)
        Qs[1] = Qtw[1]
        for l in 2:p
            Qs[l] = Qtw[p + 2 - l]
        end
        Hs[p][active:kmax, active:kmax] .= PS.T1
        oldrow = Hs[p][(kmax + 1):(kmax + 1), active:kmax]
        mul!(view(Hs[p], (kmax + 1):(kmax + 1), active:kmax), oldrow, Qs[p])

        for l in 1:(p - 1)
            Hs[l][active:kmax, active:kmax] .= PS.T[p - l]
        end
        for l in 1:p
            # no change to Vs[p][:,kmax+1] (cf. Kressner)
            copyto!(view(Vtmp, :, active:kmax), view(Vs[l], :, active:kmax))
            mul!(view(Vs[l], :, active:kmax), view(Vtmp, :, active:kmax), Qs[l])
            if active > 1
                htmp = view(Htmp, 1:(active - 1), active:kmax)
                copyto!(htmp, view(Hs[l], 1:(active - 1), active:kmax))
                mul!(view(Hs[l], 1:(active - 1), active:kmax), htmp, Qs[l])
            end
        end
        @_dbg_pk _check(PK, As, "after schur+reorder", detail = checkdet[])
        # truncate:
        Vs[1][:, k + 1] .= Vs[1][:, kmax + 1]
        Hs[p][k + 1, active:k] .= Hs[p][kmax + 1, active:k]
        Hs[p][(k + 2):(kmax + 1), 1:kmax] .= 0
        for l in 1:(p - 1)
            Hs[l][(k + 2):kmax, 1:kmax] .= 0
        end
        PK.kcur[] = k
        @_dbg_pk _check(PK, As, "after truncation", detail = checkdet2[])
        # re-evaluate convergence
        nlock = _verify_locks!(ritz, view(Hs[p], 1:(k + 1), 1:k), nlock, isconverged)

        pa_ok || break

        active = nlock + 1
        active > nev && break
    end # restart iteration loop
    pa_ok || @warn "failure detected in Arnoldi process; results are suspect"
    # in principle one could add extra converged entries, but the overhead is painful
    nconv = active - 1

    if _kry_verby[] > 0
        copyto!(ritz.ord, Base.OneTo(kmax))
        sort!(ritz.ord, 1, k, QuickSort, OrderPerm(ritz.λs, ordering))
        println("final order by preference: ", ritz.ord[1:k])
        println("converged: ", findall(i -> isconverged(i), 1:k))
    end

    Vconv = [PK.Vs[l][:, 1:nconv] for l in 1:p]
    H1conv = PK.Bs[p][1:nconv, 1:nconv]
    Hconv = [triu!(PK.Bs[l][1:nconv, 1:nconv]) for l in 1:(p - 1)]

    history = History(nprods, nconv, nconv ≥ nev, nev)
    ps = PartialPeriodicSchur(H1conv, Hconv, Vconv, ritz.λs[1:nconv])

    return ps, history
end

function _restore_hessenberg!(PK::PKrylov{T}, active, k, Vtmp, Htmp) where {T}
    Hs = PK.Bs
    Vs = PK.Vs
    p = length(Hs)
    H1x = view(Hs[p], active:(k + 1), active:k)
    Hx = [view(Hs[l], active:k, active:k) for l in 1:(p - 1)]
    nwrk = k - active + 1
    Qs = [Matrix{T}(I, nwrk, nwrk) for _ in 1:p]
    H1x, Hx = _rphessenberg!(H1x, Hx, Qs)
    for l in 1:p
        # no change to Vs[p][:,kmax+1] (cf. Kressner)
        vtmp = view(Vtmp, :, active:k)
        copyto!(vtmp, view(Vs[l], :, active:k))
        mul!(view(Vs[l], :, active:k), vtmp, Qs[l])
        if active > 1
            htmp = view(Htmp, 1:(active - 1), active:k)
            copyto!(htmp, view(Hs[l], 1:(active - 1), active:k))
            mul!(view(Hs[l], 1:(active - 1), active:k), htmp, Qs[l])
        end
    end
end

# stores residuals in ritz.rs
# march through ritz.ord[rg]; exit early if continue_test() fails
function _compute_ritz_resids!(ritz,
                               PS::PeriodicSchur{T},
                               Hpfoot,
                               rg,
                               active,
                               continue_test) where {T}
    # Convergence is tested putting one Schur pair at a time at the top of the
    # active block.
    # Actually ISTM that we could mark any leading group of eigvals
    # as converged if all the corresponding subdiags are small.
    # For now we don't try anything clever.
    #
    # Note one doesn't need to modify the Vs just to move and check, if
    # the original Hs are saved.
    #

    p = PS.period
    λprev = -ritz.λs[1]
    nwrk = length(rg)
    pairflag = 0
    j0 = active - 1
    for jo in rg
        j = ritz.ord[jo]
        ritz.rs[j] = Inf
    end
    for jo in rg
        j = ritz.ord[jo]
        λ = ritz.λs[j]
        inpair = (T <: Real) && (imag(λ) != 0)
        if inpair
            pairflag += 1
        end
        if pairflag == 2
            if !(λ ≈ conj(λprev))
                @warn "mishandled pair $j in compute_resid"
            end
            pairflag = 0
            continue
        end
        select = falses(nwrk)
        select[j - j0] = true
        if pairflag == 1
            select[j + 1 - j0] = true
            λprev = λ
        end
        PSx = deepcopy(PS)

        # fallback needs this
        # CHECKME: maybe only compute if needed
        Qpcur = PS.Z[p == 1 ? 1 : 2]
        curfoot = similar(Hpfoot)
        mul!(curfoot, Hpfoot, Qpcur)

        try
            ordschur!(PSx, select)
        catch JE
            if JE isa IllConditionedException
                # CHECKME: may need better logic for cases where ordschur fails.
                # We may want to treat some tight clusters similar to conjugate pairs.
                # But for those which converge one at a time, the existing logic works
                # (we don't arrive here).
                _kry_verby[] > 0 && @warn "punting on convergence for j=$j"
                # ritz.rs[j] = Inf
                ritz.rs[j] = maximum(abs.(view(curfoot, 1:j)))
                continue
            else
                rethrow(JE)
            end
        end
        Qs = similar(PSx.Z)
        Qs[1] = PSx.Z[1]
        for l in 2:p
            Qs[l] = PSx.Z[p + 2 - l]
        end
        newrow = similar(Hpfoot)
        mul!(newrow, Hpfoot, Qs[p])
        if pairflag > 0
            r = max(abs(newrow[1]), abs(newrow[2]))
            ritz.rs[j] = r
            ritz.rs[j + 1] = r
        else
            ritz.rs[j] = abs(newrow[1]) # as if PSx.H1[nwrk+1,1]
        end
        continue_test(j) || return nothing
    end
    nothing
end

# make fields of ritz consistent with PS after reordering
# WARNING: assumes stable sort in ordschur
function _update_ritz!(ritz, PS, select, active, kmax, kgood, ordering)
    oldrs = ritz.rs[active:kmax]
    ritz.λs[active:kmax] .= PS.values
    j0 = active - 1
    j1 = active
    nsel = count(select)
    j2 = active + nsel
    # if _kry_verby[] > 0
    #     println("in _update_ritz! active = $active, kmax = $kmax, kgood = $kgood")
    #     println("  select len = $(length(select)) nsel = $nsel drop $(findall(x -> !x, select))")
    # end
    for j in 1:(kmax - j0)
        if select[j]
            ritz.rs[j1] = popfirst!(oldrs) # ritz.rs[j0+j]
            j1 += 1
        else
            ritz.rs[j2] = popfirst!(oldrs)
            j2 += 1
        end
    end
    copyto!(ritz.ord, Base.OneTo(kmax))
    sort!(ritz.ord, active, kgood, QuickSort, OrderPerm(ritz.λs, ordering))
end

# another bit of paranoia
# make sure that the reordering operation has not degraded things so that some
# Schur pair is no longer accurate enough
function _verify_locks!(ritz, Hp::AbstractMatrix{T}, nlock, isconverged) where {T}
    k = size(Hp, 2)
    isconverged.H_frob_norm[] = norm(Hp)
    pairflag = 0
    local ri1
    for i in 1:nlock
        if T <: Real
            λ = ritz.λs[i]
            if !isreal(λ)
                pairflag += 1
            end
            if pairflag == 1
                continue
            elseif pairflag == 2
                pairflag == 0
                ritz.rs[i] = hypot(Hp[k + 1, i - 1], Hp[k + 1, i])
            else
                ritz.rs[i] = abs(Hp[k + 1, i])
            end
        else
            ritz.rs[i] = abs(Hp[k + 1, i])
        end
    end
    ncv = 0
    i = 1
    while i <= nlock
        isconverged(i) || break
        if T <: Real && !isreal(ritz.λs[i])
            i += 1
        end
        ncv = i
        i += 1
    end
    if _kry_verby[] > 0 && ncv != nlock
        println("resetting lock count to $ncv")
    end
    return ncv
end

# PartialPeriodicSchur is just a special case of PeriodicSchur
# for most purposes
"""
    eigvecs(ps::PartialPeriodicSchur, select::Vector{Bool}; shifted::Bool)

Similar to `eigvecs(ps::PeriodicSchur, select;...)`.
"""
function LinearAlgebra.eigvecs(ps0::PartialPeriodicSchur,
                               select::AbstractVector{Bool}; kwargs...)
    P = deepcopy(ps0)
    ps = PeriodicSchur(P.T1, P.T, P.Z, P.values, P.orientation, P.schurindex)
    return eigvecs(ps, select; kwargs...)
end
