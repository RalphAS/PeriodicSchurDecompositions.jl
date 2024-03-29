module PeriodicSchurDecompositions
using LinearAlgebra

# Some algorithms are configurable until tests are more comprehensive.
# These are in sections marked ALGO_CONFIG

import LinearAlgebra: lmul!, rmul!
using LinearAlgebra: checksquare, require_one_based_indexing
using LinearAlgebra: Givens, givensAlgorithm

export pschur, pschur!, phessenberg!, gpschur, PeriodicSchur, GeneralizedPeriodicSchur

# not documented until we decide what we should guarantee about it
abstract type AbstractPeriodicSchur{T} end

# Do this early in case we want to use any bits in constructors etc.
include("diagnostics.jl")

"""
    IllConditionedException <: Exception

Exception thrown when an operation on a Schur decomposition fails because of
ill-conditioning. The `info` field may be the index of an eigenvalue associated with
the failure.
"""
struct IllConditionedException <: Exception
    info::Int
end

struct NotImplemented <: Exception
    msg::String
end

"""
PeriodicSchur

Matrix factorization type of the periodic Schur factorization of a series
`A₁, A₂, ... Aₚ` of matrices. This is the return type of [`pschur!(_)`](@ref).

The `orientation` property may be `'L'`(left), corresponding to the product
 `Aₚ * Aₚ₋₁ * ... * A₂ * A₁`
or `'R'`(right), for the product `A₁ * A₂ * ... * Aₚ`.

The decomposition for the "right" orientation is
`Z₁' * A₁ * Z₂ = T₁; Z₂' * A₂ * Z₃ = T₂; ...; Zₚ' * Aₚ * Z₁ = Tₚ.`

The decomposition for the "left" orientation is
`Z₂' * A₁ * Z₁ = T₁; Z₃' * A₂ * Z₂ = T₂; ...; Z₁' * Aₚ * Zₚ = Tₚ.`

For real element types, `Tₖ` is a quasi-triangular "real Schur" matrix,
where `k` is the value of the `schurindex` field. Otherwise
the `Tⱼ` are upper triangular. The `Zⱼ` are unitary (orthogonal for reals).

Given `F::PeriodicSchur`, the (quasi) triangular Schur factor `Tₖ` can be obtained via
`F.T1`.  `F.T` is a vector of the remaining triangular `Tⱼ`.
`F.Z` is a vector of the `Zⱼ`.
`F.values` is a vector of the eigenvalues of the product of the `Aⱼ`.
"""
struct PeriodicSchur{Ty, St1 <: AbstractMatrix, St <: AbstractMatrix, Sz <: AbstractMatrix
                     } <:
       AbstractPeriodicSchur{Ty}
    T1::St1
    T::Vector{St}
    Z::Vector{Sz}
    values::Vector
    orientation::Char
    schurindex::Int
    function PeriodicSchur{Ty, St1, St, Sz}(T1::AbstractMatrix{Ty},
                                            T::Vector{<:AbstractMatrix{Ty}},
                                            Z::Vector{<:AbstractMatrix{Ty}},
                                            values::Vector,
                                            orientation::Char = 'R',
                                            schurindex::Int = 1) where {Ty, St1, St, Sz}
        new(T1, T, Z, values, orientation, schurindex)
    end
end
function PeriodicSchur(T1::St1,
                       T::Vector{<:AbstractMatrix{Ty}},
                       Z::Vector{<:AbstractMatrix{Ty}},
                       values::Vector,
                       orientation::Char = 'R',
                       schurindex::Int = 1) where {St1 <: AbstractMatrix{Ty}} where {Ty}
    PeriodicSchur{Ty, St1, eltype(T), eltype(Z)}(T1, T, Z, values, orientation, schurindex)
end
function Base.getproperty(P::PeriodicSchur, s::Symbol)
    if s == :period
        return length(P.T) + 1
    else
        return getfield(P, s)
    end
end
Base.propertynames(P::PeriodicSchur) = (:period, fieldnames(typeof(P))...)

include("householder.jl")
# const _reflector! = LinearAlgebra.reflector!
const _reflector! = _xreflector!

"""
    pschur(A::Vector{S<:StridedMatrix}, lr::Symbol) -> F::PeriodicSchur

Computes a periodic Schur decomposition of a series of general square matrices
with left (`lr=:L`) or right (`lr=:R`) product ordering.

Optional arguments `wantT` and `wantZ`, defaulting to `true`, are booleans which may
be used to save time and memory by suppressing computation of the `T` and `Z`
matrices. See [`PeriodicSchur`](@ref) for the resulting structure.
"""
function pschur(A::AbstractVector{S},
                lr::Symbol = :R;
                kwargs...) where {S <: AbstractMatrix{T}} where {T}
    Atmp = [copy(Aj) for Aj in A]
    pschur!(Atmp, lr; kwargs...)
end

"""
    pschur!(A::Vector{S<:StridedMatrix}, lr::Symbol=:R) -> F::PeriodicSchur

Same as [`pschur`](@ref) but uses the input matrices `A` as workspace.
"""
function pschur!(A::AbstractVector{S},
                 lr::Symbol = :R;
                 wantZ::Bool = true,
                 wantT::Bool = true,
                 maxitfac=30) where {S <: AbstractMatrix{T}} where {T <: Real}
    orient = char_lr(lr)
    p = length(A)
    if orient == 'L'
        Aarg = similar(A)
        for j in 1:p
            Aarg[j] = A[p + 1 - j]
        end
        H1, pH = phessenberg!(Aarg)
    else
        H1, pH = phessenberg!(A)
    end
    if wantZ
        Q = [_materializeQ(H1)]
        for j in 1:(p - 1)
            push!(Q, Matrix(pH[j].Q))
        end
    else
        Q = nothing
    end
    if p == 1
        Hs = Vector{Matrix{T}}(undef, 0)
    else
        Hs = [pH[j].R for j in 1:(p - 1)]
    end
    H1.H .= triu(H1.H, -1)
    F = pschur!(H1.H, Hs, wantT = wantT, wantZ = wantZ, Q = Q, rev = (orient == 'L'),
                maxitfac = maxitfac)
end

# orientation
function char_lr(lr::Symbol)
    if lr === :R
        return 'R'
    elseif lr === :L
        return 'L'
    else
        throw_lr()
    end
end

function sym_lr(lr::Char)
    if lr == 'R'
        return :R
    elseif lr == 'L'
        return :L
    else
        throw_lr()
    end
end

@noinline function throw_lr()
    throw(ArgumentError("orientation argument must be either :R (right) or :L (left)"))
end

using LinearAlgebra: QRPackedQ
_materializeQ(H::Hessenberg{T}) where {T <: LinearAlgebra.BlasFloat} = Matrix(H.Q)

function _materializeQ(H::Hessenberg{T}) where {T}
    A = copy(H.Q.factors)
    n = checksquare(A)
    # shift reflectors one column rightwards
    @inbounds for j in n:-1:2
        A[1:(j - 1), j] .= zero(T)
        for i in (j + 1):n
            A[i, j] = A[i, j - 1]
        end
    end
    A[2:n, 1] .= zero(T)
    A[1, 1] = one(T)
    Q1 = QRPackedQ(view(A, 2:n, 2:n), H.Q.τ)
    A[2:n, 2:n] .= Matrix{T}(Q1)
    A
end

# portions derived from SLICOT routine MB03VD
# SLICOT Copyright (c) 2002-2020 NICONET e.V.
"""
phessenberg!(A::Vector{<:AbstractMatrix}) -> (Hessenberg, Vector{QR})

reduce a series of `p` matrices `A = [A₁ A₂ ... Aₚ]` to upper Hessenberg/triangular form
via a cycle of orthogonal similarity transformations

`Q₁'A₁Q₂ = H₁`
`Q₂'A₂Q₃ = H₂`
`Qₚ'AₚQ₁ = Hₚ`

`H₁` is upper Hessenberg, the other `Hⱼ` are upper triangular.
"""
function phessenberg!(A::AbstractVector{S}) where {S <: AbstractMatrix{T}} where {T}
    p = length(A)
    n = checksquare(A[1])
    require_one_based_indexing(A[1])
    for j in 2:p
        require_one_based_indexing(A[j])
        if checksquare(A[j]) != n
            throw(DimensionMismatch())
        end
    end
    τ = Vector{Vector{T}}(undef, p)
    τ[1] = zeros(T, n - 1)
    # extra zero needed for coherent representation as QR
    for j in 2:p
        τ[j] = zeros(T, n)
    end
    for i in 1:(n - 1)
        i1 = i + 1
        i2 = min(i + 2, n)
        for j in p:-1:2
            # using a view stores the reflectors in A a la LAPACK
            ξ = view(A[j], i:n, i)
            t = _reflector!(ξ)
            H = Householder{T, typeof(ξ)}(view(ξ, 2:(n - i + 1)), t)
            τ[j][i] = H.τ
            lmul!(H', view(A[j], i:n, i1:n))
            rmul!(view(A[j - 1], :, i:n), H)
        end
        ξ = view(A[1], i1:n, i)
        t = _reflector!(ξ)
        H = Householder{T, typeof(ξ)}(view(ξ, 2:(n - i)), t)
        τ[1][i] = H.τ
        lmul!(H', view(A[1], i1:n, i1:n))
        rmul!(view(A[p], :, i1:n), H)
    end

    H1 = Hessenberg(A[1], τ[1])  # same as construction from LAPACK.gehrd!() result
    if p > 1
        pH = [QR(A[2], τ[2])]
        for j in 3:p
            push!(pH, QR(A[j], τ[j]))
        end
    else
        pH = Vector{QR{T, Matrix{T}}}(undef, 0)
    end
    return H1, pH
end

# We need to zero out the lower parts for our purposes.
_gethess!(A::StridedMatrix) = triu!(A, -1)
# WARNING: need to take data property because UpperHessenberg is not <: StridedArray
# This depends on LinearAlgebra internals
_gethess!(A::UpperHessenberg) = triu!(A.data, -1)

if isfile(joinpath(@__DIR__, "debugging.jl"))
    include("debugging.jl")
else
    macro _dbg_rpschur(expr)
        nothing
    end
    macro _dbg_gpschur(expr)
        nothing
    end
    macro _dbg_rgpschur(expr)
        nothing
    end
end

# The standard real PQZ for Hessenberg/triangular product
# Mainly translated from SLICOT routine MB03WD, by V.Sima following A.Varga
# SLICOT Copyright (c) 2002-2020 NICONET e.V.

# ALGO_CONFIG
# SLICOT has some peculiar computations for shifts; alternative is based on LAPACK
const _slicot_shifts = Ref(false)
# SLICOT's convergence criterion is often too lax; alternative is based on LAPACK
const _slicot_convg = Ref(false)

# Even the Ahues-Tisseur scheme from LAPACK seems too lax for the periodic case.
# We shrink the threshold by an empirically-based factor; the relative threshold is
# effectively ϵ^(1 + _AT_pwr16[] / 16).
# It should perhaps depend on the period.
const _AT_pwr16 = Ref(4)

# The RQ iteration for deflation seems to lack a final stage; this can enable it.
const _extra_rq = Ref(false)

# SLICOT (following LAPACK) allows limiting QR to a lower portion of the system.
# This is dangerous for some matrices, so it is suppressed by default.
const _allow_early_QR = Ref(false)

"""
    pschur!(H1,Us; Q) -> PeriodicSchur

Computes a periodic Schur decomposition of a series of Hessenberg/upper-triangular matrices.

`H1` must be upper Hessenberg, `Us` a vector of upper triangular matrices.
The argument arrays are overwritten and used in the result, a
[`PeriodicSchur`](@ref) object.

Keyword arguments:
The result corresponds to the rightwards product `H1*prod(Us)` unless
the optional argument `rev=true` is specified.
Specify `wantT=false` to compute eigenvalues, but not the Schur factors.
Specify `wantZ=false` to suppress computation of orthogonal transformation
matrices.
if `Q` is not provided, these will be the `Zⱼ` factors;
if `Q` is set to a vector of matrices `Qⱼ`, they will be the products `QⱼZⱼ`.
"""
function pschur!(H1H::S1,
                 Hs::AbstractVector{S};
                 wantT::Bool = true,
                 wantZ::Bool = true,
                 Q = nothing,
                 maxitfac = 30,
                 rev = false,
                 ) where {S1 <: Union{UpperHessenberg{T}, StridedMatrix{T}},
                                     S <: StridedMatrix{T}} where {T <: Real}
    p = length(Hs) + 1
    n = size(H1H, 1)
    if n == 1
        λ1 = H1H[1, 1]
        for j in 2:p
            λ1 *= Hs[j - 1][1, 1]
        end
        if Q === nothing
            Z = [fill(one(T), 1, 1) for j in 1:p]
        else
            Z = Q
        end
        if rev
            sT1 = fill(H1H[1, 1], 1, 1)
            sT = [fill(Hs[p - j][1, 1], 1, 1) for j in 1:(p - 1)]
            return PeriodicSchur(sT1, sT, Z, [λ1], 'L', p)
        else
            sT1 = fill(H1H[1, 1], 1, 1)
            sT = [fill(Hs[j - 1][1, 1], 1, 1) for j in 2:p]
            return PeriodicSchur(sT1, sT, Z, [λ1])
        end
    end

    dat1 = T(3) / T(4)
    dat2 = -T(7) / T(16)

    λ = zeros(complex(T), n)

    wr = zeros(T, n)
    wi = zeros(T, n)
    v = zeros(T, 3)

    unfl = floatmin(T)
    ovfl = 1 / unfl
    ulp = eps(T)
    ulpx = ulp
    AT_hi, AT_lo = divrem(_AT_pwr16[], 16)
    ulpx *= ulp^AT_hi
    s = ulp
    for iu in (8,4,2,1)
        s = sqrt(s)
        if (AT_lo & iu) != 0
            ulpx *= s
        end
    end

    smlnum = unfl * (n / ulp)

    s = ulp * n
    if n > 2
        view(H1H, 3:n, 1) .= zero(T)
    end
    # so we can index directly by j
    hnorms = zeros(T, p)
    for j in 2:p
        view(Hs[j - 1], 2:n, 1) .= zero(T)
        hnorms[j] = s * opnorm(Hs[j - 1], 1)
    end
    if wantT
        i1 = 1
        i2 = n
    end
    if wantZ
        if Q == nothing
            Z = [Matrix{T}(I, n, n) for j in 1:p]
        else
            Z = Q
        end
    end

    # The aliases aim for consistency w/ SLICOT but seem more understandable.
    hdiag = wr
    hsubdiag = wi
    hsupdiag = zeros(T, n)

    H1 = _gethess!(H1H)
    Hp = p == 1 ? H1 : Hs[p - 1]

    function showmat(str, j)
        print(str, " H[$j] ")
        show(stdout, "text/plain", j == 1 ? H1 : Hs[j - 1])
        println()
        nothing
    end

    @_dbg_rpschur fcheck = _FacChecker(H1, Hs, Z, wantZ)

    function showprod(str)
        if p > 1
            Htmp = copy(H1)
            for j in 2:p
                Htmp = Htmp * Hs[j - 1]
            end
            print(str, " ℍ ")
            show(stdout, "text/plain", Htmp)
            println()
            println("  ev: ", eigvals(Htmp)')
        end
        nothing
    end

    if verbosity[] > 2
        showprod("begin")
        showmat("begin", 1)
    end
    if verbosity[] > 3
        for j in 2:p
            showmat("begin", j)
        end
    end

    maxit = maxitfac * n
    maxitleft = maxit

    # main loop
    # i is loop index, decreasing in steps of 1 or 2
    i = n
    local splitting # flag for control flow
    local tst1
    if verbosity[] > 0
        print("Real periodic QR p=$p n=$n, ")
        println((_slicot_shifts[] ? "slicot" : "lapack")," shifts, ",
                (_slicot_convg[] ? "slicot" : "lapack"), " convg")
    end
    verbosity[] > 1 && println("AT criterion power is $(log2(ulpx)/log2(ulp))")

    # maxits and niter are just for reporting
    maxits = 0 # peak of iter count for all values of `i`
    niter = 0

    while i >= 1
        verbosity[] > 0 && println("starting main loop for block 1:$i")
        # eigvals i+1:n have converged

        # perform QR iterations on [1:i,1:i] until a submatrix of order 1 or 2 splits
        # off at the bottom

        # Let 𝕋 = H₂*H₃*...*Hₚ and ℍ = H₁ * T
        l = 1
        its = 1
        while its < maxitleft
            verbosity[] > 1 && _printsty(:cyan, "QR iteration l=$l i=$i, iter $its\n")
            splitting = false
            # initialization: compute ℍ[i,i] (and ℍ[i,i-1] if i > l)
            hp22 = one(T)
            if i > l
                hp12 = zero(T)
                hp11 = one(T)
                for j in 2:p
                    Hj = Hs[j - 1]
                    hp22 *= Hj[i, i]
                    hp12 = hp11 * Hj[i - 1, i] + hp12 * Hj[i, i]
                    hp11 *= Hj[i - 1, i - 1]
                end
                hh21 = H1[i, i - 1] * hp11
                hh22 = H1[i, i - 1] * hp12 + H1[i, i] * hp22
                hdiag[i] = hh22
                hsubdiag[i] = hh21
            else
                hp22 *= H1[i, i]
                for j in 2:p
                    hp22 *= Hs[j - 1][i, i]
                end
                hdiag[i] = hp22
            end

            # look for a single small subdiagonal
            # also compute needed current elements of diagonal,
            # first two supradiagonals of 𝕋,
            # and current elements of tridiag band of ℍ
            local klast
            found = false
            xmin = T(Inf)
            for k in i:-1:(l + 1)
                klast = k
                # evaluate ℍ[k-1,k-m:k], m=1 or 2
                hp00 = one(T)
                hp01 = zero(T)
                if k > l + 1
                    hp02 = zero(T)
                    for j in 2:p
                        Hj = Hs[j - 1]
                        hp02 = hp00 * Hj[k - 2, k] + hp01 * Hj[k - 1, k] + hp02 * Hj[k, k]
                        hp01 = hp00 * Hj[k - 2, k - 1] + hp01 * Hj[k - 1, k - 1]
                        hp00 *= Hj[k - 2, k - 2]
                    end
                    hh10 = H1[k - 1, k - 2] * hp00
                    hh11 = H1[k - 1, k - 2] * hp01 + H1[k - 1, k - 1] * hp11
                    hh12 = H1[k - 1, k - 2] * hp02 + H1[k - 1, k - 1] * hp12 +
                           H1[k - 1, k] * hp22
                    hsubdiag[k - 1] = hh10
                else
                    hh10 = zero(T)
                    hh11 = H1[k - 1, k - 1] * hp11
                    hh12 = H1[k - 1, k - 1] * hp12 + H1[k - 1, k] * hp22
                end
                hdiag[k - 1] = hh11
                hsupdiag[n - i + k - 1] = hh12

                # test for negligible subdiagonal
                if abs(hh21) < abs(xmin)
                    xmin = hh21
                end

                tst1 = abs(hh11) + abs(hh22)
                if tst1 == 0
                    tst1 = opnorm(view(H1, l:i, l:i), 1)
                end

                if _slicot_convg[]
                    found = abs(hh21) <= max(ulp*tst1, smlnum)
                else
                    # LAPACK has smlnum on the right, but that may lead to pointless
                    # iterations for tiny eigvals. We may want
                    # if abs(hh21) <= max(ulp^2 * tst1, smlnum)
                    if abs(hh21) <= smlnum
                        found = true
                    elseif abs(hh21) <= ulp * tst1
                        # The following is from LAPACK (less prone to spurious convergence)
                        ab = max(abs(hh21), abs(hh12))
                        ba = min(abs(hh21), abs(hh12))
                        aa = max(abs(hh22), abs(hh11 - hh22))
                        bb = min(abs(hh22), abs(hh11 - hh22))
                        stmp = aa + ab
                        found = ba * (ab / stmp) <= max(smlnum, ulpx * (bb * (aa / stmp)))
                        if verbosity[] > 1
                            t1 = ba * (ab / stmp)
                            t2 = ulp * (bb * (aa / stmp))
                            println((found ? "met" : "missed")
                                    * " AT criterion hh21=$hh21 vs. $(ulpx*tst1);"
                                    * "$t1 vs. $t2")
                            found || println("prod block ",[hh11 hh12; hh21 hh22])
                        end
                    end
                end
                if found
                    verbosity[] > 1 && println("found tiny hh21=$hh21 hh10=$hh10 at k=$k")
                    break
                end
                # update for next cycle
                hp22 = hp11
                hp11 = hp00
                hp12 = hp01
                hh22 = hh11
                hh21 = hh10
            end # k loop, search for small subdiagonal
            verbosity[] > 1 && !found && println("smallest subdiag: $xmin")

            # The above loop translates Fortran DO K=I,L+1,-1
            # which sets K to I even if I<=L, and sets K to L at end if unbroken.
            # (The last is true for all Fortran compilers known to me, but NOT
            # guaranteed by the Fortran standard AFAICT; this logic
            # is inherited from LAPACK.)
            # Next line translates L = K after loop
            l = (i > l) ? (found ? klast : l) : i

            # each iteration works with active block [l:i,l:i]
            # either l=1, or ℍ[l,l-1] is negligible
            if l > 1
                if wantT
                    # if H₁[l,l-1] is also negligible, zero it;
                    # otherwise the product of Hⱼ[l-1,l], j in 2:p is negligible, so
                    # annihilate subdiagonals, then restore Hⱼ to triangular (RQ step)
                    tst1 = abs(H1[l - 1, l - 1]) + abs(H1[l, l])
                    if tst1 == 0
                        tst1 = opnorm(view(H1, l:i, l:i), 1)
                    end
                    if abs(H1[l, l - 1]) > max(ulp * tst1, smlnum)
                        if verbosity[] > 1
                            _printsty(:green, "processing subdiag $l: $(H1[l, l - 1])\n")
                        end
                        for k in i:-1:l
                            for j in 1:(p - 1)
                                if j == 1
                                    Hj = H1
                                else
                                    Hj = Hs[j - 1]
                                end
                                # reflector to annihilate Hⱼ[k,k-1]
                                ξ = [Hj[k, k], Hj[k, k - 1]]
                                t = _reflector!(ξ)
                                Hj[k, (k - 1):k] .= (zero(T), ξ[1])
                                hr = HH2(ξ[2], one(T), t)
                                rmul!(view(Hj, i1:(k - 1), (k - 1):k), hr)
                                # apply to H[j+1]
                                lmul!(hr', view(Hs[j], (k - 1):k, (k - 1):i2))
                                if wantZ
                                    rmul!(view(Z[j + 1], :, (k - 1):k), hr)
                                end
                            end
                            if k < i
                                # compute reflector to annihilate Hₚ[k+1,k]
                                ξ = [Hp[k + 1, k + 1], Hp[k + 1, k]]
                                t = _reflector!(ξ)
                                Hp[k + 1, k:(k + 1)] .= (zero(T), ξ[1])
                                hr = HH2(ξ[2], one(T), t)
                                rmul!(view(Hp, i1:k, k:(k + 1)), hr)
                                # also transform rows of H₁
                                lmul!(hr', view(H1, k:(k + 1), k:i2))
                                if wantZ
                                    rmul!(view(Z[1], :, k:(k + 1)), hr)
                                end
                            end
                            verbosity[] > 2 && showprod("prep k=$k")
                        end # k loop (110)

                        if _extra_rq[]
                            # this was not in MB03WD
                            # reflector to annihilate Hₚ[l,l-1]
                            ξ = [Hp[l, l], Hp[l, l - 1]]
                            t = _reflector!(ξ)
                            Hp[l, (l - 1):l] .= (zero(T), ξ[1])
                            hr = HH2(ξ[2], one(T), t)
                            rmul!(view(Hp, i1:(l - 1), (l - 1):l), hr)
                            # apply to H1
                            lmul!(hr', view(H1, (l - 1):l, (l - 1):i2))
                            if wantZ
                                rmul!(view(Z[1], :, (l - 1):l), hr)
                            end
                            if verbosity[] > 1
                                println("after RQ H1[l, l - 1]:", H1[l, l - 1])
                            end
                        else
                            # MB03WD forces to 0, even when wrong
                            if verbosity[] > 1
                                println("after RQ Hp[l, l - 1]:", Hp[l, l - 1])
                            end
                            Hp[l, l - 1] = zero(T)
                        end
                    elseif verbosity[] > 1
                        println("  already small enough")
                    end # if subdiagonals were annihilated
                    H1[l, l - 1] = zero(T)
                    @_dbg_rpschur fcheck("after subdiag $l", H1, Hs, Z, check_Ap=true, check_A1=true)
                end # if wantT
            end # ℍ[l,l-1] was negligible (l > 1)
            # exit this loop if a submatrix of order 1 or 2 split off
            if l >= i - 1
                splitting = true
                verbosity[] > 0 && println("splitting l=$l")
                break
            end

            # bulge chasing stage (ll. 575-764)
            if !wantT
                i1 = l
                i2 = i
            end
            exc_shift = false
            if its == 10
                # exceptional shift
                exc_shift = true
                s = abs(hsubdiag[l + 1]) + abs(hsubdiag[l + 2])
                h44 = dat1 * s + hdiag[l]
                h33 = h44
                h43h34 = dat2 * s * s
                h43 = s
                h34 = dat2 * s
                verbosity[] > 0 && println("exceptional shift $h33 $h44 $h43h34")
            elseif its % 10 == 0
                # another exceptional shift, from LAPACK
                exc_shift = true
                s = abs(hsubdiag[i]) + abs(hsubdiag[i - 1])
                h44 = dat1 * s + hdiag[i]
                h33 = h44
                h43h34 = dat2 * s * s
                h43 = s
                h34 = dat2 * s
                verbosity[] > 0 && println("exceptional shift $h33 $h44 $h43h34")
            else
                # prepare for Francis' double shift
                # i.e., second degree generalized Rayleigh quotient
                h44 = hdiag[i]
                h33 = hdiag[i - 1]
                h43h34 = hsubdiag[i] * hsupdiag[n - 1]
                h43 = hsubdiag[i]
                h34 = hsupdiag[n - 1]

                if _slicot_shifts[]
                    disc = (h33 - h44) * T(0.5)
                    disc = disc * disc + h43h34
                    if disc > 0
                        # real roots: use Wilkinson's shift
                        rtdisc = sqrt(disc)
                        ave = (h33 + h44) * T(0.5)
                        if abs(h33) - abs(h44) > 0
                            h33 = h33 * h44 - h43h34
                            h44 = h33 / ((ave >= 0 ? rtdisc : -rtdisc) + ave)
                        else
                            h44 = (ave >= 0 ? rtdisc : -rtdisc) + ave
                        end
                        # use one value twice
                        h33 = h44
                        h43h34 = zero(T)
                        verbosity[] > 0 && println("slicot shift double $h33")
                    else
                        verbosity[] > 0 && println("slicot shift $h33 $h44 $h43h34")
                    end
                else
                    # the following is taken from LAPACK dlahqr
                    s = abs(h33) + abs(h34) + abs(h43) + abs(h44)
                    if s == 0
                        rt1r = zero(T)
                        rt2r = zero(T)
                        rt1i = zero(T)
                        rt2i = zero(T)
                        verbosity[] > 0 && println("zero shift")
                    else
                        verbosity[] > 1 && println("computing shift from $h33 $h44 $h43h34")
                        h33 /= s
                        h44 /= s
                        h34 /= s
                        h43 /= s
                        trc = (h33 + h44) * T(0.5)
                        disc = (h33 - trc) * (h44 - trc) - h34 * h43
                        rtdisc = sqrt(abs(disc))
                        verbosity[] > 1 && println("trc=$trc disc=$disc")
                        if disc >= 0
                            rt1r = trc * s
                            rt2r = rt1r
                            rt1i = rtdisc * s
                            rt2i = -rt1i
                            verbosity[] > 0 && println("shifts $rt1r ± $rt1i im")
                        else
                            rt1r = trc + rtdisc
                            rt2r = trc - rtdisc
                            rt1r = (abs(rt1r - h44) <= abs(rt2r - h44)) ? (rt1r*s) : (rt2r*s)
                            rt2r = rt1r
                            rt1i = rt2i = zero(T)
                            verbosity[] > 0 && println("shift $rt1r (double)")
                        end
                    end
                end
            end

            # look for two consecutive small subdiagonals and construct bulge reflector
            # Note: Fortran loop has break for M = L so no final decrement
            mmax = _allow_early_QR[] ? (i - 2) : l
            mlast = mmax # where to start QR
            for m in mmax:-1:l
                mlast = m
                # determine the effect of starting double-shift QR iteration
                # at row m; see if this would make ℍ[m,m-1] negligible
                h11 = hdiag[m]
                h12 = hsupdiag[n - i + m]
                h21 = hsubdiag[m + 1]
                h22 = hdiag[m + 1]
                if _slicot_shifts[] || exc_shift
                    h44s = h44 - h11
                    h33s = h33 - h11
                    v1 = (h33s * h44s - h43h34) / h21 + h12
                    v2 = h22 - h11 - h33s - h44s
                    v3 = hsubdiag[m + 2]
                else
                    s = abs(h11 - rt2r) + abs(rt2i) + abs(h21)
                    h21s = h21 / s
                    v1 = h21s * h12 + (h11 - rt1r) * ((h11 - rt2r) / s) - rt1i * (rt2i / s)
                    v2 = h21s * (h11 + h22 - rt1r - rt2r)
                    v3 = h21s * hsubdiag[m + 2]
                end
                s = abs(v1) + abs(v2) + abs(v3)
                v1 = v1 / s
                v2 = v2 / s
                v3 = v3 / s
                v[1:3] .= (v1, v2, v3)
                if m > l
                    tst1 = abs(v1) * (abs(hdiag[m - 1]) + abs(h11) + abs(h22))
                    if abs(hsubdiag[m]) * (abs(v2) + abs(v3)) <= ulp * tst1
                        verbosity[] > 1 && println("early QR start m=$m")
                        break
                    end
                end
            end # m loop

            # double-shift QR
            for k in mlast:(i - 1)
                # first iteration: determine reflection from vector v
                # apply symmetrically to ℍ, creating a bulge.
                # subsequently, determine reflections to restore Hessenberg form
                # in column k-1, chasing bulge towards the bottome of active submatrix

                nr = min(3, i - k + 1) # order of reflection
                nrow = min(k + nr, i) - i1 + 1
                if k > mlast
                    v[1:nr] .= H1[k:(k + nr - 1), k - 1]
                    # otherwise use v from above
                end
                ξ = view(v, 1:nr)
                τ1 = _reflector!(ξ)
                hr = Householder{T, typeof(ξ)}(view(ξ, 2:nr), τ1)
                if k > mlast
                    H1[k, k - 1] = v[1]
                    H1[k + 1, k - 1] = zero(T)
                    if k < i - 1
                        H1[k + 2, k - 1] = zero(T)
                    end
                elseif mlast > l
                    # Note: LAPACK uses τ1 to protect against underflow here and below,
                    # but I don't do that here (yet) pending study
                    # (some Julia reflectors have a different convention for τ).
                    H1[k, k - 1] = -H1[k, k - 1]
                end

                lmul!(hr', view(H1, k:(k + nr - 1), k:i2))
                str = (k == mlast) ? "bulge" : "chase"
                verbosity[] > 2 && showmat("L $str k=$k", 1)
                rmul!(view(Hp, i1:(i1 + nrow - 1), k:(k + nr - 1)), hr)
                verbosity[] > 2 && (p == 1) && showmat("R k=$k", p)
                verbosity[] > 2 && showprod("after exterior")
                if wantZ
                    rmul!(view(Z[1], 1:n, k:(k + nr - 1)), hr)
                end
                # @_dbg_rpschur fcheck("in QR $k", H1, Hs, Z; check_A1 = true)
                for j in p:-1:2
                    # transform nr by nr submatrix of H[j] at [k,k] to upper triangular
                    Hj = Hs[j - 1]
                    v[1:nr] .= Hj[k:(k + nr - 1), k]
                    ξ = view(v, 1:nr)
                    t = _reflector!(ξ)
                    hr = Householder{T, typeof(ξ)}(view(ξ, 2:nr), t)
                    Hj[k:(k + 1), k] .= (v[1], zero(T))
                    if nr == 3
                        Hj[k + 2, k] = zero(T)
                    end
                    lmul!(hr', view(Hj, k:(k + nr - 1), (k + 1):i2))
                    # and columns of H[j-1], Z
                    if j == 2
                        rmul!(view(H1, i1:(i1 + nrow - 1), k:(k + nr - 1)), hr)
                    else
                        rmul!(view(Hs[j - 2], i1:(i1 + nrow - 1), k:(k + nr - 1)), hr)
                    end
                    if wantZ
                        rmul!(view(Z[j], :, k:(k + nr - 1)), hr)
                    end
                    if nr == 3
                        v[1:2] .= Hj[(k + 1):(k + 2), k + 1]
                        ξ = view(v, 1:2)
                        t = _reflector!(ξ)
                        hr = Householder{T, typeof(ξ)}(view(ξ, 2:2), t)
                        Hj[(k + 1):(k + 2), k + 1] .= (v[1], zero(T))
                        lmul!(hr', view(Hj, (k + 1):(k + 2), (k + 2):i2))
                        # and columns of H[j-1], Z
                        if j == 2
                            rmul!(view(H1, i1:(i1 + nrow - 1), (k + 1):(k + 2)), hr)
                        else
                            rmul!(view(Hs[j - 2], i1:(i1 + nrow - 1), (k + 1):(k + 2)), hr)
                        end
                        if wantZ
                            rmul!(view(Z[j], 1:n, (k + 1):(k + 2)), hr)
                        end
                    end
                    verbosity[] > (j == 1 ? 2 : 3) && showmat("T k=$k", j)
                end # j loop (statement 140)
                @_dbg_rpschur fcheck("after QR $k", H1, Hs, Z)
                verbosity[] > 2 && showprod("k=$k")
            end # k loop (double shift QR)
            its += 1
        end # QR iteration loop (statement 160)

        # either QR iteration failed to converge, or we broke for deflation
        if !splitting
            throw(ErrorException("convergence failed at level $i"))
        end

        # deflation stage, ll. 775-939 of MB03WD.f
        if l == i
            # H₁[i,i-1] is negligible; one eigval has converged
            λ[i] = hdiag[i]
            verbosity[] > 0 && println("deflating one eigval i=$i, $(hdiag[i])")
        elseif l == i - 1
            # H₁[i-1,i-2] is negligible; a pair of eigvals have converged
            # Transform 2x2 submatrix of ℍ at [i-1,i-1] to standard Schur form.
            # Store eigvals.
            # If T matrices are not required, just use a similar submatrix from above.
            # If eigvals are real, use Givens rotations to triangularize.

            if wantT
                hp22 = one(T)
                hp12 = zero(T)
                hp11 = one(T)
                for j in 2:p
                    Hj = Hs[j - 1]
                    hp22 *= Hj[i, i]
                    hp12 = hp11 * Hj[i - 1, i] + hp12 * Hj[i, i]
                    hp11 *= Hj[i - 1, i - 1]
                end
                hh21 = H1[i, i - 1] * hp11
                hh22 = H1[i, i - 1] * hp12 + H1[i, i] * hp22
                hh11 = H1[i - 1, i - 1] * hp11
                hh12 = H1[i - 1, i - 1] * hp12 + H1[i - 1, i] * hp22
            else
                hh11 = hdiag[i - 1]
                hh12 = hsupdiag[n - 1]
                hh21 = hsubdiag[i]
                hh22 = hdiag[i]
            end # if wantT

            h2 = [hh11 hh12; hh21 hh22]
            verbosity[] > 1 && println("split block: $h2")
            G, λ[i - 1], λ[i] = _gs2x2!(h2, 2)
            verbosity[] > 0 && println("deflating two eigvals $(λ[i-1]), $(λ[i])")
            # just for consistency
            hdiag[(i - 1):i] .= real.(λ[(i - 1):i])
            hsubdiag[(i - 1):i] .= imag.(λ[(i - 1):i])
            if wantT
                # detect negligible diagonal elements  i-1 and i in Hⱼ for j>1
                jmin = 0
                jmax = 0
                for j in 2:p
                    Hj = Hs[j - 1]
                    if jmin == 0
                        if abs(Hj[i - 1, i - 1]) <= hnorms[j] # dw[n+j-2]
                            jmin = j
                        end
                    end
                    if abs(Hj[i, i]) <= hnorms[j] # dw[n+j-2]
                        jmax = j
                    end
                end
                # choose shorter path if both matched
                if jmin != 0 && jmax != 0
                    if jmin - 1 <= p - jmax + 1
                        jmax = 0
                    else
                        jmin = 0
                    end
                    verbosity[] > 0 && println("shortening path: $jmin $jmax")
                end
                if jmin != 0
                    # CHECKME: this section is rarely encountered, so not well tested
                    for j in 1:(jmin - 1)
                        if j == 1
                            Hj = H1
                        else
                            Hj = Hs[j - 1]
                        end
                        ξ = [Hj[i, i], Hj[i, i - 1]]
                        t = _reflector!(ξ)
                        hr = HH2(ξ[2], one(T), t)
                        Hj[i, (i - 1):i] .= (zero(T), ξ[2])
                        rmul!(view(Hj, i1:(i - 1), (i - 1):i), hr)
                        # and columns of H[j+1], Z
                        lmul!(hr', view(Hs[j], (i - 1):i, (i - 1):i2))
                        if wantZ
                            rmul!(view(Z[j + 1], 1:n, (i - 1):i), hr)
                        end
                    end
                else
                    # If two real eigvals and nontrivial chain, another rotation is needed
                    # otherwise use G computed above
                    replaceG = jmax > 0 && hsubdiag[i - 1] == 0
                    # If one eigval is tiny, the rotation from ℍ is also inadequate
                    # These cases are not handled in MB03WD
                    a1,a2 = abs.(λ[i-1:i])
                    if λ[i] * λ[i-1] == 0
                        replaceG = true
                    elseif hsubdiag[i - 1] == 0
                        # CHECKME: this suffices for cases seen so far, but is not
                        # supported by analysis
                        if min(a1,a2) / max(a1,a2) < eps(T)
                            replaceG = true
                        end
                    end
                    (verbosity[] > 1) && replaceG && println("nontrivial chain, reals")
                    # MB03WD only runs the 2x2 QZ once, but that can leave a (small)
                    # subdiagional in H1 so the factorization is poor after nulling.
                    for its2 in 1:20
                        if replaceG
                            G, _ = givens(H1[i - 1, i - 1], H1[i, i - 1], 1, 2)
                        end
                        # act on cols/rows i-1:i
                        lmul!(G, view(H1, (i - 1):i, (i - 1):i2))
                        rmul!(view(Hp, i1:i, (i - 1):i), G')
                        # rmul!(view(H1, :, i-1:i2), G')
                        # lmul!(G, view(H[p], i1:i, :))
                        if wantZ
                            rmul!(view(Z[1], :, (i - 1):i), G')
                        end
                        for j in p:-1:max(2, jmax + 1)
                            Hj = Hs[j - 1]
                            # reflector to annihilate Hⱼ[i,i-1] from the left
                            v[1:2] .= (Hj[i - 1, i - 1], Hj[i, i - 1])
                            ξ = view(v, 1:2)
                            t = _reflector!(ξ)
                            hr = Householder{T, typeof(ξ)}(view(ξ, 2:2), t)
                            Hj[(i - 1):i, i - 1] .= (v[1], zero(T))
                            lmul!(hr', view(Hj, (i - 1):i, i:i2))
                            # and transform columns of Hⱼ₋₁, Z
                            Hjm1 = j == 2 ? H1 : Hs[j - 2]
                            rmul!(view(Hjm1, i1:i, (i - 1):i), hr)
                            if wantZ
                                rmul!(view(Z[j], :, (i - 1):i), hr)
                            end
                        end
                        # println("new H1 block: ",H1[i-1:i,i-1:i])
                        if !replaceG || (abs(H1[i, i - 1]) < max(smlnum, ulp * max(a1, a2)))
                            break
                        end
                        replaceG = true
                    end
                    if jmax > 0
                        H1[i, i - 1] = zero(T)
                        if jmax > 1
                            Hs[jmax - 1][i, i - 1] = zero(T)
                        end
                    elseif hh21 == 0
                        H1[i, i - 1] = zero(T)
                    end
                    if replaceG
                        # sometimes the rotation swaps the eigvals
                        λ1 = H1[i - 1, i - 1]
                        λ2 = H1[i, i]
                        for j in 1:(p - 1)
                            λ1 *= Hs[j][i - 1, i - 1]
                            λ2 *= Hs[j][i, i]
                        end
                        # but the previous computation was likely more accurate
                        if abs(λ1 - λ[1]) > abs(λ1 - λ[2])
                            λ[i - 1], λ[i] = λ[i], λ[i-1]
                        end
                    end
                end # two real eigvals branch
            end # if wantT
        end # double deflation branch
        ###
        # decrement iteration limit and advance to next i
        maxitleft -= its
        i = l - 1
        maxits = max(maxits, its)
        niter += its
    end # while, main loop
    @_dbg_rpschur fcheck("after main loop", H1, Hs, Z, check_A1=true)
    # next block is not in MB02WD, but needed to clear out dust.
    # I probably applied a reflector or rotation to an extra row somewhere
    # giving roundoff instead of 0
    for i in 1:(n - 1)
        if isreal(λ[i])
            if verbosity[] > 1 && (H1[i + 1, i] != 0)
                println("forcing subdiag $i of H1 from $(H1[i + 1, i]) to 0")
            end
            H1[i + 1, i] = 0
        end
    end
    if !wantZ
        Z = [similar(H1, 0, 0)]
    end
    verbosity[] > 0 && println("total QR iterations: $niter largest was $maxits")
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
        return PeriodicSchur(H1, Hr, Zr, λ, 'L', p)
    else
        return PeriodicSchur(H1, Hs, Z, λ)
    end
end

include("rschur2x2.jl")

include("generalized.jl")
include("rgeneralized.jl")
include("utils.jl")
include("babd.jl")
include("ordschur.jl")

function pschur!(A::AbstractVector{TA},
                 lr::Symbol = :R;
                 kwargs...) where {TA <: AbstractMatrix{T}} where {T <: Complex}
    gps = pschur!(A::AbstractVector{TA}, trues(length(A)), lr; kwargs...)
    PeriodicSchur(gps.T1, gps.T, gps.Z, gps.values, gps.orientation, gps.schurindex)
end

include("krylov.jl")

include("vectors.jl")

end # module

# Local variables:
# fill-column: 84
# End:
