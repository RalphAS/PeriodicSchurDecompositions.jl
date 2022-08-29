import Base: *, eltype, size
import LinearAlgebra: adjoint, lmul!, rmul!, BlasReal

# stdlib norm2 uses a poorly implemented generic scheme for short vectors.
function _norm2(x::AbstractVector{T}) where {T<:Real}
    require_one_based_indexing(x)
    n = length(x)
    n < 1 && return zero(T)
    n == 1 && return abs(x[1])
    scale = zero(T)
    ssq = zero(T)
    for xi in x
        if !iszero(xi)
            a = abs(xi)
            if scale < a
                ssq = one(T) + ssq * (scale / a)^2
                scale = a
            else
                ssq += (a / scale)^2
            end
        end
    end
    return scale * sqrt(ssq)
end

function _norm2(x::AbstractVector{T}) where {T<:Complex}
    require_one_based_indexing(x)
    n = length(x)
    RT = real(T)
    n < 1 && return zero(RT)
    n == 1 && return abs(x[1])
    scale = zero(RT)
    ssq = zero(RT)
    for xx in x
        xr,xi = reim(xx)
        if !iszero(xr)
            a = abs(xr)
            if scale < a
                ssq = one(RT) + ssq * (scale / a)^2
                scale = a
            else
                ssq += (a / scale)^2
            end
        end
        if !iszero(xi)
            a = abs(xi)
            if scale < a
                ssq = one(RT) + ssq * (scale / a)^2
                scale = a
            else
                ssq += (a / scale)^2
            end
        end
    end
    return scale * sqrt(ssq)
end

# The reflector! code in stdlib has no underflow or accuracy protection

# These are translations of xLARFG from LAPACK
# LAPACK Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
function _xreflector!(x::AbstractVector{T}) where {T<:Real}
    require_one_based_indexing(x)
    n = length(x)
    n <= 1 && return zero(T)
    sfmin = 2floatmin(T) / eps(T)
    @inbounds begin
        α = x[1]
        xnorm = _norm2(view(x,2:n))
        if iszero(xnorm)
            return zero(T)
        end
        β = -copysign(hypot(α, xnorm), α)
        kount = 0
        smallβ = abs(β) < sfmin
        if smallβ
            # recompute xnorm and β if needed for accuracy
            rsfmin = one(T) / sfmin
            while smallβ
                kount += 1
                for j in 2:n
                    x[j] *= rsfmin
                end
                β *= rsfmin
                α *= rsfmin
                # CHECKME: is 20 adequate for BigFloat?
                smallβ = (abs(β) < sfmin) && (kount < 20)
            end
            # now β ∈ [sfmin,1]
            xnorm = _norm2(view(x,2:n))
            β = -copysign(hypot(α, xnorm), α)
        end
        τ = (β - α) / β
        t = one(T) / (α - β)
        for j in 2:n
            x[j] *= t
        end
        for j in 1:kount
            β *= sfmin
        end
        x[1] = β
    end
    return τ
end

function _xreflector!(x::AbstractVector{T}) where {T<:Complex}
    require_one_based_indexing(x)
    n = length(x)
    # we need to make subdiagonals real so the n=1 case is nontrivial for complex eltype
    n < 1 && return zero(T)
    RT = real(T)
    sfmin = floatmin(RT) / eps(RT)
    @inbounds begin
        α = x[1]
        αr, αi = reim(α)
        xnorm = _norm2(view(x,2:n))
        if iszero(xnorm) && iszero(αi)
            return zero(T)
        end
        β = -copysign(_hypot3(αr, αi, xnorm), αr)
        kount = 0
        smallβ = abs(β) < sfmin
        if smallβ
            # recompute xnorm and β if needed for accuracy
            rsfmin = one(real(T)) / sfmin
            while smallβ
                kount += 1
                for j in 2:n
                    x[j] *= rsfmin
                end
                β *= rsfmin
                αr *= rsfmin
                αi *= rsfmin
                smallβ = (abs(β) < sfmin) && (kount < 20)
            end
            # now β ∈ [sfmin,1]
            xnorm = _norm2(view(x,2:n))
            α = complex(αr, αi)
            β = -copysign(_hypot3(αr, αi, xnorm), αr)
        end
        τ = complex((β - αr) / β, -αi / β)
        t = one(T) / (α - β)
        for j in 2:n
            x[j] *= t
        end
        for j in 1:kount
            β *= sfmin
        end
        x[1] = β
    end
    return τ
end

# As of v1.5, Julia hypot() w/ >2 args is unprotected (the documentation lies),
# so we need this.
# translation of dlapy3, assuming NaN propagation
function _hypot3(x::T, y::T, z::T) where {T}
    xa = abs(x)
    ya = abs(y)
    za = abs(z)
    w = max(xa, ya, za)
    rw = one(real(T)) / w
    r::real(T) = w * sqrt((rw * xa)^2 + (rw * ya)^2 + (rw * za)^2)
    return r
end

# copied from Andreas Noack's GenericLinearAlgebra.jl, with trivial mods

"""
a Householder reflection represented as the essential part of the
vector and the normalizing factor
"""
struct Householder{T, S <: StridedVector}
    v::S
    τ::T
end

# warning; size -> length(v) in GLA, but this makes more sense to us:
size(H::Householder) = (length(H.v) + 1, length(H.v) + 1)
size(H::Householder, i::Integer) = i <= 2 ? length(H.v) + 1 : 1

eltype(H::Householder{T}) where {T} = T

adjoint(H::Householder{T}) where {T} = Adjoint{T, typeof(H)}(H)

function lmul!(H::Householder, A::StridedMatrix)
    m, n = size(A)
    nh = size(H, 1)
    nh == m || throw(DimensionMismatch("reflector length $nh; rows $m"))
    v = view(H.v, 1:(m - 1))
    τ = H.τ
    for j in 1:n
        va = A[1, j]
        Aj = view(A, 2:m, j)
        va += dot(v, Aj)
        va = τ * va
        A[1, j] -= va
        axpy!(-va, v, Aj)
    end
    A
end

function rmul!(A::StridedMatrix, H::Householder)
    m, n = size(A)
    nh = size(H, 1)
    nh == n || throw(DimensionMismatch("columns $n; reflector length $nh"))
    v = view(H.v, :)
    τ = H.τ
    a1 = view(A, :, 1)
    A1 = view(A, :, 2:n)
    x = A1 * v
    axpy!(one(τ), a1, x)
    axpy!(-τ, x, a1)
    rankUpdate!(-τ, x, v, A1)
    A
end

function lmul!(adjH::Adjoint{<:Any, <:Householder}, A::StridedMatrix)
    H = parent(adjH)
    m, n = size(A)
    size(H, 1) == m || throw(DimensionMismatch("A: $m,$n H: $(size(H))"))
    v = view(H.v, 1:(m - 1))
    τ = H.τ
    for j in 1:n
        va = A[1, j]
        Aj = view(A, 2:m, j)
        va += dot(v, Aj)
        va = τ'va
        A[1, j] -= va
        axpy!(-va, v, Aj)
    end
    A
end

function Base.convert(::Type{Matrix}, H::Householder{T}) where {T}
    lmul!(H, Matrix{T}(I, size(H, 1), size(H, 1)))
end
function Base.convert(::Type{Matrix{T}}, H::Householder{T}) where {T}
    lmul!(H, Matrix{T}(I, size(H, 1), size(H, 1)))
end

# Rank one update

## General
### BLAS
function rankUpdate!(α::T, x::StridedVector{T}, y::StridedVector{T},
                     A::StridedMatrix{T}) where {T <: BlasReal}
    BLAS.ger!(α, x, y, A)
end

### Generic
function rankUpdate!(α::Number, x::StridedVector, y::StridedVector, A::StridedMatrix)
    m, n = size(A, 1), size(A, 2)
    m == length(x) || throw(DimensionMismatch("x vector has wrong length"))
    n == length(y) || throw(DimensionMismatch("y vector has wrong length"))
    for j in 1:n
        yjc = y[j]'
        for i in 1:m
            A[i, j] += x[i] * α * yjc
        end
    end
end

# 2x2 reflector, including the whole vector
struct HH2{T}
    v1::T
    v2::T
    τ::T
end
size(H::HH2) = (2, 2)
size(H::HH2, i::Integer) = i <= 2 ? 2 : 1

eltype(H::HH2{T}) where {T} = T

adjoint(H::HH2{T}) where {T} = Adjoint{T, typeof(H)}(H)

function rmul!(A::StridedMatrix, H::HH2)
    m, n = size(A)
    n == 2 || throw(DimensionMismatch(""))
    v1, v2 = H.v1, H.v2
    t = [v1; v2] * H.τ
    @inbounds for j in 1:m
        s = A[j, 1] * v1 + A[j, 2] * v2
        A[j, 1:2] .-= s * t
    end
    A
end

function lmul!(adjH::Adjoint{<:Any, <:HH2{T}}, A::StridedMatrix) where {T}
    H = parent(adjH)
    m, n = size(A)
    m == 2 || throw(DimensionMismatch("A: $m,$n H: $(size(H))"))
    v1, v2 = H.v1, H.v2
    t = H.τ' * [v1, v2]
    @inbounds for j in 1:n
        s = v1 * A[1, j] + v2 * A[2, j]
        A[1:2, j] .-= s * t
    end
    A
end
