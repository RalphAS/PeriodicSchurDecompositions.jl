# copied from Andreas Noack's GenericLinearAlgebra.jl, with trivial mods

import Base: *, eltype, size
import LinearAlgebra: adjoint, lmul!, rmul!, BlasReal

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
