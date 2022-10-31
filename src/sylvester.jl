# periodic Sylvester equations, just for use in Schur code

# TODO:
# @inbounds
# compute scaling factors to avoid under/overflow
# replace the lot with sparse BABD solvers

# construct the cyclic Kronecker form for the periodic Sylvester operator
#    represented by the two series of matrices A,B
# typically A=A₁₁, B=A₂₂ are consecutive diagonal blocks
function _psyl2kron(A::Vector{MT1},
                    B::Vector{MT2}) where {MT1 <: AbstractMatrix{T},
                                           MT2 <: AbstractMatrix{T}} where {T}
    K = length(A)
    p1 = checksquare(A[1])
    p2 = checksquare(B[1])
    pp = p1 * p2
    n = p1 * p2 * K
    Zpse = zeros(T, n, n)
    eye1 = I(p1)
    eye2 = I(p2)
    Zpse[1:pp, 1:pp] .= kron(transpose(B[K]), -eye1)
    j1 = (K - 1) * pp
    Zpse[1:pp, (j1 + 1):(j1 + pp)] .= kron(eye2, A[K])
    for k in 1:(K - 1)
        i0 = k * pp
        j1 = (k - 1) * pp
        Zpse[(i0 + 1):(i0 + pp), (i0 + 1):(i0 + pp)] .= kron(transpose(B[k]), -eye1)
        Zpse[(i0 + 1):(i0 + pp), (j1 + 1):(j1 + pp)] .= kron(eye2, A[k])
    end
    return Zpse
end

# collect the blocks for the sparse (cyclic block bidiagonal)
# Kronecker representation of a periodic Sylvester operator
function _psyl2spkron(A::Vector{MT1},
                    B::Vector{MT2}) where {MT1 <: AbstractMatrix{T},
                                           MT2 <: AbstractMatrix{T}} where {T}
    K = length(A)
    p1 = checksquare(A[1])
    p2 = checksquare(B[1])
    eye1 = I(p1)
    eye2 = I(p2)
    Zd = [kron(transpose(B[K]), -eye1)]
    Zl = [kron(eye2, A[1])]
    for k in 1:(K - 1)
        push!(Zd, kron(transpose(B[k]), -eye1))
        push!(Zl, kron(eye2, A[k + 1]))
    end
    return Zd, Zl
end

function _pgsyl2kron(A::Vector{MT1},
                     B::Vector{MT2},
                     S::AbstractVector{Bool}) where {MT1 <: AbstractMatrix{T},
                                           MT2 <: AbstractMatrix{T}} where {T}
    K = length(A)
    p1 = checksquare(A[1])
    p2 = checksquare(B[1])
    pp = p1 * p2
    n = p1 * p2 * K
    Zpse = zeros(T, n, n)
    eye1 = I(p1)
    eye2 = I(p2)
    j1 = (K - 1) * pp
    if S[K]
        Zpse[1:pp, 1:pp] .= kron(transpose(B[K]), -eye1)
        Zpse[1:pp, (j1 + 1):(j1 + pp)] .= kron(eye2, A[K])
    else
        Zpse[1:pp, 1:pp] .= kron(eye2, A[K])
        Zpse[1:pp, (j1 + 1):(j1 + pp)] .= kron(transpose(B[K]), -eye1)
    end
    for k in 1:(K - 1)
        i0 = k * pp
        j1 = (k - 1) * pp
        if S[k]
            Zpse[(i0 + 1):(i0 + pp), (i0 + 1):(i0 + pp)] .= kron(transpose(B[k]), -eye1)
            Zpse[(i0 + 1):(i0 + pp), (j1 + 1):(j1 + pp)] .= kron(eye2, A[k])
        else
            Zpse[(i0 + 1):(i0 + pp), (i0 + 1):(i0 + pp)] .= kron(eye2, A[k])
            Zpse[(i0 + 1):(i0 + pp), (j1 + 1):(j1 + pp)] .= kron(transpose(B[k]), -eye1)
        end
    end
    return Zpse
end

# collect the blocks for the sparse (cyclic block bidiagonal)
# Kronecker representation of a periodic generalized Sylvester operator
function _pgsyl2spkron(A::Vector{MT1},
                     B::Vector{MT2},
                     S::AbstractVector{Bool}) where {MT1 <: AbstractMatrix{T},
                                           MT2 <: AbstractMatrix{T}} where {T}
    K = length(A)
    p1 = checksquare(A[1])
    p2 = checksquare(B[1])
    eye1 = I(p1)
    eye2 = I(p2)
    if S[K]
        Zd = [kron(transpose(B[K]), -eye1)]
    else
        Zd = [kron(eye2, A[K])]
    end
    if S[1]
        Zl = [kron(eye2, A[1])]
    else
        Zl = [kron(transpose(B[1]), -eye1)]
    end
    for k in 1:(K - 1)
        if S[k]
            push!(Zd, kron(transpose(B[k]), -eye1))
        else
            push!(Zd, kron(eye2, A[k]))
        end
        if S[k + 1]
            push!(Zl, kron(eye2, A[k + 1]))
        else
            push!(Zl, kron(transpose(B[k + 1]), -eye1))
        end
    end
    return Zd, Zl
end

# simple version for 1x1
function _psyl1rep(A::Vector{T}, B::Vector{T}) where {T}
    K = length(A)
    n = K
    Zpse = zeros(T, n, n)
    Zpse[1, 1] = -B[K] # transpose
    j1 = (K - 1)
    Zpse[1, j1 + 1] = A[K]
    for k in 1:(K - 1)
        i0 = k
        j1 = (k - 1)
        Zpse[i0 + 1, i0 + 1] = -B[k] # transpose
        Zpse[i0 + 1, j1 + 1] = A[k]
    end
    return Zpse
end

# generalized version for 1x1
function _pgsyl1rep(A::Vector{T}, B::Vector{T}, S::AbstractVector{Bool}) where {T}
    K = length(A)
    n = K
    Zpse = zeros(T, n, n)
    j1 = (K - 1)
    if S[K]
        Zpse[1, 1] = -B[K] # transpose
        Zpse[1, j1 + 1] = A[K]
    else
        Zpse[1, 1] = A[K]
        Zpse[1, j1 + 1] = -B[K]
    end
    for k in 1:(K - 1)
        i0 = k
        j1 = (k - 1)
        if S[k]
            Zpse[i0 + 1, i0 + 1] = -B[k] # transpose
            Zpse[i0 + 1, j1 + 1] = A[k]
        else
            Zpse[i0 + 1, i0 + 1] = A[k]
            Zpse[i0 + 1, j1 + 1] = -B[k]
        end
    end
    return Zpse
end

# solve a periodic Sylvester equation
# `AᵢXᵢ - Xᵢ₊₁Bᵢ = - Cᵢ`
# returns the Xⱼ in vector form
function _psylsolve(A::Vector{MT1},
                    B::Vector{MT2},
                    C::Vector{MTx}) where {MT1 <: AbstractMatrix{T},
                                           MT2 <: AbstractMatrix{T},
                                           MTx <: AbstractMatrix{T}} where {T}
    K = length(A)
    p1 = checksquare(A[1])
    p2 = checksquare(B[1])
    pp = p1 * p2
    scale = one(real(T))
    Zd, Zl = _psyl2spkron(A, B)
    Cv = zeros(T, pp, K)
    Cv[:, 1] .= -C[K][:]
    for k in 1:(K - 1)
        Cv[:, k + 1] .= -C[k][:]
    end
    y = vec(Cv)
    R, Zu, Zr, _ = _babd_qr!(Zd, Zl, y)
    for r in R
        _checkqr(r)
    end
    Xv = _babd_solve!(R, Zu, Zr, y)
    return Xv, scale
end

# simple version for 1x1
function _psylsolve1(A::Vector{T}, B::Vector{T}, C::Vector{T}) where {T}
    K = length(A)
    scale = one(real(T))
    Zpse = _psyl1rep(A, B)
    F = qr!(Zpse)
    _checkqr(F.R)
    Cr = -circshift(C, 1)
    Xv = F \ Cr
    return Xv, scale
end

function _pgsylsolve(A::Vector{MT1},
                     B::Vector{MT2},
                     C::Vector{MTx},
                     S::AbstractVector{Bool}
                     ) where {MT1 <: AbstractMatrix{T},
                              MT2 <: AbstractMatrix{T},
                              MTx <: AbstractMatrix{T}} where {T}
    K = length(A)
    p1 = checksquare(A[1])
    p2 = checksquare(B[1])
    pp = p1 * p2
    scale = one(real(T))
    Zd, Zl = _pgsyl2spkron(A, B, S)
    Cv = zeros(T, pp, K)
    Cv[:, 1] .= -C[K][:]
    for k in 1:(K - 1)
        Cv[:, k + 1] .= -C[k][:]
    end
    y = vec(Cv)
    R, Zu, Zr, _ = _babd_qr!(Zd, Zl, y)
    for r in R
        _checkqr(r)
    end
    Xv = _babd_solve!(R, Zu, Zr, y)
    return Xv, scale
end

# generalized 1x1
function _pgsylsolve1(A::Vector{T}, B::Vector{T}, C::Vector{T},
                     S::AbstractVector{Bool}) where {T}
    K = length(A)
    scale = one(real(T))
    Zpse = _pgsyl1rep(A, B, S)
    F = qr!(Zpse)
    _checkqr(F.R)
    Cr = -circshift(C, 1)
    Xv = F \ Cr
    return Xv, scale
end
