# periodic Sylvester equations, just for use in Schur code

# TODO:
# @inbounds

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

# construct the cyclic Kronecker form for a generalized periodic Sylvester operator
# typically A=A₁₁, B=A₂₂,C=E₁₁, D=E₂₂ are consecutive diagonal blocks
# from the matrix product Π (Eⱼ⁻¹Aⱼ)
# ordering & indexing to resemble Granat et al. BIT 2007
function _pgsyl2kron(A::Vector{MT1}, B::Vector{MT2}, C::Vector{MT3},
                     D::Vector{MT4}) where {MT1 <: AbstractMatrix{T},
                                            MT2 <: AbstractMatrix{T},
                                            MT3 <: AbstractMatrix{T},
                                            MT4 <: AbstractMatrix{T}
                                            } where {T}
    K = length(A)
    p1 = checksquare(A[1])
    p2 = checksquare(B[1])
    # if this were public, we would check sizes
    pp = p1 * p2
    n = 2 * K * p1 * p2
    Zpse = zeros(T, n, n)
    eye1 = I(p1)
    eye2 = I(p2)
    Zpse[1:pp, 1:pp] .= kron(transpose(B[1]), -eye1)
    j1 = (2 * K - 1) * pp
    Zpse[1:pp, (j1 + 1):(j1 + pp)] .= kron(eye2, A[1])
    Zpse[(pp + 1):(2 * pp), 1:pp] .= kron(transpose(D[1]), -eye1)
    Zpse[(pp + 1):(2 * pp), (pp + 1):(2 * pp)] .= kron(eye2, C[1])
    for k in 1:(K - 1)
        i0 = 2 * k * pp
        j1 = i0 - pp
        i1 = (2 * k + 1) * pp
        j2 = i1 - pp
        Zpse[(i0 + 1):(i0 + pp), (i0 + 1):(i0 + pp)] .= kron(transpose(B[k]), -eye1) # diag
        Zpse[(i0 + 1):(i0 + pp), (j1 + 1):(j1 + pp)] .= kron(eye2, A[k]) # subdiag
        Zpse[(i1 + 1):(i1 + pp), (i1 + 1):(i1 + pp)] .= kron(eye2, C[k]) # diag
        Zpse[(i1 + 1):(i1 + pp), (j2 + 1):(j2 + pp)] .= kron(transpose(D[k]), -eye1) # subdiag
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
    # TODO: compute scaling factor to avoid under/overflow
    scale = one(real(T))
    Zpse = _psyl2kron(A, B)
    F = qr!(Zpse)
    _checkqr(F)
    Cv = zeros(T, pp, K)
    Cv[:, 1] .= -C[K][:]
    for k in 1:(K - 1)
        Cv[:, k + 1] .= -C[k][:]
    end
    Xv = F \ vec(Cv)
    return Xv, scale
end

# simple version for 1x1
function _psylsolve1(A::Vector{T}, B::Vector{T}, C::Vector{T}) where {T}
    K = length(A)
    # TODO: compute scaling factor to avoid under/overflow
    scale = one(real(T))
    Zpse = _psyl1rep(A, B)
    F = qr!(Zpse)
    _checkqr(F)
    Cr = -circshift(C, 1)
    Xv = F \ Cr
    return Xv, scale
end
