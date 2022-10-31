# sparse direct solver for cyclic problems arising in reordering
# of periodic Schur decompositions.

# | D₁ 0  0 ...   Lₖ |
# | L₁ D₂ 0 ...   0  |
# | 0  L₂ D₃ ...  0  |
# | 0  0  L₃ ...  0  |
# | ...           0  |
# | 0 ...    Lₖ₋₁ Dₖ |

# QR factorization of bordered-almost-block-diagonal matrix
# algorithm from Granat et al., BIT 2007
# Zd are the diagonal blocks; they are overwritten.
# Zl are the first subdiagonal, except Zl[K] is in upper right corner;
# they are preserved.
# Zd[j] and Zl[j] are in block column j.
function _babd_qr!(Zd, Zl::AbstractVector{TM}, y; wantQ=false
                  ) where {TM <: AbstractMatrix{T}} where T
    K = length(Zl)
    m = size(Zl[1], 1)
    Zu = [zero(Zl[1]) for _ in 1:K]
    Zr = [zero(Zl[1]) for _ in 1:K]
    Zr[K-1] = Zu[K-1] # intentional alias
    # R could overwrite Zd
    R = [zero(Zl[1]) for _ in 1:K]
    zs = zeros(T,2m, m)
    qz = zeros(T,2m, m)
    w = zeros(T,2m, m)
    yt = zeros(T,2m)
    Zr[1] .= Zl[K]
    if wantQ
        Q = Matrix{T}(I, K * m, K * m)
    else
        Q = nothing
    end
    i0 = 0
    for k in 1:K-1
        zs[1:m, 1:m] .= Zd[k]
        zs[m+1:2m, 1:m] .= Zl[k]
        q, r = qr!(zs)
        R[k] .= r
        w[1:m, 1:m] .= Zu[k]
        w[m+1:2m, 1:m] .= Zd[k+1]
        mul!(qz, q', w)
        if wantQ
            lmul!(q', view(Q, i0+1:i0+2m, :))
        end
        Zu[k] .= view(qz, 1:m, 1:m)
        Zd[k+1] .= view(qz, m+1:2m, 1:m)
        if k < K - 1
            w[1:m, 1:m] .= Zr[k]
            w[m+1:2m, 1:m] .= Zr[k+1]
            mul!(qz, q', w)
            Zr[k] .= view(qz, 1:m, 1:m)
            Zr[k+1] .= view(qz, m+1:2m, 1:m)
        end
        yt .= view(y, i0+1:i0+2m)
        mul!(view(y, i0+1:i0+2m), q', yt)
        i0 += m
    end
    q,r = qr!(Zd[K])
    R[K] .= r
    i0 = (K-1) * m
    if wantQ
        lmul!(q', view(Q, i0+1:i0+m, :))
    end
    yt = y[i0+1:i0+m]
    mul!(view(y, i0+1:i0+m), q', yt)
    return R, Zu, Zr, Q
end

function _babd_solve!(R::AbstractVector{TM}, Zu, Zr, y) where {TM <: AbstractMatrix{T}} where {T}
    K = length(R)
    m = size(R[1],1)
    x = zeros(T, length(y))
    i0 = (K-1) * m
    x[i0+1:i0+m] .= R[K] \ view(y, i0+1:i0+m)
    yt = zeros(T, m)
    i1 = i0 - m
    yt .= view(y, i1+1:i1+m) - Zu[K-1] * view(x, i0+1:i0+m)

    x[i1+1:i1+m] .= R[K-1] \ yt
    xk = view(x, (K-1) * m + 1:K * m)
    for i in 1:K-2
        i0 = (i-1) * m
        mul!(view(y, i0+1:i0+m), Zr[i], xk, -1, 1)
    end
    i0 = (K-2) * m
    for i in K-2:-1:1
        i1 = i0 - m
        yt .= view(y, i1+1:i1+m) - Zu[i] * view(x, i0+1:i0+m)
        x[i1+1:i1+m] .= R[i] \ yt
        i0 -= m
    end
    return x
end
