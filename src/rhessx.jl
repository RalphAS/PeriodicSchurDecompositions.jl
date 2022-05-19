# This file implements a rowwise periodic Hessenberg reduction for left orientation,
# which is needed for periodic Krylov-Schur algorithms (to preserve the Arnoldi foot).

# rowwise Householder, mapping leading j-1 entries of row vector x to 0
# via rmul!(view(x,1:1,1:j),H')
# when constructed from reflector!(conj.(reverse(x[1:j])))
struct RHouseholder{T,S<:StridedVector}
    v::S
    τ::T
end

Base.size(H::RHouseholder) = (length(H.v)+1, length(H.v)+1)
Base.size(H::RHouseholder, i::Integer) = i <= 2 ? length(H.v)+1 : 1

Base.eltype(H::RHouseholder{T})      where T = T

LinearAlgebra.adjoint(H::RHouseholder{T}) where {T} = Adjoint{T,typeof(H)}(H)

function LinearAlgebra.lmul!(H::RHouseholder, A::StridedMatrix)
    m, n = size(A)
    nh = size(H,1)
    nh == m || throw(DimensionMismatch("reflector length $nh; rows $m"))
    v = view(H.v, 1:m - 1)
    τc = H.τ'
    for j = 1:n
        va = A[m,j]
        Aj = view(A, 1:m-1, j)
        va += dot(v, Aj)
        va = τc*va
        A[m,j] -= va
        axpy!(-va, v, Aj)
    end
    A
end

function LinearAlgebra.rmul!(A::StridedMatrix, adjH::Adjoint{<:Any,<:RHouseholder})
    H = parent(adjH)
    m, n = size(A)
    nh = size(H,1)
    nh == n || throw(DimensionMismatch("columns $n; reflector length $nh"))
    v = view(H.v, :)
    τ = H.τ
    a1 = view(A, :, n)
    A1 = view(A, :, 1:n-1)
    x = A1*v
    axpy!(one(τ), a1, x)
    axpy!(-τ, x, a1)
    rankUpdate!(-τ, x, v, A1)
    A
end

# rowwise periodic Hessenberg reduction for left orientation, after Kressner
function _rphessenberg!(Ap::AbstractMatrix{T}, A::AbstractVector{S}, Q::AbstractVector{Sq}
                        ) where {S<:AbstractMatrix{T}, Sq<:AbstractMatrix{T}} where {T}
    p = length(A)+1
    #n = checksquare(Ap)
    m,n = size(Ap)
    m in (n,n+1) || throw(ArgumentError("only implemented for square or 1 extra row"))
    require_one_based_indexing(Ap)
    for j in 2:p
        require_one_based_indexing(A[j-1])
        if checksquare(A[j-1]) != n
            throw(DimensionMismatch())
        end
    end
    if m == n+1
        i = n+1
        i1=i-1
        ξ = conj.(Ap[i,i1:-1:1])
        t = reflector!(ξ)
        ξr = ξ[i1:-1:2]
        H = RHouseholder{T,typeof(ξr)}(ξr, t)
        # lmul!(H, view(A[p-1], 1:i1, 1:i1))
        lmul!(H, view(A[p-1], 1:i1, :))
        rmul!(view(Ap, :, 1:i1), H')
        rmul!(view(Q[p], :, 1:i1), H')
    end
    for i in n:-1:2
        i1=i-1
        # i2 = max(i-2,1)
        for l in p-1:-1:1
            ξ = conj.(A[l][i,i:-1:1])
            t = reflector!(ξ)
            ξr = ξ[i:-1:2]
            H = RHouseholder{T,typeof(ξr)}(ξr, t)
            Al1 = l==1 ? Ap : A[l-1]
            # lmul!(H, view(Al1, 1:i, 1:i))
            lmul!(H, view(Al1, 1:i, :))
            rmul!(view(A[l], :, 1:i), H')
            rmul!(view(Q[l], :, 1:i), H')
        end
        ξ = conj.(Ap[i,i1:-1:1])
        t = reflector!(ξ)
        ξr = ξ[i1:-1:2]
        H = RHouseholder{T,typeof(ξr)}(ξr, t)
        lmul!(H, view(A[p-1], 1:i1, :))
        # lmul!(H, view(A[p-1], 1:i1, 1:i1))
        rmul!(view(Ap, :, 1:i1), H')
        rmul!(view(Q[p], :, 1:i1), H')
    end
    triu!(Ap,-1)
    for l in 1:p-1
        triu!(A[l])
    end
    return Ap, A
end
