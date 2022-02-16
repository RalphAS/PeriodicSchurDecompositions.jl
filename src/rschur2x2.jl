# compute Schur decomposition of real 2x2 in standard form
# return corresponding Givens and eigenvalues
# Translated from LAPACK::dlanv2
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
function _gs2x2!(H2::StridedMatrix{T},jj) where {T <: Real}
    a,b,c,d = H2[1,1], H2[1,2], H2[2,1], H2[2,2]
    sgn(x) = (x < 0) ? -one(T) : one(T) # fortran sign differs from Julia
    half = one(T) / 2
    small = 4eps(T) # how big discriminant must be for easy reality check
    if c==0
        cs = one(T)
        sn = zero(T)
    elseif b==0
        # swap rows/cols
        cs = zero(T)
        sn = one(T)
        a,b,c,d = d,-c,zero(T),a
    elseif ((a-d) == 0) && (b*c < 0)
        # nothing to do
        cs = one(T)
        sn = zero(T)
    else
        asubd = a-d
        p = half*asubd
        bcmax = max(abs(b),abs(c))
        bcmis = min(abs(b),abs(c)) * sgn(b) * sgn(c)
        scale = max(abs(p), bcmax)
        z = (p / scale) * p + (bcmax / scale) * bcmis
        # if z is of order machine accuracy: postpone decision
        if z >= small
            # real eigenvalues
            z = p + sqrt(scale) * sqrt(z) * sgn(p)
            a = d + z
            d -= (bcmax / z) * bcmis
            τ = hypot(c,z)
            cs = z / τ
            sn = c / τ
            b -= c
            c = zero(T)
        else
            # complex or almost equal real eigenvalues
            σ = b + c
            τ = hypot(σ, asubd)
            cs = sqrt(half * (one(T) + abs(σ) / τ))
            sn = -(p / (τ * cs)) * sgn(σ)
            # apply rotations
            aa = a*cs + b*sn
            bb = -a*sn + b*cs
            cc = c*cs + d*sn
            dd = -c*sn + d*cs
            a = aa*cs + cc*sn
            b = bb*cs + dd*sn
            c = -aa*sn + cc*cs
            d = -bb*sn + dd*cs
            midad = half * (a+d)
            a = midad
            d = a
            if (c != 0)
                if (b != 0)
                    if b*c >= 0
                        # real eigenvalues
                        sab = sqrt(abs(b))
                        sac = sqrt(abs(c))
                        p = sab*sac*sgn(c)
                        τ = one(T) / sqrt(abs(b+c))
                        a = midad + p
                        d = midad - p
                        b -= c
                        c = 0
                        cs1 = sab*τ
                        sn1 = sac*τ
                        cs, sn = cs*cs1 - sn*sn1, cs*sn1 + sn*cs1
                    end
                else
                    b,c = -c,zero(T)
                    cs,sn = -sn,cs
                end
            end
        end
    end

    if c==0
        w1,w2 = a,d
    else
        rti = sqrt(abs(b))*sqrt(abs(c))
        w1 = a + rti*im
        w2 = d - rti*im
    end
    H2[1,1], H2[1,2], H2[2,1], H2[2,2] = a,b,c,d
    G = Givens(jj-1,jj,cs,sn)
    return G,w1,w2
end
