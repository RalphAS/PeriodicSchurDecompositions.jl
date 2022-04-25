
const _r2verby = Ref(0)

# development hack for independently checking the wrapup logic
const _punting = Ref(false)

function _rpeigvals2x2( A1::AbstractMatrix{T},
                        As::AbstractVector{TA},
                        S::AbstractVector{Bool},
                        Aord::AbstractVector{<:Integer},
                        schurindex::Integer;
                        recip = false
                        ) where {TA <: AbstractMatrix{T}} where {T <: Real}
    Tc = complex(T)
    k = length(As) + 1
    Ax = schurindex == 1 ? A1 : As[Aord[1]]
    Xs = [Matrix{Tc}(undef,2,2) for _ in 1:k]
    copyto!(Xs[1], Ax)
    for l in 2:k
        if l == schurindex
            Ax = A1
        else
            il = l < schurindex ? l : l-1
            Ax = As[Aord[il]]
        end
        copyto!(Xs[l], Ax)
    end

    # diagnostic routines
    function showmat(str, j)
        print(str," X[$j] "); show(stdout, "text/plain", Xs[j]); println()
        nothing
    end
    function showallmats(str)
        for l in 1:k
            showmat(str, l)
        end
        nothing
    end
    function showprod(str)
            Htmp = copy(Xs[1])
            for l in 2:k
                Htmp = Htmp * Xs[l]
            end
        if _r2verby[] > 1 && k > 1
            print(str," ℍ "); show(stdout, "text/plain", Htmp); println()
            println("  ev: ",eigvals(Htmp)')
        else
            println(str," ev(ℍ): ",eigvals(Htmp)')
        end
        nothing
    end

    maxiter = 80
    ulp = eps(T)
  if punting[]
    # punt!
    Xc1 = Xs[1]
    Xcs = [Xs[l] for l in 2:k]
    ps = pschur!(Xc1, Xcs)
    copyto!(Xs[1],ps.T1)
    for l in 2:k
        copyto!(Xs[l],ps.T[l-1])
    end
    converged = true
    _r2verby[] > 0 && showprod("after pschur")
  else
    X1 = Xs[1]
    converged = false
    _r2verby[] > 0 && showprod("before PQR")
    for iter in 1:maxiter
        # test for deflation
        lhs = abs(X1[2,1])
        rhs = max(abs(X1[1,1]), abs(X1[2,2]))
        if rhs == 0
            rhs = abs(X1[1,2])
        end
        if lhs <= ulp * rhs
            converged = true
            break
        end
        if iter == 1
            # "random" start
            c,s,r = givensAlgorithm(one(T)-im*T(2),T(2)+im*T(2))
        elseif mod(iter, 40) == 0
            # ad hoc shift
            c,s,r = givensAlgorithm(T(k)+im, one(T)-im*T(2))
        else
            # QR shift
            c,s = one(Tc), zero(Tc)
            ct,st,r = givensAlgorithm(one(Tc),one(Tc))
            for l = k:-1:2
                Xl = Xs[l]
                z11 = Xl[1,1]
                z21 = Xl[2,1]
                z12 = Xl[1,2]
                z22 = Xl[2,2]
                z0 = zero(Tc)
                Z = [z11 z0 z0;
                     z0 z11 z12;
                     z0 z21 z22]
                if S[Aord[l]] != recip
                    G1 = Givens(1,3,ct,st)
                    rmul!(Z,G1')
                    G2 = Givens(1,2,complex(c),s)
                    rmul!(Z,G2')
                    ct,st,r = givensAlgorithm(Z[1,1],Z[3,1])
                    c,s,r = givensAlgorithm(z11, Z[2,1])
                else
                    G1 = Givens(1,3,complex(ct),st)
                    lmul!(G1,Z)
                    G2 = Givens(1,2,complex(c),s)
                    lmul!(G2,Z)
                    ct,st,r = givensAlgorithm(Z[3,3],Z[3,1])
                    Z[3,3] = r
                    st = -st
                    G = Givens(1,3,complex(ct),st)
                    rmul!(view(Z,1:2,:), G')
                    c,s,r = givensAlgorithm(Z[2,2], Z[2,1])
                    Z[2,2] = r
                    s = -s
                end
            end
            l = 1
            Xl = Xs[l]
            z11 = Xl[1,1]
            z21 = Xl[2,1]
            z12 = Xl[1,2]
            z22 = Xl[2,2]
            z0 = zero(Tc)
            Z = [z11 -z21 -z22;
                 z21 z0 z0]
            G = Givens(1,3,complex(ct),st)
            rmul!(Z, G')
            G = Givens(1,2,complex(c),s)
            rmul!(Z, G')
            c,s,r = givensAlgorithm(Z[1,1], Z[2,1])
        end
        _r2verby[] > 1 && println("shift c,s: $c, $s")
        ct,st = c,s
        Y = zeros(Tc,2,2)
        for l=k:-1:2
            copyto!(Y,Xs[l])
            if S[Aord[l]] != recip
                G = Givens(1,2,complex(c),s)
                rmul!(Y,G')
                c,s,r = givensAlgorithm(Y[1,1],Y[2,1])
                Y[1,1] = r
                Y[2,1] = zero(Tc)
                G = Givens(1,2,complex(c),s)
                lmul!(G, view(Y,:,2:2))
            else
                G = Givens(1,2,c,s)
                lmul!(G, Y)
                c,s,r = givensAlgorithm(Y[2,2],Y[2,1])
                Y[2,2] = r
                Y[2,1] = zero(Tc)
                # MB03BB has
                s = -s
                G = Givens(1,2,complex(c),s)
                rmul!(view(Y,1:1,:),G')

                # G = Givens(2,1,complex(c),s')
                # rmul!(view(Y,1:1,:),G')
                # s = -s
            end
            copyto!(Xs[l], Y)
        end
        l = 1
        copyto!(Y,Xs[l])
        G = Givens(1,2,complex(ct),st)
        lmul!(G, Y)
        G = Givens(1,2,complex(c),s)
        rmul!(Y, G')
        # rmul!(view(Y,1:1,:), G')
        copyto!(Xs[l], Y)
        _r2verby[] > 0 && showprod("after iter $iter")
        _r2verby[] > 2 && showallmats("after iter $iter")
    end
  end

    beta = zeros(T,2)
    scal = zeros(T,2)
    alpha = zeros(Tc,2)
    for j in 1:2
        αj = one(Tc)
        beta[j] = one(T)
        scal[j] = zero(T)
        for l in 1:k
            z = Xs[l][j,j]
            rhs = abs(z)
            if rhs != 0
                sl = floor(Int,log2(rhs))
                z *= T(2)^(-sl)
            else
                sl = 0
            end
            if S[Aord[l]]
                αj *= z
                scal[j] += sl
            elseif rhs == 0
                beta[j] = zero(T)
            else
                αj /= z
                scal[j] -= sl
            end
            if (mod(l,10) == 0) || (l == k)
                rhs = abs(αj)
                if rhs == 0
                    scal[j] = 0
                else
                    sl = floor(Int,log2(rhs))
                    αj *= T(2)^(-sl)
                    scal[j] += sl
                end
            end
        end
        alpha[j] = αj
    end
    if imag(alpha[2]) > 0
        alpha[1],alpha[2] = alpha[2],alpha[1]
        beta[1],beta[2] = beta[2],beta[1]
        scal[1],scal[2] = scal[2],scal[1]
    end
    # enforce standard for eigvals of real matrices
    good = true
    if any(imag.(alpha) .!= 0)
        sl = scal[1] - scal[2]
        if sl >= 0
            zt1 = alpha[2] * T(2)^(-sl)
            zt2 = alpha[1] - conj(zt1)
            cst = imag(alpha[1])
        else
            zt1 = alpha[1] * T(2)^(sl)
            zt2 = alpha[2] - conj(zt1)
            cst = imag(alpha[2])
        end
        misr = hypot(cst, imag(zt1))
        misc = abs(zt2) / 2
        cs = max(abs(alpha[1]), one(T), abs(alpha[2]))
        good = min(misr, misc) <= cs*sqrt(ulp)
        if !good && (_r2verby[] > 0)
            @info "misr = $misr misc = $misc cs = $cs"
        end
        if misr > misc
            # conjugate pair
            j = (scal[1] >= scal[2]) ? 1 : 2
            αt = (alpha[j] + conj(zt1)) / 2
            sl = scal[j]
            αi = abs(imag(αt))
            alpha[1] = real(αt) + im*αi
            alpha[2] = conj(alpha[1])
        else
            for j in 1:2
                alpha[j] = real(alpha[j])
            end
        end
    end
    return alpha,beta,scal,converged,good
end
