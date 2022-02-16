```@meta
CurrentModule = PeriodicSchurDecompositions
```

# PeriodicSchurDecompositions.jl

This Julia package provides implementations of the periodic Schur decomposition
of matrix products of real element types and of the generalized periodic Schur
decomposition for complex element types.

## Definitions

### Periodic Schur decomposition

Given a series of `NxN` matrices `A[j]`, `j=1...p`, a periodic Schur decomposition (PSD)
is a factorization of the form:
```julia
Q[1]'*A[1]*Q[2] = T[1]
Q[2]'*A[2]*Q[3] = T[2]
...
Q[p]'*A[p]*Q[1] = T[p]
```
where the `Q[j]` are unitary (orthogonal) and the `T[j]` are upper triangular,
except that one of the `T[j]` is quasi-triangular for real element types.
It furnishes the eigenvalues and invariant subspaces of the matrix product
`prod(A)`.

The principal reason for using the PSD is that accuracy may be lost if one
forms the product of the `A_j` before eigen-analysis. For some applications the
intermediate Schur vectors are also useful.

### Generalized Periodic Schur decomposition
Given a series of `NxN` matrices `A[j]`, `j=1...p`, and a signature vector
`S` where `S[j]` is `1` or `-1`, a generalized periodic Schur decomposition (GPSD)
is a factorization of the formal product `A[1]^(S[1]*A[2]^(S[2]*...*A[p]^(S[p])`:
`Q[j]' * A[j] * Q[j+1] = T[j]` if `S[j] == 1` and
`Q[j+1]' * A[j] * Q[j] = T[j]` if `S[j] == -1`.

The GPSD is an extension of the QZ decomposition used for generalized eigenvalue
problems.

## References

A. Bojanczyk, G. Golub, and P. Van Dooren, "The periodic Schur decomposition.
Algorithms and applications," Proc. SPIE 1996.

D. Kressner, thesis and assorted articles.

## Acknowledgements

The meat of this package is mainly a translation of implementations in [the SLICOT library](https://github.com/SLICOT/SLICOT-Reference.git).
