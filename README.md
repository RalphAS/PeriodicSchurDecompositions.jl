# PeriodicSchurDecompositions

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
<!-- ![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg) -->
[![Build Status](https://github.com/RalphAS/PeriodicSchurDecompositions.jl/workflows/CI/badge.svg)](https://github.com/RalphAS/PeriodicSchurDecompositions.jl/actions)
[![Coverage](https://codecov.io/gh/RalphAS/PeriodicSchurDecompositions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/RalphAS/PeriodicSchurDecompositions.jl)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://RalphAS.github.io/PeriodicSchurDecompositions.jl/dev)

## Periodic Schur decomposition

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

This package provides a straightforward PSD for real and complex element types.

The basic API is as follows:
```julia
p = period_of_your_problem()
Aarg = [your_matrix(j) for j in 1:p]
pS = pschur!(Aarg, :R)
your_eigvals = pS.values
```
The result `pS` is a `PeriodicSchur` object. Computation of the Schur vectors is
fairly expensive, so it is an option set by keyword argument;
see the `pschur!` docstring for further details.

### Operator ordering
The `:R` argument above indicates that the product represented by `pS` is `prod(Aarg)`
i.e., counting rightwards. In many applications it is more convenient to number
the matrices leftwards (`A[p]*...*A[2]*A[1]`); this interpretation is available
with an orientation argument `:L`.


## Generalized Periodic Schur decomposition
Given a series of `NxN` matrices `A[j]`, `j=1...p`, and a signature vector
`S` where `S[j]` is `1` or `-1`, a generalized periodic Schur decomposition (GPSD)
is a factorization of the formal product `A[1]^(S[1]*A[2]^(S[2]*...*A[p]^(S[p])`:
`Q[j]' * A[j] * Q[j+1] = T[j]` if `S[j] == 1` and
`Q[j+1]' * A[j] * Q[j] = T[j]` if `S[j] == -1`.

The GPSD is an extension of the QZ decomposition used for generalized eigenvalue
problems.

This package currently provides a GPSD for complex element types. The PSD is obviously
a special case of the GPSD.

The basic API is as follows:
```julia
p = period_of_your_problem()
Aarg = [your_complex_matrix(j) for j in 1:p]
S = [sign_for_your_problem(j) for j in 1:p]
gpS = pschur!(Aarg, S, :R)
your_eigvals = gpS.values
```
The result `gpS` is a `GeneralizedPeriodicSchur` object
(see the docstring for further details; note that eigenvalues are stored in a
decomposed form in this case).

## Reordering eigenvalues

Selected eigenvalues and their associated subspace can be moved to the top of
a periodic Schur decomposition with `ordschur!` methods.

Currently reordering is only implemented for standard, not generalized, decompositions.

## Large problems: periodic Krylov-Schur

If only a few exterior eigenvalues (and corresponding Schur vectors) of a standard
periodic system are needed, it may be
more efficient to use the Krylov-Schur algorithm implemented as

```julia
    pps, hist = partial_pschur(Avec, nev, which; kw...)
```
where `Avec` is a vector of either matrices or linear maps.
The result is a `PartialPeriodicSchur` object `pps`, with a summary `hist` of the iteration.
`pps` usually includes the `nev` eigenvalues near the edge of the convex hull of the
spectrum specified by `which`.
The interface is derived from the `ArnoldiMethod` package,
q.v. for additional details.

Note: `partial_pschur` is currently **only implemented for the leftwards orientation**.

## Status
This is a new package (2022) implementing complicated algorithms, so caveat emptor.
Although tests to date indicate that the package is largely correct, there may be
surprises; if you find any please file issues.

Little effort has gone into optimization so far, and the package takes a while to compile.

The API in v0.1 should be considered tentative; constructive suggestions for changes are
welcome.


## References

A. Bojanczyk, G. Golub, and P. Van Dooren, "The periodic Schur decomposition.
Algorithms and applications," Proc. SPIE 1996.

D. Kressner, thesis and assorted articles.

## Acknowledgements

Many functions in this package are translations of implementations in [the SLICOT library](https://github.com/SLICOT/SLICOT-Reference.git).

Special thanks to Dr. A. Varga for making SLICOT available with a liberal license.

A few types and methods have been adapted from A. Noack's GenericLinearAlgebra package.
