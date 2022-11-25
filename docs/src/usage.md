# Usage

For ordinary periodic Schur decompositions, the basic API ([`pschur`](@ref)) is as follows:

```julia
p = period_of_your_problem()
Aarg = [your_matrix(j) for j in 1:p]
pS = pschur!(Aarg, :R)
your_eigvals = pS.values
```
The result `pS` is a [`PeriodicSchur`](@ref) object.
Computation of the Schur vectors is
fairly expensive, so it may be suppressed via keyword arguments (`wantZ=false`).


For generalized PSD, the basic API is as follows:

```julia
p = period_of_your_problem()
Aarg = [your_complex_matrix(j) for j in 1:p]
S = [sign_for_your_problem(j) for j in 1:p] # a vector of `Bool`, true for positive.
gpS = pschur!(Aarg, S, :R)
your_eigvals = gpS.values
```
The result `gpS` is a [`GeneralizedPeriodicSchur`](@ref) object. For the common
case of lefwards alternating `A` (not inverted) and `B` inverted, see [`gpschur`](@ref).

For a partial periodic Schur decomposition, the basic API ([`partial_pschur`](@ref)) is

```julia
    pps, hist = partial_pschur(Aarg, nev, which; kw...)
```
The result is a [`PartialPeriodicSchur`](@ref) object `pps`, with a summary `hist` of the
iteration.
`pps` usually includes the `nev` eigenvalues nearest the edge of the convex hull of the
spectrum specified by `which`. The interface is derived from the `ArnoldiMethod` package,
q.v. for additional details.

## Operator ordering
The `:R` argument indicates that the product represented by `pS` is `prod(Aarg)`
i.e., counting rightwards. In many applications it is more convenient to number
the matrices leftwards (`A[p]*...*A[2]*A[1]`), corresponding to an orientation
argument `:L`.

At present, `partial_pschur` is only implemented for the left orientation.

## Operations on the decompositions

For eigenvectors, see [`eigvecs`](@ref). For reordering of subspaces, see
[`ordschur!`](@ref).

## Krylov-Schur with GPUArrays

!!! Warning
    This capability should be considered experimental

The Krylov-Schur code can be made to do Schur/Ritz-vector operations
(mainly, multiplication by the `A` matrices or operators) on a GPU
by use of `LinearMap`s.  Currently it is necessary to inform the code
that unusual arrays are in use by providing an initial vector of the
appropriate type as the starting vector for the iteration, as follows:

```
using LinearMaps

As_d = [cu(A) for A in As]
n = size(As_d[1], 1)
T = eltype(As_d[1])
Amaps = [LinearMap{T}(v -> A*v, n, n) for A in As_d]
v0_d = cu(rand(T, n))
ps, history = partial_pschur(Amaps, nev, LM(); u1=v0_d)
```

Note that the first run will take a very long time because of all the
compilation.