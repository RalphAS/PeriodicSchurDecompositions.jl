# Usage

For ordinary periodic Schur decompositions, the basic API is as follows:

```julia
p = period_of_your_problem()
Aarg = [your_matrix(j) for j in 1:p]
pS = pschur!(Aarg, :R)
your_eigvals = pS.values
```
The result `pS` is a `PeriodicSchur` object.
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
The result `gpS` is a `GeneralizedPeriodicSchur` object.

For a partial periodic Schur decomposition, the basic API is

```julia
    pps, hist = partial_pschur(Aarg, nev, which; kw...)
```
The result is a `PartialPeriodicSchur` object `pps`, with a summary `hist` of the iteration.
`pps` usually includes the `nev` eigenvalues nearest the edge of the convex hull of the
spectrum specified by `which`. The interface is derived from the `ArnoldiMethod` package,
q.v. for additional details.

## Operator ordering
The `:R` argument indicates that the product represented by `pS` is `prod(Aarg)`
i.e., counting rightwards. In many applications it is more convenient to number
the matrices leftwards (`A[p]*...*A[2]*A[1]`), corresponding to an orientation
argument `:L`.

At present, `partial_pschur` is only implemented for the left orientation.
