# Usage

For ordinary periodic Schur decompositions, the basic API is as follows:

```julia
p = period_of_your_problem()
Aarg = [your_matrix(j) for j in 1:p]
pS = pschur!(Aarg)
your_eigvals = pS.values
```
The result `pS` is a `PeriodicSchur` object.
Computation of the Schur vectors is
fairly expensive, so it may be suppressed via keyword arguments.

For generalized PSD, the basic API is as follows:

```julia
p = period_of_your_problem()
Aarg = [your_complex_matrix(j) for j in 1:p]
S = [sign_for_your_problem(j) for j in 1:p]
gpS = pschur!(Aarg, S)
your_eigvals = gpS.values
```
The result `gpS` is a `GeneralizedPeriodicSchur` object.

