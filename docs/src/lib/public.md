# Functions

## Periodic Schur decompositions
```@docs
pschur
```

```@docs
pschur!
```

```@docs
gpschur
```

### Reordering

```@docs
LinearAlgebra.ordschur!(P::AbstractPeriodicSchur{T}, select::AbstractVector{Bool}) where {T <: Complex}
```

## Partial periodic Schur decompositions
```@docs
partial_pschur
```

## Eigenvectors
Eigenvector computations are provided for standard and partial PSDs.
In future releases, they may also be available for some kinds of generalized PSDs.

Most practical applications of PSDs are too ill-conditioned for the standard
back-solve approaches to computing eigenvectors, so the code in this package
uses more stable subspace-reordering steps instead. This has the feature
that attempting to get eigenvectors out of ill-conditioned subspaces
(eigenvalues too close together) will fail instead of the usual misleading
redundancy. This approach is fairly expensive, so the user should only ask for
vectors which will be needed.

```@docs
LinearAlgebra.eigvecs(ps0::PeriodicSchur{T}, select::AbstractVector{Bool}) where {T}
```

## Periodic Hessenberg decompositions
```@docs
phessenberg!
```
