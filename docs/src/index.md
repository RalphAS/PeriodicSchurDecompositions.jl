```@meta
CurrentModule = PeriodicSchurDecompositions
```

# PeriodicSchurDecompositions.jl

This Julia package provides implementations of the periodic Schur decomposition
of matrix products of real element types and of the generalized periodic Schur
decomposition for complex element types.

## Definitions

### Periodic Schur decomposition

Given a series of $N\times N$ matrices $A_j,\ j=1,\ldots,p$, a periodic Schur decomposition (PSD)
is a factorization of the form:

$$\begin{aligned}
Q_1^\prime A_1 Q_2 &= T_1 \\
Q_2^\prime A_2 Q_3 &= T_2 \\
\vdots& \\
Q_p^\prime A_p Q_1 &= T_p
\end{aligned}$$

where the $Q_j$ are unitary (orthogonal) and the $T_j$ are upper triangular,
except that one of the $T_j$ is quasi-triangular for real element types.
It furnishes the eigenvalues and invariant subspaces of the matrix product
$\Pi_{j=1}^p A_j$.

The principal reason for using the PSD is that accuracy may be lost if one
forms the product of the $A_j$ before eigen-analysis. For some applications the
intermediate Schur vectors are also useful.

### Operator ordering

For many applications it is more natural to pose the matrix product in the form
$A_p A_{p-1}\ldots A_2 A_1$. In this case the more useful factorization is

$$\begin{aligned}
Q_2^\prime A_1 Q_1 &= T_1 \\
Q_3^\prime A_2 Q_2 &= T_2 \\
\vdots& \\
Q_1^\prime A_p Q_p &= T_p.
\end{aligned}$$

This ordering is accommodated with the ':L' (left) orientation argument to `pschur!`.

### Generalized periodic Schur decomposition
Given a series of $N\times N$ matrices $A_j,\ j=1,\ldots,p$, and a signature vector
$S$ where $s_j\in \{1,-1\}$, a generalized periodic Schur decomposition (GPSD)
is a factorization of the formal product $\Pi_{j=1}^p A_j^{s_j}$ so that
$Q_j^\prime A_j  Q_{j+1} = T_j$ if $s_j = 1$ and
$Q_{j+1}^\prime  A_j  Q_j = T_j$ if $s_j = -1$.

The GPSD is an extension of the QZ decomposition used for generalized eigenvalue
problems. Thus formally infinite eigenvalues are not problematic.

## References

A. Bojanczyk, G. Golub, and P. Van Dooren, "The periodic Schur decomposition.
Algorithms and applications," Proc. SPIE 1996.

D. Kressner, thesis and assorted articles.

## Acknowledgements

The meat of this package is mainly a translation of implementations in [the SLICOT library](https://github.com/SLICOT/SLICOT-Reference.git).
