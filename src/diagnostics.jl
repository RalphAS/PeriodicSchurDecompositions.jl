# verbosity values: 0 - silent, 1 - steps, 2 - convergence info, 3 - various matrices
const verbosity = Ref(0)

# Diagnosing the Krylov-Schur code is a special adventure, so we handle it separately.
const _kry_verby = Ref(0)

setverbosity(j) = verbosity[] = j

# Styling is sometimes awkward so make it optional.
const _dgn_styled = Ref(true)

_printsty(c, xs...) =
    if _dgn_styled[]
        printstyled(xs...; color = c)
    else
        print(xs...)
    end

