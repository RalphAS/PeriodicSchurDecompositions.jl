using PeriodicSchurDecompositions
using Documenter

DocMeta.setdocmeta!(PeriodicSchurDecompositions, :DocTestSetup, :(using PeriodicSchurDecompositions); recursive=true)

makedocs(;
    modules=[PeriodicSchurDecompositions],
    authors="Ralph A. Smith and contributors",
    repo="https://github.com/RalphAS/PeriodicSchurDecompositions.jl/blob/{commit}{path}#{line}",
    sitename="PeriodicSchurDecompositions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://RalphAS.github.io/PeriodicSchurDecompositions.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Usage" => "usage.md",
        "Interface" => Any[
            "lib/public.md",
        ]
    ],
)

deploydocs(;
    repo="github.com/RalphAS/PeriodicSchurDecompositions.jl",
    devbranch="main",
)
