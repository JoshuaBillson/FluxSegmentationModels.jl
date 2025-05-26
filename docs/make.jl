using FluxSegmentationModels
using Documenter

DocMeta.setdocmeta!(FluxSegmentationModels, :DocTestSetup, :(using FluxSegmentationModels); recursive=true)

makedocs(;
    modules=[FluxSegmentationModels],
    authors="Joshua Billson",
    sitename="FluxSegmentationModels.jl",
    format=Documenter.HTML(;
        canonical="https://JoshuaBillson.github.io/FluxSegmentationModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JoshuaBillson/FluxSegmentationModels.jl",
    devbranch="main",
)
