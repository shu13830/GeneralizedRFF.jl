using GeneralizedRFFs
using Documenter

DocMeta.setdocmeta!(GeneralizedRFFs, :DocTestSetup, :(using GeneralizedRFFs); recursive=true)

makedocs(;
    modules=[GeneralizedRFFs],
    authors="Shuichi Miyazawa",
    sitename="GeneralizedRFFs.jl",
    format=Documenter.HTML(;
        canonical="https://shu13830.github.io/GeneralizedRFFs.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/shu13830/GeneralizedRFFs.jl",
    devbranch="main",
)
