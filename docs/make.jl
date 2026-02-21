using GeneralizedRFF
using Documenter

DocMeta.setdocmeta!(GeneralizedRFF, :DocTestSetup, :(using GeneralizedRFF); recursive=true)

makedocs(;
    modules=[GeneralizedRFF],
    authors="Shuichi Miyazawa",
    sitename="GeneralizedRFF.jl",
    format=Documenter.HTML(;
        canonical="https://shu13830.github.io/GeneralizedRFF.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/shu13830/GeneralizedRFF.jl",
    devbranch="main",
)
