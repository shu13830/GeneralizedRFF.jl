using GeneralizedRandomFourierFeatures
using Documenter

DocMeta.setdocmeta!(GeneralizedRandomFourierFeatures, :DocTestSetup, :(using GeneralizedRandomFourierFeatures); recursive=true)

makedocs(;
    modules=[GeneralizedRandomFourierFeatures],
    authors="Shuichi Miyazawa",
    sitename="GeneralizedRandomFourierFeatures.jl",
    format=Documenter.HTML(;
        canonical="https://shu13830.github.io/GeneralizedRandomFourierFeatures.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/shu13830/GeneralizedRandomFourierFeatures.jl",
    devbranch="main",
)
