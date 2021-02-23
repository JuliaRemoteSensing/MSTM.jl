using MSTM
using Documenter

DocMeta.setdocmeta!(MSTM, :DocTestSetup, :(using MSTM); recursive=true)

makedocs(;
    modules=[MSTM],
    authors="Gabriel Wu <qqbbnease1004@126.com> and contributors",
    repo="https://github.com/JuliaRemoteSensing/MSTM.jl/blob/{commit}{path}#{line}",
    sitename="MSTM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaRemoteSensing.github.io/MSTM.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaRemoteSensing/MSTM.jl",
    devbranch = "main",
)
