using Strada

include(joinpath(Pkg.dir("Strada"), "examples", "svm-net.jl"))

net = make_svm(30)
