using Strada
using Base.Test
using Lint

include("apollotests.jl")
include("basictests.jl")
include("caffetests.jl")
include("test-grad-mnist.jl")
include("test-grad-cifar.jl")
include("test-grad-svm.jl")

@test isempty(lintpkg("Strada", returnMsgs=true))
