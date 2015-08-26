using Strada

include("cifar-net.jl")
include("load-cifar.jl")

# the following model has been trained with caffe (see the tutorial)
model = joinpath(Pkg.dir("Strada"), "data", "cifar10_full.caffemodel")

Strada.load_caffemodel(net, model)

directory = joinpath(Pkg.dir("Strada"), "data")

(Xtrain, ytrain) = load_cifar_10(directory; data_set=:train)
Xtrain = permutedims(Xtrain, (2, 3, 1, 4))

(objective, theta) = make_objective(net, Float32)
predictor = make_predictor(net, Float32, "ip1")

result = zeros(Int, batchsize, 1)

predictor((Xtrain[:,:,:,1:batchsize], ytrain[:,1:batchsize]), theta; result=result)

println("Recognized ", sum(result .== ytrain[:,1:batchsize]'), " of ", batchsize)
