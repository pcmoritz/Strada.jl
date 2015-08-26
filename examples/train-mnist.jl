using Strada
using Logging

include("mnist-net.jl");
include("load-mnist.jl")

# Strada.set_mode_gpu(net)

directory = joinpath(Pkg.dir("Strada"), "data")

(Xtrain, ytrain) = load_mnist(directory; data_set=:train)
(Xtest, ytest) = load_mnist(directory; data_set=:test)

data = minibatch_stream(Xtrain, ytrain; batchsize=batchsize)
testset = minibatch_stream(Xtest, ytest, batchsize=batchsize)

(objective, theta) = make_objective(net, Float32)

Logging.configure(level=INFO)

predictor = make_predictor(net, Float32, "ip2")

sgd(objective, data, theta; predictor=predictor, testset=testset, lr_schedule=InvLR(0.01, 0.0001, 0.75, 0.9), epochs=5, verbose=true)


