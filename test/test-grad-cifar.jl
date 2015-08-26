using Strada
using Base.Test

include(joinpath(Pkg.dir("Strada"), "examples", "cifar-net.jl"))

function test_gradient(net, lambda, batchsize)
	(objective, theta) = make_objective(net, Float32; lambda=lambda)

	X = rand(Float32, 3, 32, 32, batchsize)
	y = floor(10*rand(Float32, 1, batchsize))

	@test grad_check(objective, theta, (X, y), 5e-4) <= 0.05
end

test_gradient(net, 0.0, batchsize)
test_gradient(net, 1e-3, batchsize)
