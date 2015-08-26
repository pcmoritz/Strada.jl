using Strada
using Base.Test

include(joinpath(Pkg.dir("Strada"), "examples", "svm-net.jl"))

p = 50
batchsize = 100
net = make_svm(p)

function test_gradient(net, lambda, batchsize)
	(objective, theta) = make_objective(net, Float32; lambda=lambda)

	X = rand(Float32, p, 1, 1, batchsize)
	y = floor(2*rand(Float32, 1, batchsize))

	@test grad_check(objective, theta, (X, y), 5e-3; abs_error=false)  < 0.001
end

test_gradient(net, 0.0, batchsize)
test_gradient(net, 1.0, batchsize)
