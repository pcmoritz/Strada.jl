using Strada
using Base.Test

include(joinpath(Pkg.dir("Strada"), "examples", "mnist-net.jl"))

function test_gradient(net, lambda, batchsize)
	(objective, theta) = make_objective(net, Float32; lambda=lambda)

	X = rand(Float32, 1, 28, 28, batchsize)
	y = floor(10*rand(Float32, 1, batchsize))

	val = Float64[]
	for i = 1:20
		push!(val, grad_check(objective, theta, (X, y), 5e-6; abs_error=false))
	end

	@test mean(val) <= 0.05
end

test_gradient(net, 0.0, batchsize)
test_gradient(net, 1e-3, batchsize)
