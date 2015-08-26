using Strada
using Base.Test

net = Net("SimpleNet"; log_level=3)

function run_simple(net)
	reset(net)
	forward(net, DataLayer("data", data=ones(Float32, 1, 1)))
	forward(net, DataLayer("label", data=ones(Float32, 1, 1)))
	forward(net, LinearLayer("ip1", ["data"]; n_filter=1, param_names=["ip1_weights", "ip1_bias"]))
	forward(net, ActivationLayer("relu1", ["ip1"]; activation=ReLU))
	forward(net, LinearLayer("ip2", ["relu1"]; n_filter=1, param_names=["ip2_weights", "ip2_bias"]))
	forward(net, EuclideanLoss("loss", ["ip2", "label"]))
end

run_simple(net)

@test all(sort(keys(net.params.data)) .== sort(["ip1_weights", "ip1_bias", "ip2_weights", "ip2_bias"]))
@test all(sort(keys(net.blobs.data)) .== sort(["data", "label", "ip1", "relu1", "ip2", "loss"]))

@test size(net.blobs.data["ip1"]) == (1,1)

# Test round trip
blob = net.params.data
theta = rand(Float32, length(blob))
copy!(blob, theta)
newtheta = zeros(length(blob))
copy!(newtheta, blob)
@test norm(newtheta - theta) <= 1e-8

