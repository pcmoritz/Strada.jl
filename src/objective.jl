using Compat
using ArrayViews

function make_objective(net::CaffeNet, F::DataType; lambda::Float64=5e-4, loss_layer_name::Tag="loss")
	@assert F == Float32 # so far, only Float32 is supported
	lambda = convert(F, lambda)
	N = length(net.data_layer_assoc)
	function objective(data::Data{F,N}, theta::Vector{F}; grad::Vector{F}=zeros(F, 0))
		for i = 1:length(net.data_layer_assoc)
			set_data!(net, data[i], i)
		end
		zero!(net.layers.diff)
		copy!(net.layers.data, theta)
		l2 = convert(F, 0.5) * lambda * dot(theta, theta)
		forward(net)
		if length(grad) > 0 # calculate gradient
			backward(net)
		end
		if net.on_gpu
		   net.blobs = get_blobs(net.state)
		   net.layers = get_layer_blobs(net.state)
		end
		if length(grad) > 0 # calculate gradient
			copy!(grad, net.layers.diff)
			BLAS.axpy!(lambda, theta, grad) # add regularizer
		end
		return net.blobs.data.vars[loss_layer_name][1][1] + l2
	end

	blob = net.layers.data
	theta = zeros(F, length(blob))
	copy!(theta, blob)

	return (objective, theta)
end

function argmax_zero!(a::DenseArray, result::Array)
    for i = 1:length(result)
     	result[i] = indmax(view(a, :, i)) - 1
    end
end

function make_predictor(net::CaffeNet, F::DataType, output_layer_name::String; loss_layer_name::Tag="loss")
	@assert F == Float32 # again, only Float32 is supported at the moment
	N = length(net.data_layer_assoc)
	# for result, the shape is (T, N)
	function predictor(data::Data{F,N}, theta::Vector{F}; result::Matrix{Int}=zeros(Int, 0, 0))
		for i = 1:length(net.data_layer_assoc)
			set_data!(net, data[i], i)
		end
		copy!(net.layers.data, theta)
		forward(net)
		if net.on_gpu
		   net.blobs = get_blobs(net.state)
		   net.layers = get_layer_blobs(net.state)
		end
		pred = net.blobs.data[output_layer_name][1]
		if length(result) > 0
			argmax_zero!(pred, result)
		end
		return net.blobs.data[loss_layer_name][1][1]
	end
	return predictor
end
