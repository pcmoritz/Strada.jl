using DataStructures

type CaffeNet
	name::ASCIIString
	param::NetParameter
	layer_defs::Vector{Layer}
	state::Ptr{Void}
	blobs::NetData{CaffeDict}
	layers::NetData{CaffeDict}
	# data_layer_assoc[i] is the index of the i-th data layer in the constructed net
	# (need not be consecutive because of split layers that Caffe inserts automatically)
	data_layer_assoc::Vector{Int}
	on_gpu::Bool
end

function make_net_param(layers::Vector{LayerParameter})
	net_param = NetParameter()
	set_field!(net_param, :layer, layers)
	return net_param
end

@doc "Load a model from a caffe compatible .caffemodel file (for example from the caffe model zoo)." ->
function Net(name::ASCIIString, layers::Vector{Layer}; use_gpu::Bool=false, log_level::Int=3)
	param = make_net_param([layer.param for layer in layers])
	buffer = PipeBuffer()
	writeproto(buffer, param)
	state = ccall((:init_caffenet, "libjlcaffe.so"), Ptr{Void}, (Ptr{Void}, Int, Int), buffer.data, length(buffer.data), log_level)
	if use_gpu
		ccall((:set_use_gpu, "libjlcaffe.so"), Void, ())
	end
	num_data_layers = sum([x.memory_layer for x in layers])
	data_layer_names = [x.name for x in layers[1:num_data_layers]]
	(layer_names, layer_blobs) = get_layer_info(state)
	data_layer_assoc = Int[]
	# TODO: Make n^2 -> n log(n) or better
	for (i, layer_name) in enumerate(layer_names)
		if layer_name in data_layer_names
			push!(data_layer_assoc, i)
		end
	end
	return CaffeNet(name, param, layers, state, get_blobs(state), layer_blobs, data_layer_assoc, false)
end

@doc "Activate CPU mode." ->
function set_mode_cpu(net::CaffeNet)
	ccall((:set_mode_cpu, "libjlcaffe.so"), Void, ())
	net.on_gpu = false
end

@doc "Activate CPU mode." ->
function set_mode_gpu(net::CaffeNet)
	ccall((:set_mode_gpu, "libjlcaffe.so"), Void, ())
	net.on_gpu = true
end

@doc "Load a model from a caffe compatible .caffemodel file (for example from the caffe model zoo)." ->
function load_caffemodel(net::CaffeNet, filename::String)
	ccall((:load_from_caffemodel, "libjlcaffe.so"), Void, (Ptr{Void}, Ptr{UInt8}), net.state, filename)
end

function set_data!{N}(net::CaffeNet, data::Array{Float32, N}, layer_idx::Int)
	stream_axis = net.layer_defs[layer_idx].stream_axis
	stream_axis = stream_axis == -1 ? ndims(data) : stream_axis
	n = size(data, stream_axis)
	# make layer_idx compatible with caffe's zero indexing
	caffe_layer_idx = net.data_layer_assoc[layer_idx]
	ccall((:set_data, "libjlcaffe.so"), Void, (Ptr{Void}, Int, Ptr{Float32}, Int), net.state, caffe_layer_idx - 1, pointer(data), n)
end

function get_batchsize(net::CaffeNet)
	return net.param.input_shape[1].dim[end]
end

@doc "Run a forward pass through the whole network." ->
function forward(net::CaffeNet)
	ccall((:forward, "libjlcaffe.so"), Void, (Ptr{Void},), net.state)
end

@doc "Run a backward pass through the whole network." ->
function backward(net::CaffeNet)
	ccall((:backward, "libjlcaffe.so"), Void, (Ptr{Void},), net.state)
end

function get_layer_info(state::Ptr{Void})
	layers = NetData(CaffeDict(), CaffeDict())
	layer_names = String[]
	num_layers = ccall((:num_layers, "libjlcaffe.so"), Int, (Ptr{Void},), state)
	for layer = 0:num_layers-1
		name = ccall((:layer_name, "libjlcaffe.so"), Ptr{UInt8}, (Ptr{Void}, Int), state, layer)
		num_axis = ccall((:num_blob_axis, "libjlcaffe.so"), Int, (Ptr{Void}, Int), state, layer)
		push!(layer_names, bytestring(name))
		for axis in 0:num_axis-1
			blob = ccall((:get_weight_blob, "libjlcaffe.so"), Ptr{Void}, (Ptr{Void}, Int, Int), state, layer, axis)
			if blob != C_NULL
				assign_blob!(blob, bytestring(name), layers)
			end
		end
	end
	return (layer_names, layers)
end

function get_blobs(state::Ptr{Void})
	result = NetData(CaffeDict(), CaffeDict())
	num_blobs = ccall((:num_blobs, "libjlcaffe.so"), Int, (Ptr{Void},), state)
	for blob_idx = 0:num_blobs-1
		name = bytestring(ccall((:get_blob_name, "libjlcaffe.so"), Ptr{UInt8}, (Ptr{Void}, Int), state, blob_idx))
		blob = ccall((:get_blob, "libjlcaffe.so"), Ptr{Void}, (Ptr{Void}, Int), state, blob_idx)
		assign_blob!(blob, bytestring(name), result)
	end
	return result
end

function push_data!(blob::CaffeDict, field::ASCIIString, data::Ptr{Float32}, shape::Tuple)
	if !haskey(blob, field)
		blob[field] = []
	end
	push!(blob[field], pointer_to_array(data, Base.reverse(shape))) # row major and column major
end

function assign_blob!(source::Ptr{Void}, field::ASCIIString, target::NetData{CaffeDict})
	shape = get_shape(source)
	dataptr = ccall((:get_data, "libjlcaffe.so"), Ptr{Float32}, (Ptr{Void},), source)
	diffptr = ccall((:get_diff, "libjlcaffe.so"), Ptr{Float32}, (Ptr{Void},), source)
	push_data!(target.data, field, dataptr, shape)
	push_data!(target.diff, field, diffptr, shape)
end
