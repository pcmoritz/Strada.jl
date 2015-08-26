using DataStructures

type ApolloNet
	name::String
	state::Ptr{Void}
	blobs::NetData{ApolloDict}
	params::NetData{ApolloDict}
	loss::Float32
end

@doc "Create an empty ApolloNet. A log_level of 0 prints full caffe debug information, a log_level of 3 prints nothing." ->
function Net(name::ASCIIString; log_level::Int=3)
	state = ccall((:init_apollonet, "libjlcaffe.so"), Ptr{Void}, (Int,), log_level)
	blobs = NetData{ApolloDict}(ApolloDict(state, true, true), ApolloDict(state, true, false))
	params = NetData{ApolloDict}(ApolloDict(state, false, true), ApolloDict(state, false, false))
	return ApolloNet(name, state, blobs, params, 0.0f0)
end

@doc "Clear the active layers and active parameters of the net so a new forward pass can be run." ->
function reset(net::ApolloNet)
	ccall((:apollo_reset, "libjlcaffe.so"), Void, (Ptr{Void},), net.state)
	net.loss = 0.0f0
end

@doc "Run a forward pass of a single layer." ->
function forward(net::ApolloNet, layer::Layer)
	buffer = PipeBuffer()
	writeproto(buffer, layer.param)
	result = ccall((:apollo_forward, "libjlcaffe.so"), Float32, (Ptr{Void}, Ptr{Void}, Int), net.state, buffer.data, length(buffer.data))
	net.loss += result
	return result
end

@doc "Run a backward pass through the whole network." ->
function backward(net::ApolloNet)
	ccall((:apollo_backward, "libjlcaffe.so"), Void, (Ptr{Void},), net.state)
end

@doc "Activate CPU mode." ->
function set_mode_cpu(net::ApolloNet)
	ccall((:set_mode_cpu, "libjlcaffe.so"), Void, ())
end

@doc "Activate GPU mode." ->
function set_mode_gpu(net::ApolloNet)
	ccall((:set_mode_gpu, "libjlcaffe.so"), Void, ())
end
