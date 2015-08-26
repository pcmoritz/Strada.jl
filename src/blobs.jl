using DataStructures
import Base: haskey, get, get!, getkey, empty!, setindex!, getindex, length, isempty,
		start, next, done, keys, values, fill!, copy!

@doc """A CaffeDict is a collection of blobs with names, each name is associated with
		an arbitrary number of floating point arrays. Example: The name 'conv1' could
		map to a vector containing the biases and weights of a convolution.""" ->
type CaffeDict
	vars::OrderedDict{ASCIIString, Vector{Any}}
end

@delegate CaffeDict.vars [ haskey, get, get!, getkey, empty!, setindex!, getindex,
	isempty, start, next, done, keys, values ]

CaffeDict() = begin
	return CaffeDict(OrderedDict{ASCIIString, Vector{Any}}())
end

@doc "Number of parameters in the blob." ->
function length(blob::CaffeDict)
	len = 0
	for key in keys(blob)
		for b in blob[key]
			len += length(b)
		end
	end
	return len
end

@doc "Copy a binary blob into a flat parameter vector." ->
function copy!(output::Vector, input::CaffeDict)
	curr_idx = 1
	for key in keys(input)
		for b in input[key]
			curr_len = length(b)
			copy!(sub(output, curr_idx:(curr_idx+curr_len)-1), b)
			curr_idx += curr_len
		end
	end
end

@doc "Copy a flat parameter vector into a binary blob." ->
function copy!(output::CaffeDict, input::Vector)
	curr_idx = 1
	for key in keys(output)
		for b in output[key]
			curr_len = length(b)
			copy!(b, sub(input, curr_idx:(curr_idx+curr_len)-1))
			curr_idx += curr_len
		end
	end
end

@doc "Fill a binary blob with zeros." ->
function zero!(A::CaffeDict)
	for key in keys(A)
		for b in A[key]
			fill!(b, 0.0)
		end
	end
end

@doc """A collection of blobs in a network. Grouped into 'data' (the actual parameters)
		and 'diff' (the gradients) so they can be treated as vectors that can be added together.""" ->
type NetData{D}
	data::D
	diff::D
end

@doc """An ApolloDict is a collection of blobs with names, each name is associated with
		one floating point array. Example: The name 'ip_weights' could map to the weights of a linear layer.""" ->
type ApolloDict
	state::Ptr{Void} # a pointer to an apollonet_state
	access_blob::Bool # are we acessing blobs (true) or params (false)?
	access_data::Bool # are we acessing data (true) or diffs (false)?
end

start(blob::ApolloDict) = (keys(blob), 1)
@lintpragma("Ignore unused blob")
done(blob::ApolloDict, pos) = pos[2] > length(pos[1])
next(blob::ApolloDict, pos) = begin
	(thekeys, key_id) = pos
	return ((thekeys[key_id], blob[thekeys[key_id]]), (thekeys, key_id + 1))
end

keys(blob::ApolloDict) = begin
	result = ASCIIString[]
	while true
		nameptr = C_NULL
		if blob.access_blob
			nameptr = ccall((:apollo_next_blob_name, "libjlcaffe.so"), Ptr{Uint8}, (Ptr{Void},), blob.state)				
		else
			nameptr = ccall((:apollo_next_param_name, "libjlcaffe.so"), Ptr{Uint8}, (Ptr{Void},), blob.state)
		end
		if nameptr == C_NULL
			return result
		end
		push!(result, bytestring(nameptr))
	end
end

function get_shape(blob)
	num_axes = ccall((:get_num_axes, "libjlcaffe.so"), Int, (Ptr{Void},), blob)
	result = Int[]
	for axis = 0:num_axes-1
		axis_shape = ccall((:get_axis_shape, "libjlcaffe.so"), Int, (Ptr{Void},Int), blob, axis)
		push!(result, axis_shape)
	end
	return tuple(result...)
end

getkey(blob::ApolloDict, key, default) = begin
	blobptr = C_NULL
	if blob.access_blob
		blobptr = ccall((:apollo_get_blob, "libjlcaffe.so"), Ptr{Void}, (Ptr{Void}, Ptr{Void}), blob.state, key)
	else
		blobptr = ccall((:apollo_get_param, "libjlcaffe.so"), Ptr{Void}, (Ptr{Void}, Ptr{Void}), blob.state, key)
	end
	blobptr == C_NULL && return default
	shape = get_shape(blobptr)
	dataptr = C_NULL
	if blob.access_data
		dataptr = ccall((:get_data, "libjlcaffe.so"), Ptr{Float32}, (Ptr{Void},), blobptr)
	else
		dataptr = ccall((:get_diff, "libjlcaffe.so"), Ptr{Float32}, (Ptr{Void},), blobptr)
	end
	dataptr == C_NULL && return default
	return pointer_to_array(dataptr, Base.reverse(shape)) # row major and column major
end

const secret_apollo_dict_token = :unique_identifier_61320927118372926144

function getindex(blob::ApolloDict, key)
	v = getkey(blob, key, secret_apollo_dict_token)
	if is(v, secret_apollo_dict_token)
		throw(KeyError(key))
	end
	return v
end

@doc "The total number of variables in a binary blob." ->
function length(blob::ApolloDict)
	len = 0
	for key in keys(blob)
		len += length(blob[key])
	end
	return len
end

@doc "Copy a binary blob into a flat parameter vector." ->
function copy!(output::Vector, input::ApolloDict)
	curr_idx = 1
	for key in keys(input)
		curr_len = length(input[key])
		copy!(sub(output, curr_idx:(curr_idx+curr_len)-1), input[key])
		curr_idx += curr_len
	end
end

@doc "Copy a flat parameter vector into a binary blob." ->
function copy!(output::ApolloDict, input::Vector)
	curr_idx = 1
	for key in keys(output)
		curr_len = length(output[key])
		copy!(output[key], sub(input, curr_idx:(curr_idx+curr_len)-1))
		curr_idx += curr_len
	end
end

@doc "Fill a binary blob with zeros." ->
function zero!(A::ApolloDict)
	for key in keys(A)
		fill!(A[key], 0.0)
	end
end
