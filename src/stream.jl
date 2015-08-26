
@doc """
Data to be fed into a CaffeNet is kept in a `Data{F,N}` structure where `F` is the type of floating point number used to store the data (Float32 or Float64) and `N` is the number of data layers of the network. We represent Data{F,N} as a tuple, where the dimension i holds data that will be fed into data layer i of the network. A canonical example is for supervised learning, where N is 2, the first component representing the image (say) and the second component representing the label. Each dimension of the tuple typically holds an array whose last dimension corresponds to the index in the minibatch.
""" ->
typealias Data{F,N} NTuple{N, AbstractArray{F}}

@doc """
A data stream represents a data source for a neural network. It could be data held in memory, in a database on disk, or a network socket for example.
""" ->
abstract DataStream

@doc """
An empty stream represents a data source with no data.
""" ->
type EmptyStream <: DataStream
end

getfull(x::AbstractVector, I) = x[I]
getfull(x::AbstractMatrix, I) = x[:, I]
getfull{F}(x::AbstractArray{F, 3}, I) = x[:, :, I]
getfull{F}(x::AbstractArray{F, 4}, I) = x[:, :, :, I]

@doc """
A MinibatchStream is a collection of data represented in memory that has been partitioned into minibatches.
""" ->
type MinibatchStream <: DataStream
	data::Vector{Data}
    batchsize::Int

	function MinibatchStream(input::Data, batchsize::Int, drop::Bool=true)
		for i in 1:length(input)-1
			size(input[i], ndims(input[i])) == size(input[i+1], ndims(input[i+1])) ||
				throw(DimensionMismatch("All input arrays must have the same number of samples."))
		end
		n = size(input[1], ndims(input[1]))
		@assert mod(n, batchsize) == 0 || drop
		data = typeof(input)[]
		for i = 1:div(n, batchsize)
			therange = ((i-1) * batchsize + 1) : i * batchsize
			push!(data, tuple([getfull(input[j], therange) for j in 1:length(input)]...))
		end
		new(data, batchsize)
	end
end

@doc "Construct a MinibatchStream from a tuple of data arrays with full batch size." ->
minibatch_stream(args::AbstractArray...; batchsize::Int=1) = begin
	for i in 1:length(args) - 1
		eltype(args[i]) == eltype(args[i+1]) ||
			error("All input arrays must have the same type")
	end
	MinibatchStream(args, batchsize, true)
end

@lintpragma("Ignore unused str")
start(str::MinibatchStream) = 1
done(str::MinibatchStream, s) = num_batches(str) + 1 == s
next(str::MinibatchStream, s) = (str.data[s], s+1)

@doc "Number of minibatches in the MinibatchStream" ->
function num_batches(str::MinibatchStream)
	return length(str.data)
end

@doc "Number of datapoints in the MinibatchStream" ->
function size(str::MinibatchStream)
    return num_batches(str) * str.batchsize
end

@doc "Batchsize of the MinibatchStream" ->
function get_batchsize(str::MinibatchStream)
	return str.batchsize
end

@doc "Get a random minibatch from the MinibatchStream" ->
function getminibatch(str::MinibatchStream)
	i = rand(1:size(str.xs, 1))
	return (str.xs[i], str.ys[i])
end
