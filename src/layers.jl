using Compat

typealias Tag ASCIIString

type Layer
	name::Tag
	bottoms::Vector{Tag}
	tops::Vector{Tag}
	param_name::Symbol
	param::LayerParameter
	stream_axis::Int
	memory_layer::Bool # Is this layer a memory layer?
end

function Layer(name, bottoms, tops, param_name, param, stream_axis)
	return Layer(name, bottoms, tops, param_name, param, stream_axis, false)
end

function make_param(lr_mult::Float64)
	param = ParamSpec()
	set_field!(param, :lr_mult, convert(Float32, lr_mult))
	return param
end

function make_layer(typ::ASCIIString, name::Tag, bottoms::Vector{Tag}, tops::Vector{Tag};
		has_weights=false, param_names=Tag[])
	layer = LayerParameter()
	set_field!(layer, :name, string(name))
	set_field!(layer, :_type, typ)
	bottom_names = length(bottoms) == 0 ? ASCIIString[] : map(string, bottoms)
	top_names = length(tops) == 0 ? ASCIIString[] : map(string, tops)
	set_field!(layer, :bottom, convert(Vector{AbstractString}, bottom_names))
	set_field!(layer, :top, convert(Vector{AbstractString}, top_names))
	if has_weights
		param1 = make_param(1.0)
		param2 = make_param(2.0)
		set_field!(layer, :param, [param1, param2])
	end
	if length(param_names) > 0
		params = ParamSpec[]
		for param_name in param_names
			param = ParamSpec()
			set_field!(param, :name, string(param_name))
			push!(params, param)
		end
		set_field!(layer, :param, params)
	end
	return layer
end

@doc """
Fillers are random number generators that fills a blob using the specified algorithm. The algorithm is specified by

* `name`: Can be `:gaussian`, `:uniform`, `:xavier` or `:constant`

The parameters are given by keyword arguments:

* `value`: Gives the value for a constant filler

* `min` and `max`: Range for a uniform filler

* `mean` and `std`: Mean and standard deviation of a Gaussian filler

""" ->
function filler(name::Symbol; value::Float64=NaN, min::Float64=NaN, max::Float64=NaN, mean::Float64=NaN, std::Float64=NaN)
	result = FillerParameter()
	function set_num!(field::Symbol, value::Float64)
		if !isnan(value)
			set_field!(result, field, convert(Float32, value))
		end
	end
	set_field!(result, :_type, string(name))
	set_num!(:value, value)
	set_num!(:min, min)
	set_num!(:max, max)
	set_num!(:mean, mean)
	set_num!(:std, std)
	return result
end

@doc """
The Convolution layer convolves the input image with a set of learnable filters, each producing one feature map in the output image. keyword parameters are

* `n_filter`: The number of filters (required)

* `kernel`: A tuple specifying height and width of each filter (required)

* `stride`: A tuple which specifies the intervals at which to apply the filters to the input (horizontally and vertically)

* `pad`: A tuple which specifies the number of pixels to (implicitly) add to each side of the input

* `group` (default 1): If g > 1, we restrict the connectivity of each filter to a subset of the input. Specifically, the input and output channels are separated into g groups, and the ith output group channels will be only connected to the ith input group channels.

The input is of shape `n * c_i * h_i * w_i` and the output is of shape `n * c_o * h_o * w_o`, where `h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1` and `w_o` likewise.

""" ->
function ConvLayer(name::Tag, bottoms::Vector{Tag};
		kernel::NTuple{2,Int}=(1,1), stride::NTuple{2,Int}=(1,1), pad::NTuple{2,Int}=(0,0), n_filter::Int=1, group::Int=1,
		weight_filler::FillerParameter=filler(:xavier),
		bias_filler::FillerParameter=filler(:constant))
	layer = make_layer("Convolution", name, bottoms, [name]; has_weights=true)
	conv_param = ConvolutionParameter()
	set_field!(conv_param, :num_output, convert(UInt32, n_filter))
	set_field!(conv_param, :group, convert(UInt32, group))
	set_field!(conv_param, :kernel_h, convert(UInt32, kernel[1]))
	set_field!(conv_param, :kernel_w, convert(UInt32, kernel[2]))
	set_field!(conv_param, :stride_h, convert(UInt32, stride[1]))
	set_field!(conv_param, :stride_w, convert(UInt32, stride[2]))
	set_field!(conv_param, :pad_h, convert(UInt32, pad[1]))
	set_field!(conv_param, :pad_w, convert(UInt32, pad[2]))
	set_field!(conv_param, :weight_filler, weight_filler)
	set_field!(conv_param, :bias_filler, bias_filler)
	set_field!(conv_param, :bias_term, true)
	set_field!(layer, :convolution_param, conv_param)
	return Layer(name, bottoms, [name], :convolution_param, layer, 0)
end

@doc """
The InnerProduct layer (also usually referred to as the fully connected layer) treats the input as a simple vector and produces an output in the form of a single vector (with the blob’s height and width set to 1). The keyword parameters are

* `n_filter`: The number of filters (required)

The input is of shape `n * c_i * h_i * w_i` and the output of shape `n * c_o * 1 * 1`.

""" ->
function LinearLayer(name::Tag, bottoms::Vector{Tag};
		n_filter::Int=1, weight_filler::FillerParameter=filler(:xavier),
		bias_filler::FillerParameter=filler(:constant; value=0.0), axis::Int=-1, param_names=Tag[])
	layer = make_layer("InnerProduct", name, bottoms, [name]; has_weights=true, param_names=param_names)
	linear_param = InnerProductParameter()
	set_field!(linear_param, :num_output, convert(UInt32, n_filter))
	set_field!(linear_param, :weight_filler, weight_filler)
	set_field!(linear_param, :bias_filler, bias_filler)
	set_field!(linear_param, :bias_term, true)
	if axis != -1
		set_field!(linear_param, :axis, convert(Int32, axis))
	end
	set_field!(layer, :inner_product_param, linear_param)
	return Layer(name, bottoms, [name], :inner_product_param, layer, 0)
end

@doc """The PoolLayer partitions the input image into a set of non-overlapping rectangles and, for each such sub-region, outputs the maximum or average value. The keyword parameters are

* `method` (default `MAX`): The pooling method. Can be `MAX`, `AVE` or `STOCHASTIC`.

* `pad` (default 0): Specifies the number of pixels to (implicitly) add to each side of the input

* `stride` (default 1): Specifies the intervals at which to apply the filters to the input

The input is of shape `n * c * h_i * w_i` and the output of shape `n * c * h_o * w_o` where `h_o` and `w_o` are computed in the same way as for the convolution.
""" ->
function PoolLayer(name::Tag, bottoms::Vector{Tag};
		method::Int=0, kernel::NTuple{2, Int}=(1,1), stride::NTuple{2, Int} = (1,1), pad::NTuple{2,Int}=(0,0))
	layer = make_layer("Pooling", name, bottoms, [name])
	pool_param = PoolingParameter()
	set_field!(pool_param, :pool, convert(Int32, method))
	set_field!(pool_param, :kernel_h, convert(UInt32, kernel[1]))
	set_field!(pool_param, :kernel_w, convert(UInt32, kernel[2]))
	set_field!(pool_param, :stride_h, convert(UInt32, stride[1]))
	set_field!(pool_param, :stride_w, convert(UInt32, stride[2]))
	set_field!(pool_param, :pad_h, convert(UInt32, pad[1]))
	set_field!(pool_param, :pad_w, convert(UInt32, pad[2]))
	set_field!(layer, :pooling_param, pool_param)
	return Layer(name, bottoms, [name], :pooling_param, layer, 0)
end

immutable Activation
	activation::Tag
	param::Symbol
end

ReLU = Activation("ReLU", :relu_param)
Sigmoid = Activation("Sigmoid", :sigmoid_param)
TanH = Activation("TanH", :tanh_param)

@doc """
Activation layers compute element-wise function, taking one bottom blob as input and producing one top blob of the same size. Parameters are

* `activation` (default `Sigmoid`): The nonlinear function applied. Can be `ReLU`, `Sigmoid` or `TanH`.

Both input and output are of shape `n * c * h * w`.
""" ->
function ActivationLayer(name::Tag, bottoms::Vector{Tag};
	activation::Activation=Sigmoid)
	layer = make_layer(string(activation.activation), name, bottoms, [name]) 
	return Layer(name, bottoms, [name], activation.param, layer, 0)
end

const ACROSS_CHANNELS = 0 # TODO: Read that out from the proto file
const WITHIN_CHANNEL = 1 # TODO: Read that out from the proto file

@doc """
 The local response normalization layer performs a kind of “lateral inhibition” by normalizing over local input regions. In `ACROSS_CHANNELS` mode, the local regions extend across nearby channels, but have no spatial extent (i.e., they have shape `local_size` x 1 x 1). In `WITHIN_CHANNEL` mode, the local regions extend spatially, but are in separate channels (i.e., they have shape 1 x `local_size` x `local_size`). Each input value is divided by `(1+(α/n) sum_i x_i^2)β)`, where n is the size of each local region, and the sum is taken over the region centered at that value (zero padding is added where necessary). It accepts the following keyword arguments:

 * `local_size` (default 3): Size of the region the normalization is computed over

 * `alpha` (default `5e-5`): Value of the parameter α

 * `beta` (default `0.75`): Value of the parameter β

 * `norm_region`: Mode of the local contrast normalization. Can be `ACROSS_CHANNELS` or `WITHIN_CHANNEL`. 
 """ ->
function LRNLayer(name::Tag, bottoms::Vector{Tag};
		local_size::Int=3, alpha::Float64=5e-5, beta::Float64=0.75, norm_region=ACROSS_CHANNELS)
	layer = make_layer("LRN", name, bottoms, [name])
	lrn_param = LRNParameter()
	set_field!(lrn_param, :local_size, convert(UInt32, local_size))
	set_field!(lrn_param, :alpha, convert(Float32, alpha))
	set_field!(lrn_param, :beta, convert(Float32, beta))
	set_field!(lrn_param, :norm_region, convert(Int32, norm_region))
	set_field!(layer, :lrn_param, lrn_param)
	return Layer(name, bottoms, [name], :lrn_param, layer, 0)
end

@doc """
The dropout layer is a regularizer that randomly sets input values to zero.
""" ->
function DropoutLayer(name::Tag, bottoms::Vector{Tag};
	dropout_ratio::Float64=0.5)
	layer = make_layer("Dropout", name, bottoms, [name])
	dropout_param = DropoutParameter()
	set_field!(dropout_param, :dropout_ratio, convert(Float32, dropout_ratio))
	set_field!(layer, :dropout_param, dropout_param)
	return Layer(name, bottoms, [name], :dropout_param, layer, 0)
end

function RecurrentLayer(name::Tag, bottoms::Vector{Tag};
		num_output::Int=1, weight_filler::FillerParameter=filler(:xavier), bias_filler::FillerParameter=filler(:constant), param_names=Tag[])
	layer = make_layer("LSTM", name, bottoms, [name]; param_names=param_names)
	recurrent_param = RecurrentParameter()
	set_field!(recurrent_param, :num_output, convert(UInt32, num_output))
	set_field!(recurrent_param, :weight_filler, weight_filler)
	set_field!(recurrent_param, :bias_filler, bias_filler)
	set_field!(layer, :recurrent_param, recurrent_param)
	return Layer(name, bottoms, [name], :recurrent_param, layer, 0)
end

@doc """
The LstmLayer is an LSTM unit. It takes two blobs as input, the current LSTM input and the previous memory cell content. It outputs the new hidden state and the updated memory cell.
""" ->
function LstmLayer(name::Tag, bottoms::Vector{Tag}, tops::Vector{Tag}=Tag[];
	num_cells::Int=1, weight_filler::FillerParameter=filler(:uniform, min=-0.1, max=0.1), param_names=Tag[])
	layer = make_layer("LstmUnit", name, bottoms, length(tops) == 0 ? [name] : tops; param_names=param_names)
	lstm_unit_param = LstmUnitParameter()
	set_field!(lstm_unit_param, :num_cells, convert(UInt32, num_cells))
	set_field!(lstm_unit_param, :input_weight_filler, weight_filler) # TODO: make separate fillers
	set_field!(lstm_unit_param, :input_gate_weight_filler, weight_filler) # TODO: make separate fillers
	set_field!(lstm_unit_param, :forget_gate_weight_filler, weight_filler) # TODO: make separate fillers
	set_field!(lstm_unit_param, :output_gate_weight_filler, weight_filler) # TODO: make separate fillers
	set_field!(layer, :lstm_unit_param, lstm_unit_param)
	return Layer(name, bottoms, tops, :lstm_unit_param, layer, 0)
end

@doc """
The WordvecLayer turns positive integers (indexes) between 0 and `vocab_size - 1` into dense vectors of fixed size `dimension`. The input is of size `n` where `n` is the batchsize and the output is of size `n * dimension`.
""" ->
function WordvecLayer(name::Tag, bottoms::Vector{Tag};
		dimension::Int=1, vocab_size::Int=1, weight_filler::FillerParameter=filler(:uniform, min=-0.1, max=0.1), param_names=Tag[])
	layer = make_layer("Wordvec", name, bottoms, [name]; param_names=param_names)
	wordvec_param = WordvecParameter()
	set_field!(wordvec_param, :dimension, convert(UInt32, dimension))
	set_field!(wordvec_param, :vocab_size, convert(UInt32, vocab_size))
	set_field!(wordvec_param, :weight_filler, weight_filler)
	set_field!(layer, :wordvec_param, wordvec_param)
	return Layer(name, bottoms, [name], :wordvec_param, layer, 0)
end

function EmbedLayer(name::Tag, bottoms::Vector{Tag};
		input_dim::Int=1, num_output::Int=1)
	layer = make_layer("Embed", name, bottoms, [name])
	embed_param = EmbedParameter()
	set_field!(embed_param, :input_dim, convert(UInt32, input_dim))
	set_field!(embed_param, :num_output, convert(UInt32, num_output))
	set_field!(layer, :embed_param, embed_param)
	return Layer(name, bottoms, [name], :embed_param, layer, 0)
end

@doc """
The Concat layer is a utility layer that concatenates its multiple input blobs to one single output blob. It takes one keyword parameter, `axis`. The input shape of the bottoms are `n_i * c_i * h * w` for `i = 1, ..., K` and the output shape is

* `(n_1 + n_2 + ... + n_K) * c_1 * h * w` if `axis = 0` in which case all `c_i` should be the same and

* `n_1 * (c_1 + c_2 + ... + c_K) * h * w` if `axis = 1` in which case all `n_i` should be the same.
""" ->
function ConcatLayer(name::Tag, bottoms::Vector{Tag}; axis::Int=1)
	layer = make_layer("Concat", name, bottoms, [name])
	concat_param = ConcatParameter()
	set_field!(concat_param, :axis, convert(Int32, axis))
	set_field!(layer, :concat_param, concat_param)
	return Layer(name, bottoms, [name], :concat_param, layer, 0)
end

@doc """
The softmax loss layer computes the multinomial logistic loss of the softmax of its inputs. It’s conceptually identical to a softmax layer followed by a multinomial logistic loss layer, but provides a more numerically stable gradient. Its parameters are

* `ignore_label` (default -1): Label does not contribute to the loss

This layer expects two bottom blobs, the actual data of size `n * c * h * w` and a label of size `n * 1 * 1 * 1`.
""" ->
function SoftmaxWithLoss(name::Tag, bottoms::Vector{Tag}; axis::Int=-1, ignore_label::Int=-1)
	layer = make_layer("SoftmaxWithLoss", name, bottoms, [name])
	loss_param = LossParameter()
	set_field!(loss_param, :ignore_label, convert(Int32, ignore_label))
	set_field!(layer, :loss_weight, [convert(Float32, 1.0)])
	set_field!(layer, :loss_param, loss_param)
	softmax_param = SoftmaxParameter()
	if axis != -1
		set_field!(softmax_param, :axis, convert(Int32, axis))
		set_field!(layer, :softmax_param, softmax_param)
	end
	return Layer(name, bottoms, [name], :softmax_param, layer, 0)
end

@doc """
Computes the softmax of the input. The parameter `axis` specifies which axis the softmax is computed over.
""" ->
function Softmax(name::Tag, bottoms::Vector{Tag}; axis::Int=-1)
	layer = make_layer("Softmax", name, bottoms, [name])
	softmax_param = SoftmaxParameter()
	if axis != -1
		set_field!(softmax_param, :axis, convert(Int32, axis))
		set_field!(layer, :softmax_param, softmax_param)
	end
	return Layer(name, bottoms, [name], :softmax_param, layer, 0)
end

function HingeLoss(name::Tag, bottoms::Vector{Tag})
	layer = make_layer("HingeLoss", name, bottoms, [name])
	return Layer(name, bottoms, [name], :hinge_loss_param, layer, 0)
end

@doc "The Euclidean loss layer computes the sum of squares of differences of its two inputs" ->
function EuclideanLoss(name::Tag, bottoms::Vector{Tag})
	layer = make_layer("EuclideanLoss", name, bottoms, [name])
	return Layer(name, bottoms, [name], :euclidean_loss_param, layer, 0)
end

@doc """The MemoryLayer presents data to Caffe through a pointer (it is implemented as a new Caffe Layer called PointerData), which can be set using `set_data!` method of `CaffeNet`. It is the preferred way to fill `CaffeNet` with data. As each MemoryLayer provides exactly one top blob, you will typically have multiple of these (in the supervised setting, one for labels and one for images for example). In `set_data!`, you can specify with an integer index which of the layers will be filled with the data provided.
""" ->
function MemoryLayer{N}(name::Tag; shape::NTuple{N, Int}=(1,1,1,1), stream_axis::Int=-1)
	layer = make_layer("PointerData", name, Tag[], [name])
	memory_data_param = PointerDataParameter()
	shape_param = BlobShape()
	set_field!(shape_param, :dim, [shape...])
	set_field!(memory_data_param, :shape, shape_param)
	set_field!(layer, :pointer_data_param, memory_data_param)
	return Layer(name, Tag[], [name], :pointer_data_param, layer, stream_axis, true)
end

@doc """The DataLayer makes it easy to propagate data through the network while doing computation. The data is being stored in Google Protocol Buffers and transferred to Caffe in this way. Its only keyword argument is `data` which is an array that will be presented to the next layer through the top blob. It is meant to be used only with `ApolloNet`s.
""" ->
function DataLayer{N}(name::Tag; data::Array{Float32, N}=zeros(Float32, tuple(zeros(Int, N)...)))
	layer = make_layer("NumpyData", name, Tag[], [name])
	numpy_data_param = NumpyDataParameter()
	set_field!(numpy_data_param, :shape, uint32([size(data)...]))
	set_field!(numpy_data_param, :data, data[:])
	runtime_param = RuntimeParameter()
	set_field!(runtime_param, :numpy_data_param, numpy_data_param)
	set_field!(layer, :rp, runtime_param)
	return Layer(name, Tag[], [name], :rp, layer, -1, true)
end
