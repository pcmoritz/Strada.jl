module Strada

export
	Layer,
	MemoryLayer,
	DataLayer,
	ConvLayer,
	PoolLayer,
	LinearLayer,
	ActivationLayer,
	ReLU,
	Sigmoid,
	LRNLayer,
	WITHIN_CHANNEL,
	ACROSS_CHANNELS,
	ConcatLayer,
	RecurrentLayer,
	WordvecLayer,
	EmbedLayer,
	LstmLayer,
	SoftmaxWithLoss,
	Softmax,
	HingeLoss,
	EuclideanLoss,
	DropoutLayer,
	filler,
	CaffeNet,
	Net,
	set_mode_cpu,
	set_mode_gpu,
	forward,
	backward,
	reset,
	reverse,
	set_data!,
	layers,
	blobs,
	Data,
	copy!,
	zero!,
	length,
	DataStream,
	minibatch_stream,
	num_batches,
	get_batchsize,
	getminibatch,
	sgd,
	InvLR,
	ConstLR,
	grad_check,
	make_objective,
	make_predictor,
	load_caffemodel,
	read_svmlight

import Base: start, done, next, size, reset
import Logging.info
using Lint
using Compat

using Docile

@docstrings

function include_without_linting(filename)
	eval(parse("include(\"$filename\")"))
end

include_without_linting("caffe_pb.jl")
include("stream.jl")
include("delegate.jl")
include("error.jl")
include("layers.jl")
include("blobs.jl")
include("apollonet.jl")
include("caffenet.jl")
include("prettyprint.jl")
include("objective.jl")
include("gradcheck.jl")
include("svmlight.jl")
include("utils.jl")
include("sgd.jl")

libpath = joinpath(Pkg.dir("Strada"), "deps", "usr", "lib")
@linux? Libdl.dlopen(joinpath(libpath, "libjlcaffe.so")) : dlopen(joinpath(libpath, "libjlcaffe.so.dylib"))

# load blas and export symbols
Libdl.dlopen(Base.libblas_name, Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND | Libdl.RTLD_GLOBAL)

# Initialize caffe
ccall((:init_jlcaffe, "libjlcaffe.so"), Void, ())
ccall((:set_global_error_callback, "libjlcaffe.so"), Void, (Ptr{Void},), error_callback_c)

end # module
