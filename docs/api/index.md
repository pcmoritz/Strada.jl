# API-INDEX


## MODULE: Strada

---

## Methods [Exported]

[ActivationLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1})](Strada.md#method__activationlayer.1)  Activation layers compute element-wise function, taking one bottom blob as input and producing one top blob of the same size. Parameters are

[ConcatLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1})](Strada.md#method__concatlayer.1)  The Concat layer is a utility layer that concatenates its multiple input blobs to one single output blob. It takes one keyword parameter, `axis`. The input shape of the bottoms are `n_i * c_i * h * w` for `i = 1, ..., K` and the output shape is

[ConvLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1})](Strada.md#method__convlayer.1)  The Convolution layer convolves the input image with a set of learnable filters, each producing one feature map in the output image. keyword parameters are

[DataLayer(name::ASCIIString)](Strada.md#method__datalayer.1)  The DataLayer makes it easy to propagate data through the network while doing computation. The data is being stored in Google Protocol Buffers and transferred to Caffe in this way. Its only keyword argument is `data` which is an array that will be presented to the next layer through the top blob. It is meant to be used only with `ApolloNet`s.

[DropoutLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1})](Strada.md#method__dropoutlayer.1)  The dropout layer is a regularizer that randomly sets input values to zero.

[EuclideanLoss(name::ASCIIString, bottoms::Array{ASCIIString, 1})](Strada.md#method__euclideanloss.1)  The Euclidean loss layer computes the sum of squares of differences of its two inputs

[LRNLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1})](Strada.md#method__lrnlayer.1)  The local response normalization layer performs a kind of “lateral inhibition” by normalizing over local input regions. In `ACROSS_CHANNELS` mode, the local regions extend across nearby channels, but have no spatial extent (i.e., they have shape `local_size` x 1 x 1). In `WITHIN_CHANNEL` mode, the local regions extend spatially, but are in separate channels (i.e., they have shape 1 x `local_size` x `local_size`). Each input value is divided by `(1+(α/n) sum_i x_i^2)β)`, where n is the size of each local region, and the sum is taken over the region centered at that value (zero padding is added where necessary). It accepts the following keyword arguments:

[LinearLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1})](Strada.md#method__linearlayer.1)  The InnerProduct layer (also usually referred to as the fully connected layer) treats the input as a simple vector and produces an output in the form of a single vector (with the blob’s height and width set to 1). The keyword parameters are

[LstmLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1})](Strada.md#method__lstmlayer.1)  The LstmLayer is an LSTM unit. It takes two blobs as input, the current LSTM input and the previous memory cell content. It outputs the new hidden state and the updated memory cell.

[LstmLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1}, tops::Array{ASCIIString, 1})](Strada.md#method__lstmlayer.2)  The LstmLayer is an LSTM unit. It takes two blobs as input, the current LSTM input and the previous memory cell content. It outputs the new hidden state and the updated memory cell.

[MemoryLayer(name::ASCIIString)](Strada.md#method__memorylayer.1)  The MemoryLayer presents data to Caffe through a pointer (it is implemented as a new Caffe Layer called PointerData), which can be set using `set_data!` method of `CaffeNet`. It is the preferred way to fill `CaffeNet` with data. As each MemoryLayer provides exactly one top blob, you will typically have multiple of these (in the supervised setting, one for labels and one for images for example). In `set_data!`, you can specify with an integer index which of the layers will be filled with the data provided.

[Net(name::ASCIIString)](Strada.md#method__net.1)  Create an empty ApolloNet. A log_level of 0 prints full caffe debug information, a log_level of 3 prints nothing.

[Net(name::ASCIIString, layers::Array{Layer, 1})](Strada.md#method__net.2)  Load a model from a caffe compatible .caffemodel file (for example from the caffe model zoo).

[PoolLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1})](Strada.md#method__poollayer.1)  The PoolLayer partitions the input image into a set of non-overlapping rectangles and, for each such sub-region, outputs the maximum or average value. The keyword parameters are

[Softmax(name::ASCIIString, bottoms::Array{ASCIIString, 1})](Strada.md#method__softmax.1)  Computes the softmax of the input. The parameter `axis` specifies which axis the softmax is computed over.

[SoftmaxWithLoss(name::ASCIIString, bottoms::Array{ASCIIString, 1})](Strada.md#method__softmaxwithloss.1)  The softmax loss layer computes the multinomial logistic loss of the softmax of its inputs. It’s conceptually identical to a softmax layer followed by a multinomial logistic loss layer, but provides a more numerically stable gradient. Its parameters are

[WordvecLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1})](Strada.md#method__wordveclayer.1)  The WordvecLayer turns positive integers (indexes) between 0 and `vocab_size - 1` into dense vectors of fixed size `dimension`. The input is of size `n` where `n` is the batchsize and the output is of size `n * dimension`.

[backward(net::ApolloNet)](Strada.md#method__backward.1)  Run a backward pass through the whole network.

[backward(net::CaffeNet)](Strada.md#method__backward.2)  Run a backward pass through the whole network.

[copy!(output::ApolloDict, input::Array{T, 1})](Strada.md#method__copy.1)  Copy a flat parameter vector into a binary blob.

[copy!(output::Array{T, 1}, input::ApolloDict)](Strada.md#method__copy.2)  Copy a binary blob into a flat parameter vector.

[copy!(output::Array{T, 1}, input::CaffeDict)](Strada.md#method__copy.3)  Copy a binary blob into a flat parameter vector.

[copy!(output::CaffeDict, input::Array{T, 1})](Strada.md#method__copy.4)  Copy a flat parameter vector into a binary blob.

[filler(name::Symbol)](Strada.md#method__filler.1)  Fillers are random number generators that fills a blob using the specified algorithm. The algorithm is specified by

[forward(net::ApolloNet, layer::Layer)](Strada.md#method__forward.1)  Run a forward pass of a single layer.

[forward(net::CaffeNet)](Strada.md#method__forward.2)  Run a forward pass through the whole network.

[get_batchsize(str::MinibatchStream)](Strada.md#method__get_batchsize.1)  Batchsize of the MinibatchStream

[getminibatch(str::MinibatchStream)](Strada.md#method__getminibatch.1)  Get a random minibatch from the MinibatchStream

[grad_check{F}(objective::Function, theta::Array{F, 1}, data, epsilon::Float64)](Strada.md#method__grad_check.1)  Check gradients using symmetric finite differences. See the tests for example how to run.

[length(blob::ApolloDict)](Strada.md#method__length.1)  The total number of variables in a binary blob.

[length(blob::CaffeDict)](Strada.md#method__length.2)  Number of parameters in the blob.

[load_caffemodel(net::CaffeNet, filename::String)](Strada.md#method__load_caffemodel.1)  Load a model from a caffe compatible .caffemodel file (for example from the caffe model zoo).

[minibatch_stream(args::AbstractArray{T, N}...)](Strada.md#method__minibatch_stream.1)  Construct a MinibatchStream from a tuple of data arrays with full batch size.

[num_batches(str::MinibatchStream)](Strada.md#method__num_batches.1)  Number of minibatches in the MinibatchStream

[read_svmlight(filename::String)](Strada.md#method__read_svmlight.1)  Load a dataset from the libsvm compatible svmlight file format into a sparse matrix.

[read_svmlight(filename::String, Dtype::DataType)](Strada.md#method__read_svmlight.2)  Load a dataset from the libsvm compatible svmlight file format into a sparse matrix.

[reset(net::ApolloNet)](Strada.md#method__reset.1)  Clear the active layers and active parameters of the net so a new forward pass can be run.

[set_mode_cpu(net::ApolloNet)](Strada.md#method__set_mode_cpu.1)  Activate CPU mode.

[set_mode_cpu(net::CaffeNet)](Strada.md#method__set_mode_cpu.2)  Activate CPU mode.

[set_mode_gpu(net::ApolloNet)](Strada.md#method__set_mode_gpu.1)  Activate GPU mode.

[set_mode_gpu(net::CaffeNet)](Strada.md#method__set_mode_gpu.2)  Activate CPU mode.

[sgd{F}(objective!::Function, data::DataStream, theta::Array{F, 1})](Strada.md#method__sgd.1)  Run the stochastic gradient descent method on the objective. If a testset is provided, generalization performance will also periodically be evaluated.

[zero!(A::ApolloDict)](Strada.md#method__zero.1)  Fill a binary blob with zeros.

[zero!(A::CaffeDict)](Strada.md#method__zero.2)  Fill a binary blob with zeros.

---

## Types [Exported]

[DataStream](Strada.md#type__datastream.1)  A data stream represents a data source for a neural network. It could be data held in memory, in a database on disk, or a network socket for example.

---

## Typealiass [Exported]

[Data](Strada.md#typealias__data.1)  Data to be fed into a CaffeNet is kept in a `Data{F,N}` structure where `F` is the type of floating point number used to store the data (Float32 or Float64) and `N` is the number of data layers of the network. We represent Data{F,N} as a tuple, where the dimension i holds data that will be fed into data layer i of the network. A canonical example is for supervised learning, where N is 2, the first component representing the image (say) and the second component representing the label. Each dimension of the tuple typically holds an array whose last dimension corresponds to the index in the minibatch.

---

## Methods [Internal]

[calc_full_gradient{F}(objective!::Function, data::DataStream, theta::Array{F, 1}, grad::Array{F, 1})](Strada.md#method__calc_full_gradient.1)  Calculate the full gradient of the model at parameters `theta` over the dataset `data`. The gradient will be stored in `grad`.

[calc_full_prediction{F}(predictor::Function, data::DataStream, theta::Array{F, 1})](Strada.md#method__calc_full_prediction.1)  Calculate prediction performance of the model with parameters `theta` over a whole dataset `data`

[size(str::MinibatchStream)](Strada.md#method__size.1)  Number of datapoints in the MinibatchStream

---

## Types [Internal]

[ApolloDict](Strada.md#type__apollodict.1)  An ApolloDict is a collection of blobs with names, each name is associated with

[CaffeDict](Strada.md#type__caffedict.1)  A CaffeDict is a collection of blobs with names, each name is associated with

[EmptyStream](Strada.md#type__emptystream.1)  An empty stream represents a data source with no data.

[MinibatchStream](Strada.md#type__minibatchstream.1)  A MinibatchStream is a collection of data represented in memory that has been partitioned into minibatches.

[NetData{D}](Strada.md#type__netdata.1)  A collection of blobs in a network. Grouped into 'data' (the actual parameters)

