# Strada

## Exported

---

<a id="method__activationlayer.1" class="lexicon_definition"></a>
#### ActivationLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1}) [¶](#method__activationlayer.1)
Activation layers compute element-wise function, taking one bottom blob as input and producing one top blob of the same size. Parameters are

* `activation` (default `Sigmoid`): The nonlinear function applied. Can be `ReLU`, `Sigmoid` or `TanH`.

Both input and output are of shape `n * c * h * w`.


*source:*
[Strada/src/layers.jl:182](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L182)

---

<a id="method__concatlayer.1" class="lexicon_definition"></a>
#### ConcatLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1}) [¶](#method__concatlayer.1)
The Concat layer is a utility layer that concatenates its multiple input blobs to one single output blob. It takes one keyword parameter, `axis`. The input shape of the bottoms are `n_i * c_i * h * w` for `i = 1, ..., K` and the output shape is

* `(n_1 + n_2 + ... + n_K) * c_1 * h * w` if `axis = 0` in which case all `c_i` should be the same and

* `n_1 * (c_1 + c_2 + ... + c_K) * h * w` if `axis = 1` in which case all `n_i` should be the same.


*source:*
[Strada/src/layers.jl:284](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L284)

---

<a id="method__convlayer.1" class="lexicon_definition"></a>
#### ConvLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1}) [¶](#method__convlayer.1)
The Convolution layer convolves the input image with a set of learnable filters, each producing one feature map in the output image. keyword parameters are

* `n_filter`: The number of filters (required)

* `kernel`: A tuple specifying height and width of each filter (required)

* `stride`: A tuple which specifies the intervals at which to apply the filters to the input (horizontally and vertically)

* `pad`: A tuple which specifies the number of pixels to (implicitly) add to each side of the input

* `group` (default 1): If g > 1, we restrict the connectivity of each filter to a subset of the input. Specifically, the input and output channels are separated into g groups, and the ith output group channels will be only connected to the ith input group channels.

The input is of shape `n * c_i * h_i * w_i` and the output is of shape `n * c_o * h_o * w_o`, where `h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1` and `w_o` likewise.



*source:*
[Strada/src/layers.jl:96](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L96)

---

<a id="method__datalayer.1" class="lexicon_definition"></a>
#### DataLayer(name::ASCIIString) [¶](#method__datalayer.1)
The DataLayer makes it easy to propagate data through the network while doing computation. The data is being stored in Google Protocol Buffers and transferred to Caffe in this way. Its only keyword argument is `data` which is an array that will be presented to the next layer through the top blob. It is meant to be used only with `ApolloNet`s.


*source:*
[Strada/src/layers.jl:351](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L351)

---

<a id="method__dropoutlayer.1" class="lexicon_definition"></a>
#### DropoutLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1}) [¶](#method__dropoutlayer.1)
The dropout layer is a regularizer that randomly sets input values to zero.


*source:*
[Strada/src/layers.jl:217](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L217)

---

<a id="method__euclideanloss.1" class="lexicon_definition"></a>
#### EuclideanLoss(name::ASCIIString, bottoms::Array{ASCIIString, 1}) [¶](#method__euclideanloss.1)
The Euclidean loss layer computes the sum of squares of differences of its two inputs

*source:*
[Strada/src/layers.jl:332](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L332)

---

<a id="method__lrnlayer.1" class="lexicon_definition"></a>
#### LRNLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1}) [¶](#method__lrnlayer.1)
The local response normalization layer performs a kind of “lateral inhibition” by normalizing over local input regions. In `ACROSS_CHANNELS` mode, the local regions extend across nearby channels, but have no spatial extent (i.e., they have shape `local_size` x 1 x 1). In `WITHIN_CHANNEL` mode, the local regions extend spatially, but are in separate channels (i.e., they have shape 1 x `local_size` x `local_size`). Each input value is divided by `(1+(α/n) sum_i x_i^2)β)`, where n is the size of each local region, and the sum is taken over the region centered at that value (zero padding is added where necessary). It accepts the following keyword arguments:

* `local_size` (default 3): Size of the region the normalization is computed over

* `alpha` (default `5e-5`): Value of the parameter α

* `beta` (default `0.75`): Value of the parameter β

* `norm_region`: Mode of the local contrast normalization. Can be `ACROSS_CHANNELS` or `WITHIN_CHANNEL`. 


*source:*
[Strada/src/layers.jl:202](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L202)

---

<a id="method__linearlayer.1" class="lexicon_definition"></a>
#### LinearLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1}) [¶](#method__linearlayer.1)
The InnerProduct layer (also usually referred to as the fully connected layer) treats the input as a simple vector and produces an output in the form of a single vector (with the blob’s height and width set to 1). The keyword parameters are

* `n_filter`: The number of filters (required)

The input is of shape `n * c_i * h_i * w_i` and the output of shape `n * c_o * 1 * 1`.



*source:*
[Strada/src/layers.jl:125](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L125)

---

<a id="method__lstmlayer.1" class="lexicon_definition"></a>
#### LstmLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1}) [¶](#method__lstmlayer.1)
The LstmLayer is an LSTM unit. It takes two blobs as input, the current LSTM input and the previous memory cell content. It outputs the new hidden state and the updated memory cell.


*source:*
[Strada/src/layers.jl:240](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L240)

---

<a id="method__lstmlayer.2" class="lexicon_definition"></a>
#### LstmLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1}, tops::Array{ASCIIString, 1}) [¶](#method__lstmlayer.2)
The LstmLayer is an LSTM unit. It takes two blobs as input, the current LSTM input and the previous memory cell content. It outputs the new hidden state and the updated memory cell.


*source:*
[Strada/src/layers.jl:240](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L240)

---

<a id="method__memorylayer.1" class="lexicon_definition"></a>
#### MemoryLayer(name::ASCIIString) [¶](#method__memorylayer.1)
The MemoryLayer presents data to Caffe through a pointer (it is implemented as a new Caffe Layer called PointerData), which can be set using `set_data!` method of `CaffeNet`. It is the preferred way to fill `CaffeNet` with data. As each MemoryLayer provides exactly one top blob, you will typically have multiple of these (in the supervised setting, one for labels and one for images for example). In `set_data!`, you can specify with an integer index which of the layers will be filled with the data provided.


*source:*
[Strada/src/layers.jl:339](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L339)

---

<a id="method__net.1" class="lexicon_definition"></a>
#### Net(name::ASCIIString) [¶](#method__net.1)
Create an empty ApolloNet. A log_level of 0 prints full caffe debug information, a log_level of 3 prints nothing.

*source:*
[Strada/src/apollonet.jl:11](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/apollonet.jl#L11)

---

<a id="method__net.2" class="lexicon_definition"></a>
#### Net(name::ASCIIString, layers::Array{Layer, 1}) [¶](#method__net.2)
Load a model from a caffe compatible .caffemodel file (for example from the caffe model zoo).

*source:*
[Strada/src/caffenet.jl:22](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/caffenet.jl#L22)

---

<a id="method__poollayer.1" class="lexicon_definition"></a>
#### PoolLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1}) [¶](#method__poollayer.1)
The PoolLayer partitions the input image into a set of non-overlapping rectangles and, for each such sub-region, outputs the maximum or average value. The keyword parameters are

* `method` (default `MAX`): The pooling method. Can be `MAX`, `AVE` or `STOCHASTIC`.

* `pad` (default 0): Specifies the number of pixels to (implicitly) add to each side of the input

* `stride` (default 1): Specifies the intervals at which to apply the filters to the input

The input is of shape `n * c * h_i * w_i` and the output of shape `n * c * h_o * w_o` where `h_o` and `w_o` are computed in the same way as for the convolution.


*source:*
[Strada/src/layers.jl:151](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L151)

---

<a id="method__softmax.1" class="lexicon_definition"></a>
#### Softmax(name::ASCIIString, bottoms::Array{ASCIIString, 1}) [¶](#method__softmax.1)
Computes the softmax of the input. The parameter `axis` specifies which axis the softmax is computed over.


*source:*
[Strada/src/layers.jl:316](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L316)

---

<a id="method__softmaxwithloss.1" class="lexicon_definition"></a>
#### SoftmaxWithLoss(name::ASCIIString, bottoms::Array{ASCIIString, 1}) [¶](#method__softmaxwithloss.1)
The softmax loss layer computes the multinomial logistic loss of the softmax of its inputs. It’s conceptually identical to a softmax layer followed by a multinomial logistic loss layer, but provides a more numerically stable gradient. Its parameters are

* `ignore_label` (default -1): Label does not contribute to the loss

This layer expects two bottom blobs, the actual data of size `n * c * h * w` and a label of size `n * 1 * 1 * 1`.


*source:*
[Strada/src/layers.jl:299](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L299)

---

<a id="method__wordveclayer.1" class="lexicon_definition"></a>
#### WordvecLayer(name::ASCIIString, bottoms::Array{ASCIIString, 1}) [¶](#method__wordveclayer.1)
The WordvecLayer turns positive integers (indexes) between 0 and `vocab_size - 1` into dense vectors of fixed size `dimension`. The input is of size `n` where `n` is the batchsize and the output is of size `n * dimension`.


*source:*
[Strada/src/layers.jl:256](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L256)

---

<a id="method__backward.1" class="lexicon_definition"></a>
#### backward(net::ApolloNet) [¶](#method__backward.1)
Run a backward pass through the whole network.

*source:*
[Strada/src/apollonet.jl:34](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/apollonet.jl#L34)

---

<a id="method__backward.2" class="lexicon_definition"></a>
#### backward(net::CaffeNet) [¶](#method__backward.2)
Run a backward pass through the whole network.

*source:*
[Strada/src/caffenet.jl:79](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/caffenet.jl#L79)

---

<a id="method__copy.1" class="lexicon_definition"></a>
#### copy!(output::ApolloDict, input::Array{T, 1}) [¶](#method__copy.1)
Copy a flat parameter vector into a binary blob.

*source:*
[Strada/src/blobs.jl:160](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/blobs.jl#L160)

---

<a id="method__copy.2" class="lexicon_definition"></a>
#### copy!(output::Array{T, 1}, input::ApolloDict) [¶](#method__copy.2)
Copy a binary blob into a flat parameter vector.

*source:*
[Strada/src/blobs.jl:150](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/blobs.jl#L150)

---

<a id="method__copy.3" class="lexicon_definition"></a>
#### copy!(output::Array{T, 1}, input::CaffeDict) [¶](#method__copy.3)
Copy a binary blob into a flat parameter vector.

*source:*
[Strada/src/blobs.jl:30](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/blobs.jl#L30)

---

<a id="method__copy.4" class="lexicon_definition"></a>
#### copy!(output::CaffeDict, input::Array{T, 1}) [¶](#method__copy.4)
Copy a flat parameter vector into a binary blob.

*source:*
[Strada/src/blobs.jl:42](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/blobs.jl#L42)

---

<a id="method__filler.1" class="lexicon_definition"></a>
#### filler(name::Symbol) [¶](#method__filler.1)
Fillers are random number generators that fills a blob using the specified algorithm. The algorithm is specified by

* `name`: Can be `:gaussian`, `:uniform`, `:xavier` or `:constant`

The parameters are given by keyword arguments:

* `value`: Gives the value for a constant filler

* `min` and `max`: Range for a uniform filler

* `mean` and `std`: Mean and standard deviation of a Gaussian filler



*source:*
[Strada/src/layers.jl:64](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/layers.jl#L64)

---

<a id="method__forward.1" class="lexicon_definition"></a>
#### forward(net::ApolloNet, layer::Layer) [¶](#method__forward.1)
Run a forward pass of a single layer.

*source:*
[Strada/src/apollonet.jl:25](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/apollonet.jl#L25)

---

<a id="method__forward.2" class="lexicon_definition"></a>
#### forward(net::CaffeNet) [¶](#method__forward.2)
Run a forward pass through the whole network.

*source:*
[Strada/src/caffenet.jl:74](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/caffenet.jl#L74)

---

<a id="method__get_batchsize.1" class="lexicon_definition"></a>
#### get_batchsize(str::MinibatchStream) [¶](#method__get_batchsize.1)
Batchsize of the MinibatchStream

*source:*
[Strada/src/stream.jl:70](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/stream.jl#L70)

---

<a id="method__getminibatch.1" class="lexicon_definition"></a>
#### getminibatch(str::MinibatchStream) [¶](#method__getminibatch.1)
Get a random minibatch from the MinibatchStream

*source:*
[Strada/src/stream.jl:75](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/stream.jl#L75)

---

<a id="method__grad_check.1" class="lexicon_definition"></a>
#### grad_check{F}(objective::Function, theta::Array{F, 1}, data, epsilon::Float64) [¶](#method__grad_check.1)
Check gradients using symmetric finite differences. See the tests for example how to run.

*source:*
[Strada/src/gradcheck.jl:11](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/gradcheck.jl#L11)

---

<a id="method__length.1" class="lexicon_definition"></a>
#### length(blob::ApolloDict) [¶](#method__length.1)
The total number of variables in a binary blob.

*source:*
[Strada/src/blobs.jl:141](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/blobs.jl#L141)

---

<a id="method__length.2" class="lexicon_definition"></a>
#### length(blob::CaffeDict) [¶](#method__length.2)
Number of parameters in the blob.

*source:*
[Strada/src/blobs.jl:19](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/blobs.jl#L19)

---

<a id="method__load_caffemodel.1" class="lexicon_definition"></a>
#### load_caffemodel(net::CaffeNet, filename::String) [¶](#method__load_caffemodel.1)
Load a model from a caffe compatible .caffemodel file (for example from the caffe model zoo).

*source:*
[Strada/src/caffenet.jl:56](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/caffenet.jl#L56)

---

<a id="method__minibatch_stream.1" class="lexicon_definition"></a>
#### minibatch_stream(args::AbstractArray{T, N}...) [¶](#method__minibatch_stream.1)
Construct a MinibatchStream from a tuple of data arrays with full batch size.

*source:*
[Strada/src/stream.jl:46](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/stream.jl#L46)

---

<a id="method__num_batches.1" class="lexicon_definition"></a>
#### num_batches(str::MinibatchStream) [¶](#method__num_batches.1)
Number of minibatches in the MinibatchStream

*source:*
[Strada/src/stream.jl:60](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/stream.jl#L60)

---

<a id="method__read_svmlight.1" class="lexicon_definition"></a>
#### read_svmlight(filename::String) [¶](#method__read_svmlight.1)
Load a dataset from the libsvm compatible svmlight file format into a sparse matrix.

*source:*
[Strada/src/svmlight.jl:3](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/svmlight.jl#L3)

---

<a id="method__read_svmlight.2" class="lexicon_definition"></a>
#### read_svmlight(filename::String, Dtype::DataType) [¶](#method__read_svmlight.2)
Load a dataset from the libsvm compatible svmlight file format into a sparse matrix.

*source:*
[Strada/src/svmlight.jl:3](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/svmlight.jl#L3)

---

<a id="method__reset.1" class="lexicon_definition"></a>
#### reset(net::ApolloNet) [¶](#method__reset.1)
Clear the active layers and active parameters of the net so a new forward pass can be run.

*source:*
[Strada/src/apollonet.jl:19](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/apollonet.jl#L19)

---

<a id="method__set_mode_cpu.1" class="lexicon_definition"></a>
#### set_mode_cpu(net::ApolloNet) [¶](#method__set_mode_cpu.1)
Activate CPU mode.

*source:*
[Strada/src/apollonet.jl:39](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/apollonet.jl#L39)

---

<a id="method__set_mode_cpu.2" class="lexicon_definition"></a>
#### set_mode_cpu(net::CaffeNet) [¶](#method__set_mode_cpu.2)
Activate CPU mode.

*source:*
[Strada/src/caffenet.jl:44](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/caffenet.jl#L44)

---

<a id="method__set_mode_gpu.1" class="lexicon_definition"></a>
#### set_mode_gpu(net::ApolloNet) [¶](#method__set_mode_gpu.1)
Activate GPU mode.

*source:*
[Strada/src/apollonet.jl:44](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/apollonet.jl#L44)

---

<a id="method__set_mode_gpu.2" class="lexicon_definition"></a>
#### set_mode_gpu(net::CaffeNet) [¶](#method__set_mode_gpu.2)
Activate CPU mode.

*source:*
[Strada/src/caffenet.jl:50](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/caffenet.jl#L50)

---

<a id="method__sgd.1" class="lexicon_definition"></a>
#### sgd{F}(objective!::Function, data::DataStream, theta::Array{F, 1}) [¶](#method__sgd.1)
Run the stochastic gradient descent method on the objective. If a testset is provided, generalization performance will also periodically be evaluated.

*source:*
[Strada/src/sgd.jl:22](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/sgd.jl#L22)

---

<a id="method__zero.1" class="lexicon_definition"></a>
#### zero!(A::ApolloDict) [¶](#method__zero.1)
Fill a binary blob with zeros.

*source:*
[Strada/src/blobs.jl:170](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/blobs.jl#L170)

---

<a id="method__zero.2" class="lexicon_definition"></a>
#### zero!(A::CaffeDict) [¶](#method__zero.2)
Fill a binary blob with zeros.

*source:*
[Strada/src/blobs.jl:54](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/blobs.jl#L54)

---

<a id="type__datastream.1" class="lexicon_definition"></a>
#### DataStream [¶](#type__datastream.1)
A data stream represents a data source for a neural network. It could be data held in memory, in a database on disk, or a network socket for example.


*source:*
[Strada/src/stream.jl:9](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/stream.jl#L9)

---

<a id="typealias__data.1" class="lexicon_definition"></a>
#### Data [¶](#typealias__data.1)
Data to be fed into a CaffeNet is kept in a `Data{F,N}` structure where `F` is the type of floating point number used to store the data (Float32 or Float64) and `N` is the number of data layers of the network. We represent Data{F,N} as a tuple, where the dimension i holds data that will be fed into data layer i of the network. A canonical example is for supervised learning, where N is 2, the first component representing the image (say) and the second component representing the label. Each dimension of the tuple typically holds an array whose last dimension corresponds to the index in the minibatch.


*source:*
[Strada/src/stream.jl:4](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/stream.jl#L4)

## Internal

---

<a id="method__calc_full_gradient.1" class="lexicon_definition"></a>
#### calc_full_gradient{F}(objective!::Function, data::DataStream, theta::Array{F, 1}, grad::Array{F, 1}) [¶](#method__calc_full_gradient.1)
Calculate the full gradient of the model at parameters `theta` over the dataset `data`. The gradient will be stored in `grad`.

*source:*
[Strada/src/utils.jl:43](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/utils.jl#L43)

---

<a id="method__calc_full_prediction.1" class="lexicon_definition"></a>
#### calc_full_prediction{F}(predictor::Function, data::DataStream, theta::Array{F, 1}) [¶](#method__calc_full_prediction.1)
Calculate prediction performance of the model with parameters `theta` over a whole dataset `data`

*source:*
[Strada/src/utils.jl:57](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/utils.jl#L57)

---

<a id="method__size.1" class="lexicon_definition"></a>
#### size(str::MinibatchStream) [¶](#method__size.1)
Number of datapoints in the MinibatchStream

*source:*
[Strada/src/stream.jl:65](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/stream.jl#L65)

---

<a id="type__apollodict.1" class="lexicon_definition"></a>
#### ApolloDict [¶](#type__apollodict.1)
An ApolloDict is a collection of blobs with names, each name is associated with
one floating point array. Example: The name 'ip_weights' could map to the weights of a linear layer.

*source:*
[Strada/src/blobs.jl:71](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/blobs.jl#L71)

---

<a id="type__caffedict.1" class="lexicon_definition"></a>
#### CaffeDict [¶](#type__caffedict.1)
A CaffeDict is a collection of blobs with names, each name is associated with
an arbitrary number of floating point arrays. Example: The name 'conv1' could
map to a vector containing the biases and weights of a convolution.

*source:*
[Strada/src/blobs.jl:7](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/blobs.jl#L7)

---

<a id="type__emptystream.1" class="lexicon_definition"></a>
#### EmptyStream [¶](#type__emptystream.1)
An empty stream represents a data source with no data.


*source:*
[Strada/src/stream.jl:14](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/stream.jl#L14)

---

<a id="type__minibatchstream.1" class="lexicon_definition"></a>
#### MinibatchStream [¶](#type__minibatchstream.1)
A MinibatchStream is a collection of data represented in memory that has been partitioned into minibatches.


*source:*
[Strada/src/stream.jl:25](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/stream.jl#L25)

---

<a id="type__netdata.1" class="lexicon_definition"></a>
#### NetData{D} [¶](#type__netdata.1)
A collection of blobs in a network. Grouped into 'data' (the actual parameters)
and 'diff' (the gradients) so they can be treated as vectors that can be added together.

*source:*
[Strada/src/blobs.jl:64](https://github.com/pcmoritz/Strada.jl/tree/e5894dea6e68013b0cea9a57fd518cad4fdc05b4/src/blobs.jl#L64)

