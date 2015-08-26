


## ActivationLayer

Activation layers compute element-wise function, taking one bottom blob as input and producing one top blob of the same size. Parameters are

* `activation` (default `Sigmoid`): The nonlinear function applied. Can be `ReLU`, `Sigmoid` or `TanH`.

Both input and output are of shape `n * c * h * w`.

[Strada/src/layers.jl:182](file:///home/pcmoritz/.julia/v0.3/Strada/src/layers.jl)



## ConcatLayer

The Concat layer is a utility layer that concatenates its multiple input blobs to one single output blob. It takes one keyword parameter, `axis`. The input shape of the bottoms are `n_i * c_i * h * w` for `i = 1, ..., K` and the output shape is

* `(n_1 + n_2 + ... + n_K) * c_1 * h * w` if `axis = 0` in which case all `c_i` should be the same and

* `n_1 * (c_1 + c_2 + ... + c_K) * h * w` if `axis = 1` in which case all `n_i` should be the same.

[Strada/src/layers.jl:284](file:///home/pcmoritz/.julia/v0.3/Strada/src/layers.jl)



## ConvLayer

The Convolution layer convolves the input image with a set of learnable filters, each producing one feature map in the output image. keyword parameters are

* `n_filter`: The number of filters (required)

* `kernel`: A tuple specifying height and width of each filter (required)

* `stride`: A tuple which specifies the intervals at which to apply the filters to the input (horizontally and vertically)

* `pad`: A tuple which specifies the number of pixels to (implicitly) add to each side of the input

* `group` (default 1): If g > 1, we restrict the connectivity of each filter to a subset of the input. Specifically, the input and output channels are separated into g groups, and the ith output group channels will be only connected to the ith input group channels.

The input is of shape `n * c_i * h_i * w_i` and the output is of shape `n * c_o * h_o * w_o`, where `h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1` and `w_o` likewise.


[Strada/src/layers.jl:96](file:///home/pcmoritz/.julia/v0.3/Strada/src/layers.jl)



## DataLayer

The DataLayer makes it easy to propagate data through the network while doing computation. The data is being stored in Google Protocol Buffers and transferred to Caffe in this way. Its only keyword argument is `data` which is an array that will be presented to the next layer through the top blob. It is meant to be used only with `ApolloNet`s.

[Strada/src/layers.jl:351](file:///home/pcmoritz/.julia/v0.3/Strada/src/layers.jl)



## DropoutLayer

The dropout layer is a regularizer that randomly sets input values to zero.

[Strada/src/layers.jl:217](file:///home/pcmoritz/.julia/v0.3/Strada/src/layers.jl)



## EuclideanLoss

The Euclidean loss layer computes the sum of squares of differences of its two inputs
[Strada/src/layers.jl:332](file:///home/pcmoritz/.julia/v0.3/Strada/src/layers.jl)



## LRNLayer

The local response normalization layer performs a kind of “lateral inhibition” by normalizing over local input regions. In `ACROSS_CHANNELS` mode, the local regions extend across nearby channels, but have no spatial extent (i.e., they have shape `local_size` x 1 x 1). In `WITHIN_CHANNEL` mode, the local regions extend spatially, but are in separate channels (i.e., they have shape 1 x `local_size` x `local_size`). Each input value is divided by `(1+(α/n) sum_i x_i^2)β)`, where n is the size of each local region, and the sum is taken over the region centered at that value (zero padding is added where necessary). It accepts the following keyword arguments:

* `local_size` (default 3): Size of the region the normalization is computed over

* `alpha` (default `5e-5`): Value of the parameter α

* `beta` (default `0.75`): Value of the parameter β

* `norm_region`: Mode of the local contrast normalization. Can be `ACROSS_CHANNELS` or `WITHIN_CHANNEL`. 

[Strada/src/layers.jl:202](file:///home/pcmoritz/.julia/v0.3/Strada/src/layers.jl)



## LinearLayer

The InnerProduct layer (also usually referred to as the fully connected layer) treats the input as a simple vector and produces an output in the form of a single vector (with the blob’s height and width set to 1). The keyword parameters are

* `n_filter`: The number of filters (required)

The input is of shape `n * c_i * h_i * w_i` and the output of shape `n * c_o * 1 * 1`.


[Strada/src/layers.jl:125](file:///home/pcmoritz/.julia/v0.3/Strada/src/layers.jl)



## LstmLayer

The LstmLayer is an LSTM unit. It takes two blobs as input, the current LSTM input and the previous memory cell content. It outputs the new hidden state and the updated memory cell.

[Strada/src/layers.jl:240](file:///home/pcmoritz/.julia/v0.3/Strada/src/layers.jl)



## LstmLayer

The LstmLayer is an LSTM unit. It takes two blobs as input, the current LSTM input and the previous memory cell content. It outputs the new hidden state and the updated memory cell.

[Strada/src/layers.jl:240](file:///home/pcmoritz/.julia/v0.3/Strada/src/layers.jl)



## MemoryLayer

The MemoryLayer presents data to Caffe through a pointer (it is implemented as a new Caffe Layer called PointerData), which can be set using `set_data!` method of `CaffeNet`. It is the preferred way to fill `CaffeNet` with data. As each MemoryLayer provides exactly one top blob, you will typically have multiple of these (in the supervised setting, one for labels and one for images for example). In `set_data!`, you can specify with an integer index which of the layers will be filled with the data provided.

[Strada/src/layers.jl:339](file:///home/pcmoritz/.julia/v0.3/Strada/src/layers.jl)



## PoolLayer

The PoolLayer partitions the input image into a set of non-overlapping rectangles and, for each such sub-region, outputs the maximum or average value. The keyword parameters are

* `method` (default `MAX`): The pooling method. Can be `MAX`, `AVE` or `STOCHASTIC`.

* `pad` (default 0): Specifies the number of pixels to (implicitly) add to each side of the input

* `stride` (default 1): Specifies the intervals at which to apply the filters to the input

The input is of shape `n * c * h_i * w_i` and the output of shape `n * c * h_o * w_o` where `h_o` and `w_o` are computed in the same way as for the convolution.

[Strada/src/layers.jl:151](file:///home/pcmoritz/.julia/v0.3/Strada/src/layers.jl)



## Softmax

Computes the softmax of the input. The parameter `axis` specifies which axis the softmax is computed over.

[Strada/src/layers.jl:316](file:///home/pcmoritz/.julia/v0.3/Strada/src/layers.jl)



## SoftmaxWithLoss

The softmax loss layer computes the multinomial logistic loss of the softmax of its inputs. It’s conceptually identical to a softmax layer followed by a multinomial logistic loss layer, but provides a more numerically stable gradient. Its parameters are

* `ignore_label` (default -1): Label does not contribute to the loss

This layer expects two bottom blobs, the actual data of size `n * c * h * w` and a label of size `n * 1 * 1 * 1`.

[Strada/src/layers.jl:299](file:///home/pcmoritz/.julia/v0.3/Strada/src/layers.jl)



## WordvecLayer

The WordvecLayer turns positive integers (indexes) between 0 and `vocab_size - 1` into dense vectors of fixed size `dimension`. The input is of size `n` where `n` is the batchsize and the output is of size `n * dimension`.

[Strada/src/layers.jl:256](file:///home/pcmoritz/.julia/v0.3/Strada/src/layers.jl)

