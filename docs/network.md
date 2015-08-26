# Defining Networks

Deep networks are compositional models that are naturally represented as a collection of inter-connected layers that work on chunks of data. In Strada, models can be conveniently described in Julia, either statically at the beginning of the computation (`CaffeNet`) or dynamically at runtime (`ApolloNet`). The network defines the entire model bottom-to-top from input data to loss. As data and derivatives flow through the network in the forward and backward passes Caffe stores, communicates, and manipulates the information as blobs: the blob is the standard array and unified memory interface for the framework. The layer comes next as the foundation of both model and computation. The net follows as the collection and connection of layers. The details of blob describe how information is stored and communicated in and across layers and nets.

## Layers and Blobs

The layer is the essence of a model and the fundamental unit of computation. Layers convolve filters, pool, take inner products, apply nonlinearities like rectified-linear and sigmoid and other elementwise transformations, normalize, load data, and compute losses like softmax and hinge. See the [layer catalogue](api/layers.md) for all operations. Most of the types needed for state-of-the-art deep learning tasks are there.

A Blob is a wrapper over the actual data being processed and passed along by Caffe, and also under the hood provides synchronization capability between the CPU and the GPU. Below, we describe how the Caffe blobs are mapped to Julia types. Each layer takes its input from so called `bottom` blobs and puts the output into `top` blobs.

Layer definitions in Julia looks like
```julia
ConvLayer("conv1", ["data"]; kernel=(11,11), stride=(4,4), n_filter=96)
```
The first parameter is the name of the layer, the second parameter is the name of the bottom blobs. There is an optional third parameter where you can specify the names of the top blobs. If it is omitted, we assume one top blob with the same name as the layer. Next, we describe how layers can be combined to form networks.

## Networks in Strada

In Strada, there are two ways to specify a network: `CaffeNet` and `ApolloNet`. The former is designed to be compatible with Caffe so that all the features and models implemented for Caffe can easily be ported to Strada. It requires the specification of the network architecture before any computation is performed. The latter is more suited to implementing recurrent networks and allows building the network on the fly.

This is how a `CaffeNet` is defined:

```julia
layers = [
	MemoryLayer("data", data=ones(Float32, 10, 10)),
	MemoryLayer("label", data=ones(Float32, 10, 1)),
	LinearLayer("ip1", ["data"]; n_filter=1),
	ActivationLayer("relu1", ["ip1"]; activation=ReLU),
	LinearLayer("ip2", ["relu1"]; n_filter=1),
	SoftmaxWithLoss("loss", ["ip2", "label"])
]

net = Net("SimpleNet", layers; log_level=3)
```

This is how you can run an `ApolloNet`:

```julia
net = Net("SimpleNet"; log_level=3)

function run_simple(net)
	reset(net)
	forward(net, DataLayer("data", data=ones(Float32, 10, 10)))
	forward(net, DataLayer("label", data=ones(Float32, 10)))
	forward(net, LinearLayer("ip1", ["data"]; n_filter=1, param_names=["ip1_weights", "ip1_bias"]))
	forward(net, ActivationLayer("relu1", ["ip1"]; activation=ReLU))
	forward(net, LinearLayer("ip2", ["relu1"]; n_filter=1, param_names=["ip2_weights", "ip2_bias"]))
	forward(net, SoftmaxWithLoss("loss", ["ip2", "label"]))
end

run_simple(net)
```

Not that an insignificant amount of runtime is being spent on the construction of the network (on the recurrent networks I tried this amounted to about 10% of the total runtime).

## Wrapping Caffe blobs in Strada

For a `CaffeNet`, intermediate results from layer computations are stored in `net.blobs`, which has two fields: `net.blobs.data` for the data and `net.blobs.diff` for the gradients. Both these fields are dictionaries that map layer names to their top blobs, which are Julia arrays. Parameters are stored in `net.layers`, they also have a `net.layers.data` and `net.layers.diff` field.

For an `ApolloNet`, parameters are named and stored in the `net.params` which also has a subfield `data` and `diff`. This makes sharing of parameters between parts of the network very easy, a crucial feature for recurrent networks. The tops of the layers are stored in `net.blobs`.