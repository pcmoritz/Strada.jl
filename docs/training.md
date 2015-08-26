# Training networks

Strada facilitates implementing your own training mechanisms for neural networks in Julia. This makes it very convenient to use your own optimization procedures, which is very convenient if you are for example working on reinforcement learning. You can tap into the training mechanism on two levels: By manually loading data into the network and calling its `forward` and `backward` method and then getting the gradient blobs out, or using a slightly higher level interface which should be familiar if you have used a package for numerical optimization before. We describe the latter approach here.

As an example, let us consider how to train a convolutional network that can recognize MNIST digits. First, let us define the model:

```julia
batchsize = 64

layers = [
	MemoryLayer("data"; shape=(batchsize, 1, 28, 28)),
	MemoryLayer("label"; shape=(batchsize, 1)),
	ConvLayer("conv1", ["data"]; kernel=(5,5), n_filter=20),
	PoolLayer("pool1", ["conv1"]; kernel=(2,2), stride=(2,2)),
	ConvLayer("conv2", ["pool1"]; kernel=(5,5), n_filter=50),
	PoolLayer("pool2", ["conv2"]; kernel=(2,2), stride=(2,2)),
	LinearLayer("ip1", ["pool2"]; n_filter=500),
	ActivationLayer("relu1", ["ip1"]; activation=ReLU),
	LinearLayer("ip2", ["relu1"]; n_filter=10),
	SoftmaxWithLoss("loss", ["ip2", "label"])
]

net = Net("LeNet", layers; log_level=3);
```

## Creating objective functions and predictors

You can now create an objective function that will be optimized by calling

```julia
(objective, theta) = make_objective(net, Float32)
```

Here, `Float32` is the floating point type used by the network, `theta` is a flat vector containing the initial parameters and `objective` is a Julia function with signature

```julia
function objective(data::Data{F,N}, theta::Vector{F}; grad::Vector{F}=zeros(F, 0))
	# If length(grad) != 0, store the gradient of the loss function in grad.
	# The caller needs to guarantee that length(grad) = length(theta)
	# In any case, return the loss of the network computed on the minibatch data
end
```

`Data{F,N}` is the datatype representing a minibatch (see its documentation in the [API](api/Strada.md)). Here, `N` is the number of data layers in the network. `Data{F,N}` is an `N` tuple where each component is an array that will be fed into the corresponding data layer of the network. In the case of MNIST, `N = 2` which means `Data{F,N}` is of type `NTuple{Array{Float32, 4}, Array{Float32, 2}}`. The first array in the tuple corresponds to images and the second one to labels.

Now we create a function that can compute a prediction on a new digit (once the network has been trained):

```julia
predictor = make_predictor(net, Float32, "ip2")
```

Here `"ip2"` is the name of the last layer before the softmax. The predictor has signature

```julia
function predictor(data::Data{F,N}, theta::Vector{F}; result::Matrix{Int}=zeros(Int, 0, 0))
	# Store the predicted label of the n-th example from minibatch data in result[n, 1]
end
```
The result is a matrix here, because we also support predicting sequences.

## Loading the data

We can now load the dataset. Let us assume we have a function ``load_mnist`` that outputs arrays with shape (1, 28, 28, 50000) and (1, 50000) for the training set. Using the `minibatch_stream` constructor, this data can then be loaded into a `MinibatchStream`, which is a collection of `Data{F,N}` tuples of minibatch size that can be iterated over.

```julia
(Xtrain, ytrain) = load_mnist(directory; data_set=:train)
(Xtest, ytest) = load_mnist(directory; data_set=:test)

data = minibatch_stream(Xtrain, ytrain; batchsize=batchsize)
testset = minibatch_stream(Xtest, ytest, batchsize=batchsize)
```

## Training the network

In this case, we train the model using SGD:

```julia
sgd(objective, data, theta; predictor=predictor, testset=testset,
    lr_schedule=InvLR(0.01, 0.0001, 0.75, 0.9), epochs=5, verbose=true)
```