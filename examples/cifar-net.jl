using Strada

batchsize = 100

layers = [
	MemoryLayer("data"; shape=(batchsize, 3, 32, 32))
	MemoryLayer("label"; shape=(batchsize, 1))
	ConvLayer("conv1", ["data"]; kernel=(5,5), pad=(2,2), n_filter=32, weight_filler=filler(:gaussian; std=0.0001))
	PoolLayer("pool1", ["conv1"]; kernel=(3,3), stride=(2,2))
	ActivationLayer("relu1", ["pool1"]; activation=ReLU)
	LRNLayer("norm1", ["relu1"]; local_size=3, alpha=5e-5, beta=0.75, norm_region=WITHIN_CHANNEL)
	ConvLayer("conv2", ["norm1"]; kernel=(5,5), pad=(2,2), n_filter=32, weight_filler=filler(:gaussian; std=0.01))
	ActivationLayer("relu2", ["conv2"]; activation=ReLU)
	PoolLayer("pool2", ["relu2"]; kernel=(3,3), stride=(2,2))
	LRNLayer("norm2", ["pool2"]; local_size=3, alpha=5e-5, beta=0.75, norm_region=WITHIN_CHANNEL)
	ConvLayer("conv3", ["norm2"]; kernel=(5,5), pad=(2,2), n_filter=64, weight_filler=filler(:gaussian; std=0.01))
	ActivationLayer("relu3", ["conv3"]; activation=ReLU)
	PoolLayer("pool3", ["relu3"]; kernel=(3,3), stride=(2,2))
	LinearLayer("ip1", ["pool3"]; n_filter=10, weight_filler=filler(:gaussian; std=0.01))
	SoftmaxWithLoss("loss", ["ip1", "label"])
]

net = Net("CIFAR10", layers; log_level=3)
