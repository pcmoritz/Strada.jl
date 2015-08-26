using Strada

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
