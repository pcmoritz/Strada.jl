layers = [
	MemoryLayer("data"; shape=(batchsize, 3, 227, 227)),
	MemoryLayer("label"; shape=(batchsize, 1)),
	ConvLayer("conv1", ["data"]; kernel=(11,11), stride=(4,4), n_filter=96),
	ActivationLayer("relu1", ["conv1"]; activation=ReLU),
	PoolLayer("pool1", ["relu1"]; kernel=(3,3), stride=(2,2)),
	LRNLayer("norm1", ["pool1"]; local_size=5, alpha=0.0001, beta=0.75),
	ConvLayer("conv2", ["norm1"]; kernel=(5,5), pad=(2,2), n_filter=256, group=2),
	ActivationLayer("relu2", ["conv2"]; activation=ReLU),
	PoolLayer("pool2", ["relu2"]; kernel=(3,3), stride=(2,2)),
	LRNLayer("norm2", ["pool2"]; local_size=5, alpha=0.0001, beta=0.75),
	ConvLayer("conv3", ["norm2"]; kernel=(3,3), pad=(1,1), n_filter=384),
	ActivationLayer("relu3", ["conv3"]; activation=ReLU),
	ConvLayer("conv4", ["relu3"]; kernel=(3,3), pad=(1,1), n_filter=384, group=2),
	ActivationLayer("relu4", ["conv4"]; activation=ReLU),
	ConvLayer("conv5", ["relu4"]; kernel=(3,3), pad=(1,1), n_filter=256, group=2),
	ActivationLayer("relu5", ["conv5"]; activation=ReLU),
	PoolLayer("pool5", ["relu5"]; kernel=(3,3), stride=(2,2)),
	LinearLayer("fc6", ["pool5"]; n_filter=4096),
	ActivationLayer("relu6", ["fc6"]; activation=ReLU),
	DropoutLayer("drop6", ["relu6"]; dropout_ratio=0.5),
	LinearLayer("fc7", ["drop6"]; n_filter=4096),
	ActivationLayer("relu7", ["fc7"]; activation=ReLU),
	DropoutLayer("drop7", ["relu7"]; dropout_ratio=0.5),
	LinearLayer("fc8", ["drop7"]; n_filter=1000),
	SoftmaxWithLoss("loss", ["fc8", "label"])
]

net = Net("CaffeNet", layers; log_level=3)

