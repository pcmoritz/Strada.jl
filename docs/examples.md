# Examples

Scripts to train or evaluate these examples are located in the `examples` subdirectory. This is only a very small collection, but it is straightforward to port any model available in Caffe. If you are interested in porting over more models from the [Caffe model zoo](http://caffe.berkeleyvision.org/model_zoo.html) (see also the [model zoo wiki](https://github.com/BVLC/caffe/wiki/Model-Zoo)), you should open an issue on github or create a pull request.

## An SVM like shallow network

```julia
function make_svm(p::Int; batchsize::Int=100)
	layers = [
		MemoryLayer("data"; shape=(batchsize, 1, 1, p)),
		MemoryLayer("label"; shape=(batchsize, 1)),
		LinearLayer("linear", ["data"]; n_filter=3, weight_filler=filler(:constant)),
		SoftmaxWithLoss("loss", ["linear", "label"])
	]
	return Net("SVMNet", layers; log_level=3)
end
```

## A convolutional network for MNIST

```julia
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

## A convolutional network for CIFAR10

```julia
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
```

## A version of AlexNet for Imagenet

```julia
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
```

## A recurrent network for character prediction

```julia
net = Net("CharRNN"; log_level=3)

function run_lstm(net, data; test=false, maxlen=100)
	reset(net)
	forward(net, DataLayer("lstm_seed", data=zeros(Float32,batchsize,dimension)))
	for i = 0:min(size(data, 2)-1, maxlen)
		prev_hidden = i == 0 ? "lstm_seed" : "lstm$(i-1)_hidden"
		prev_mem = i == 0 ? "lstm_seed" : "lstm$(i-1)_mem"
		word = i == 0 ? zeros(Float32, batchsize) : data[:,i]
		if test && i == 0
			word = fill(1.0f0 * '.', batchsize)
		end
		forward(net, DataLayer("word$i", data=word))
		forward(net, WordvecLayer("wordvec$i", ["word$i"];
			param_names=["wordvec_param"],
			dimension=dimension, vocab_size=vocab_size,
			weight_filler=filler(:uniform; min=-0.1, max=0.1)))
		forward(net, ConcatLayer("lstm_concat$i", [prev_hidden, "wordvec$i"]))
		forward(net, LstmLayer("lstm$i", ["lstm_concat$i", prev_mem], ["lstm$(i)_hidden", "lstm$(i)_mem"];
			param_names=["lstm_input_value", "lstm_input_gate", "lstm_forget_gate", "lstm_output_gate"],
			num_cells=dimension))
		forward(net, DropoutLayer("dropout$i", ["lstm$(i)_hidden"]; dropout_ratio=0.16))
		forward(net, LinearLayer("ip$i", ["dropout$i"];
			param_names=["softmax_ip_weights", "softmax_ip_bias"],
			n_filter=vocab_size,
			weight_filler=filler(:constant; value=0.0)))
		if test
			forward(net, Softmax("softmax$i", ["ip$i"]))
			softmax_choice(net.blobs.data["softmax$i"], sub(data, :,i+1))
		else
			forward(net, DataLayer("label$i", data=data[:,i+1]))
			forward(net, SoftmaxWithLoss("loss$i", ["ip$i", "label$i"]; ignore_label=vocab_size-1))
		end
	end
end
```