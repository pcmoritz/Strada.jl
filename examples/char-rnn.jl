using Strada
using JSON
using Logging
using Distributions

# Task and hyperparameters from the original Apollo repo

batchsize = 32
vocab_size = 256
zero_symbol = vocab_size - 1
dimension = 300
clip_gradients = 10.0

net = Net("CharRNN"; log_level=3)

function softmax_choice(data, out)
	for i = 1:size(data, 2)
		p = float64(data[:,i])
		p = p ./ sum(p)
		out[i] = rand(Categorical(p)) - 1
	end
end

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
		forward(net, WordvecLayer("wordvec$i", ["word$i"]; dimension=dimension, vocab_size=vocab_size, param_names=["wordvec_param"], weight_filler=filler(:uniform; min=-0.1, max=0.1)))
		forward(net, ConcatLayer("lstm_concat$i", [prev_hidden, "wordvec$i"]))
		forward(net, LstmLayer("lstm$i", ["lstm_concat$i", prev_mem], ["lstm$(i)_hidden", "lstm$(i)_mem"]; num_cells=dimension, param_names=["lstm_input_value", "lstm_input_gate", "lstm_forget_gate", "lstm_output_gate"]))
		forward(net, DropoutLayer("dropout$i", ["lstm$(i)_hidden"]; dropout_ratio=0.16))
		forward(net, LinearLayer("ip$i", ["dropout$i"]; n_filter=vocab_size, param_names=["softmax_ip_weights", "softmax_ip_bias"], weight_filler=filler(:constant; value=0.0)))
		if test
			ip = net.blobs.data["ip$i"]
			ip *= 1.5
			forward(net, Softmax("softmax$i", ["ip$i"]))
			softmax_choice(net.blobs.data["softmax$i"], sub(data, :,i+1))
		else
			forward(net, DataLayer("label$i", data=data[:,i+1]))
			forward(net, SoftmaxWithLoss("loss$i", ["ip$i", "label$i"]; ignore_label=vocab_size-1))
		end
	end
end

filename = joinpath(Pkg.dir("Strada"), "data", "reddit_ml.txt")

function data_stream(filename)
	while true
		file = open(filename, "r")
		for line in readlines(file)
			data = JSON.parse(line)
			if length(data["body"]) == 0
				continue
			else
				produce(data["body"])
			end
		end
		close(file)
	end
end

function make_data(producer::Task, batchsize::Int)
	batch = String[]
	for i = 1:batchsize
		entry = consume(producer)
		push!(batch, entry)
	end
	maxlen = maximum(map(length, batch))
	result = zero_symbol * ones(Float32, batchsize, maxlen)
	for (i, sentence) in enumerate(batch)
		for (j, c) in enumerate(sentence)
			if j > maxlen
				break
			end	
			result[i,j] = min(c, 255) * 1.0f0
		end
	end
	return result
end

producer = Task(() -> data_stream(filename))

Logging.configure(level=INFO)

base_lr = 0.15

run_lstm(net, zeros(Float32, batchsize, 100))
theta = zeros(Float32, length(net.params.data))
copy!(theta, net.params.data)
grad = zeros(Float32, length(theta))

for i in 1:10000
  	data = make_data(producer, batchsize)
   	run_lstm(net, data)
   	zero!(net.params.diff)
   	backward(net)

   	Logging.info("[", i, "] ", net.loss)
   	lr = base_lr * 0.8 ^ (i // 2500)
    copy!(grad, net.params.diff)
    gradnorm = norm(grad)
    clip_scale = 1.0
    if gradnorm > clip_gradients
     	clip_scale = clip_gradients / gradnorm
    end
    theta -= clip_scale * lr * grad
    copy!(net.params.data, theta)

    if mod(i, 20) == 0
    	data = zeros(Float32, batchsize, 150)
     	run_lstm(net, data; test=true, maxlen=150)
     	println(join(map(char, data[1,:])))
     end
  end
