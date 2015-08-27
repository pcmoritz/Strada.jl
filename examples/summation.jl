using Strada
using Logging

# Task and hyperparameters from the original Apollo repo

batch_size = 32
init_range = 0.1
base_lr = 0.03
momentum = 0.9
clip_gradients = 0.1
mem_cells = 1000

function run_summation_lstm(net; test=false)
	reset(net)
	len = rand(5:15)
	uniform_filler = filler(:uniform, min=-init_range, max=init_range)
	forward(net, DataLayer("lstm_seed", data=zeros(Float32, batch_size, mem_cells)))
	accum = zeros(Float32, batch_size)
	for i in 1:len
		value = rand(Float32, batch_size, 1)
		forward(net, DataLayer("value$i", data=value))
		accum += value[:]
		prev_hidden = i == 1 ? "lstm_seed" : "lstm$(i-1)_hidden"
		prev_mem = i == 1 ? "lstm_seed" : "lstm$(i-1)_mem"
		forward(net, ConcatLayer("lstm_concat$i", [prev_hidden, "value$i"]))
		forward(net, LstmLayer("lstm$i", ["lstm_concat$i", prev_mem], ["lstm$(i)_hidden", "lstm$(i)_mem"];
			num_cells=mem_cells, param_names=["lstm_input_value", "lstm_input_gate", "lstm_forget_gate", "lstm_output_gate"],
			weight_filler=uniform_filler))
	end
	forward(net, LinearLayer("ip", ["lstm$(len)_hidden"]; n_filter=1, weight_filler=uniform_filler))
	if test
		println("test")
		println([accum[:] net.blobs.data["ip"][:]])
	else
		forward(net, DataLayer("label", data=accum))
		loss = forward(net, EuclideanLoss("loss", ["ip", "label"]))
		return loss/len
	end
end

net = Net("SummationRNN"; log_level=3)

Logging.configure(level=INFO)

run_summation_lstm(net)
theta = zeros(Float32, length(net.params.data))
copy!(theta, net.params.data)
grad = zeros(Float32, length(theta))

for i = 1:10000
	loss = run_summation_lstm(net)
	zero!(net.params.diff)
	backward(net)

	Logging.info("[", i, "] ", loss)

   	lr = base_lr * 0.5 ^ (i // 1000)
    copy!(grad, net.params.diff)
    gradnorm = norm(grad)
    clip_scale = 1.0
    if gradnorm > clip_gradients
     	clip_scale = clip_gradients / gradnorm
    end
    theta -= clip_scale * lr * grad
    copy!(net.params.data, theta)
    if i % 10 == 0
    	run_summation_lstm(net; test=true)
    end
end