using Logging

function log_to_screen(str::String)
	Logging.info(str)
end

function log_to_hdf{F}(epoch::Int, loss::F, gradnorm::Union(F, Nothing), theta::Vector{F}, hdfname::String)
	h5open(hdfname, "r+") do hdf
		diagnostics = hdf["diagnostics"]
		diagnostics["obj/" * @sprintf "%04d" epoch] = loss
		diagnostics["theta/" * @sprintf "%04d" epoch] = theta
        if gradnorm != nothing
            diagnostics["gradnorm/" * @sprintf "%04d" epoch] = gradnorm
        end
	end
	log_to_screen(epoch, loss, gradnorm, theta)
end

@lintpragma("Ignore unused theta")
function log_to_screen{F}(epoch::Int, loss::F, gradnorm::Union(F, Nothing), theta::Vector{F};
		test_loss::F=convert(F, NaN), num_correct::Int=-1, num_total::Int=-1)
	if gradnorm != nothing && !isnan(test_loss)
		num_digits = length(string(num_total))
		correct_pred = lpad(string(num_correct), num_digits, " ")
		logstr = @sprintf "%4d | %4.4E %4.4E %4.4E %s/%d" epoch loss gradnorm test_loss correct_pred num_total
	elseif gradnorm != nothing
		logstr = @sprintf "%4d | %4.4E %4.4E" epoch loss gradnorm
	else
		logstr = @sprintf "%4d | %4.4E" epoch loss
	end
	Logging.info(logstr)
end

function make_net_logger(net::CaffeNet)
	function log_net_to_screen{F}(epoch::Int, loss::F, gradnorm::Union(F, Nothing), theta::Vector{F};
		test_loss::F=F(NaN), num_correct::Int=-1, num_total::Int=-1)
		log_to_screen(epoch, loss, gradnorm, theta; test_loss=test_loss, num_correct=num_correct, num_total=num_total)
		log_to_screen(string(net.layers))
	end
	return log_net_to_screen
end

@doc "Calculate the full gradient of the model at parameters `theta` over the dataset `data`. The gradient will be stored in `grad`." ->
function calc_full_gradient{F}(objective!::Function, data::DataStream, theta::Vector{F}, grad::Vector{F})
	result = convert(F, 0.0)
    p = length(theta)
    scratch = zeros(F, p)
    fill!(grad, 0.0)
	for batch in data
		result += objective!(batch, theta; grad=scratch)
        BLAS.axpy!(1.0, scratch, grad)
	end
	scale!(grad, 1.0 / num_batches(data))
	return result / num_batches(data)
end

@doc "Calculate prediction performance of the model with parameters `theta` over a whole dataset `data`" ->
function calc_full_prediction{F}(predictor::Function, data::DataStream, theta::Vector{F})
	n = get_batchsize(data)
	t = size(data.data[1][2], 1) # TODO: make better interface for this
	num_correct = 0
	loss = convert(F, 0.0)
	scratch = zeros(Int, t, n)
	for batch in data
		loss += predictor(batch, theta; result=scratch)
		num_correct += sum(abs(scratch - batch[2]) .< 0.5) # batch[2] is the labels, same TODO as above applies
	end
	return (loss / num_batches(data), num_correct)
end
