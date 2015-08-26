abstract LearningRate

type InvLR <: LearningRate
	base_lr::Float64
	gamma::Float64
	power::Float64
	momentum::Float64
end

type ConstLR <: LearningRate
	base_lr::Float64
	momentum::Float64
end

@lintpragma("Ignore unused iter")
get_lr(lr::ConstLR, iter::Int) = lr.base_lr
get_lr(lr::InvLR, iter::Int) = lr.base_lr * (1.0 + lr.gamma * iter) ^ (-lr.power)

function empty()
end

@doc """Run the stochastic gradient descent method on the objective. If a testset is provided, generalization performance will also periodically be evaluated.""" ->
function sgd{F}(objective!::Function, data::DataStream, theta::Vector{F};
	testset::DataStream=EmptyStream(), predictor::Function=empty, lr_schedule::LearningRate=InvLR(0.1, 1.0, 0.5, 0.9),
	epochs::Int=30, verbose::Bool=true, diagnostics::Function=log_to_screen)
	
	log_to_screen("======= running the stochastic gradient method =======")
	log_to_screen(@sprintf "learning schedule: %s" string(lr_schedule))
	log_to_screen(@sprintf "batchsize:         %d" get_batchsize(data))
	log_to_screen(@sprintf "epochs:            %d" epochs)
	log_to_screen("------------------------------------------------------")
	log_to_screen("iter | trainloss  traingrad  testloss   classification")

	total_steps = 1
	theta = copy(theta)
	grad = Array(F, length(theta))
    grad_scratch = Array(F, length(theta))
    velocity = zeros(F, length(theta))

	for epoch in 1:epochs
		if verbose
        	loss = calc_full_gradient(objective!, data, theta, grad_scratch)
        	if typeof(testset) != EmptyStream
        		(test_loss, num_correct) = calc_full_prediction(predictor, testset, theta)
        		diagnostics(epoch, loss, norm(grad_scratch), theta; test_loss=test_loss, num_correct=num_correct, num_total=size(testset))
        	else
				diagnostics(epoch, loss, norm(grad_scratch), theta)
			end
		end
		for (batch, minibatch) in enumerate(data)
			objective!(minibatch, theta; grad=grad)
			velocity *= lr_schedule.momentum
			velocity -= get_lr(lr_schedule, total_steps) * grad
			theta += velocity
	    	total_steps += 1
		end
	end
	if verbose
		log_to_screen("Optimization done.")
	end
	return theta
end
