function make_stencil{F}(theta::Vector{F}, epsilon::F)
	delta = deepcopy(theta)
	rand!(delta)
	fst = deepcopy(theta)
	BLAS.axpy!(epsilon, delta, fst)
	snd = deepcopy(theta)
	BLAS.axpy!(-epsilon, delta, snd)
	return (delta, fst, snd)
end

@doc "Check gradients using symmetric finite differences. See the tests for example how to run." ->
function grad_check{F}(objective::Function, theta::Vector{F}, data, epsilon::Float64; abs_error::Bool=true, verbose::Bool=false)
	epsilon = convert(F, epsilon)
	(delta, fst, snd) = make_stencil(theta, epsilon)
	deriv_approx = (objective(data, fst) - objective(data, snd))/(2*epsilon)
	grad = deepcopy(theta)
	objective(data, theta; grad=grad)
	deriv_exact = dot(grad, delta)
	if verbose
		println("exact  derivative: ", deriv_exact)
		println("finite diff derivative: ", deriv_approx)
	end
	if abs_error
		return abs(deriv_approx - deriv_exact)
	else
		return abs((deriv_approx - deriv_exact)/deriv_exact)
	end
end
