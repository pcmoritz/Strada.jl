using Strada

# An SVM like network for classification

function make_svm(p::Int; batchsize::Int=100)
	layers = [
		MemoryLayer("data"; shape=(batchsize, 1, 1, p)),
		MemoryLayer("label"; shape=(batchsize, 1)),
		LinearLayer("linear", ["data"]; n_filter=3, weight_filler=filler(:constant)),
		SoftmaxWithLoss("loss", ["linear", "label"])
	]
	return Net("SVMNet", layers; log_level=3)
end
