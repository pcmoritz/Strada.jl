using GZip

@doc "Load a dataset from the libsvm compatible svmlight file format into a sparse matrix." ->
function read_svmlight(filename::String, Dtype::DataType=Float32)
	f = GZip.open(filename)
	lines = readlines(f)
	lines = lines[1:end-1]
	n = length(lines)
	y = zeros(Dtype, n)
	Idx = Int[]
	J = Int[]
	V = zeros(Dtype, 0)
	for (i, line) in enumerate(lines)
		data = split(line)
		y[i] = parse(Dtype, data[1])
		for d in data[2:end]
			(idx, val) = split(d, ":")
			push!(Idx, parse(Int, idx))
			push!(J, i)
			push!(V, parse(Dtype, val))
		end
	end
	close(f)
	return (sparse(Idx, J, V), y)
end
