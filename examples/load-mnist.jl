# Modified from Mocha.jl

using Compat

function load_digits_and_labels(label_fn, data_fn)
	label_f = open(label_fn)
	data_f  = open(data_fn)

	label_header = read(label_f, Int32, 2)
	@assert ntoh(label_header[1]) == 2049
	n_label = @compat Int(ntoh(label_header[2]))
	data_header = read(data_f, Int32, 4)
	@assert ntoh(data_header[1]) == 2051
	n_data = @compat Int(ntoh(data_header[2]))
	@assert n_label == n_data
	h = @compat Int(ntoh(data_header[3]))
	w = @compat Int(ntoh(data_header[4]))

	X = zeros(Float32, 1, h, w, n_data)
	y = zeros(Float32, 1, n_data)

	n_batch = 1
	@assert n_data % n_batch == 0
	batch_size = div(n_data, n_batch)

	for i = 1:n_batch
		idx = (i-1)*batch_size+1:i*batch_size
		idx = collect(idx)
		rp = randperm(length(idx))
		
		img = readbytes(data_f, batch_size * h*w)
		img = convert(Array{Float32},img) / 256 # scale into [0,1)
		class = readbytes(label_f, batch_size)
		class = convert(Array{Int},class)
		
		for j = 1:length(idx)
			r_idx = rp[j]

			X[1,:,:,idx[j]] = img[(r_idx-1)*h*w+1:r_idx*h*w]
			y[1, idx[j]] = class[r_idx]
		end
	end

	close(label_f)
	close(data_f)

	return (X, y)
end

function load_mnist(directory; data_set=:train)
	if data_set == :train
		label_fn = joinpath(directory, "train-labels-idx1-ubyte")
		data_fn = joinpath(directory, "train-images-idx3-ubyte")
	elseif data_set == :test
		label_fn = joinpath(directory, "t10k-labels-idx1-ubyte")
		data_fn = joinpath(directory, "t10k-images-idx3-ubyte")
	else
		error("Dataset unknown")
	end

	return load_digits_and_labels(label_fn, data_fn)
end
