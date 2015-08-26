using Strada
using Images
using Color

batchsize = 1

include(joinpath(Pkg.dir("Strada"), "examples", "caffenet-net.jl"))

dir = joinpath(Pkg.dir("Strada"), "deps", "src", "caffe")
load_caffemodel(net, joinpath(dir, "models", "bvlc_reference_caffenet", "bvlc_reference_caffenet.caffemodel"))
img = Images.imread(joinpath(dir, "examples", "images", "cat.jpg"))
mu = 114.45155662981172 # mean of training images in imagenet, compute with ilsvrc_2012_mean.npy from caffe

function preprocess(img)
	if spatialorder(img) == ["y","x"]
      	img = permutedims(img, (2,1,3)) # permute to row-major
	end
	img = Images.imresize(img, (256, 256))
	img = convert(Image{RGB}, img)
	img = separate(img)
	data = convert(Array, img)
	data = permutedims(data, (2,1,3))
	data *= 255.0f0
	data -= mu
	data[:,:,1], data[:,:,3] = data[:,:,3], data[:,:,1] # convert from RGB to BGR (as expected by the caffe model)
	return convert(Array{Float32}, data)
end

data = preprocess(img)

data = reshape(data[14:240,14:240,:], (227, 227, 3, 1)) # take middle patch from image

(objective, theta) = make_objective(net, Float32)
predictor = make_predictor(net, Float32, "fc8")

result = zeros(Int, batchsize, 1)
predictor((data, zeros(Float32, 1, 50)), theta; result=result)

@assert result[1] == 281
