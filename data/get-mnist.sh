#!/usr/bin/env sh

for dset in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
	t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
do
	wget http://yann.lecun.com/exdb/mnist/$dset
	gunzip $dset
done
