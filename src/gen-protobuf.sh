protoc -I=../deps/src/caffe/src/caffe/proto --julia_out=. ../deps/src/caffe/src/caffe/proto/caffe.proto
rm caffe.jl
