# TODO

* Port to Mac OS X (should be quite easy, you just have to make sure the right binary dependencies are installed using BinDeps)

* Implement a Julia layer similar to [Caffe's Python layer](https://github.com/BVLC/caffe/pull/1020)

* Wrap Caffe's SGD implementation and using that make it possible to train AlexNet on ImageNet; this would also guarantee that all the caffe models can be trained with exactly the same parameters as in the Caffe model zoo.

* Integrate Caffe's [multi GPU support](https://github.com/BVLC/caffe/pull/2870)

* Port over more Caffe models and wrap all the remaining layers
