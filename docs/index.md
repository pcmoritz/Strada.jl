# Strada

Strada is a Deep Learning library for Julia, based on the popular [Caffe](http://caffe.berkeleyvision.org/) framework developed by BVLC and the Berkeley computer vision community. It supports convolutional and recurrent neural netwok training, both on the CPU and GPU. Some highlights:

- **Simplicity** both for advanced users and novices. It is easy to install and also a good platform for teaching.

- **Flexibility** expecially for doing research in various different domains like mathematical optimization, computer vision, natural language and reinforcement learning.

- **Integration with Julia**: Strada is distributed with a version of Caffe that has minimal dependencies and was integrated with Julia's linear algebra subroutines, Julia's tensor manipulation routines and Julia's error handling system.

- **Support of Caffe features**: It is easy to rebase Strada to a different Caffe version with additional pull request integrated (for example with multi GPU support)

- **Open source**: Strada is distributed under a BSD licence.

We make crucial use of a number of projects:

* [Caffe](http://caffe.berkeleyvision.org/), which was created by Yangqing Jia and is developed by the Berkeley Vision and Learning Center and by community contributors.

* [Apollo](https://github.com/BVLC/caffe/pull/2932), an extension of Caffe that makes it easy to work with recurrent neural networks and was developed in the Stanford NLP group

* [Mocha](https://github.com/pluskid/Mocha.jl), another deep learning framework for Julia developed by Chiyuan Zhang at MIT

* The [Julia language](http://julialang.org/) initially developed by Jeff Bezanson, Alan Edelman, Stefan Karpinski and Viral B. Shah at MIT and now by community contributers from all over the world. We make heavy use of the [ProtoBuf.jl](https://github.com/tanmaykm/ProtoBuf.jl) package by Tanmay Mohapatra.

This documentation is partly adapted from [Caffe's documentation](http://caffe.berkeleyvision.org/tutorial/), which you should also consider looking into: Most of the techniques and terminology is shared between Strada and Caffe. Also, [Mocha's documentation](https://readthedocs.org/projects/mochajl/) might be helpful.

Parts of this package were developed while working on [AMPLab](https://amplab.cs.berkeley.edu/) projects, I gratefully acknowledge funding.
