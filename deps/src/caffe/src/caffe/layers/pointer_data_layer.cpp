#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void PointerDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  const BlobShape& shape = this->layer_param_.pointer_data_param().shape();
  shape_ = shape;
  batch_size_ = shape.dim(0);
  size_ = 1;
  for(int i = 1; i < shape.dim_size(); ++i) { // batch_size is not part of size
    size_ *= shape.dim(i);
  }
  CHECK_GT(batch_size_ * size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  top[0]->Reshape(shape_);
  data_ = NULL;
}

template <typename Dtype>
void PointerDataLayer<Dtype>::Reset(Dtype* data, int n) {
  CHECK(data);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  // Warn with transformation parameters since a memory array is meant to
  // be generic and no transformations are done with Reset().
  if (this->layer_param_.has_transform_param()) {
    LOG(WARNING) << this->type() << " does not transform array data on Reset()";
  }
  data_ = data;
  n_ = n;
  pos_ = 0;
}

template <typename Dtype>
void PointerDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(data_) << "MemoryDataLayer needs to be initalized by calling Reset";
  top[0]->Reshape(shape_);
  top[0]->set_cpu_data(data_ + pos_ * size_);
  pos_ = (pos_ + batch_size_) % n_;
  if (pos_ == 0)
    has_new_data_ = false;
}

INSTANTIATE_CLASS(PointerDataLayer);
REGISTER_LAYER_CLASS(PointerData);

}  // namespace caffe
