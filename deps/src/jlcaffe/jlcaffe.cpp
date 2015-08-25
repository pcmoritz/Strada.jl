#include "caffe/net.hpp"
#include "caffe/apollonet.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include <string>
#include <vector>
#include <stdexcept>
#include <boost/shared_ptr.hpp>

#include <glog/logging.h>

#include "jlcaffe.h"

using boost::shared_ptr;
using std::vector;

struct caffenet_state {
	caffe::Net<DTYPE> *net;
};

void init_jlcaffe() {
	google::InitGoogleLogging("");
}

caffenet_state* init_caffenet(const char* net_param, int net_param_len, int log_verbosity) {
	google::SetStderrLogging(log_verbosity);
	caffenet_state *state = new caffenet_state();
	caffe::NetParameter param;
	param.ParseFromString(std::string(net_param, net_param_len));
	state->net = new caffe::Net<DTYPE>(param);
	return state;
}

const char* layer_name(caffenet_state* state, int i) {
	return state->net->layers()[i]->layer_param().name().c_str();
}

int num_layers(caffenet_state* state) {
	return state->net->layers().size();
}

int num_blob_axis(caffenet_state* state, int layer_idx) {
	return state->net->layers()[layer_idx]->blobs().size();
}

void* get_blob(caffenet_state* state, int blob_idx) {
	return state->net->blobs()[blob_idx].get();
}

const char* get_blob_name(caffenet_state* state, int blob_idx) {
	return state->net->blob_names()[blob_idx].c_str();
}

void* get_weight_blob(caffenet_state* state, int layer_idx, int blob_idx) {
	return state->net->layers()[layer_idx]->blobs()[blob_idx].get();
}

int num_blobs(caffenet_state* state) {
	return state->net->blobs().size();
}

int num_output_blobs(caffenet_state* state) {
	return state->net->num_outputs();
}

#define TO_BLOB(blob) ((caffe::Blob<DTYPE>*)blob)

DTYPE* get_data(void* blob) {
	return TO_BLOB(blob)->mutable_cpu_data();
}

DTYPE* get_diff(void* blob) {
	return TO_BLOB(blob)->mutable_cpu_diff();
}

DTYPE* get_inc_data(void* blob) {
	// return TO_BLOB(blob)->mutable_cpu_inc_data();
}

DTYPE* get_inc_diff(void* blob) {
	// return TO_BLOB(blob)->mutable_cpu_inc_diff();
}

int get_num_axes(void* blob) {
	return TO_BLOB(blob)->num_axes();
}

int get_axis_shape(void* blob, int axis) {
	return TO_BLOB(blob)->shape(axis);
}

void set_data(caffenet_state* state, int layer_idx, DTYPE* data_arr, int num_data) {
	if (state->net->layers().size() == 0) {
		throw std::runtime_error("tried to set data of empty network");
	}
	boost::shared_ptr<caffe::PointerDataLayer<DTYPE> > md_layer =
		boost::dynamic_pointer_cast<caffe::PointerDataLayer<DTYPE> >(state->net->layers()[layer_idx]);
	if (!md_layer) {
		throw std::runtime_error("set_input_arrays may only be called if the"
		" respective layer is a MemoryDataLayer");
	}
	md_layer->Reset(data_arr, num_data);
}

void forward(caffenet_state* state) {
	int start_ind = 0;
	int end_ind = state->net->layers().size() - 1;
	state->net->ForwardFromTo(start_ind, end_ind);
}

void backward(caffenet_state* state) {
	int start_ind = 0;
	int end_ind = state->net->layers().size() - 1;
	state->net->BackwardFromTo(end_ind, start_ind);
}

void reverse(caffenet_state* state) {
	// state->net->RvForwardBackward();
}

void destroy_net(caffenet_state* state) {
	delete state->net;
	delete state;
}

void set_global_error_callback(callback c) {
	global_caffe_error_callback = c;
}

void set_mode_gpu() {
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
}

void set_mode_cpu() {
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
}

void load_from_caffemodel(caffenet_state* state, const char* filename) {
	state->net->CopyTrainedLayersFrom(filename);
}

void gpu_to_cpu(caffenet_state* state) {
  caffe::Net<DTYPE> *net = state->net;
  for(int i = 0; i < net->layers().size(); ++i) {
    vector<shared_ptr<caffe::Blob<DTYPE> > >& blobs =
      net->layers()[i]->blobs();
    for (int j = 0; j < blobs.size(); ++j) {
      blobs[j]->data()->cpu_mem();
      blobs[j]->diff()->cpu_mem();
      // TODO: copy data for reverse computation
    }
  }
}

struct apollonet_state {
	caffe::ApolloNet<DTYPE> *net;
	bool dirty; // need to reset iterators for apollo_next_param_name and apollo_next_blob_name
};

apollonet_state* init_apollonet(int log_verbosity) {
	google::SetStderrLogging(log_verbosity);
	apollonet_state *state = new apollonet_state();
	state->net = new caffe::ApolloNet<DTYPE>();
	state->dirty = true;
	return state;
}

DTYPE apollo_forward(apollonet_state* state, const char* layer_param, int layer_param_len) {
	state->dirty = true;
	return state->net->ForwardLayer(std::string(layer_param, layer_param_len));
}

void apollo_backward(apollonet_state* state) {
	state->dirty = true;
	state->net->Backward();
}

void apollo_reset(apollonet_state* state) {
	state->dirty = true;
	state->net->ResetForward();
}

const char* apollo_next_param_name(apollonet_state* state) {
	if(state->net->active_param_names().empty())
		return NULL;
	static std::set<std::string>::const_iterator it = state->net->active_param_names().begin();
	if(state->dirty) {
		state->dirty = false;
		it = state->net->active_param_names().begin();
	}
	if(it == state->net->active_param_names().end()) {
		it = state->net->active_param_names().begin();
		return NULL;
	}
	const char* result = it->c_str();
	it++;
	return result;
}

const char* apollo_next_blob_name(apollonet_state* state) {
	if(state->net->blobs().empty())
		return NULL;
	static std::map<std::string, shared_ptr<caffe::Blob<DTYPE> > >::const_iterator it = state->net->blobs().begin();
	if(state->dirty) {
		state->dirty = false;
		it = state->net->blobs().begin();
	}
	if(it == state->net->blobs().end()) {
		it = state->net->blobs().begin();
		return NULL;
	}
	const char* result = it->first.c_str();
	it++;
	return result;
}

void* apollo_get_param(apollonet_state* state, const char* param_name) {
	return state->net->params()[param_name].get();
}

void* apollo_get_blob(apollonet_state* state, const char* blob_name) {
	return state->net->blobs()[blob_name].get();
}

void apollo_update(apollonet_state* state, DTYPE lr, DTYPE momentum, DTYPE clip_gradients, DTYPE weight_decay) {
	state->net->Update(lr, momentum, clip_gradients, weight_decay);
}
