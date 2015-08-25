#define DTYPE float

extern "C" {
	typedef void (*callback)(const char *msg);

	callback global_caffe_error_callback;

	struct caffenet_state;

	void init_jlcaffe();

	caffenet_state* init_caffenet(const char* net_param, int net_param_len, int log_verbosity);

	void set_data(caffenet_state* state, int layer_idx, DTYPE* data_arr, int num_data);
	const char * layer_name(caffenet_state* state, int i);
	void forward(caffenet_state* state);
	void backward(caffenet_state* state);
	void reverse(caffenet_state* state);

	int num_layers(caffenet_state* state);
	int num_blob_axis(caffenet_state* state, int layer_idx);

	int num_blobs(caffenet_state* state);

	void* get_blob(caffenet_state* state, int blob_idx);
	const char* get_blob_name(caffenet_state* state, int blob_idx);
	void* get_weight_blob(caffenet_state* state, int layer_idx, int blob_idx);

	DTYPE* get_data(void* blob);
	DTYPE* get_diff(void* blob);
	DTYPE* get_inc_data(void* blob);
	DTYPE* get_inc_diff(void* blob);

	int get_num_axes(void* blob);
	int get_axis_shape(void* blob, int axis);

	void set_global_error_callback(callback c);
	
	void set_mode_cpu();
	void set_mode_gpu();

	void load_from_caffemodel(caffenet_state* state, const char* filename);

  	void gpu_to_cpu(caffenet_state* state);

	struct apollonet_state;

	apollonet_state* init_apollonet(int log_verbosity);

	DTYPE apollo_forward(apollonet_state* state, const char* layer_param, int layer_param_len);
	void apollo_backward(apollonet_state* state);
	void apollo_reset(apollonet_state* state);

	const char* apollo_next_param_name(apollonet_state* state);
	const char* apollo_next_blob_name(apollonet_state* state);

	void* apollo_get_param(apollonet_state* state, const char* param_name);
	void* apollo_get_blob(apollonet_state* state, const char* blob_name);
}
