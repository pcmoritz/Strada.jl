#ifndef CAFFE_ERROR_HPP_
#define CAFFE_ERROR_HPP_

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <glog/logging.h>

#ifdef CAFFE_HEADLESS

// Remember the following definition:
// #define LOG(severity) COMPACT_GOOGLE_LOG_ ## severity.stream()

#undef COMPACT_GOOGLE_LOG_FATAL

extern "C" {
	extern void (*global_caffe_error_callback)(const char *msg);
}

struct ErrorStream : public std::ostream {
	std::ostringstream stream;
	template<size_t n>
	ErrorStream& operator<<(const char (&message)[n]) {
		stream << message;
		return *this;
	}
	virtual ErrorStream& operator<<(const char *message) {
		stream << message;
		return *this;
	}
	virtual ErrorStream& operator<<(const std::string& message) {
		stream << message;
		return *this;
	}
	virtual ErrorStream& operator<<(size_t message) {
		stream << message;
		return *this;
	}
	virtual ErrorStream& operator<<(std::ostream&(*f)(std::ostream&))
	{
		stream << std::endl;
		return *this;
	}
};

struct ErrorCallback {
	ErrorStream thestream;
	ErrorStream& stream() {
		return thestream;
	}
	ErrorCallback(void (*callback)(const char *msg)) {}
	ErrorCallback(void (*callback)(const char *msg), const char* file, int line, const google::CheckOpString& result) { 
		thestream << file << ":" << line << " [" << *(result.str_) << "] "; 
	}
	~ErrorCallback() { global_caffe_error_callback(thestream.stream.str().c_str()); }
};

#define COMPACT_GOOGLE_LOG_FATAL ErrorCallback(global_caffe_error_callback)

#define GET_LOG(FILE, LINE, RESULT) ErrorCallback(global_caffe_error_callback, FILE, LINE, RESULT)

#undef CHECK_OP
#define CHECK_OP(name, op, val1, val2) \
	CHECK_OP_LOG(name, op, val1, val2, GET_LOG)

#define ASSERT(condition, stream) CHECK(condition) << stream

#endif // CAFFE_HEADLESS

#endif
