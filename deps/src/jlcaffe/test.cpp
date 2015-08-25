// A little test program for the shared library
// to test the linking requirements

// g++ test.cpp -L../../build/ -ljlcaffe

#include "jlcaffe.h"

int main() {
	init_jlcaffe();
}
