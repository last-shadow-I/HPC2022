__global__ void addKernel(float* c, float* a, float* b, unsigned int size) {
	int count_threads = gridDim.x * blockDim.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = index; i < size; i+= count_threads){
		c[i] = a[i] + b[i];
	}
}
#include "mainGPU.h"