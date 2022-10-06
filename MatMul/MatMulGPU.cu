__global__ void MatMulKer(float* c, float* a, float* b, unsigned int m, unsigned int n, unsigned int k) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((index_x < m) && (index_y < k))
	{
		for (int i = 0; i < n; i++)
		{
			c[index_x*k+index_y]+=a[index_x*n+i]*b[i*k+index_y];
		}
	}
}
#include "mainGPU.h"