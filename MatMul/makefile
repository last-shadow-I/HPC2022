# The size of shared memory block size
NVCC = nvcc 
CFLAGS = -g -G -O0 
MatMul: MatMulGPU.cu mainGPU.h
	$(NVCC) $(CFLAGS) $< -o $@