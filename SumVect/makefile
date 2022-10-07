# The size of shared memory block size
NVCC = nvcc 
CFLAGS = -g -G -O0 
Add: addGPU.cu mainGPU.h
	$(NVCC) $(CFLAGS) $< -o $@