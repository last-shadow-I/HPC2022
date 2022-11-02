
CFLAGS = -g -G -O3 

start: EasyBMP.o startGPU.o
	nvcc $(CFLAGS) EasyBMP.o startGPU.o -o BilFiltr

EasyBMP.o: EasyBMP.cpp EasyBMP*.h
	nvcc $(CFLAGS) -c EasyBMP.cpp

startGPU.o: startGPU.cu 
	nvcc -c startGPU.cu 

clean: 
	rm EasyBMP*.h
	rm EasyBMP.cpp
	rm EasyBMPtest*
	rm EasyBMPoutput*.bmp
	rm -f *.o
