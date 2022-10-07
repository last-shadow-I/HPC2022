//#include <cublas_v2.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h> //clock_gettime

//Сложение векторов последовательное. C = A+B, где размеры векторов ­- 1*n
void SumVect (float* c, float* a, float* b, unsigned int n){
	for(int i = 0; i < n; i++){
		c[i] = a[i]+b[i];
	}
}

int main(int argc, char* argv[])
{
    int n = atoi(argv[1]);
	int BLOCK_SIZE = atoi(argv[2]);
	int GRID_SIZE = atoi(argv[3]);
	bool autoFill = atoi(argv[4]); // флаг автоматического заполнения
	int borderOut = atoi(argv[5]); // Сколько элементов выводить
	int nb = n * sizeof(float);

    //Определяем размер грида и блоков
    BLOCK_SIZE = (BLOCK_SIZE > 0) ? BLOCK_SIZE : 1;
    GRID_SIZE = (GRID_SIZE > 0) ? GRID_SIZE : 1;

	printf("n = %d\n", n);
	printf("BLOCK_SIZE: %d GRID_SIZE: %d CountThreads: %d\n", BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE*GRID_SIZE);
    printf("\n");

    // Выделение памяти на хосте-CPU
    float* a = (float*)calloc(n, sizeof(float));
    float* b = (float*)calloc(n, sizeof(float));
    float* c = (float*)calloc(n, sizeof(float));

	float* cCpu = (float*)calloc(n, sizeof(float));

    // Инициализация массивов
	if (autoFill) {
		srand(time(NULL));
    	for (int i = 0; i < n; i++) {
        	a[i] = (float)(rand() % 100) / (float)(rand() % 100);
			b[i] = (float)(rand() % 100) / (float)(rand() % 100);
    	}
	}
    else {
		for (int i = 0; i < n; i++) {
	   		scanf("%f", &a[i]); 
    	}
    	for (int i = 0; i < n; i++) {
	  	  	scanf("%f", &b[i]);
    	}
    	printf("\n");
	}

    // Выделение памяти на устройстве
    float* adev = NULL;
    cudaError_t cuerr = cudaMalloc((void**)&adev, nb);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for a: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    float* bdev = NULL;
    cuerr = cudaMalloc((void**)&bdev, nb);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for b: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    float* cdev = NULL;
    cuerr = cudaMalloc((void**)&cdev, nb);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for c: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Создание обработчиков событий
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cuerr = cudaEventCreate(&start);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot create CUDA start event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    cuerr = cudaEventCreate(&stop);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot create CUDA end event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Копирование данных с хоста на девайс
    cuerr = cudaMemcpy(adev, a, nb, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy a array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    cuerr = cudaMemcpy(bdev, b, nb, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy b array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Установка точки старта
    cuerr = cudaEventRecord(start, 0);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot record CUDA event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    //Запуск ядра
    addKernel <<< GRID_SIZE, BLOCK_SIZE >>> (cdev, adev, bdev, n);

    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    // Синхронизация устройств
    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Установка точки окончания
    cuerr = cudaEventRecord(stop, 0);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Копирование результата на хост
    cuerr = cudaMemcpy(c, cdev, nb, cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

	struct timespec mt1, mt2; 
    //Переменная для расчета дельты времени
    long int tt;

	clock_gettime(CLOCK_REALTIME, &mt1);
    SumVect(cCpu, a, b, n);
    clock_gettime(CLOCK_REALTIME, &mt2);
    
    for (int i = 0; i < ((n<borderOut ) ? n :borderOut); i++) {
        printf("[%d] a: %f b: %f cGpu: %f cCpu: %f\n", i, a[i], b[i], c[i], cCpu[i]);
    }
	printf("\n");

    // Расчет времени
	tt=1000000000*(mt2.tv_sec - mt1.tv_sec)+(mt2.tv_nsec - mt1.tv_nsec);
	printf("time spent executing cpu: %ld nanoseconds\n",tt);
    cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
    printf("time spent executing %s: %.9f seconds\n", "kernel", gpuTime / 1000);
	printf("\n");

	float maxEl = 0;
	for(int i = 0; i< i; i++) {
		cCpu[i] -= c[i];
		if(abs(cCpu[i]) > maxEl) maxEl = abs(cCpu[i]);
	}
	printf("maximum error: %.9f \n", maxEl);
	printf("SpeedUp: %.9f \n", tt/(gpuTime*1000000));
	printf("\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
    free(a);
    free(b);
    free(c);
	free(cCpu);
    return 0;
}
